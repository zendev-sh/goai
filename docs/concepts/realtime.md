---
title: Realtime Voice
description: "Open bidirectional voice sessions with GoAI: stream audio in and out, handle interruptions, and let the model call tools mid-conversation with Gemini Live on the Gemini API or Vertex AI."
---

# Realtime Voice

A realtime session is a long-lived, bidirectional connection to a voice model. Audio flows in both directions at once, the user can interrupt the model, and the model can call tools in the middle of the conversation. It is a different shape from `GenerateText` and `StreamText`, which send one request and read one response, so GoAI models it as its own family: `provider.RealtimeModel`.

> **Experimental.** The realtime API may change incompatibly while more providers are added.

| Provider | Constructor | Backend |
| --- | --- | --- |
| Gemini Live | `google.Live("gemini-3.8-live")` | Gemini API |
| Gemini Live | `google.Live(id, google.WithVertex(project, location))` or `vertex.Live(id)` | Vertex AI |

Sessions connect from your Go process to the provider over WebSocket. Capturing and playing audio, and how your own clients reach your backend, are up to your application.

## Quick Start

```go
model := google.Live("gemini-3.8-live", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

session, err := goai.OpenRealtimeSession(ctx, model,
    goai.WithRealtimeSystem("You are a concise voice assistant."),
    goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
)
if err != nil {
    return err
}
defer session.Close()

// Send microphone frames from another goroutine.
go func() {
    for frame := range microphone { // 16 kHz mono PCM16
        if err := session.InputAudio(ctx, frame); err != nil {
            return
        }
    }
}()

for ev := range session.Events() {
    switch ev.Type {
    case provider.RealtimeOutputAudio:
        speaker.Play(ev.Audio) // 24 kHz mono PCM16, see ev.AudioFormat
    case provider.RealtimeInputSpeechStarted:
        speaker.Stop() // the user started talking: barge-in
    case provider.RealtimeOutputTranscript:
        if ev.IsFinal {
            fmt.Println("assistant:", ev.Text)
        }
    }
}
if err := session.Err(); err != nil {
    return err
}
```

See [`examples/realtime-voice`](https://github.com/zendev-sh/goai/tree/main/examples/realtime-voice) for a runnable program.

## Two Layers

- **`provider.RealtimeModel`** opens a raw `provider.RealtimeConnection`: send normalized client events, read normalized provider events. Use it when you want full control.
- **`goai.OpenRealtimeSession`** wraps a connection with lifecycle management, a bounded event queue, hooks, and optional automatic tool execution. Most applications want this layer.

## Audio

Audio travels as raw bytes in the format the session negotiated. GoAI never transcodes: a format the model does not support fails with `provider.ErrUnsupportedAudioFormat` when the session opens.

```go
caps := provider.RealtimeCapabilitiesOf(model)
fmt.Println(caps.SupportedInputAudio, caps.SupportedOutputAudio)
```

Gemini Live takes 16 kHz mono PCM16 and returns 24 kHz mono PCM16. Each `RealtimeOutputAudio` event carries its `AudioFormat`, so playback never has to guess the sample rate.

## Turns and Interruptions

`WithRealtimeTurnDetection` selects who decides when the user's turn ends:

| Mode | Behavior |
| --- | --- |
| `TurnModeProviderVAD` (default) | The provider detects speech and responds when the user stops. |
| `TurnModeClientVAD` | Your app signals speech with `StartInputActivity` / `EndInputActivity`. |
| `TurnModeManual` | Your app commits input and asks for a response explicitly (provider permitting). |

The zero value of `provider.TurnDetection` keeps provider defaults. Set `DisableInterruption` to keep the model talking when the user starts to speak, or `DisableAutoResponse` to answer only on request.

When the user interrupts, stop local playback immediately; do not wait for the network. The response then finishes with `FinishReason == provider.FinishInterrupted`. Providers that support truncation accept a `provider.PlaybackCursor` in `Interrupt` so the conversation history matches what the user actually heard.

Operations a model does not support return `provider.ErrRealtimeUnsupported`; nothing is emulated. Check `RealtimeCapabilities` to adapt your UI.

## Tools

Tools registered with `WithRealtimeTools` are executed automatically when the model calls them, and their results go back into the same session:

```go
session, err := goai.OpenRealtimeSession(ctx, model,
    goai.WithRealtimeTools(weatherTool),
    goai.WithRealtimeToolTimeout(10*time.Second),
    goai.WithRealtimeMaxParallelTools(2),
    goai.WithRealtimeOnBeforeToolExecute(func(info goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
        if info.ToolName == "send_email" && !userConfirmed() {
            return goai.BeforeToolExecuteResult{Skip: true, Result: "The user declined."}
        }
        return goai.BeforeToolExecuteResult{}
    }),
)
```

- Tools run on their own goroutines, so a slow tool never blocks audio.
- When the provider cancels a call (for example after an interruption), the tool's context is cancelled and its result is dropped.
- Results are sent as JSON. Text that is not JSON is wrapped as `{"text": "..."}`; errors as `{"error": "..."}`. Keep secrets and internal details out of both.
- The model decides which tool to call. Authorization and human confirmation belong in `WithRealtimeOnBeforeToolExecute`.

Gemini 3.8 Live runs functions without blocking the conversation by default. Choose per tool with `google.LiveToolOptions`:

```go
tool.ProviderDefinedOptions = google.LiveToolOptions(google.LiveBlocking, "")       // wait for the result
tool.ProviderDefinedOptions = google.LiveToolOptions(google.LiveNonBlocking, google.LiveSilent) // add it quietly
```

## Event Reference

| Event | Meaning |
| --- | --- |
| `RealtimeSessionReady` | Setup confirmed; the session accepts input. |
| `RealtimeInputSpeechStarted` / `Stopped` | Voice activity detected by the provider. |
| `RealtimeInputTranscript` / `RealtimeOutputTranscript` | Transcripts. `IsFinal == false` carries a delta; `IsFinal == true` carries the full turn. |
| `RealtimeResponseStarted` / `RealtimeResponseFinished` | A model response begins / ends, with its `FinishReason`. |
| `RealtimeOutputAudio` / `RealtimeTextDelta` | Model output. |
| `RealtimeToolCallEvent` / `RealtimeToolCallCancelled` | Tool calls and cancellations. |
| `RealtimeUsageUpdated` | Token usage reported during the session. |
| `RealtimeSessionGoAway` | The provider will close the connection after `TimeToClose`. |
| `RealtimeSessionResumeState` | A `ResumeToken` that can resume the session on a new connection. |

## Backpressure and Lifecycle

- `Events()` is a bounded queue (`WithRealtimeEventBuffer`). If your consumer stops draining it for longer than `WithRealtimeBackpressureTimeout` (one second by default), the session ends with `provider.ErrRealtimeBackpressure` instead of piling up stale audio.
- `Close` is idempotent and waits for internal goroutines, including running tools. After `Events()` closes, `Err()` reports why; it is nil after a normal close.
- The context passed to `OpenRealtimeSession` bounds the whole session.
- GoAI never reconnects automatically: replaying a conversation could repeat tool side effects. Gemini sends `RealtimeSessionGoAway` before its connection limit and, when enabled, `RealtimeSessionResumeState` tokens; reconnecting with a token is an explicit application decision:

```go
goai.WithRealtimeProviderOptions(map[string]any{"google": map[string]any{
    "sessionResumption":        map[string]any{"handle": lastResumeToken},
    "contextWindowCompression": map[string]any{"slidingWindow": map[string]any{}},
}})
```

## Security

Keep API keys and cloud credentials on your backend. Treat transcripts and tool arguments as user data: decide what you store and for how long. Tool results are sent to the provider, so return only what the model needs.
