---
title: Realtime
description: "API reference for GoAI realtime voice sessions: OpenRealtimeSession, RealtimeSession, realtime options, and the provider.RealtimeModel contract."
---

# Realtime

Reference for realtime voice sessions. For concepts, audio formats, and turn handling, see [Realtime Voice](../concepts/realtime.md).

> **Experimental.** The realtime API may change incompatibly while more providers are added.

## OpenRealtimeSession

```go
func OpenRealtimeSession(ctx context.Context, model provider.RealtimeModel, opts ...RealtimeOption) (*RealtimeSession, error)
```

Opens a provider connection, sends the initial configuration, and starts the event loop. `ctx` bounds the whole session. Invalid options fail with `provider.ErrInvalidRealtimeConfig`; unsupported audio formats with `provider.ErrUnsupportedAudioFormat`; unsupported behavior with `provider.ErrRealtimeUnsupported`.

## RealtimeSession

All methods are safe for concurrent use. Input buffers are copied before sending.

| Method | Description |
| --- | --- |
| `Events() <-chan provider.RealtimeEvent` | Normalized events in provider order. Closed when the session ends. |
| `InputAudio(ctx, audio []byte) error` | Appends audio in the negotiated input format. |
| `SendText(ctx, text string) error` | Sends text into the conversation. |
| `StartInputActivity(ctx) error` / `EndInputActivity(ctx) error` | Marks user speech in `TurnModeClientVAD`. |
| `CommitInput(ctx) error` | Ends the buffered input turn in `TurnModeManual`. |
| `RequestResponse(ctx) error` | Asks for a response explicitly. |
| `Interrupt(ctx, cursor *provider.PlaybackCursor) error` | Cancels the active response; `cursor` reports how much audio was actually played. |
| `Close() error` | Ends the session and waits for internal goroutines. Idempotent. |
| `Err() error` | Terminal error after `Events()` closes; nil after a normal close. |

After `Close` or a terminal error, sending returns `provider.ErrRealtimeSessionClosed`.

## Options

| Option | Description |
| --- | --- |
| `WithRealtimeSystem(string)` | System instructions. |
| `WithRealtimeTools(...Tool)` | Tools offered to the model; those with `Execute` run automatically. |
| `WithRealtimeToolChoice(string)` | Tool choice, where the provider supports it. |
| `WithRealtimeAudio(input, output provider.AudioFormat)` | Audio formats; zero values use the model defaults. |
| `WithRealtimeVoice(string)` | Provider voice name. |
| `WithRealtimeTurnDetection(provider.TurnDetection)` | Turn mode and interruption behavior. |
| `WithRealtimeInputTranscription(provider.TranscriptionConfig)` | Transcribe user audio. |
| `WithRealtimeOutputTranscription(provider.TranscriptionConfig)` | Transcribe model audio. |
| `WithRealtimeMaxOutputTokens(int)` | Response token limit. |
| `WithRealtimeProviderOptions(map[string]any)` | Provider-specific session fields. |
| `WithRealtimeEventBuffer(int)` | Size of the event queue (default 64). |
| `WithRealtimeBackpressureTimeout(time.Duration)` | How long a full queue may wait before failing (default 1s). |
| `WithRealtimeToolTimeout(time.Duration)` | Per-tool timeout (default 30s; ≤ 0 disables). |
| `WithRealtimeMaxParallelTools(int)` | Concurrent tool limit (default 4). |
| `WithRealtimeSequentialTools(bool)` | Run tools one at a time. |
| `WithRealtimeOnEvent(func(provider.RealtimeEvent))` | Observe every event before it is queued. |
| `WithRealtimeOnToolCallStart`, `WithRealtimeOnToolCall` | Tool observability hooks. |
| `WithRealtimeOnBeforeToolExecute`, `WithRealtimeOnAfterToolExecute` | Tool interceptor hooks (authorization, redaction). |
| `WithRealtimeOnPanic(func(PanicInfo))` | Recovered panics from hooks and tools. |

## provider Contract

```go
type RealtimeModel interface {
    ModelID() string
    OpenRealtime(ctx context.Context, params RealtimeParams) (RealtimeConnection, error)
}

type RealtimeConnection interface {
    Events() <-chan RealtimeEvent
    Send(ctx context.Context, event RealtimeClientEvent) error
    Close() error
    Err() error
}

type RealtimeCapableModel interface {
    RealtimeCapabilities() RealtimeCapabilities
}
```

`provider.RealtimeCapabilitiesOf(model)` returns the declared capabilities, or the zero value when the model does not implement `RealtimeCapableModel`.

### Key Types

| Type | Purpose |
| --- | --- |
| `AudioFormat` | Encoding (`AudioPCM16`, `AudioG711U`, `AudioG711A`), sample rate, channels. |
| `RealtimeParams` | Initial session configuration built from the options above. |
| `TurnDetection` | `Mode`, `DisableAutoResponse`, `DisableInterruption`, provider options. |
| `TranscriptionConfig` | `Enabled`, `Language`, `Prompt`, provider options. |
| `RealtimeClientEvent` | Commands: audio, activity, text, tool result, response create/cancel. |
| `RealtimeEvent` | Events: see the [event reference](../concepts/realtime.md#event-reference). |
| `RealtimeToolCall` / `RealtimeToolResult` | Tool call from the model and its JSON result. |
| `PlaybackCursor` | Response, item, and played duration for truncation on interrupt. |
| `RealtimeCapabilities` | What a model supports: audio formats, turn modes, tools, resumption, and more. |

### Errors

| Error | When |
| --- | --- |
| `ErrRealtimeUnsupported` | The model does not support the requested operation or setting. |
| `ErrRealtimeSessionClosed` | Sending after the session closed. |
| `ErrUnsupportedAudioFormat` | The audio format is invalid or not supported by the model. |
| `ErrRealtimeBackpressure` | The consumer stopped draining events. |
| `ErrInvalidRealtimeConfig` | Inconsistent configuration. |
