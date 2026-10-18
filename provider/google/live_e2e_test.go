//go:build e2e

package google

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// liveE2EModel returns the Live model under test, skipping when no key is set.
// GEMINI_LIVE_MODEL overrides the default model ID.
func liveE2EModel(t *testing.T) provider.RealtimeModel {
	t.Helper()
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	id := os.Getenv("GEMINI_LIVE_MODEL")
	if id == "" {
		id = "gemini-3.8-live"
	}
	return Live(id)
}

func TestE2E_LiveTextToAudio(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()

	session, err := goai.OpenRealtimeSession(ctx, liveE2EModel(t),
		goai.WithRealtimeSystem("Answer in one short sentence."),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	if err := session.SendText(ctx, "Say hello."); err != nil {
		t.Fatal(err)
	}

	var audioBytes int
	var transcript string
	for ev := range session.Events() {
		switch ev.Type {
		case provider.RealtimeOutputAudio:
			audioBytes += len(ev.Audio)
		case provider.RealtimeOutputTranscript:
			if ev.IsFinal {
				transcript = ev.Text
			}
		case provider.RealtimeResponseFinished:
			t.Logf("finished (%s): %d audio bytes, transcript %q", ev.FinishReason, audioBytes, transcript)
			if audioBytes == 0 {
				t.Fatal("no audio received")
			}
			return
		}
	}
	t.Fatalf("session ended before the response finished: %v", session.Err())
}

func TestE2E_LiveToolCall(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()

	called := make(chan string, 1)
	weather := goai.Tool{
		Name:                   "get_weather",
		Description:            "Returns the current weather for a city.",
		InputSchema:            json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		ProviderDefinedOptions: LiveToolOptions(LiveBlocking, ""),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			called <- string(input)
			return `{"condition":"sunny","celsius":21}`, nil
		},
	}
	session, err := goai.OpenRealtimeSession(ctx, liveE2EModel(t),
		goai.WithRealtimeTools(weather),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	if err := session.SendText(ctx, "What is the weather in Lima right now? Use the tool."); err != nil {
		t.Fatal(err)
	}

	toolDone := false
	for ev := range session.Events() {
		switch ev.Type {
		case provider.RealtimeToolCallEvent:
			t.Logf("tool call %s %s", ev.ToolCall.Name, ev.ToolCall.Input)
		case provider.RealtimeOutputTranscript:
			if ev.IsFinal {
				t.Logf("transcript: %q", ev.Text)
			}
		case provider.RealtimeResponseFinished:
			if toolDone && ev.FinishReason != provider.FinishToolCalls {
				return
			}
		}
		select {
		case input := <-called:
			t.Logf("tool executed with %s", input)
			toolDone = true
		default:
		}
	}
	t.Fatalf("session ended early (tool executed: %v): %v", toolDone, session.Err())
}
