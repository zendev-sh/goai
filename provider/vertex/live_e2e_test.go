//go:build e2e

package vertex

import (
	"context"
	"os"
	"testing"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// TestE2E_VertexLive needs GOOGLE_CLOUD_PROJECT and Application Default
// Credentials. GOOGLE_CLOUD_LOCATION and GEMINI_LIVE_MODEL are optional.
func TestE2E_VertexLive(t *testing.T) {
	if os.Getenv("GOOGLE_CLOUD_PROJECT") == "" {
		t.Skip("GOOGLE_CLOUD_PROJECT not set")
	}
	model := os.Getenv("GEMINI_LIVE_MODEL")
	if model == "" {
		model = "gemini-3.8-live"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()

	session, err := goai.OpenRealtimeSession(ctx, Live(model),
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
	for ev := range session.Events() {
		switch ev.Type {
		case provider.RealtimeOutputAudio:
			audioBytes += len(ev.Audio)
		case provider.RealtimeResponseFinished:
			if audioBytes == 0 {
				t.Fatal("no audio received")
			}
			t.Logf("finished with %d audio bytes", audioBytes)
			return
		}
	}
	t.Fatalf("session ended early: %v", session.Err())
}
