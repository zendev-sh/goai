//go:build ignore

// Example: realtime voice session with tool calling.
//
// Opens a Gemini Live session, sends either a text prompt or a recorded audio
// file, writes the spoken reply to a raw PCM file, and prints transcripts. The
// model can call a local tool; the session runtime executes it and returns the
// result automatically.
//
// Audio is raw PCM16 mono: 16 kHz in, 24 kHz out. To hear the reply:
//
//	ffplay -f s16le -ar 24000 -ac 1 reply.pcm
//
// To record a prompt:
//
//	ffmpeg -i question.wav -f s16le -ar 16000 -ac 1 question.pcm
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run ./examples/realtime-voice/main.go -text "What time is it in Tokyo?"
//	go run ./examples/realtime-voice/main.go -in question.pcm
//
// On Vertex AI (Application Default Credentials):
//
//	export GOOGLE_CLOUD_PROJECT=my-project
//	go run ./examples/realtime-voice/main.go -backend vertex -text "Hello"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/google"
	"github.com/zendev-sh/goai/provider/vertex"
)

func main() {
	backend := flag.String("backend", "gemini", "gemini (AI Studio) or vertex")
	modelID := flag.String("model", "gemini-3.8-live", "Live model ID")
	text := flag.String("text", "", "text prompt to send")
	in := flag.String("in", "", "16 kHz mono PCM16 file to send as speech")
	out := flag.String("out", "reply.pcm", "file for the 24 kHz mono PCM16 reply")
	flag.Parse()
	if (*text == "") == (*in == "") {
		log.Fatal("pass exactly one of -text or -in")
	}

	var model provider.RealtimeModel
	switch *backend {
	case "gemini":
		model = google.Live(*modelID) // reads GEMINI_API_KEY
	case "vertex":
		model = vertex.Live(*modelID) // reads GOOGLE_CLOUD_PROJECT and ADC
	default:
		log.Fatalf("unknown backend %q", *backend)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	session, err := goai.OpenRealtimeSession(ctx, model,
		goai.WithRealtimeSystem("You are a concise voice assistant."),
		goai.WithRealtimeTools(clockTool()),
		goai.WithRealtimeInputTranscription(provider.TranscriptionConfig{Enabled: true}),
		goai.WithRealtimeOutputTranscription(provider.TranscriptionConfig{Enabled: true}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	reply, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	defer reply.Close()

	// Send input concurrently: a real app streams microphone frames here.
	go func() {
		var err error
		if *text != "" {
			err = session.SendText(ctx, *text)
		} else {
			err = streamFile(ctx, session, *in)
		}
		if err != nil {
			log.Printf("send: %v", err)
		}
	}()

	for ev := range session.Events() {
		switch ev.Type {
		case provider.RealtimeOutputAudio:
			if _, err := reply.Write(ev.Audio); err != nil {
				log.Fatal(err)
			}
		case provider.RealtimeInputTranscript, provider.RealtimeOutputTranscript:
			if ev.IsFinal {
				fmt.Printf("%s: %s\n", ev.Type, ev.Text)
			}
		case provider.RealtimeToolCallEvent:
			fmt.Printf("tool call: %s %s\n", ev.ToolCall.Name, ev.ToolCall.Input)
		case provider.RealtimeResponseFinished:
			if ev.FinishReason == provider.FinishToolCalls {
				continue // the model speaks again once the tool result arrives
			}
			fmt.Printf("reply written to %s\n", *out)
			return
		}
	}
	if err := session.Err(); err != nil {
		log.Fatal(err)
	}
}

// streamFile sends a PCM file in 100 ms frames at real-time pace, then a second
// of silence so the provider's voice activity detection ends the turn.
func streamFile(ctx context.Context, session *goai.RealtimeSession, path string) error {
	audio, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	const frame = 16_000 * 2 / 10 // 100 ms of 16 kHz PCM16
	audio = append(audio, make([]byte, 10*frame)...)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for start := 0; start < len(audio); start += frame {
		end := min(start+frame, len(audio))
		if err := session.InputAudio(ctx, audio[start:end]); err != nil {
			return err
		}
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func clockTool() goai.Tool {
	return goai.Tool{
		Name:        "current_time",
		Description: "Returns the current time in an IANA time zone, such as Asia/Tokyo.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct {
				Zone string `json:"zone"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return "", err
			}
			loc, err := time.LoadLocation(args.Zone)
			if err != nil {
				return "", fmt.Errorf("unknown time zone %q", args.Zone)
			}
			return time.Now().In(loc).Format("15:04 on Monday"), nil
		},
	}
}
