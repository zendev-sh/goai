package ollama_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/ollama"
)

func TestChatNDJSONErrors(t *testing.T) {
	const wireError = `{"error":"fixture runner crashed"}`
	const text = `{"message":{"role":"assistant","content":"partial"},"done":false}`
	const reasoning = `{"message":{"role":"assistant","thinking":"thinking"},"done":false}`
	const tool = `{"message":{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{}}}]},"done":false}`
	const done = `{"message":{"role":"assistant","content":"later"},"done":true,"done_reason":"stop"}`
	cases := []struct {
		name   string
		body   string
		before []provider.StreamChunkType
	}{
		{"error only", wireError, nil},
		{"after text", text + "\n" + wireError, []provider.StreamChunkType{provider.ChunkText}},
		{"after reasoning", reasoning + "\n" + wireError, []provider.StreamChunkType{provider.ChunkReasoning}},
		{"after tool", tool + "\n" + wireError, []provider.StreamChunkType{provider.ChunkToolCallStreamStart, provider.ChunkToolCall}},
		{"before later data", wireError + "\n" + done, nil},
		{"same frame content and finish", `{"error":"fixture runner crashed","message":{"content":"must not leak","thinking":"nor this","tool_calls":[{"function":{"name":"f","arguments":{}}}]},"done":true}`, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/x-ndjson")
				_, _ = fmt.Fprintln(w, tc.body)
			}))
			defer server.Close()
			model := ollama.Chat("fixture", ollama.WithBaseURL(server.URL))
			result, err := model.DoGenerate(t.Context(), provider.GenerateParams{})
			if result != nil || err == nil || err.Error() != "ollama: fixture runner crashed" {
				t.Fatalf("DoGenerate = (%+v, %v), want nil and provider error", result, err)
			}
			stream, err := model.DoStream(t.Context(), provider.GenerateParams{})
			if err != nil {
				t.Fatal(err)
			}
			var types []provider.StreamChunkType
			for chunk := range stream.Stream {
				types = append(types, chunk.Type)
				if chunk.Type == provider.ChunkError && (chunk.Error == nil || chunk.Error.Error() != "ollama: fixture runner crashed") {
					t.Fatalf("stream error = %v", chunk.Error)
				}
			}
			want := append(append([]provider.StreamChunkType{}, tc.before...), provider.ChunkError)
			if !reflect.DeepEqual(types, want) {
				t.Fatalf("chunk types = %v, want %v", types, want)
			}
		})
	}
}

func TestStreamTextReportsOllamaNDJSONError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, `{"message":{"role":"assistant","content":"partial"},"done":false}`)
		_, _ = fmt.Fprintln(w, `{"error":"runner failed"}`)
		_, _ = fmt.Fprintln(w, `{"message":{"content":"must not leak"},"done":true}`)
	}))
	defer server.Close()
	model := ollama.Chat("fixture", ollama.WithBaseURL(server.URL))
	stream, err := goai.StreamText(t.Context(), model, goai.WithPrompt("fixture"), goai.WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for part := range stream.TextStream() {
		text.WriteString(part)
	}
	if text.String() != "partial" {
		t.Fatalf("text = %q, want partial", text.String())
	}
	if stream.Err() == nil || stream.Err().Error() != "ollama: runner failed" {
		t.Fatalf("Err = %v, want provider error", stream.Err())
	}
}
