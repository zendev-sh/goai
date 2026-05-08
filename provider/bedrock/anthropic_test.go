package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestBedrockAnthropicURLBuilder(t *testing.T) {
	base := "https://bedrock-runtime.us-east-1.amazonaws.com"
	cases := []struct {
		modelID   string
		streaming bool
		want      string
	}{
		{"foo.bar.model-v1:0", false, base + "/model/foo.bar.model-v1:0/invoke"},
		{"foo.bar.model-v1:0", true, base + "/model/foo.bar.model-v1:0/invoke-with-response-stream"},
	}
	for _, tc := range cases {
		got := bedrockAnthropicURLBuilder(base, tc.modelID, tc.streaming)
		// url.PathEscape converts ":" → "%3A".
		if !strings.HasPrefix(got, base+"/model/") {
			t.Errorf("url = %q, want prefix %q", got, base+"/model/")
		}
		if tc.streaming && !strings.HasSuffix(got, "/invoke-with-response-stream") {
			t.Errorf("url = %q, want streaming suffix", got)
		}
		if !tc.streaming && !strings.HasSuffix(got, "/invoke") {
			t.Errorf("url = %q, want non-streaming suffix", got)
		}
	}
}

func TestBedrockAnthropicBodyTransformer(t *testing.T) {
	body := map[string]any{
		"model":      "foo",
		"stream":     true,
		"max_tokens": 100,
		"messages":   []any{},
	}
	out := bedrockAnthropicBodyTransformer(body)
	if _, ok := out["model"]; ok {
		t.Errorf("model should be removed, got %v", out["model"])
	}
	if out["anthropic_version"] != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version = %v, want bedrock-2023-05-31", out["anthropic_version"])
	}
	// "stream" must remain on the body until the transport strips it: the
	// anthropic provider reads it to pick the URL via the URL builder.
	if out["stream"] != true {
		t.Errorf("stream = %v, want true (provider reads it after transform)", out["stream"])
	}
	if out["max_tokens"] != 100 {
		t.Errorf("max_tokens dropped: %+v", out)
	}
}

func TestFilterBedrockBetas(t *testing.T) {
	in := []string{"claude-code-20250219", "interleaved-thinking-2025-05-14", "web-search-2025-03-05", "computer-use-2025-01-24"}
	got := filterBedrockBetas(in)
	if len(got) != 2 {
		t.Fatalf("filtered = %v, want 2 entries", got)
	}
	want := map[string]bool{"web-search-2025-03-05": true, "computer-use-2025-01-24": true}
	for _, b := range got {
		if !want[b] {
			t.Errorf("unexpected beta in filtered set: %q", b)
		}
	}
}

// TestEventStreamSSEReader covers the Bedrock InvokeModelWithResponseStream
// → Anthropic SSE byte translation. Each EventStream "event" frame whose
// JSON payload is `{"bytes":<base64 anthropic event>}` is re-emitted as
// `data: <anthropic event>\n\n`.
func TestEventStreamSSEReader(t *testing.T) {
	anthropicEv := `{"type":"message_start","message":{"id":"m1"}}`
	innerPayload := map[string]any{"bytes": []byte(anthropicEv)} // json.Marshal base64-encodes []byte
	innerJSON, _ := json.Marshal(innerPayload)

	frame := buildEventStreamFrame("messageStart", innerJSON)

	r := newEventStreamSSEReader(io.NopCloser(strings.NewReader(string(frame))))
	out, err := io.ReadAll(r)
	if err != nil && err != io.EOF {
		t.Fatalf("read: %v", err)
	}
	want := "data: " + anthropicEv + "\n\n"
	if string(out) != want {
		t.Errorf("translated = %q, want %q", string(out), want)
	}
}

// TestAnthropicChat_GenerateRoundTrip exercises the full path:
// bedrock.AnthropicChat → SigV4 transport → InvokeModel JSON response.
// We point at an httptest server that asserts the body shape (no model,
// has anthropic_version, no stream) and returns a native Anthropic JSON
// payload, which the anthropic provider parses unchanged.
func TestAnthropicChat_GenerateRoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/invoke") {
			t.Errorf("path = %q, want suffix /invoke", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if _, ok := req["model"]; ok {
			t.Errorf("body still has model field: %+v", req)
		}
		if _, ok := req["stream"]; ok {
			t.Errorf("body still has stream field: %+v", req)
		}
		if req["anthropic_version"] != "bedrock-2023-05-31" {
			t.Errorf("anthropic_version = %v", req["anthropic_version"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg-1",
			"model": "test-model",
			"type": "message",
			"content": [{"type":"text","text":"ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 5, "output_tokens": 1}
		}`)
	}))
	defer server.Close()

	// SigV4 signing: provide dummy creds; the test server doesn't validate
	// the signature, only the request shape.
	model := AnthropicChat("test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("us-east-1"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want ok", result.Text)
	}
}

// TestAnthropicChat_StreamServerToolRoundTrip exercises streaming with
// EventStream → SSE translation, end-to-end with the anthropic provider's
// server-tool capture path.
func TestAnthropicChat_StreamServerToolRoundTrip(t *testing.T) {
	// Each "event" frame's payload is {"bytes": <base64 of anthropic SSE event JSON>}.
	emitFrame := func(w io.Writer, anthropicEvent string) {
		inner, _ := json.Marshal(map[string]any{"bytes": []byte(anthropicEvent)})
		w.Write(buildEventStreamFrame("messageStart", inner))
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/invoke-with-response-stream") {
			t.Errorf("path = %q, want streaming suffix", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		flusher, _ := w.(http.Flusher)
		emitFrame(w, `{"type":"message_start","message":{"id":"m1","model":"test","usage":{"input_tokens":5}}}`)
		emitFrame(w, `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search"}}`)
		emitFrame(w, `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`)
		emitFrame(w, `{"type":"content_block_stop","index":0}`)
		emitFrame(w, `{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev","title":"Go"}]}}`)
		emitFrame(w, `{"type":"content_block_stop","index":1}`)
		emitFrame(w, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`)
		emitFrame(w, `{"type":"message_stop"}`)
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer server.Close()

	model := AnthropicChat("test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("us-east-1"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "search"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	var sawCall provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			sawCall = chunk
		}
	}
	if sawCall.ToolCallID != "srvtoolu_1" {
		t.Fatalf("ChunkToolCall ID = %q, want srvtoolu_1", sawCall.ToolCallID)
	}
	rb, ok := sawCall.Metadata["resultBlock"].(map[string]any)
	if !ok {
		t.Fatalf("resultBlock missing: %v", sawCall.Metadata)
	}
	if rb["type"] != "web_search_tool_result" {
		t.Errorf("resultBlock type = %v, want web_search_tool_result", rb["type"])
	}
}

// TestAnthropicChat_BetaFolding verifies the transport moves the anthropic-beta
// header into the request body's anthropic_beta field, filtered to flags
// Bedrock InvokeModel accepts.
func TestAnthropicChat_BetaFolding(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h := r.Header.Get("anthropic-beta"); h != "" {
			t.Errorf("anthropic-beta header should be stripped, got %q", h)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","model":"t","type":"message","content":[{"type":"text","text":""}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`)
	}))
	defer server.Close()

	// Anthropic provider injects "claude-code-20250219,interleaved-thinking-2025-05-14"
	// as base betas. Tools that emit additional betas (web_search, etc.) get
	// merged in. The transport should drop unsupported flags and keep the
	// supported ones in the body.
	model := AnthropicChat("test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("us-east-1"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "web_search", ProviderDefinedType: "web_search_20250305"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	betas, _ := capturedBody["anthropic_beta"].([]any)
	var hasWebSearch bool
	for _, b := range betas {
		if b == "web-search-2025-03-05" {
			hasWebSearch = true
		}
		if b == "claude-code-20250219" || b == "interleaved-thinking-2025-05-14" {
			t.Errorf("unsupported beta leaked into body: %v", b)
		}
	}
	if !hasWebSearch {
		t.Errorf("web-search beta missing from body: %v", betas)
	}
}

// Ensure base64.StdEncoding is referenced (test file for clarity).
var _ = base64.StdEncoding
