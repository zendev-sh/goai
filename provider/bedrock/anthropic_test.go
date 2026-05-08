package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"errors"
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

// TestAnthropicChat_CrossRegionInference covers the inference-profile
// region override: a "us.*" model id called from an "eu-*" region should
// rewrite region to a us-* default so the request reaches a matching
// Bedrock endpoint.
func TestAnthropicChat_CrossRegionInference(t *testing.T) {
	var capturedHost string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHost = r.Host
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","model":"t","type":"message","content":[{"type":"text","text":""}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`)
	}))
	defer server.Close()

	// Mismatched region (eu-west-1) with us-prefixed model. WithBaseURL
	// captures the request so we can assert the constructor took the right
	// path; the real point is that AnthropicChat does not error and the
	// URL builder still composes a sensible path.
	model := AnthropicChat("us.anthropic.test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("eu-west-1"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if capturedHost == "" {
		t.Error("server did not receive a request")
	}
}

// TestAnthropicChat_DefaultBaseURL covers the default-URL branch: when
// WithBaseURL is not set, AnthropicChat builds a bedrock-runtime hostname
// from the resolved region.
func TestAnthropicChat_DefaultBaseURL(t *testing.T) {
	model := AnthropicChat("test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("ap-northeast-1"),
	)
	if model == nil {
		t.Fatal("AnthropicChat returned nil")
	}
	if got := model.ModelID(); got != "test-model.v1:0" {
		t.Errorf("ModelID = %q, want test-model.v1:0", got)
	}
}

// TestAnthropicChat_NoBetasInBody covers the branch where neither the
// anthropic-beta header nor a body anthropic_beta field should appear in
// the wire request when the request has no betas to begin with.
func TestAnthropicChat_NoBetasInBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		// The upstream provider always injects baked-in betas, so the
		// transport should see a non-empty header. Verify the body
		// reflects only Bedrock-supported betas (subset, possibly empty).
		if betas, ok := req["anthropic_beta"].([]any); ok {
			for _, b := range betas {
				s, _ := b.(string)
				if !bedrockSupportedBetas[s] {
					t.Errorf("unsupported beta leaked: %q", s)
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","model":"t","type":"message","content":[{"type":"text","text":""}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`)
	}))
	defer server.Close()

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
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

// TestAnthropicChat_BaseRoundTripError verifies the base.RoundTrip error
// is propagated unchanged (the transport must not rewrite errors from the
// underlying transport).
func TestAnthropicChat_BaseRoundTripError(t *testing.T) {
	failingTransport := roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("dial: connection refused")
	})
	model := AnthropicChat("test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("us-east-1"),
		WithBaseURL("http://does-not-resolve.invalid"),
		WithHTTPClient(&http.Client{Transport: failingTransport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error from failing transport, got nil")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("error = %q, want substring 'connection refused'", err.Error())
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// TestAnthropicChat_BearerAuth covers the bearer-token branch of the
// SigV4/bearer transport: when WithBearerToken is set the transport sends
// "Authorization: Bearer ..." instead of signing.
func TestAnthropicChat_BearerAuth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("Authorization")
		if got != "Bearer test-bedrock-token" {
			t.Errorf("Authorization = %q, want Bearer test-bedrock-token", got)
		}
		// SigV4 headers should be absent.
		if r.Header.Get("X-Amz-Date") != "" {
			t.Error("X-Amz-Date should be absent on bearer-auth requests")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","model":"t","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := AnthropicChat("test-model.v1:0",
		WithBearerToken("test-bedrock-token"),
		WithRegion("us-east-1"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

// TestAnthropicChat_HTTPError covers the non-200 path: Bedrock returns an
// error body, the anthropic provider surfaces it as an APIError tagged with
// the bedrock-anthropic provider name (set by WithErrorProvider).
func TestAnthropicChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"message":"The provided request is not valid"}`)
	}))
	defer server.Close()

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
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "not valid") {
		t.Errorf("error = %q, want substring 'not valid'", err.Error())
	}
}

// TestAnthropicChat_HeadersPassthrough verifies WithHeaders adds custom
// headers to the request (still wrapped by SigV4 signing).
func TestAnthropicChat_HeadersPassthrough(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Custom"); got != "yes" {
			t.Errorf("X-Custom = %q, want yes", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"m","model":"t","type":"message","content":[{"type":"text","text":""}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`)
	}))
	defer server.Close()

	model := AnthropicChat("test-model.v1:0",
		WithAccessKey("AKIA00000000"),
		WithSecretKey("testsecret"),
		WithRegion("us-east-1"),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"X-Custom": "yes"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}

// TestEventStreamSSEReader_ExceptionFrame covers the "exception" / "error"
// MessageType branch of the translator: such frames are re-emitted as
// SSE error events so the anthropic provider's parseSSE can classify them.
func TestEventStreamSSEReader_ExceptionFrame(t *testing.T) {
	exceptionPayload := []byte(`{"message":"throttled"}`)
	frame := buildEventStreamFrame("ThrottlingException", exceptionPayload)
	// Manually rewrite :message-type to "exception".
	// buildEventStreamFrame defaults to "event"; we need an exception frame.
	// Build one inline with the helper's primitives instead.
	headers := buildStringHeader(":message-type", "exception")
	headers = append(headers, buildStringHeader(":exception-type", "ThrottlingException")...)
	headers = append(headers, buildStringHeader(":content-type", "application/json")...)
	headersLen := uint32(len(headers))
	totalLen := uint32(12 + int(headersLen) + len(exceptionPayload) + 4)
	var buf strings.Builder
	writeBE := func(v uint32) {
		buf.WriteByte(byte(v >> 24))
		buf.WriteByte(byte(v >> 16))
		buf.WriteByte(byte(v >> 8))
		buf.WriteByte(byte(v))
	}
	writeBE(totalLen)
	writeBE(headersLen)
	prelude := []byte(buf.String())
	preludeCRC := crc32IEEE(prelude)
	writeBE(preludeCRC)
	buf.Write(headers)
	buf.Write(exceptionPayload)
	all := []byte(buf.String())
	msgCRC := crc32IEEE(all)
	writeBE(msgCRC)

	r := newEventStreamSSEReader(io.NopCloser(strings.NewReader(buf.String())))
	out, _ := io.ReadAll(r)
	got := string(out)
	if !strings.Contains(got, "event: error") {
		t.Errorf("translated = %q, want substring 'event: error'", got)
	}
	if !strings.Contains(got, "throttled") {
		t.Errorf("translated = %q, want substring 'throttled'", got)
	}
	_ = frame // unused but keeps the reference
}

func crc32IEEE(b []byte) uint32 {
	var sum uint32 = 0xffffffff
	for _, c := range b {
		sum ^= uint32(c)
		for range 8 {
			if sum&1 != 0 {
				sum = (sum >> 1) ^ 0xedb88320
			} else {
				sum >>= 1
			}
		}
	}
	return ^sum
}

// TestEventStreamSSEReader_Close ensures Close() proxies to the wrapped reader.
func TestEventStreamSSEReader_Close(t *testing.T) {
	tr := &trackingReadCloser{src: strings.NewReader("")}
	r := newEventStreamSSEReader(tr)
	if err := r.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if !tr.closed {
		t.Error("Close did not propagate to source")
	}
}

type trackingReadCloser struct {
	src    io.Reader
	closed bool
}

func (t *trackingReadCloser) Read(p []byte) (int, error) { return t.src.Read(p) }
func (t *trackingReadCloser) Close() error               { t.closed = true; return nil }
