package bedrock

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// === EventStream test helpers ===

// buildEventStreamFrame builds an AWS EventStream binary frame for testing.
func buildEventStreamFrame(eventType string, payload []byte) []byte {
	// Build headers.
	headers := buildStringHeader(":message-type", "event")
	headers = append(headers, buildStringHeader(":event-type", eventType)...)
	headers = append(headers, buildStringHeader(":content-type", "application/json")...)

	headersLen := uint32(len(headers))
	totalLen := 4 + 4 + 4 + headersLen + uint32(len(payload)) + 4 // prelude(12) + headers + payload + msgCRC

	var buf bytes.Buffer
	// Prelude: totalLength + headersLength.
	_ = binary.Write(&buf, binary.BigEndian, totalLen)
	_ = binary.Write(&buf, binary.BigEndian, headersLen)
	// Prelude CRC.
	prelude := buf.Bytes()
	preludeCRC := crc32.ChecksumIEEE(prelude)
	_ = binary.Write(&buf, binary.BigEndian, preludeCRC)
	// Headers + payload.
	buf.Write(headers)
	buf.Write(payload)
	// Message CRC (over entire frame so far).
	msgCRC := crc32.ChecksumIEEE(buf.Bytes())
	_ = binary.Write(&buf, binary.BigEndian, msgCRC)

	return buf.Bytes()
}

// buildStringHeader builds a single EventStream string header.
func buildStringHeader(name, value string) []byte {
	var buf bytes.Buffer
	buf.WriteByte(byte(len(name)))
	buf.WriteString(name)
	buf.WriteByte(7) // string type tag
	_ = binary.Write(&buf, binary.BigEndian, uint16(len(value)))
	buf.WriteString(value)
	return buf.Bytes()
}

// converseResponse builds a Converse API JSON response.
func converseResponse(text string, stopReason string, inputTokens, outputTokens int) string {
	return fmt.Sprintf(`{
		"output": {"message": {"role": "assistant", "content": [{"text": %q}]}},
		"stopReason": %q,
		"usage": {"inputTokens": %d, "outputTokens": %d, "totalTokens": %d}
	}`, text, stopReason, inputTokens, outputTokens, inputTokens+outputTokens)
}

// === Tests ===

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/model/") {
			t.Errorf("path = %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.Path, "/converse-stream") {
			t.Errorf("path should contain /converse-stream, got %s", r.URL.Path)
		}
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256") {
			t.Errorf("auth = %s", auth)
		}
		if r.Header.Get("x-amz-date") == "" {
			t.Error("missing x-amz-date")
		}
		if r.Header.Get("x-amz-content-sha256") == "" {
			t.Error("missing x-amz-content-sha256")
		}

		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		// Text delta frame.
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"Hello"}}`)))
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}`)))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AKIAIOSFODNN7EXAMPLE"),
		WithSecretKey("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Errorf("texts = %v, want [Hello]", texts)
	}
}

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/converse") {
			t.Errorf("path should contain /converse, got %s", r.URL.Path)
		}
		if strings.Contains(r.URL.Path, "/converse-stream") {
			t.Errorf("non-streaming should use /converse, not /converse-stream")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Requestid", "test-request-id-123")
		_, _ = fmt.Fprint(w, converseResponse("Hello world", "end_turn", 10, 5))
	}))
	defer server.Close()

	model := Chat("test",
		WithAccessKey("AKIAIOSFODNN7EXAMPLE"),
		WithSecretKey("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want 'Hello world'", result.Text)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
	// Response metadata: ID from x-amzn-requestid header, Model echoed from Chat().
	if result.Response.ID != "test-request-id-123" {
		t.Errorf("Response.ID = %q, want 'test-request-id-123'", result.Response.ID)
	}
	if result.Response.Model != "test" {
		t.Errorf("Response.Model = %q, want 'test'", result.Response.Model)
	}
}

func TestChat_MissingCredentials(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	m := Chat("model")
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("model",
		WithAccessKey("AKIAIOSFODNN7EXAMPLE"),
		WithSecretKey("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithSessionToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-amz-security-token") != "session-tok" {
			t.Error("missing session token header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithSessionToken("session-tok"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithAccessKey("AK"), WithSecretKey("SK"), WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestDefaultRegion(t *testing.T) {
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "")
	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"))
	cm := model.(*chatModel)
	if cm.opts.region != "us-east-1" {
		t.Errorf("region = %q, want us-east-1", cm.opts.region)
	}
}

func TestCustomRegion(t *testing.T) {
	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithRegion("eu-west-1"))
	cm := model.(*chatModel)
	if cm.opts.region != "eu-west-1" {
		t.Errorf("region = %q, want eu-west-1", cm.opts.region)
	}
}

func TestCrossRegionInferenceProfile(t *testing.T) {
	// Cross-region inference profiles (us., eu., ap.) must be called from
	// a matching geo's endpoint. global. works from any region.
	tests := []struct {
		modelID    string
		region     string
		wantRegion string
	}{
		// US prefix + non-US region → override to us-east-1
		{"us.anthropic.claude-sonnet-4-6", "ap-northeast-1", "us-east-1"},
		// US prefix + US region → keep
		{"us.anthropic.claude-sonnet-4-6", "us-west-2", "us-west-2"},
		// EU prefix + non-EU region → override to eu-west-1
		{"eu.anthropic.claude-sonnet-4-6", "us-east-1", "eu-west-1"},
		// EU prefix + EU region → keep
		{"eu.anthropic.claude-sonnet-4-6", "eu-central-1", "eu-central-1"},
		// AP prefix + non-AP region → override to ap-southeast-1
		{"ap.anthropic.claude-sonnet-4-6", "us-east-1", "ap-southeast-1"},
		// Global prefix → keep any region
		{"global.anthropic.claude-sonnet-4-6", "ap-northeast-1", "ap-northeast-1"},
		// No prefix → keep original region
		{"anthropic.claude-sonnet-4-v1:0", "eu-west-1", "eu-west-1"},
	}
	for _, tt := range tests {
		t.Run(tt.modelID+"_"+tt.region, func(t *testing.T) {
			model := Chat(tt.modelID, WithAccessKey("AK"), WithSecretKey("SK"), WithRegion(tt.region))
			cm := model.(*chatModel)
			if cm.opts.region != tt.wantRegion {
				t.Errorf("region = %q, want %q", cm.opts.region, tt.wantRegion)
			}
		})
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("anthropic.claude-sonnet-4-v1:0", WithAccessKey("AK"), WithSecretKey("SK"))
	if model.ModelID() != "anthropic.claude-sonnet-4-v1:0" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestSHA256Hex(t *testing.T) {
	got := sha256Hex([]byte("test"))
	want := "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
	if got != want {
		t.Errorf("sha256Hex = %q, want %q", got, want)
	}
}

func TestToolCallStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		toolConfig, _ := req["toolConfig"].(map[string]any)
		tools, _ := toolConfig["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools count = %d, want 1", len(tools))
		}

		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		// contentBlockStart with toolUse.
		_, _ = w.Write(buildEventStreamFrame("contentBlockStart", []byte(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"get_weather"}}}`)))
		time.Sleep(10 * time.Millisecond)
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\""}}}`)))
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":": \"Paris\"}"}}}`)))
		// contentBlockStop.
		_, _ = w.Write(buildEventStreamFrame("contentBlockStop", []byte(`{"contentBlockIndex":0}`)))
		// messageStop and metadata.
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"tool_use"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":10,"outputTokens":20,"totalTokens":30}}`)))
	}))
	defer server.Close()

	model := Chat("test", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather in Paris"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get weather", InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string"}}}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var gotToolCall bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			gotToolCall = true
			if chunk.ToolName != "get_weather" {
				t.Errorf("ToolName = %q", chunk.ToolName)
			}
		}
	}
	if !gotToolCall {
		t.Error("no tool call chunk received")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// errReader is an io.ReadCloser that returns an error on Read.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, fmt.Errorf("forced read error") }
func (errReader) Close() error             { return nil }

type errBodyTransport struct{}

func (t *errBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       errReader{},
		Header:     make(http.Header),
	}, nil
}

func TestDoGenerate_ReadError(t *testing.T) {
	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestDoGenerate_ParseError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{invalid json`)
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected parse error")
	}
	if !strings.Contains(err.Error(), "parsing bedrock response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEnvFallback(t *testing.T) {
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
	t.Setenv("AWS_ACCESS_KEY_ID", "ENV_AK")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "ENV_SK")
	t.Setenv("AWS_SESSION_TOKEN", "ENV_ST")
	model := Chat("m")
	cm := model.(*chatModel)
	if cm.opts.region != "ap-southeast-1" {
		t.Errorf("region = %q, want ap-southeast-1", cm.opts.region)
	}
	if cm.opts.accessKey != "ENV_AK" {
		t.Errorf("accessKey = %q", cm.opts.accessKey)
	}
	if cm.opts.secretKey != "ENV_SK" {
		t.Errorf("secretKey = %q", cm.opts.secretKey)
	}
	if cm.opts.sessionToken != "ENV_ST" {
		t.Errorf("sessionToken = %q", cm.opts.sessionToken)
	}
}

type urlCapturingTransport struct {
	captured string
}

func (t *urlCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.captured = req.URL.String()
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(converseResponse("ok", "end_turn", 1, 1))),
		Header:     make(http.Header),
	}, nil
}

func TestDefaultURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithRegion("us-west-2"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-sonnet-4-v1:0/converse"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestStreamURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	// Override to return EventStream for streaming.
	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithRegion("us-west-2"),
		WithHTTPClient(&http.Client{Transport: &streamCapturingTransport{urlCapturing: transport}}),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
	expected := "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-sonnet-4-v1:0/converse-stream"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

type streamCapturingTransport struct {
	urlCapturing *urlCapturingTransport
}

func (t *streamCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.urlCapturing.captured = req.URL.String()
	// Build minimal EventStream response.
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(&buf),
		Header:     make(http.Header),
	}, nil
}

func TestRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-Header") != "from-params" {
			t.Errorf("X-Request-Header = %q", r.Header.Get("X-Request-Header"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"X-Request-Header": "from-params"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestHTTPError_DoGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

// === BF.18 Tests ===

func TestWithAdditionalModelRequestFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Additional fields should be under additionalModelRequestFields.
		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		if additional["custom_field"] != "custom_value" {
			t.Errorf("custom_field = %v, want custom_value", additional["custom_field"])
		}
		nested, ok := additional["nested_config"].(map[string]any)
		if !ok || nested["key"] != "val" {
			t.Errorf("nested_config = %v", additional["nested_config"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAdditionalModelRequestFields(map[string]any{
			"custom_field":  "custom_value",
			"nested_config": map[string]any{"key": "val"},
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithAdditionalModelRequestFields_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		if additional["extra"] != "data" {
			t.Errorf("extra = %v, want data", additional["extra"])
		}

		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"ok"}}`)))
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	}))
	defer server.Close()

	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAdditionalModelRequestFields(map[string]any{"extra": "data"}),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
}

func TestWithReasoningConfig_AnthropicEnabled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		thinking, ok := additional["thinking"].(map[string]any)
		if !ok {
			t.Fatalf("missing thinking field, body = %s", string(body))
		}
		if thinking["type"] != "enabled" {
			t.Errorf("thinking.type = %v, want enabled", thinking["type"])
		}
		budget, ok := thinking["budget_tokens"].(float64)
		if !ok || int(budget) != 8192 {
			t.Errorf("thinking.budget_tokens = %v, want 8192", thinking["budget_tokens"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithReasoningConfig(ReasoningConfig{
			Type:         ReasoningEnabled,
			BudgetTokens: 8192,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithReasoningConfig_AnthropicAdaptive(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		thinking, ok := additional["thinking"].(map[string]any)
		if !ok {
			t.Fatalf("missing thinking field, body = %s", string(body))
		}
		if thinking["type"] != "adaptive" {
			t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
		}
		if _, hasBudget := thinking["budget_tokens"]; hasBudget {
			t.Error("adaptive should not have budget_tokens")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithReasoningConfig(ReasoningConfig{
			Type: ReasoningAdaptive,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithReasoningConfig_AnthropicEffort(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		outputConfig, ok := additional["output_config"].(map[string]any)
		if !ok {
			t.Fatalf("missing output_config, body = %s", string(body))
		}
		if outputConfig["effort"] != "high" {
			t.Errorf("output_config.effort = %v, want high", outputConfig["effort"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithReasoningConfig(ReasoningConfig{
			MaxReasoningEffort: ReasoningEffortHigh,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithReasoningConfig_NonAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		rc, ok := additional["reasoningConfig"].(map[string]any)
		if !ok {
			t.Fatalf("missing reasoningConfig, body = %s", string(body))
		}
		if rc["maxReasoningEffort"] != "medium" {
			t.Errorf("maxReasoningEffort = %v, want medium", rc["maxReasoningEffort"])
		}
		if rc["type"] != "enabled" {
			t.Errorf("type = %v, want enabled", rc["type"])
		}

		// Should NOT have "thinking" field.
		if _, hasThinking := additional["thinking"]; hasThinking {
			t.Error("non-anthropic model should not have thinking field")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("amazon.nova-pro-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithReasoningConfig(ReasoningConfig{
			Type:               ReasoningEnabled,
			MaxReasoningEffort: ReasoningEffortMedium,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithReasoningConfig_DisabledNoFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Disabled type with no effort should not add additionalModelRequestFields.
		if additional, ok := req["additionalModelRequestFields"].(map[string]any); ok {
			if _, has := additional["thinking"]; has {
				t.Error("disabled should not have thinking field")
			}
			if _, has := additional["reasoningConfig"]; has {
				t.Error("disabled with no effort should not have reasoningConfig")
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithReasoningConfig(ReasoningConfig{
			Type: ReasoningDisabled,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithAnthropicBeta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// anthropicBeta now goes in the request body as anthropic_beta, not as an HTTP header.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		betas, ok := additional["anthropic_beta"].([]any)
		if !ok {
			t.Fatalf("missing anthropic_beta in additionalModelRequestFields, body = %s", string(body))
		}
		if len(betas) != 2 || betas[0] != "extended-thinking-2025-04-14" || betas[1] != "interleaved-thinking-2025-05-14" {
			t.Errorf("anthropic_beta = %v", betas)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAnthropicBeta([]string{"extended-thinking-2025-04-14", "interleaved-thinking-2025-05-14"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithAnthropicBeta_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify anthropicBeta is in request body, not header.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields")
		}
		betas, ok := additional["anthropic_beta"].([]any)
		if !ok || len(betas) != 1 || betas[0] != "pdfs-2024-09-25" {
			t.Errorf("anthropic_beta = %v, want [pdfs-2024-09-25]", betas)
		}

		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"ok"}}`)))
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAnthropicBeta([]string{"pdfs-2024-09-25"}),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
}

func TestWithAnthropicBeta_Empty(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify no anthropic_beta in request body when not configured.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if additional, ok := req["additionalModelRequestFields"].(map[string]any); ok {
			if _, hasBeta := additional["anthropic_beta"]; hasBeta {
				t.Error("expected no anthropic_beta in body")
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCombined_ReasoningAndBeta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}

		// Verify anthropic_beta is in body, not header.
		betas, ok := additional["anthropic_beta"].([]any)
		if !ok || len(betas) != 1 || betas[0] != "extended-thinking-2025-04-14" {
			t.Errorf("anthropic_beta = %v", betas)
		}

		thinking, ok := additional["thinking"].(map[string]any)
		if !ok {
			t.Fatalf("missing thinking, body = %s", string(body))
		}
		if thinking["type"] != "enabled" {
			t.Errorf("thinking.type = %v", thinking["type"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAnthropicBeta([]string{"extended-thinking-2025-04-14"}),
		WithReasoningConfig(ReasoningConfig{
			Type:         ReasoningEnabled,
			BudgetTokens: 4096,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestProviderDefinedToolBeta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}

		betas, ok := additional["anthropic_beta"].([]any)
		if !ok {
			t.Fatalf("missing anthropic_beta, additional = %v", additional)
		}
		// Should contain both user-supplied and tool-auto-detected betas.
		hasCU := false
		hasET := false
		for _, b := range betas {
			if b == "computer-use-2025-01-24" {
				hasCU = true
			}
			if b == "extended-thinking-2025-04-14" {
				hasET = true
			}
		}
		if !hasCU {
			t.Errorf("missing computer-use beta, got %v", betas)
		}
		if !hasET {
			t.Errorf("missing user-supplied beta, got %v", betas)
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Requestid", "test-123")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAnthropicBeta([]string{"extended-thinking-2025-04-14"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{
				Name:                "computer",
				ProviderDefinedType: "computer_20250124",
				ProviderDefinedOptions: map[string]any{
					"display_width_px":  1920,
					"display_height_px": 1080,
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestToolBetaForType(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"computer_20241022", "computer-use-2024-10-22"},
		{"bash_20250124", "computer-use-2025-01-24"},
		{"text_editor_20250429", "computer-use-2025-01-24"},
		{"unknown", ""},
		{"", ""},
	}
	for _, tt := range tests {
		got := toolBetaForType(tt.input)
		if got != tt.want {
			t.Errorf("toolBetaForType(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestCombined_AdditionalFieldsAndReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		additional, ok := req["additionalModelRequestFields"].(map[string]any)
		if !ok {
			t.Fatalf("missing additionalModelRequestFields, body = %s", string(body))
		}
		if additional["custom_param"] != "value" {
			t.Errorf("custom_param = %v", additional["custom_param"])
		}
		thinking, ok := additional["thinking"].(map[string]any)
		if !ok {
			t.Fatalf("missing thinking, body = %s", string(body))
		}
		if thinking["type"] != "enabled" {
			t.Errorf("thinking.type = %v", thinking["type"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("ok", "end_turn", 1, 1))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
		WithAdditionalModelRequestFields(map[string]any{"custom_param": "value"}),
		WithReasoningConfig(ReasoningConfig{
			Type:         ReasoningEnabled,
			BudgetTokens: 2048,
		}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// === EventStream Decoder Unit Tests ===

func TestEventStreamDecoder_BasicFrame(t *testing.T) {
	payload := []byte(`{"text":"hello"}`)
	frame := buildEventStreamFrame("contentBlockDelta", payload)

	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	f, err := decoder.Next()
	if err != nil {
		t.Fatal(err)
	}
	if f.MessageType != "event" {
		t.Errorf("MessageType = %q, want event", f.MessageType)
	}
	if f.EventType != "contentBlockDelta" {
		t.Errorf("EventType = %q, want contentBlockDelta", f.EventType)
	}
	if f.ContentType != "application/json" {
		t.Errorf("ContentType = %q, want application/json", f.ContentType)
	}
	if string(f.Payload) != string(payload) {
		t.Errorf("Payload = %q, want %q", f.Payload, payload)
	}
}

func TestEventStreamDecoder_MultipleFrames(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"delta":{"text":"A"}}`)))
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"delta":{"text":"B"}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))

	decoder := newEventStreamDecoder(&buf)
	count := 0
	for {
		_, err := decoder.Next()
		if err != nil {
			break
		}
		count++
	}
	if count != 3 {
		t.Errorf("frame count = %d, want 3", count)
	}
}

func TestEventStreamDecoder_EOF(t *testing.T) {
	decoder := newEventStreamDecoder(bytes.NewReader(nil))
	_, err := decoder.Next()
	if err != io.EOF {
		t.Errorf("expected io.EOF, got %v", err)
	}
}

func TestConverseRequestFormat(t *testing.T) {
	temp := 0.7
	body := buildConverseRequest(provider.GenerateParams{
		System: "You are helpful",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Temperature:     &temp,
		MaxOutputTokens: 100,
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "a test tool", InputSchema: []byte(`{"type":"object"}`)},
		},
	}, "anthropic.claude-sonnet-4-v1:0")

	// Verify system.
	sys, ok := body["system"].([]map[string]any)
	if !ok || len(sys) != 1 || sys[0]["text"] != "You are helpful" {
		t.Errorf("system = %v", body["system"])
	}

	// Verify messages.
	msgs, ok := body["messages"].([]map[string]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages = %v", body["messages"])
	}
	if msgs[0]["role"] != "user" {
		t.Errorf("role = %v", msgs[0]["role"])
	}

	// Verify inferenceConfig.
	ic, ok := body["inferenceConfig"].(map[string]any)
	if !ok {
		t.Fatalf("missing inferenceConfig")
	}
	if ic["maxTokens"] != 100 {
		t.Errorf("maxTokens = %v", ic["maxTokens"])
	}
	if ic["temperature"] != 0.7 {
		t.Errorf("temperature = %v", ic["temperature"])
	}

	// Verify toolConfig.
	tc, ok := body["toolConfig"].(map[string]any)
	if !ok {
		t.Fatalf("missing toolConfig")
	}
	tools, ok := tc["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", tc["tools"])
	}
	toolSpec, ok := tools[0]["toolSpec"].(map[string]any)
	if !ok {
		t.Fatalf("missing toolSpec")
	}
	if toolSpec["name"] != "test" {
		t.Errorf("toolSpec.name = %v", toolSpec["name"])
	}
}

func TestConvertMessages_ToolResultMerge(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "call tool"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartToolCall, ToolCallID: "c1", ToolName: "test", ToolInput: []byte(`{}`)}}},
		{Role: "tool", Content: []provider.Part{{Type: provider.PartToolResult, ToolCallID: "c1", ToolOutput: "result"}}},
	})
	// Tool result should be under "user" role (merged or separate).
	if len(msgs) != 3 {
		t.Fatalf("messages count = %d, want 3", len(msgs))
	}
	if msgs[2]["role"] != "user" {
		t.Errorf("tool result role = %v, want user", msgs[2]["role"])
	}
}

func TestParseConverseResponse_ToolUse(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [
			{"text": "I'll check"},
			{"toolUse": {"toolUseId": "tc_1", "name": "search", "input": {"query": "test"}}}
		]}},
		"stopReason": "tool_use",
		"usage": {"inputTokens": 50, "outputTokens": 30, "totalTokens": 80}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "I'll check" {
		t.Errorf("Text = %q", result.Text)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "search" {
		t.Errorf("ToolCall.Name = %q", result.ToolCalls[0].Name)
	}
	if result.ToolCalls[0].ID != "tc_1" {
		t.Errorf("ToolCall.ID = %q, want tc_1", result.ToolCalls[0].ID)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
}

func TestMapStopReason_AllCases(t *testing.T) {
	tests := []struct {
		reason string
		want   provider.FinishReason
	}{
		{"", provider.FinishStop},
		{"end_turn", provider.FinishStop},
		{"stop", provider.FinishStop},
		{"stop_sequence", provider.FinishStop},
		{"tool_use", provider.FinishToolCalls},
		{"tool-calls", provider.FinishToolCalls},
		{"max_tokens", provider.FinishLength},
		{"length", provider.FinishLength},
		{"content_filtered", provider.FinishContentFilter},
		{"content-filter", provider.FinishContentFilter},
		{"guardrail_intervened", provider.FinishContentFilter},
		{"unknown_reason", provider.FinishOther},
	}
	for _, tt := range tests {
		got := mapStopReason(tt.reason)
		if got != tt.want {
			t.Errorf("mapStopReason(%q) = %q, want %q", tt.reason, got, tt.want)
		}
	}
}

func TestConvertParts_ImageDataURL(t *testing.T) {
	parts := []provider.Part{
		{Type: provider.PartImage, URL: "data:image/jpeg;base64,/9j/abc123"},
	}
	blocks := convertParts(parts, make(map[string]int))
	if len(blocks) != 1 {
		t.Fatalf("blocks = %d, want 1", len(blocks))
	}
	img, ok := blocks[0]["image"].(map[string]any)
	if !ok {
		t.Fatal("missing image block")
	}
	if img["format"] != "jpeg" {
		t.Errorf("format = %v, want jpeg", img["format"])
	}
	src, _ := img["source"].(map[string]any)
	if src["bytes"] != "/9j/abc123" {
		t.Errorf("bytes = %v, want /9j/abc123 (prefix should be stripped)", src["bytes"])
	}
}

func TestConvertParts_ImageWithMediaType(t *testing.T) {
	parts := []provider.Part{
		{Type: provider.PartImage, URL: "data:image/png;base64,iVBOR", MediaType: "image/webp"},
	}
	blocks := convertParts(parts, make(map[string]int))
	img, _ := blocks[0]["image"].(map[string]any)
	// MediaType should take precedence over data URL format.
	if img["format"] != "webp" {
		t.Errorf("format = %v, want webp (from MediaType)", img["format"])
	}
	src, _ := img["source"].(map[string]any)
	if src["bytes"] != "iVBOR" {
		t.Errorf("bytes = %v, want iVBOR", src["bytes"])
	}
}

func TestConvertParts_Document(t *testing.T) {
	parts := []provider.Part{
		{Type: provider.PartFile, URL: "base64data", MediaType: "application/pdf", Filename: "report.pdf"},
	}
	blocks := convertParts(parts, make(map[string]int))
	if len(blocks) != 1 {
		t.Fatalf("blocks = %d, want 1", len(blocks))
	}
	doc, ok := blocks[0]["document"].(map[string]any)
	if !ok {
		t.Fatal("missing document block")
	}
	if doc["format"] != "pdf" {
		t.Errorf("format = %v, want pdf", doc["format"])
	}
	if doc["name"] != "report-pdf" {
		t.Errorf("name = %v, want report-pdf (sanitized from report.pdf)", doc["name"])
	}
}

func TestParseConverseResponse_CacheTokens(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
		"stopReason": "end_turn",
		"usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120,
		           "cacheReadInputTokens": 50, "cacheWriteInputTokens": 10}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.CacheReadTokens != 50 {
		t.Errorf("CacheReadTokens = %d, want 50", result.Usage.CacheReadTokens)
	}
	if result.Usage.CacheWriteTokens != 10 {
		t.Errorf("CacheWriteTokens = %d, want 10", result.Usage.CacheWriteTokens)
	}
}

func TestParseConverseResponse_Reasoning(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [
			{"reasoningContent": {"reasoningText": {"text": "thinking...", "signature": "sig123"}}},
			{"text": "answer"}
		]}},
		"stopReason": "end_turn",
		"usage": {"inputTokens": 10, "outputTokens": 5}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "answer" {
		t.Errorf("Text = %q, want answer", result.Text)
	}
	bedrock, ok := result.ProviderMetadata["bedrock"]
	if !ok {
		t.Fatal("missing bedrock provider metadata")
	}
	reasoning, ok := bedrock["reasoning"].([]map[string]any)
	if !ok || len(reasoning) != 1 {
		t.Fatalf("reasoning = %v", bedrock["reasoning"])
	}
	if reasoning[0]["text"] != "thinking..." {
		t.Errorf("reasoning text = %v", reasoning[0]["text"])
	}
	if reasoning[0]["signature"] != "sig123" {
		t.Errorf("reasoning signature = %v", reasoning[0]["signature"])
	}
}

// === Coverage gap tests ===

// --- #2: DoStream tool-streaming retry path ---

func TestDoStream_ToolStreamingRetry(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		if callCount == 1 {
			// First call: send an exception frame about tool streaming not supported.
			_, _ = w.Write(buildExceptionFrame("validationException", `tool streaming is not supported for this model`))
		} else {
			// Second call (retry without tools): succeed.
			_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"retried"}}`)))
			_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
			_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
		}
	}))
	defer server.Close()

	model := Chat("nvidia.nemotron", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "retried" {
		t.Errorf("texts = %v, want [retried]", texts)
	}
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2", callCount)
	}
}

// buildExceptionFrame builds an EventStream frame with message-type=exception.
func buildExceptionFrame(exceptionType string, message string) []byte {
	headers := buildStringHeader(":message-type", "exception")
	headers = append(headers, buildStringHeader(":event-type", exceptionType)...)
	headers = append(headers, buildStringHeader(":content-type", "application/json")...)

	payload := []byte(message)
	headersLen := uint32(len(headers))
	totalLen := 4 + 4 + 4 + headersLen + uint32(len(payload)) + 4

	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, totalLen)
	_ = binary.Write(&buf, binary.BigEndian, headersLen)
	prelude := buf.Bytes()
	preludeCRC := crc32.ChecksumIEEE(prelude)
	_ = binary.Write(&buf, binary.BigEndian, preludeCRC)
	buf.Write(headers)
	buf.Write(payload)
	msgCRC := crc32.ChecksumIEEE(buf.Bytes())
	_ = binary.Write(&buf, binary.BigEndian, msgCRC)
	return buf.Bytes()
}

func TestDoStream_ToolStreamingRetryError(t *testing.T) {
	// When retry itself fails, DoStream should return the error.
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
			_, _ = w.Write(buildExceptionFrame("validationException", `tool streaming is not supported for this model`))
		} else {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = fmt.Fprint(w, `{"error":{"message":"internal error"}}`)
		}
	}))
	defer server.Close()

	model := Chat("nvidia.nemotron", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err == nil {
		t.Fatal("expected error on retry")
	}
}

func TestDoStream_NonToolErrorBreak(t *testing.T) {
	// Non-tool error should break peeking, not retry.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(buildExceptionFrame("modelError", `some random model error`))
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error chunk in stream")
	}
}

func TestDoStream_CtxDoneInForwarding(t *testing.T) {
	// Test that ctx.Done in forwarding goroutine properly exits.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		// Send a text chunk, then many more to trigger ctx cancellation.
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"A"}}`)))
		for i := 0; i < 100; i++ {
			_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"B"}}`)))
		}
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(t.Context())
	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Read one chunk then cancel.
	<-result.Stream
	cancel()
	// Drain to avoid goroutine leak; count remaining chunks.
	var drained int
	for range result.Stream {
		drained++
	}
	// Server sends 1 + 100 content chunks + stop + metadata = 102 remaining after
	// reading the first one. If ctx cancellation worked, we must have drained fewer.
	const maxRemaining = 102
	if drained >= maxRemaining {
		t.Errorf("drained %d chunks after cancel - context cancellation had no effect (want < %d)", drained, maxRemaining)
	}
}

func TestDoStream_ToolCallStreamStartBreak(t *testing.T) {
	// ToolCallStreamStart should break peeking loop.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(buildEventStreamFrame("contentBlockStart", []byte(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"c1","name":"test"}}}`)))
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{}"}}}`)))
		_, _ = w.Write(buildEventStreamFrame("contentBlockStop", []byte(`{"contentBlockIndex":0}`)))
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"tool_use"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var gotToolStart bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCallStreamStart {
			gotToolStart = true
		}
	}
	if !gotToolStart {
		t.Error("expected tool call stream start chunk")
	}
}

func TestDoStream_FinishBreak(t *testing.T) {
	// ChunkFinish should break peeking loop.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	}))
	defer server.Close()

	model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
}

// --- #3: isStreamChunkToolError ---

func TestIsStreamChunkToolError(t *testing.T) {
	if isStreamChunkToolError(nil) {
		t.Error("nil should return false")
	}
	if !isStreamChunkToolError(fmt.Errorf("tool streaming is not supported")) {
		t.Error("should match tool streaming message")
	}
	if isStreamChunkToolError(fmt.Errorf("random error")) {
		t.Error("random error should return false")
	}
}

// --- #4: doRequest fallback paths ---

// apiErrorTransport returns an API error response to trigger fallbacks.
type apiErrorTransport struct {
	calls     int
	responses []apiErrorResponse
}
type apiErrorResponse struct {
	status  int
	message string
}

func (t *apiErrorTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	idx := t.calls
	t.calls++
	if idx < len(t.responses) {
		r := t.responses[idx]
		body := fmt.Sprintf(`{"message":"%s"}`, r.message)
		return &http.Response{
			StatusCode: r.status,
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     make(http.Header),
		}, nil
	}
	// Default: success.
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(converseResponse("ok", "end_turn", 1, 1))),
		Header:     make(http.Header),
	}, nil
}

func TestDoRequest_CrossRegionFallback(t *testing.T) {
	transport := &apiErrorTransport{
		responses: []apiErrorResponse{
			{400, "model identifier is invalid"},
			// Second call succeeds (default).
		},
	}
	model := Chat("meta.llama3-70b",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithRegion("us-east-1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	cm := model.(*chatModel)
	if !strings.HasPrefix(cm.id, "us.") {
		t.Errorf("expected us. prefix after cross-region fallback, got %q", cm.id)
	}
	if transport.calls != 2 {
		t.Errorf("expected 2 calls, got %d", transport.calls)
	}
}

func TestDoRequest_CrossRegionFallback_OnDemandThroughput(t *testing.T) {
	transport := &apiErrorTransport{
		responses: []apiErrorResponse{
			{400, "on-demand throughput isn't supported"},
			// Second call succeeds.
		},
	}
	model := Chat("meta.llama3-70b",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithRegion("us-east-1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDoRequest_BareUSFallback(t *testing.T) {
	transport := &apiErrorTransport{
		responses: []apiErrorResponse{
			{400, "model identifier is invalid"}, // triggers cross-region
			{400, "model identifier is invalid"}, // triggers bare-US fallback
			// Third call succeeds.
		},
	}
	model := Chat("ai21.jamba-instruct",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithRegion("us-east-1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	cm := model.(*chatModel)
	if cm.id != "ai21.jamba-instruct" {
		t.Errorf("expected bare ID after fallback, got %q", cm.id)
	}
	if transport.calls != 3 {
		t.Errorf("expected 3 calls, got %d", transport.calls)
	}
}

func TestDoRequest_MaxTokensFallback(t *testing.T) {
	transport := &apiErrorTransport{
		responses: []apiErrorResponse{
			{400, "maximum tokens of 4096 exceeds the model limit"},
			// Second call succeeds.
		},
	}
	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		MaxOutputTokens: 4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	if transport.calls != 2 {
		t.Errorf("expected 2 calls, got %d", transport.calls)
	}
}

func TestDoRequest_ToolStreamingFallback(t *testing.T) {
	transport := &apiErrorTransport{
		responses: []apiErrorResponse{
			{400, "tool streaming is not supported for this model"},
			// Second call succeeds.
		},
	}
	model := Chat("m",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
}

// --- #5: isToolStreamingError non-APIError path ---

func TestIsToolStreamingError_NonAPIError(t *testing.T) {
	if isToolStreamingError(fmt.Errorf("not an APIError")) {
		t.Error("non-APIError should return false")
	}
}

// --- #6: matchesToolStreamingMessage patterns ---

func TestMatchesToolStreamingMessage(t *testing.T) {
	tests := []struct {
		msg  string
		want bool
	}{
		{"tool streaming is not supported", true},
		{"tool in stream mode doesn't support this", true},
		{"tool in stream mode does not support", true},
		{"mantle streaming error", true},
		{"random error", false},
		{"tool error without stream keyword", false},
	}
	for _, tt := range tests {
		got := matchesToolStreamingMessage(tt.msg)
		if got != tt.want {
			t.Errorf("matchesToolStreamingMessage(%q) = %v, want %v", tt.msg, got, tt.want)
		}
	}
}

// --- #7: tryBareUSFallback ---

func TestTryBareUSFallback_NotFallbackDone(t *testing.T) {
	m := &chatModel{id: "m", originalID: "m", opts: options{region: "us-east-1"}}
	// fallbackDone is false, should return false.
	err := &goai.APIError{Message: "model identifier is invalid"}
	if m.tryBareUSFallback(err) {
		t.Error("should return false when fallbackDone is false")
	}
}

func TestTryBareUSFallback_SameID(t *testing.T) {
	m := &chatModel{id: "us.m", originalID: "us.m", fallbackDone: true, opts: options{region: "us-east-1"}}
	err := &goai.APIError{Message: "model identifier is invalid"}
	if m.tryBareUSFallback(err) {
		t.Error("should return false when id == originalID")
	}
}

func TestTryBareUSFallback_NonAPIError(t *testing.T) {
	m := &chatModel{id: "us.m", originalID: "m", fallbackDone: true, opts: options{region: "us-east-1"}}
	if m.tryBareUSFallback(fmt.Errorf("not api error")) {
		t.Error("should return false for non-APIError")
	}
}

func TestTryBareUSFallback_WrongMessage(t *testing.T) {
	m := &chatModel{id: "us.m", originalID: "m", fallbackDone: true, opts: options{region: "us-east-1"}}
	err := &goai.APIError{Message: "some other error"}
	if m.tryBareUSFallback(err) {
		t.Error("should return false for wrong message")
	}
}

func TestTryBareUSFallback_Success(t *testing.T) {
	m := &chatModel{id: "us.m", originalID: "m", fallbackDone: true, opts: options{region: "us-east-1"}}
	err := &goai.APIError{Message: "model identifier is invalid"}
	if !m.tryBareUSFallback(err) {
		t.Error("should return true")
	}
	if m.id != "m" {
		t.Errorf("id = %q, want m", m.id)
	}
}

// --- #8: tryCrossRegionFallback ---

func TestTryCrossRegionFallback_NoError(t *testing.T) {
	m := &chatModel{id: "m", opts: options{region: "us-east-1"}}
	if m.tryCrossRegionFallback(nil) {
		t.Error("nil error should return false")
	}
}

func TestTryCrossRegionFallback_AlreadyHasPrefix(t *testing.T) {
	m := &chatModel{id: "us.m", opts: options{region: "us-east-1"}}
	err := &goai.APIError{Message: "model identifier is invalid"}
	if m.tryCrossRegionFallback(err) {
		t.Error("already-prefixed model should return false")
	}
}

func TestTryCrossRegionFallback_NonAPIError(t *testing.T) {
	m := &chatModel{id: "m", opts: options{region: "us-east-1"}}
	if m.tryCrossRegionFallback(fmt.Errorf("not api error")) {
		t.Error("non-APIError should return false")
	}
}

func TestTryCrossRegionFallback_WrongMessage(t *testing.T) {
	m := &chatModel{id: "m", opts: options{region: "us-east-1"}}
	err := &goai.APIError{Message: "rate limit exceeded"}
	if m.tryCrossRegionFallback(err) {
		t.Error("wrong message should return false")
	}
}

func TestTryCrossRegionFallback_OnDemandThroughput(t *testing.T) {
	m := &chatModel{id: "m", originalID: "m", opts: options{region: "us-east-1"}}
	err := &goai.APIError{Message: "on-demand throughput isn\u2019t supported"}
	if !m.tryCrossRegionFallback(err) {
		t.Error("should return true for on-demand throughput error")
	}
	if m.id != "us.m" {
		t.Errorf("id = %q, want us.m", m.id)
	}
}

// --- #9: applyBedrockOptions ---

func TestApplyBedrockOptions_NonAnthropicReasoningNoType(t *testing.T) {
	// Non-anthropic with effort but no type.
	m := &chatModel{id: "amazon.nova", opts: options{
		reasoningConfig: &ReasoningConfig{
			MaxReasoningEffort: ReasoningEffortHigh,
		},
	}}
	body := map[string]any{}
	m.applyBedrockOptions(body, nil, nil)
	additional, ok := body["additionalModelRequestFields"].(map[string]any)
	if !ok {
		t.Fatal("missing additionalModelRequestFields")
	}
	rc, ok := additional["reasoningConfig"].(map[string]any)
	if !ok {
		t.Fatal("missing reasoningConfig")
	}
	if rc["maxReasoningEffort"] != "high" {
		t.Errorf("maxReasoningEffort = %v", rc["maxReasoningEffort"])
	}
	if _, hasType := rc["type"]; hasType {
		t.Error("should not have type when Type is empty")
	}
}

func TestParseReasoningConfig_Float64Budget(t *testing.T) {
	// JSON numbers decode as float64 - verify conversion to int.
	cfg := parseReasoningConfig(map[string]any{
		"reasoningConfig": map[string]any{
			"type":         "enabled",
			"budgetTokens": float64(2048),
		},
	})
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if cfg.Type != ReasoningEnabled {
		t.Errorf("Type = %q, want %q", cfg.Type, ReasoningEnabled)
	}
	if cfg.BudgetTokens != 2048 {
		t.Errorf("BudgetTokens = %d, want 2048", cfg.BudgetTokens)
	}
}

func TestParseReasoningConfig_Missing(t *testing.T) {
	cfg := parseReasoningConfig(map[string]any{"other": "value"})
	if cfg != nil {
		t.Errorf("expected nil for missing reasoningConfig, got %+v", cfg)
	}
}

func TestApplyBedrockOptions_ThinkingEnabled_NoBudgetTokens(t *testing.T) {
	// Anthropic thinking enabled without BudgetTokens.
	m := &chatModel{id: "anthropic.claude", opts: options{
		reasoningConfig: &ReasoningConfig{
			Type: ReasoningEnabled,
		},
	}}
	body := map[string]any{
		"inferenceConfig": map[string]any{
			"temperature": 0.7,
			"topK":        40,
			"topP":        0.9,
		},
	}
	m.applyBedrockOptions(body, nil, nil)
	ic := body["inferenceConfig"].(map[string]any)
	if _, ok := ic["temperature"]; ok {
		t.Error("temperature should be removed when thinking enabled")
	}
	if _, ok := ic["topK"]; ok {
		t.Error("topK should be removed when thinking enabled")
	}
	if _, ok := ic["topP"]; ok {
		t.Error("topP should be removed when thinking enabled")
	}
	// No maxTokens should be set since BudgetTokens is 0.
	if _, ok := ic["maxTokens"]; ok {
		t.Error("maxTokens should not be set without budget")
	}
}

func TestApplyBedrockOptions_ThinkingEnabled_WithMaxTokens(t *testing.T) {
	// Anthropic thinking enabled with BudgetTokens and existing maxTokens.
	m := &chatModel{id: "anthropic.claude", opts: options{
		reasoningConfig: &ReasoningConfig{
			Type:         ReasoningEnabled,
			BudgetTokens: 2048,
		},
	}}
	body := map[string]any{
		"inferenceConfig": map[string]any{
			"maxTokens": 1024,
		},
	}
	m.applyBedrockOptions(body, nil, nil)
	ic := body["inferenceConfig"].(map[string]any)
	if ic["maxTokens"] != 1024+2048 {
		t.Errorf("maxTokens = %v, want %d", ic["maxTokens"], 1024+2048)
	}
}

func TestApplyBedrockOptions_ThinkingEnabled_NoMaxTokens(t *testing.T) {
	// Anthropic thinking enabled with BudgetTokens but no existing maxTokens.
	m := &chatModel{id: "anthropic.claude", opts: options{
		reasoningConfig: &ReasoningConfig{
			Type:         ReasoningEnabled,
			BudgetTokens: 2048,
		},
	}}
	body := map[string]any{
		"inferenceConfig": map[string]any{},
	}
	m.applyBedrockOptions(body, nil, nil)
	ic := body["inferenceConfig"].(map[string]any)
	if ic["maxTokens"] != 2048+4096 {
		t.Errorf("maxTokens = %v, want %d", ic["maxTokens"], 2048+4096)
	}
}

func TestApplyBedrockOptions_AnthropicEffort(t *testing.T) {
	m := &chatModel{id: "anthropic.claude", opts: options{
		reasoningConfig: &ReasoningConfig{
			MaxReasoningEffort: ReasoningEffortMedium,
		},
	}}
	body := map[string]any{}
	m.applyBedrockOptions(body, nil, nil)
	additional := body["additionalModelRequestFields"].(map[string]any)
	oc, ok := additional["output_config"].(map[string]any)
	if !ok {
		t.Fatal("missing output_config")
	}
	if oc["effort"] != "medium" {
		t.Errorf("effort = %v", oc["effort"])
	}
}

// --- #10: regionMatchesGeo global prefix ---

func TestRegionMatchesGeo_Global(t *testing.T) {
	// Global prefix should match any region.
	if !regionMatchesGeo("ap-northeast-1", "global.meta.llama") {
		t.Error("global prefix should match any region")
	}
	if !regionMatchesGeo("eu-west-1", "global.meta.llama") {
		t.Error("global prefix should match eu region")
	}
}

// --- #11: buildConverseRequest ---

func TestBuildConverseRequest_PromptCaching(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		System:        "sys",
		PromptCaching: true,
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}, "anthropic.claude")
	sys := body["system"].([]map[string]any)
	if len(sys) != 2 {
		t.Fatalf("system blocks = %d, want 2", len(sys))
	}
	cp, ok := sys[1]["cachePoint"].(map[string]any)
	if !ok {
		t.Fatal("missing cachePoint")
	}
	if cp["type"] != "default" {
		t.Errorf("cachePoint type = %v", cp["type"])
	}
}

func TestBuildConverseRequest_StopSequences(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		StopSequences: []string{"END", "STOP"},
	}, "m")
	ic := body["inferenceConfig"].(map[string]any)
	ss := ic["stopSequences"].([]string)
	if len(ss) != 2 || ss[0] != "END" {
		t.Errorf("stopSequences = %v", ss)
	}
}

func TestBuildConverseRequest_TopK(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{"topK": 40},
	}, "m")
	ic := body["inferenceConfig"].(map[string]any)
	if ic["topK"] != 40 {
		t.Errorf("topK = %v", ic["topK"])
	}
}

func TestBuildConverseRequest_ToolChoiceAuto(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "t", Description: "d", InputSchema: []byte(`{}`)},
		},
		ToolChoice: "auto",
	}, "m")
	tc := body["toolConfig"].(map[string]any)
	choice := tc["toolChoice"].(map[string]any)
	if _, ok := choice["auto"]; !ok {
		t.Error("expected auto tool choice")
	}
}

func TestBuildConverseRequest_ToolChoiceAny(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "t", Description: "d", InputSchema: []byte(`{}`)},
		},
		ToolChoice: "any",
	}, "m")
	tc := body["toolConfig"].(map[string]any)
	choice := tc["toolChoice"].(map[string]any)
	if _, ok := choice["any"]; !ok {
		t.Error("expected any tool choice")
	}
}

func TestBuildConverseRequest_ToolChoiceNone(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "t", Description: "d", InputSchema: []byte(`{}`)},
		},
		ToolChoice: "none",
	}, "m")
	if _, ok := body["toolConfig"]; ok {
		t.Error("toolConfig should not be set for none")
	}
}

func TestBuildConverseRequest_ToolChoiceSpecific(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "d", InputSchema: []byte(`{}`)},
		},
		ToolChoice: "get_weather",
	}, "m")
	tc := body["toolConfig"].(map[string]any)
	choice := tc["toolChoice"].(map[string]any)
	tool, ok := choice["tool"].(map[string]any)
	if !ok {
		t.Fatal("expected tool choice")
	}
	if tool["name"] != "get_weather" {
		t.Errorf("tool name = %v", tool["name"])
	}
}

// --- #12: convertMessages ---

func TestConvertMessages_ToolRole(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: "tool", Content: []provider.Part{{Type: provider.PartToolResult, ToolCallID: "c1", ToolOutput: "result"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("msgs = %d, want 1", len(msgs))
	}
	if msgs[0]["role"] != "user" {
		t.Errorf("role = %v, want user", msgs[0]["role"])
	}
}

func TestConvertMessages_MergeSameRole(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "world"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("msgs = %d, want 1 (merged)", len(msgs))
	}
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 2 {
		t.Errorf("content parts = %d, want 2", len(content))
	}
}

// --- #13: convertParts ---

func TestConvertParts_ReasoningWithSignature(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type: provider.PartReasoning,
			Text: "thinking...",
			ProviderOptions: map[string]any{
				"signature": "sig123",
			},
		},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	rc := parts[0]["reasoningContent"].(map[string]any)
	rt := rc["reasoningText"].(map[string]any)
	if rt["text"] != "thinking..." {
		t.Errorf("text = %v", rt["text"])
	}
	if rt["signature"] != "sig123" {
		t.Errorf("signature = %v", rt["signature"])
	}
}

func TestConvertParts_ReasoningWithoutSignature(t *testing.T) {
	// Reasoning from other providers (e.g. Gemini) may lack signature.
	// These should be skipped to avoid Bedrock validation errors.
	parts := convertParts([]provider.Part{
		{Type: provider.PartReasoning, Text: "gemini thinking"},
		{Type: provider.PartText, Text: "answer"},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1 (reasoning without signature skipped)", len(parts))
	}
	if parts[0]["text"] != "answer" {
		t.Errorf("part[0] = %v, want text answer", parts[0])
	}
}

func TestConvertParts_ReasoningRedacted(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type: provider.PartReasoning,
			Text: "", // no text = redacted
			ProviderOptions: map[string]any{
				"redactedData": "encrypted123",
			},
		},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	rc := parts[0]["reasoningContent"].(map[string]any)
	rd := rc["redactedReasoning"].(map[string]any)
	if rd["data"] != "encrypted123" {
		t.Errorf("data = %v", rd["data"])
	}
}

func TestConvertParts_ToolCall(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:       provider.PartToolCall,
			ToolCallID: "tc1",
			ToolName:   "search",
			ToolInput:  []byte(`{"q":"test"}`),
		},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d", len(parts))
	}
	tu := parts[0]["toolUse"].(map[string]any)
	if tu["toolUseId"] != "tc1" {
		t.Errorf("toolUseId = %v", tu["toolUseId"])
	}
	if tu["name"] != "search" {
		t.Errorf("name = %v", tu["name"])
	}
}

func TestConvertParts_ToolResult(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:       provider.PartToolResult,
			ToolCallID: "tc1",
			ToolOutput: "result text",
		},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d", len(parts))
	}
	tr := parts[0]["toolResult"].(map[string]any)
	if tr["toolUseId"] != "tc1" {
		t.Errorf("toolUseId = %v", tr["toolUseId"])
	}
}

func TestConvertParts_File(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:      provider.PartFile,
			URL:       "base64data",
			MediaType: "text/csv",
			Filename:  "data.csv",
		},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d", len(parts))
	}
	doc := parts[0]["document"].(map[string]any)
	if doc["format"] != "csv" {
		t.Errorf("format = %v, want csv", doc["format"])
	}
}

func TestConvertParts_FileDuplicateNames(t *testing.T) {
	docNames := make(map[string]int)
	parts := convertParts([]provider.Part{
		{Type: provider.PartFile, URL: "data1", MediaType: "application/pdf", Filename: "report.pdf"},
		{Type: provider.PartFile, URL: "data2", MediaType: "application/pdf", Filename: "report.pdf"},
		{Type: provider.PartFile, URL: "data3", MediaType: "application/pdf", Filename: "report.pdf"},
	}, docNames)
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want 3", len(parts))
	}
	names := make([]string, 3)
	for i, p := range parts {
		names[i] = p["document"].(map[string]any)["name"].(string)
	}
	if names[0] != "report-pdf" {
		t.Errorf("first name = %q, want report-pdf", names[0])
	}
	if names[1] != "report-pdf-2" {
		t.Errorf("second name = %q, want report-pdf-2", names[1])
	}
	if names[2] != "report-pdf-3" {
		t.Errorf("third name = %q, want report-pdf-3", names[2])
	}
}

func TestConvertParts_FileDataURLStripped(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:      provider.PartFile,
			URL:       "data:application/pdf;base64,AQIDBA==",
			MediaType: "application/pdf",
			Filename:  "report.pdf",
		},
	}, make(map[string]int))
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	doc := parts[0]["document"].(map[string]any)
	src := doc["source"].(map[string]any)
	if src["bytes"] != "AQIDBA==" {
		t.Errorf("source.bytes = %v, want raw base64 without data URL prefix", src["bytes"])
	}
}

func TestConvertParts_FileNoFilename(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:      provider.PartFile,
			URL:       "base64data",
			MediaType: "application/pdf",
		},
	}, make(map[string]int))
	doc := parts[0]["document"].(map[string]any)
	name := doc["name"].(string)
	if !strings.HasPrefix(name, "document-") {
		t.Errorf("name = %v, want document-N", name)
	}
}

func TestSanitizeDocumentName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"report.pdf", "report-pdf"},
		{"my_report.docx", "my-report-docx"},
		{"path/to/report.pdf", "path-to-report-pdf"},
		{"report@2024!.pdf", "report-2024-pdf"},
		{"my-report", "my-report"},
		{"report", "report"},
		{"My Report (draft) [v2]", "My Report (draft) [v2]"},
		{"file  with   spaces", "file with spaces"},
		{"my.report.v2.final.pdf", "my-report-v2-final-pdf"},
		{"archive.tar.gz", "archive-tar-gz"},
		{"báo_cáo.pdf", "b-o-c-o-pdf"},                                // Vietnamese with diacritics → keep ASCII parts
		{"レポート2024.pdf", "2024-pdf"},                                  // Japanese + number → keep number + ext
		{strings.Repeat("a", 250) + ".pdf", strings.Repeat("a", 200)}, // 250 + "-pdf" = 254, truncated to 200
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := sanitizeDocumentName(tt.input)
			if got != tt.want {
				t.Errorf("sanitizeDocumentName(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestSanitizeDocumentName_NonLatin(t *testing.T) {
	// Pure non-Latin filename → only extension survives as ASCII.
	name := sanitizeDocumentName("報告書.pdf")
	if name != "pdf" {
		t.Errorf("sanitizeDocumentName(Japanese.pdf) = %q, want pdf", name)
	}
	// Pure non-Latin without extension → fallback to document-N.
	name2 := sanitizeDocumentName("報告書")
	if !strings.HasPrefix(name2, "document-") {
		t.Errorf("sanitizeDocumentName(Japanese) = %q, want document-N", name2)
	}
}

func TestSanitizeDocumentName_Empty(t *testing.T) {
	name1 := sanitizeDocumentName("")
	name2 := sanitizeDocumentName("")
	if name1 == name2 {
		t.Errorf("expected unique names, got %q and %q", name1, name2)
	}
	if !strings.HasPrefix(name1, "document-") {
		t.Errorf("expected document-N, got %q", name1)
	}
}

func TestConvertParts_CacheControl(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:         provider.PartText,
			Text:         "cached text",
			CacheControl: "ephemeral",
		},
	}, make(map[string]int))
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2 (text + cachePoint)", len(parts))
	}
	cp, ok := parts[1]["cachePoint"].(map[string]any)
	if !ok {
		t.Fatal("missing cachePoint")
	}
	if cp["type"] != "default" {
		t.Errorf("cachePoint type = %v", cp["type"])
	}
}

// --- #14: parseConverseResponse ---

func TestParseConverseResponse_RedactedReasoning(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [
			{"reasoningContent": {"redactedReasoning": {"data": "enc123"}}},
			{"text": "answer"}
		]}},
		"stopReason": "end_turn",
		"usage": {"inputTokens": 10, "outputTokens": 5}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	reasoning := result.ProviderMetadata["bedrock"]["reasoning"].([]map[string]any)
	if reasoning[0]["type"] != "redacted_reasoning" {
		t.Errorf("type = %v", reasoning[0]["type"])
	}
	if reasoning[0]["data"] != "enc123" {
		t.Errorf("data = %v", reasoning[0]["data"])
	}
}

func TestParseConverseResponse_AdditionalFields(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
		"stopReason": "end_turn",
		"usage": {"inputTokens": 1, "outputTokens": 1},
		"additionalModelResponseFields": {"stop_sequence": "\n"},
		"trace": {"guardrail": {"assessment": "safe"}}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	bedrock := result.ProviderMetadata["bedrock"]
	additional := bedrock["additionalModelResponseFields"].(map[string]any)
	if additional["stop_sequence"] != "\n" {
		t.Errorf("stop_sequence = %v", additional["stop_sequence"])
	}
	trace := bedrock["trace"].(map[string]any)
	guardrail := trace["guardrail"].(map[string]any)
	if guardrail["assessment"] != "safe" {
		t.Errorf("assessment = %v", guardrail["assessment"])
	}
}

// --- #15: parseEventStream ---

func TestParseEventStream_Exception(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildExceptionFrame("validationException", `invalid request`))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotError bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkError {
			gotError = true
			if !strings.Contains(chunk.Error.Error(), "validationException") {
				t.Errorf("error = %v", chunk.Error)
			}
		}
	}
	if !gotError {
		t.Error("expected error chunk for exception")
	}
}

func TestParseEventStream_UnknownEventType(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("unknownEvent", []byte(`{}`)))
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"hello"}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotText bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkText && chunk.Text == "hello" {
			gotText = true
		}
	}
	if !gotText {
		t.Error("should skip unknown event and continue processing")
	}
}

// buildNonEventFrame builds an EventStream frame with a non-event, non-exception type.
func buildNonEventFrame() []byte {
	headers := buildStringHeader(":message-type", "initial-response")
	headers = append(headers, buildStringHeader(":content-type", "application/json")...)

	payload := []byte(`{}`)
	headersLen := uint32(len(headers))
	totalLen := 4 + 4 + 4 + headersLen + uint32(len(payload)) + 4

	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, totalLen)
	_ = binary.Write(&buf, binary.BigEndian, headersLen)
	prelude := buf.Bytes()
	preludeCRC := crc32.ChecksumIEEE(prelude)
	_ = binary.Write(&buf, binary.BigEndian, preludeCRC)
	buf.Write(headers)
	buf.Write(payload)
	msgCRC := crc32.ChecksumIEEE(buf.Bytes())
	_ = binary.Write(&buf, binary.BigEndian, msgCRC)
	return buf.Bytes()
}

func TestParseEventStream_NonEventMessageType(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildNonEventFrame())
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"ok"}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotText bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkText && chunk.Text == "ok" {
			gotText = true
		}
	}
	if !gotText {
		t.Error("should skip non-event message type and continue")
	}
}

func TestParseEventStream_ContentBlockStartText(t *testing.T) {
	// contentBlockStart for a text block (not toolUse).
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockStart", []byte(`{"contentBlockIndex":0,"start":{}}`)))
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"hello"}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotText bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkText && chunk.Text == "hello" {
			gotText = true
		}
	}
	if !gotText {
		t.Error("expected text chunk")
	}
}

func TestParseEventStream_ReasoningRedacted(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"reasoningContent":{"redactedReasoning":{"data":"enc456"}}}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotRedacted bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkReasoning && chunk.Metadata != nil {
			if chunk.Metadata["redactedData"] == "enc456" {
				gotRedacted = true
			}
		}
	}
	if !gotRedacted {
		t.Error("expected redacted reasoning chunk")
	}
}

func TestParseEventStream_ReasoningPlainText(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"thinking step"}}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotReasoning bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkReasoning && chunk.Text == "thinking step" {
			gotReasoning = true
		}
	}
	if !gotReasoning {
		t.Error("expected plain text reasoning chunk")
	}
}

func TestParseEventStream_ContentBlockStopTool(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockStart", []byte(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"c1","name":"test"}}}`)))
	buf.Write(buildEventStreamFrame("contentBlockStop", []byte(`{"contentBlockIndex":0}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"tool_use"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotToolCall bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkToolCall && chunk.ToolCallID == "c1" {
			gotToolCall = true
		}
	}
	if !gotToolCall {
		t.Error("expected tool call chunk on contentBlockStop")
	}
}

func TestParseEventStream_CacheWriteTokens(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":100,"outputTokens":10,"totalTokens":110,"cacheWriteInputTokens":25}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotCacheWrite bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkFinish && chunk.Metadata != nil {
			if chunk.Metadata["cacheWriteInputTokens"] == 25 {
				gotCacheWrite = true
			}
		}
	}
	if !gotCacheWrite {
		t.Error("expected cacheWriteInputTokens in metadata")
	}
}

// --- #16: strVal key not found ---

func TestStrVal_KeyNotFound(t *testing.T) {
	m := map[string]any{"a": "b"}
	if strVal(m, "missing") != "" {
		t.Error("missing key should return empty string")
	}
}

func TestStrVal_NonString(t *testing.T) {
	m := map[string]any{"a": 123}
	if strVal(m, "a") != "" {
		t.Error("non-string value should return empty string")
	}
}

// --- #17: bedrockDocumentFormat ---

func TestBedrockDocumentFormat(t *testing.T) {
	tests := []struct {
		mime string
		want string
	}{
		{"application/pdf", "pdf"},
		{"text/csv", "csv"},
		{"application/msword", "doc"},
		{"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"},
		{"application/vnd.ms-excel", "xls"},
		{"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"},
		{"text/html", "html"},
		{"text/plain", "txt"},
		{"text/markdown", "md"},
		{"application/octet-stream", "txt"}, // default
	}
	for _, tt := range tests {
		got := bedrockDocumentFormat(tt.mime)
		if got != tt.want {
			t.Errorf("bedrockDocumentFormat(%q) = %q, want %q", tt.mime, got, tt.want)
		}
	}
}

// --- #18: EventStream decoder error paths ---

func TestEventStreamDecoder_PreludeCRCMismatch(t *testing.T) {
	// Build a frame, then corrupt the prelude CRC.
	frame := buildEventStreamFrame("contentBlockDelta", []byte(`{}`))
	// Byte 11 is the last byte of prelude CRC (bytes 8-11).
	frame[11] ^= 0xff

	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "prelude CRC mismatch") {
		t.Errorf("expected prelude CRC mismatch error, got %v", err)
	}
}

func TestEventStreamDecoder_InvalidTotalLength(t *testing.T) {
	// Build a frame with totalLen set to something less than 16 (12 prelude + 4 message CRC).
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, uint32(10)) // totalLen < 16
	_ = binary.Write(&buf, binary.BigEndian, uint32(0))  // headersLen
	prelude := buf.Bytes()
	preludeCRC := crc32.ChecksumIEEE(prelude)
	_ = binary.Write(&buf, binary.BigEndian, preludeCRC)

	decoder := newEventStreamDecoder(&buf)
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "invalid total length") {
		t.Errorf("expected invalid total length error, got %v", err)
	}
}

func TestEventStreamDecoder_FrameBodyReadError(t *testing.T) {
	// Valid prelude, but not enough data for the frame body.
	var buf bytes.Buffer
	totalLen := uint32(100) // claims 100 bytes total
	headersLen := uint32(0)
	_ = binary.Write(&buf, binary.BigEndian, totalLen)
	_ = binary.Write(&buf, binary.BigEndian, headersLen)
	prelude := buf.Bytes()
	preludeCRC := crc32.ChecksumIEEE(prelude)
	_ = binary.Write(&buf, binary.BigEndian, preludeCRC)
	// Only write a few bytes, not enough.
	buf.Write([]byte{0, 0, 0})

	decoder := newEventStreamDecoder(&buf)
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "reading frame body") {
		t.Errorf("expected frame body read error, got %v", err)
	}
}

func TestEventStreamDecoder_MessageCRCMismatch(t *testing.T) {
	frame := buildEventStreamFrame("contentBlockDelta", []byte(`{}`))
	// Corrupt the last 4 bytes (message CRC).
	frame[len(frame)-1] ^= 0xff

	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "message CRC mismatch") {
		t.Errorf("expected message CRC mismatch error, got %v", err)
	}
}

func TestEventStreamDecoder_HeaderNameOverflow(t *testing.T) {
	// Build a frame with a header whose nameLen exceeds remaining header bytes.
	frame := buildFrameWithRawHeaders([]byte{255}) // nameLen=255, but no name bytes
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "header name overflow") {
		t.Errorf("expected header name overflow error, got %v", err)
	}
}

func TestEventStreamDecoder_MissingTypeTag(t *testing.T) {
	// Header with valid name but no type tag.
	header := []byte{1, 'x'} // nameLen=1, name="x", then nothing
	frame := buildFrameWithRawHeaders(header)
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "missing header type tag") {
		t.Errorf("expected missing type tag error, got %v", err)
	}
}

func TestEventStreamDecoder_StringValueLengthOverflow(t *testing.T) {
	// String header (type 7) with value length that overflows.
	header := []byte{1, 'x', 7} // nameLen=1, name="x", type=7 (string), then no length bytes
	frame := buildFrameWithRawHeaders(header)
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "string header value length overflow") {
		t.Errorf("expected string value length overflow error, got %v", err)
	}
}

func TestEventStreamDecoder_StringValueOverflow(t *testing.T) {
	// String header with valid length prefix but not enough data.
	var header bytes.Buffer
	header.WriteByte(1)                                      // nameLen
	header.WriteByte('x')                                    // name
	header.WriteByte(7)                                      // type = string
	_ = binary.Write(&header, binary.BigEndian, uint16(100)) // valLen=100, but no value bytes
	frame := buildFrameWithRawHeaders(header.Bytes())
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "string header value overflow") {
		t.Errorf("expected string value overflow error, got %v", err)
	}
}

func TestEventStreamDecoder_BytesHeaderType(t *testing.T) {
	// Bytes header (type 6) with valid length.
	var header bytes.Buffer
	header.WriteByte(1)                                    // nameLen
	header.WriteByte('b')                                  // name
	header.WriteByte(6)                                    // type = bytes
	_ = binary.Write(&header, binary.BigEndian, uint16(3)) // bLen=3
	header.Write([]byte{1, 2, 3})                          // data
	frame := buildFrameWithRawHeaders(header.Bytes())
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	f, err := decoder.Next()
	if err != nil {
		t.Fatal(err)
	}
	if f == nil {
		t.Fatal("expected frame")
	}
}

func TestEventStreamDecoder_BytesHeaderOverflow(t *testing.T) {
	// Bytes header (type 6) but no length bytes.
	header := []byte{1, 'b', 6} // nameLen=1, name="b", type=6 (bytes), no length
	frame := buildFrameWithRawHeaders(header)
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "bytes header length overflow") {
		t.Errorf("expected bytes header length overflow error, got %v", err)
	}
}

func TestEventStreamDecoder_UnknownHeaderType(t *testing.T) {
	header := []byte{1, 'x', 99} // nameLen=1, name="x", type=99 (unknown)
	frame := buildFrameWithRawHeaders(header)
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err == nil || !strings.Contains(err.Error(), "unknown header type tag") {
		t.Errorf("expected unknown header type tag error, got %v", err)
	}
}

func TestEventStreamDecoder_BoolHeaderType(t *testing.T) {
	// Bool true (type 0) and bool false (type 1).
	var header bytes.Buffer
	header.WriteByte(1)   // nameLen
	header.WriteByte('t') // name
	header.WriteByte(0)   // type = bool true
	header.WriteByte(1)   // nameLen
	header.WriteByte('f') // name
	header.WriteByte(1)   // type = bool false
	frame := buildFrameWithRawHeaders(header.Bytes())
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err != nil {
		t.Fatal(err)
	}
}

func TestEventStreamDecoder_ShortIntLongTimestampUUIDHeaders(t *testing.T) {
	var header bytes.Buffer
	// byte (type 2)
	header.WriteByte(1)
	header.WriteByte('a')
	header.WriteByte(2) // byte
	header.WriteByte(42)
	// short (type 3)
	header.WriteByte(1)
	header.WriteByte('b')
	header.WriteByte(3) // short
	_ = binary.Write(&header, binary.BigEndian, int16(100))
	// int (type 4)
	header.WriteByte(1)
	header.WriteByte('c')
	header.WriteByte(4) // int
	_ = binary.Write(&header, binary.BigEndian, int32(1000))
	// long (type 5)
	header.WriteByte(1)
	header.WriteByte('d')
	header.WriteByte(5) // long
	_ = binary.Write(&header, binary.BigEndian, int64(10000))
	// timestamp (type 8)
	header.WriteByte(1)
	header.WriteByte('e')
	header.WriteByte(8) // timestamp
	_ = binary.Write(&header, binary.BigEndian, int64(1234567890))
	// uuid (type 9)
	header.WriteByte(1)
	header.WriteByte('f')
	header.WriteByte(9) // uuid
	header.Write(make([]byte, 16))

	frame := buildFrameWithRawHeaders(header.Bytes())
	decoder := newEventStreamDecoder(bytes.NewReader(frame))
	_, err := decoder.Next()
	if err != nil {
		t.Fatal(err)
	}
}

// buildFrameWithRawHeaders creates a valid EventStream frame with custom raw headers.
func buildFrameWithRawHeaders(rawHeaders []byte) []byte {
	payload := []byte{}
	headersLen := uint32(len(rawHeaders))
	totalLen := 4 + 4 + 4 + headersLen + uint32(len(payload)) + 4

	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, totalLen)
	_ = binary.Write(&buf, binary.BigEndian, headersLen)
	prelude := buf.Bytes()
	preludeCRC := crc32.ChecksumIEEE(prelude)
	_ = binary.Write(&buf, binary.BigEndian, preludeCRC)
	buf.Write(rawHeaders)
	buf.Write(payload)
	msgCRC := crc32.ChecksumIEEE(buf.Bytes())
	_ = binary.Write(&buf, binary.BigEndian, msgCRC)
	return buf.Bytes()
}

// --- parseEventStream reasoning with signature ---

func TestParseEventStream_ReasoningWithSignature(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"reasoningContent":{"reasoningText":{"text":"step1","signature":"sig999"}}}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotSig bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkReasoning && chunk.Metadata != nil {
			if chunk.Metadata["signature"] == "sig999" {
				gotSig = true
			}
		}
	}
	if !gotSig {
		t.Error("expected reasoning chunk with signature")
	}
}

func TestParseEventStream_ReasoningStreamingFormat(t *testing.T) {
	// Bedrock streaming sends reasoning as flat fields in reasoningContent:
	// - text: reasoningContent.text (string)
	// - signature: reasoningContent.signature (string, separate delta)
	// This matches the actual Bedrock API behavior observed in production.
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"thinking..."}}}`)))
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":""}}}`)))
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"EqsBCkgICxABtest"}}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotText, gotSig bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkReasoning {
			if chunk.Text == "thinking..." {
				gotText = true
			}
			if chunk.Metadata != nil && chunk.Metadata["signature"] == "EqsBCkgICxABtest" {
				gotSig = true
			}
		}
	}
	if !gotText {
		t.Error("expected reasoning text chunk")
	}
	if !gotSig {
		t.Error("expected reasoning signature chunk")
	}
}

// === Remaining coverage gap tests ===

// bedrock.go:247 -- drain goroutine in tool-streaming retry (already covered by TestDoStream_ToolStreamingRetry)
// bedrock.go:280 -- ctx.Done in forwarding goroutine for remaining innerCh

func TestDoStream_CtxDoneInForwardingRemaining(t *testing.T) {
	// Test ctx.Done in the forwarding goroutine. Cancel context immediately
	// after DoStream returns, before any chunks are read from the output channel.
	// This ensures the forwarding goroutine hits ctx.Done when trying to send
	// buffered chunks. Run multiple times to increase probability of hitting
	// the race condition on the select.
	for attempt := 0; attempt < 10; attempt++ {
		func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				// Many text chunks to fill buffers.
				for i := 0; i < 100; i++ {
					_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(fmt.Sprintf(`{"contentBlockIndex":0,"delta":{"text":"chunk%d"}}`, i))))
				}
				_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
				_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

			}))
			defer server.Close()

			ctx, cancel := context.WithCancel(t.Context())
			model := Chat("m", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
			result, err := model.DoStream(ctx, provider.GenerateParams{
				Messages: []provider.Message{
					{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
				},
				Tools: []provider.ToolDefinition{
					{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
				},
			})
			if err != nil {
				// May error on some attempts, that's fine.
				cancel()
				return
			}
			// Cancel immediately -- forwarding goroutine may hit ctx.Done.
			cancel()
			// Drain to let goroutine exit.
			for range result.Stream {
			}
		}()
	}
}

// bedrock.go:686 -- regionMatchesGeo returns true for no-prefix model
func TestRegionMatchesGeo_NoPrefix(t *testing.T) {
	if !regionMatchesGeo("eu-west-1", "anthropic.claude-sonnet-4-v1:0") {
		t.Error("no-prefix model should match any region")
	}
}

// converse.go:40-42 -- TopP in buildConverseRequest
func TestBuildConverseRequest_TopP(t *testing.T) {
	topP := 0.9
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		TopP: &topP,
	}, "m")
	ic := body["inferenceConfig"].(map[string]any)
	if ic["topP"] != 0.9 {
		t.Errorf("topP = %v, want 0.9", ic["topP"])
	}
}

// converse.go:72-74 -- invalid JSON in tool InputSchema
func TestBuildConverseRequest_InvalidToolSchema(t *testing.T) {
	body := buildConverseRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "d", InputSchema: []byte(`not valid json`)},
		},
	}, "m")
	tc := body["toolConfig"].(map[string]any)
	tools := tc["tools"].([]map[string]any)
	toolSpec := tools[0]["toolSpec"].(map[string]any)
	inputSchema := toolSpec["inputSchema"].(map[string]any)
	// Should fallback to default object schema on invalid JSON.
	schema, ok := inputSchema["json"].(map[string]any)
	if !ok {
		t.Fatal("expected fallback schema")
	}
	if schema["type"] != "object" {
		t.Errorf("expected {type:object} fallback schema, got %v", schema)
	}
}

// converse.go:120-121 -- empty content skip in convertMessages
func TestConvertMessages_EmptyContent(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{}}, // empty content
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("msgs = %d, want 1 (empty should be skipped)", len(msgs))
	}
	if msgs[0]["role"] != "user" {
		t.Errorf("role = %v", msgs[0]["role"])
	}
}

// converse.go:211-213 -- ToolInput nil fallback to empty map
func TestConvertParts_ToolCallNilInput(t *testing.T) {
	parts := convertParts([]provider.Part{
		{
			Type:       provider.PartToolCall,
			ToolCallID: "tc1",
			ToolName:   "test",
			ToolInput:  nil, // nil input
		},
	}, make(map[string]int))
	tu := parts[0]["toolUse"].(map[string]any)
	input, ok := tu["input"].(map[string]any)
	if !ok {
		t.Fatal("expected empty map fallback for nil ToolInput")
	}
	if len(input) != 0 {
		t.Errorf("expected empty input, got %v", input)
	}
}

// converse.go:277-279 -- unmarshal error in parseConverseResponse content block
func TestParseConverseResponse_InvalidContentBlock(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": ["not a json object", {"text": "valid"}]}},
		"stopReason": "end_turn",
		"usage": {"inputTokens": 1, "outputTokens": 1}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "valid" {
		t.Errorf("Text = %q, want valid", result.Text)
	}
}

// converse.go:296-297 -- missing input in toolUse content block
func TestParseConverseResponse_ToolUseNoInput(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [
			{"toolUse": {"toolUseId": "tc1", "name": "test"}}
		]}},
		"stopReason": "tool_use",
		"usage": {"inputTokens": 1, "outputTokens": 1}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Input != nil {
		t.Errorf("Input should be nil when not present, got %v", result.ToolCalls[0].Input)
	}
}

// converse.go:300-302 -- multiple text blocks (result.Text non-empty append)
func TestParseConverseResponse_MultipleTextBlocks(t *testing.T) {
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [
			{"text": "hello "},
			{"text": "world"}
		]}},
		"stopReason": "end_turn",
		"usage": {"inputTokens": 1, "outputTokens": 1}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello world" {
		t.Errorf("Text = %q, want 'hello world'", result.Text)
	}
}

// converse.go:410-411 -- invalid JSON payload in event stream
func TestParseEventStream_InvalidJSONPayload(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`not valid json`)))
	buf.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"ok"}}`)))
	buf.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
	buf.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))

	ch := make(chan provider.StreamChunk, 64)
	go parseEventStream(t.Context(), io.NopCloser(&buf), ch, provider.ResponseMetadata{})

	var gotText bool
	for chunk := range ch {
		if chunk.Type == provider.ChunkText && chunk.Text == "ok" {
			gotText = true
		}
	}
	if !gotText {
		t.Error("should skip invalid JSON payload and continue")
	}
}

// converse.go:277 -- parseConverseResponse with invalid JSON body
func TestParseConverseResponse_InvalidJSON(t *testing.T) {
	_, err := parseConverseResponse([]byte(`not json`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "parsing bedrock response") {
		t.Errorf("error = %v", err)
	}
}

func TestWithBearerToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer my-bearer-token" {
			t.Errorf("auth = %q, want Bearer my-bearer-token", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, converseResponse("hello", "end_turn", 10, 5))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-20250514-v1:0",
		WithBearerToken("my-bearer-token"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" {
		t.Errorf("Text = %q", result.Text)
	}
}

func TestChat_EnvVarBearerToken(t *testing.T) {
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer")
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	t.Setenv("AWS_REGION", "us-west-2")
	m := Chat("anthropic.claude-sonnet-4-20250514-v1:0")
	cm := m.(*chatModel)
	if cm.opts.bearerToken != "env-bearer" {
		t.Errorf("bearerToken = %q, want env-bearer", cm.opts.bearerToken)
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "ak")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "sk")
	t.Setenv("AWS_BEDROCK_BASE_URL", "https://custom.bedrock.com")
	m := Chat("anthropic.claude-sonnet-4-20250514-v1:0")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.bedrock.com" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_EnvVarCredentials(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "env-ak")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "env-sk")
	t.Setenv("AWS_SESSION_TOKEN", "env-st")
	t.Setenv("AWS_REGION", "eu-west-1")
	m := Chat("anthropic.claude-sonnet-4-20250514-v1:0")
	cm := m.(*chatModel)
	if cm.opts.accessKey != "env-ak" {
		t.Errorf("accessKey = %q", cm.opts.accessKey)
	}
	if cm.opts.secretKey != "env-sk" {
		t.Errorf("secretKey = %q", cm.opts.secretKey)
	}
	if cm.opts.sessionToken != "env-st" {
		t.Errorf("sessionToken = %q", cm.opts.sessionToken)
	}
}

func TestApplyBedrockOptions_NonAnthropicReasoning(t *testing.T) {
	m := &chatModel{
		id: "amazon.nova-pro-v1:0",
		opts: options{
			reasoningConfig: &ReasoningConfig{
				Type:               ReasoningEnabled,
				MaxReasoningEffort: ReasoningEffortHigh,
			},
		},
	}
	body := map[string]any{
		"inferenceConfig": map[string]any{"maxTokens": 1024},
	}
	m.applyBedrockOptions(body, nil, nil)
	additional, ok := body["additionalModelRequestFields"].(map[string]any)
	if !ok {
		t.Fatal("missing additionalModelRequestFields")
	}
	rc, ok := additional["reasoningConfig"].(map[string]any)
	if !ok {
		t.Fatal("missing reasoningConfig in additional fields")
	}
	if rc["maxReasoningEffort"] != string(ReasoningEffortHigh) {
		t.Errorf("maxReasoningEffort = %v", rc["maxReasoningEffort"])
	}
	if rc["type"] != string(ReasoningEnabled) {
		t.Errorf("type = %v", rc["type"])
	}
}

func TestApplyBedrockOptions_ProviderOptsReasoningConfig_IntBudget(t *testing.T) {
	m := &chatModel{
		id:   "anthropic.claude-sonnet-4-20250514-v1:0",
		opts: options{},
	}
	body := map[string]any{
		"inferenceConfig": map[string]any{"maxTokens": 1024},
	}
	providerOpts := map[string]any{
		"reasoningConfig": map[string]any{
			"type":         "enabled",
			"budgetTokens": int(3000),
		},
	}
	m.applyBedrockOptions(body, nil, providerOpts)
	additional := body["additionalModelRequestFields"].(map[string]any)
	thinking := additional["thinking"].(map[string]any)
	if thinking["budget_tokens"] != 3000 {
		t.Errorf("budget_tokens = %v, want 3000", thinking["budget_tokens"])
	}
}

func TestApplyBedrockOptions_ProviderOptsReasoningConfig(t *testing.T) {
	m := &chatModel{
		id:   "anthropic.claude-sonnet-4-20250514-v1:0",
		opts: options{},
	}
	body := map[string]any{
		"inferenceConfig": map[string]any{"maxTokens": 1024, "temperature": 0.5, "topP": 0.9, "topK": float64(40)},
	}
	providerOpts := map[string]any{
		"reasoningConfig": map[string]any{
			"type":               "enabled",
			"budgetTokens":       float64(5000),
			"maxReasoningEffort": "high",
		},
	}
	m.applyBedrockOptions(body, nil, providerOpts)

	additional, ok := body["additionalModelRequestFields"].(map[string]any)
	if !ok {
		t.Fatal("missing additionalModelRequestFields")
	}
	thinking, ok := additional["thinking"].(map[string]any)
	if !ok {
		t.Fatal("missing thinking in additional fields")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v", thinking["type"])
	}
	if thinking["budget_tokens"] != 5000 {
		t.Errorf("budget_tokens = %v", thinking["budget_tokens"])
	}

	// Verify inferenceConfig was cleaned (temperature/topP/topK removed for thinking).
	ic, ok := body["inferenceConfig"].(map[string]any)
	if !ok {
		t.Fatal("missing inferenceConfig")
	}
	if _, exists := ic["temperature"]; exists {
		t.Error("temperature should be removed for thinking-enabled")
	}
	if _, exists := ic["topP"]; exists {
		t.Error("topP should be removed for thinking-enabled")
	}
	if _, exists := ic["topK"]; exists {
		t.Error("topK should be removed for thinking-enabled")
	}
}

func TestApplyBedrockOptions_AdaptiveThinking(t *testing.T) {
	m := &chatModel{
		id: "anthropic.claude-sonnet-4-20250514-v1:0",
		opts: options{
			reasoningConfig: &ReasoningConfig{
				Type: ReasoningAdaptive,
			},
		},
	}
	body := map[string]any{
		"inferenceConfig": map[string]any{"maxTokens": 1024},
	}
	m.applyBedrockOptions(body, nil, nil)
	additional := body["additionalModelRequestFields"].(map[string]any)
	thinking := additional["thinking"].(map[string]any)
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
	}
}

func TestApplyBedrockOptions_AnthropicBeta(t *testing.T) {
	m := &chatModel{
		id: "anthropic.claude-sonnet-4-20250514-v1:0",
		opts: options{
			anthropicBeta: []string{"custom-beta"},
		},
	}
	body := map[string]any{}
	m.applyBedrockOptions(body, nil, nil)
	additional := body["additionalModelRequestFields"].(map[string]any)
	betas, ok := additional["anthropic_beta"].([]string)
	if !ok {
		t.Fatal("missing anthropic_beta")
	}
	if len(betas) != 1 || betas[0] != "custom-beta" {
		t.Errorf("betas = %v", betas)
	}
}

func TestParseEventStream_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in parseEventStream with a
	// cancelled context and unbuffered channel.

	mustJSON := func(v any) []byte {
		b, _ := json.Marshal(v)
		return b
	}

	tests := []struct {
		name   string
		frames [][]byte
	}{
		{
			// Decoder error (line 432) -- corrupt binary data causes CRC mismatch.
			name: "decode_error",
			frames: [][]byte{
				{0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00},
			},
		},
		{
			// Exception frame (line 440)
			name: "exception_frame",
			frames: [][]byte{
				buildExceptionFrame("validationException", "invalid request"),
			},
		},
		{
			// contentBlockStart tool_use (line 470)
			name: "tool_start",
			frames: [][]byte{
				buildEventStreamFrame("contentBlockStart", mustJSON(map[string]any{
					"contentBlockIndex": 0,
					"start": map[string]any{
						"toolUse": map[string]any{
							"toolUseId": "t1",
							"name":      "fn",
						},
					},
				})),
			},
		},
		{
			// contentBlockDelta text (line 486)
			name: "text_delta",
			frames: [][]byte{
				buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
					"contentBlockIndex": 0,
					"delta": map[string]any{
						"text": "hello",
					},
				})),
			},
		},
		{
			// contentBlockDelta reasoningContent.text (line 509)
			name: "reasoning_text",
			frames: [][]byte{
				buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
					"contentBlockIndex": 0,
					"delta": map[string]any{
						"reasoningContent": map[string]any{
							"text": "thinking",
						},
					},
				})),
			},
		},
		{
			// contentBlockDelta reasoningContent.signature (line 514)
			name: "reasoning_signature",
			frames: [][]byte{
				buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
					"contentBlockIndex": 0,
					"delta": map[string]any{
						"reasoningContent": map[string]any{
							"signature": "sig123",
						},
					},
				})),
			},
		},
		{
			// contentBlockDelta reasoningContent.reasoningText (line 527)
			name: "reasoning_text_wrapper",
			frames: [][]byte{
				buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
					"contentBlockIndex": 0,
					"delta": map[string]any{
						"reasoningContent": map[string]any{
							"reasoningText": map[string]any{
								"text":      "wrapped",
								"signature": "sig456",
							},
						},
					},
				})),
			},
		},
		{
			// contentBlockDelta reasoningContent.redactedReasoning (line 533)
			name: "redacted_reasoning",
			frames: [][]byte{
				buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
					"contentBlockIndex": 0,
					"delta": map[string]any{
						"reasoningContent": map[string]any{
							"redactedReasoning": map[string]any{
								"data": "encrypted",
							},
						},
					},
				})),
			},
		},
		{
			// messageStop (line 557)
			name: "message_stop",
			frames: [][]byte{
				buildEventStreamFrame("messageStop", mustJSON(map[string]any{
					"stopReason": "end_turn",
				})),
			},
		},
		{
			// metadata ChunkFinish (line 584)
			name: "metadata_finish",
			frames: [][]byte{
				buildEventStreamFrame("metadata", mustJSON(map[string]any{
					"usage": map[string]any{
						"inputTokens":  10,
						"outputTokens": 5,
						"totalTokens":  15,
					},
				})),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			var buf bytes.Buffer
			for _, f := range tc.frames {
				buf.Write(f)
			}

			out := make(chan provider.StreamChunk) // unbuffered
			done := make(chan struct{})
			go func() {
				parseEventStream(ctx, io.NopCloser(&buf), out, provider.ResponseMetadata{})
				close(done)
			}()
			<-done
			for range out {
			}
		})
	}

	// Nested: contentBlockDelta toolUse (line 500) requires tool start TrySend to succeed.
	t.Run("tool_delta_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		var buf bytes.Buffer
		buf.Write(buildEventStreamFrame("contentBlockStart", mustJSON(map[string]any{
			"contentBlockIndex": 0,
			"start": map[string]any{
				"toolUse": map[string]any{
					"toolUseId": "t1",
					"name":      "fn",
				},
			},
		})))
		buf.Write(buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
			"contentBlockIndex": 0,
			"delta": map[string]any{
				"toolUse": map[string]any{
					"input": "{\"a\":1}",
				},
			},
		})))

		done := make(chan struct{})
		go func() {
			parseEventStream(ctx, io.NopCloser(&buf), out, provider.ResponseMetadata{})
			close(done)
		}()

		<-out // tool start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: contentBlockStop tool (line 546) requires tool start to succeed.
	t.Run("tool_stop_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		var buf bytes.Buffer
		buf.Write(buildEventStreamFrame("contentBlockStart", mustJSON(map[string]any{
			"contentBlockIndex": 0,
			"start": map[string]any{
				"toolUse": map[string]any{
					"toolUseId": "t1",
					"name":      "fn",
				},
			},
		})))
		buf.Write(buildEventStreamFrame("contentBlockStop", mustJSON(map[string]any{
			"contentBlockIndex": 0,
		})))

		done := make(chan struct{})
		go func() {
			parseEventStream(ctx, io.NopCloser(&buf), out, provider.ResponseMetadata{})
			close(done)
		}()

		<-out // tool start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: messageStop (line 557) after text delta.
	t.Run("message_stop_after_text_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		var buf bytes.Buffer
		buf.Write(buildEventStreamFrame("contentBlockDelta", mustJSON(map[string]any{
			"contentBlockIndex": 0,
			"delta": map[string]any{
				"text": "hi",
			},
		})))
		buf.Write(buildEventStreamFrame("messageStop", mustJSON(map[string]any{
			"stopReason": "end_turn",
		})))

		done := make(chan struct{})
		go func() {
			parseEventStream(ctx, io.NopCloser(&buf), out, provider.ResponseMetadata{})
			close(done)
		}()

		<-out // text delta
		cancel()
		<-done
		for range out {
		}
	})
}

// === Coverage gap tests for 100% ===

// --- #19: DoGenerate + injectResponseFormatTool + extractResponseFormatResult ---

func TestDoGenerate_ResponseFormat(t *testing.T) {
	// Tests DoGenerate with ResponseFormat set, which triggers injectResponseFormatTool
	// and extractResponseFormatResult.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify the synthetic tool was injected.
		tc, ok := req["toolConfig"].(map[string]any)
		if !ok {
			t.Fatalf("missing toolConfig, body = %s", string(body))
		}
		tools, _ := tc["tools"].([]any)
		if len(tools) == 0 {
			t.Fatal("expected injected __json_response tool")
		}
		firstTool := tools[0].(map[string]any)
		toolSpec := firstTool["toolSpec"].(map[string]any)
		if toolSpec["name"] != "__json_response" {
			t.Errorf("first tool name = %v, want __json_response", toolSpec["name"])
		}

		// Verify tool choice was set to the synthetic tool name.
		choice, _ := tc["toolChoice"].(map[string]any)
		toolChoice, _ := choice["tool"].(map[string]any)
		if toolChoice["name"] != "__json_response" {
			t.Errorf("toolChoice = %v, want __json_response", toolChoice["name"])
		}

		// Respond with a tool call for __json_response.
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Requestid", "rf-test")
		_, _ = fmt.Fprint(w, `{
			"output": {"message": {"role": "assistant", "content": [
				{"toolUse": {"toolUseId": "tc_rf", "name": "__json_response", "input": {"name": "Alice", "age": 30}}}
			]}},
			"stopReason": "tool_use",
			"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
		}`)
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Return a person"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// extractResponseFormatResult should convert the tool call to text.
	if result.Text == "" {
		t.Error("expected Text to be populated from synthetic tool call")
	}
	// The synthetic tool call should be removed.
	if len(result.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %d, want 0 (synthetic tool should be removed)", len(result.ToolCalls))
	}
	// FinishReason should be stop (not tool_calls) since the only tool call was the synthetic one.
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
}

func TestDoGenerate_ResponseFormat_WithOtherToolCalls(t *testing.T) {
	// When __json_response is in the tool calls but other tool calls remain,
	// FinishReason should NOT be changed to stop.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"output": {"message": {"role": "assistant", "content": [
				{"toolUse": {"toolUseId": "tc_other", "name": "search", "input": {"q": "test"}}},
				{"toolUse": {"toolUseId": "tc_rf", "name": "__json_response", "input": {"answer": "42"}}}
			]}},
			"stopReason": "tool_use",
			"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
		}`)
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Text should be extracted from __json_response tool call.
	if result.Text == "" {
		t.Error("expected Text from synthetic tool call")
	}
	// Other tool call should remain.
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "search" {
		t.Errorf("ToolCalls = %v, want [search]", result.ToolCalls)
	}
	// FinishReason should stay as tool_calls since other tools remain.
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want tool_calls", result.FinishReason)
	}
}

func TestExtractResponseFormatResult_NoMatch(t *testing.T) {
	// When no __json_response tool call exists, nothing should change.
	result := &provider.GenerateResult{
		ToolCalls: []provider.ToolCall{
			{ID: "tc1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
		},
		FinishReason: provider.FinishToolCalls,
	}
	extractResponseFormatResult(result)
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
}

// --- #20: DoStream with ResponseFormat ---

func TestDoStream_ResponseFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify the synthetic tool was injected.
		tc, ok := req["toolConfig"].(map[string]any)
		if !ok {
			t.Errorf("missing toolConfig, body = %s", string(body))
		} else {
			tools, _ := tc["tools"].([]any)
			if len(tools) > 0 {
				firstTool := tools[0].(map[string]any)
				toolSpec := firstTool["toolSpec"].(map[string]any)
				if toolSpec["name"] != "__json_response" {
					t.Errorf("first tool name = %v, want __json_response", toolSpec["name"])
				}
			}
		}

		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"ok"}}`)))
		_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
		_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
	}))
	defer server.Close()

	model := Chat("anthropic.claude-sonnet-4-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
}

// --- #21: toolBetaForType remaining branches ---

func TestToolBetaForType_AllBranches(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		// Already tested:
		{"computer_20241022", "computer-use-2024-10-22"},
		{"bash_20241022", "computer-use-2024-10-22"},
		{"text_editor_20241022", "computer-use-2024-10-22"},
		{"computer_20250124", "computer-use-2025-01-24"},
		{"bash_20250124", "computer-use-2025-01-24"},
		{"text_editor_20250124", "computer-use-2025-01-24"},
		{"text_editor_20250429", "computer-use-2025-01-24"},
		// Uncovered branches:
		{"computer_20251124", "computer-use-2025-11-24"},
		{"text_editor_20250728", ""},
		{"code_execution_20250825", "code-execution-2025-08-25"},
		{"code_execution_20260120", ""},
		{"web_search_20260209", ""},
		{"web_fetch_20260209", ""},
		// Default:
		{"unknown", ""},
		{"", ""},
	}
	for _, tt := range tests {
		got := toolBetaForType(tt.input)
		if got != tt.want {
			t.Errorf("toolBetaForType(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// --- #22: parseConverseResponse tool input marshal error ---

func TestParseConverseResponse_ToolInputMarshalError(t *testing.T) {
	// The json.Marshal error in parseConverseResponse (line 353-355) is triggered when
	// tu["input"] contains a value that cannot be marshaled. In practice, JSON
	// unmarshal produces Go values that are always re-marshalable, so this path
	// is nearly impossible to hit through normal flow. We test via direct call
	// with a crafted JSON that has valid but complex nested input.
	// Actually, the error path requires json.Marshal to fail, which can't happen
	// with json.Unmarshal output. This is dead code - but let's verify the
	// happy path with input present.
	body := []byte(`{
		"output": {"message": {"role": "assistant", "content": [
			{"toolUse": {"toolUseId": "tc1", "name": "fn", "input": {"nested": {"deep": true}}}}
		]}},
		"stopReason": "tool_use",
		"usage": {"inputTokens": 1, "outputTokens": 1}
	}`)
	result, err := parseConverseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "fn" {
		t.Errorf("Name = %q", result.ToolCalls[0].Name)
	}
}

// --- #23: DoStream drain goroutine ctx.Done during tool streaming retry ---

func TestDoStream_ToolStreamingRetry_DrainCtxDone(t *testing.T) {
	// This tests the ctx.Done path in the drain goroutine (line 329-330)
	// that runs when the original stream is being drained during a tool
	// streaming retry. The first connection must stay open (blocking the
	// drain goroutine on innerCh) while context is cancelled.
	firstConnHold := make(chan struct{}) // signals first handler to finish
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		if callCount == 1 {
			// Send a tool streaming error exception.
			_, _ = w.Write(buildExceptionFrame("validationException", `tool streaming is not supported`))
			w.(http.Flusher).Flush()
			// Hold connection open so drain goroutine blocks on innerCh.
			<-firstConnHold
		} else {
			_, _ = w.Write(buildEventStreamFrame("contentBlockDelta", []byte(`{"contentBlockIndex":0,"delta":{"text":"retried"}}`)))
			_, _ = w.Write(buildEventStreamFrame("messageStop", []byte(`{"stopReason":"end_turn"}`)))
			_, _ = w.Write(buildEventStreamFrame("metadata", []byte(`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)))
		}
	}))
	defer server.Close()
	defer close(firstConnHold)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	model := Chat("nvidia.nemotron", WithAccessKey("AK"), WithSecretKey("SK"), WithBaseURL(server.URL))
	result, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "test", Description: "t", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Cancel context while drain goroutine is blocked on innerCh
	// (first connection is held open).
	cancel()
	// Drain the result stream.
	for range result.Stream {
	}
}

func TestModelSupportsTools(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		{"anthropic.claude-sonnet-4-20250514-v1:0", true},
		{"anthropic.claude-3-5-haiku-20241022-v1:0", true},
		{"deepseek.r1-v1:0", false},
		{"deepseek.v3-v1:0", true},
		{"deepseek.v3.2", true},
		{"amazon.titan-text-lite-v1", false},
		{"amazon.titan-text-express-v1", false},
		{"amazon.nova-pro-v1:0", true},
		{"meta.llama3-2-90b-instruct-v1:0", true},
		{"us.deepseek.r1-v1:0", false},
		{"us.deepseek.v3-v1:0", true},
	}
	for _, tt := range tests {
		got := modelSupportsTools(tt.modelID)
		if got != tt.want {
			t.Errorf("modelSupportsTools(%q) = %v, want %v", tt.modelID, got, tt.want)
		}
	}
}

func TestCapabilities_PerModel(t *testing.T) {
	m := Chat("deepseek.r1-v1:0").(*chatModel)
	caps := m.Capabilities()
	if caps.ToolCall {
		t.Error("deepseek.r1-v1:0 should have ToolCall=false")
	}
	m2 := Chat("deepseek.v3-v1:0").(*chatModel)
	caps2 := m2.Capabilities()
	if !caps2.ToolCall {
		t.Error("deepseek.v3-v1:0 should have ToolCall=true")
	}
}

func TestEnsureToolConfigForHistory_NoToolBlocks(t *testing.T) {
	body := map[string]any{}
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
	}
	ensureToolConfigForHistory(body, msgs)
	if _, ok := body["toolConfig"]; ok {
		t.Error("should not add toolConfig when no tool blocks")
	}
}

func TestEnsureToolConfigForHistory_WithToolBlocks(t *testing.T) {
	body := map[string]any{}
	msgs := []provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolName: "bash", ToolCallID: "1"},
		}},
	}
	ensureToolConfigForHistory(body, msgs)
	tc, ok := body["toolConfig"]
	if !ok {
		t.Fatal("should add toolConfig when tool blocks present")
	}
	tools := tc.(map[string]any)["tools"].([]map[string]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	spec := tools[0]["toolSpec"].(map[string]any)
	if spec["name"] != "bash" {
		t.Errorf("expected tool name 'bash', got %v", spec["name"])
	}
}

func TestEnsureToolConfigForHistory_SkipsIfExists(t *testing.T) {
	body := map[string]any{"toolConfig": map[string]any{"tools": []any{}}}
	msgs := []provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolName: "bash", ToolCallID: "1"},
		}},
	}
	ensureToolConfigForHistory(body, msgs)
	tools := body["toolConfig"].(map[string]any)["tools"].([]any)
	if len(tools) != 0 {
		t.Error("should not modify existing toolConfig")
	}
}

func TestEnsureToolConfigForHistory_MultipleTools(t *testing.T) {
	body := map[string]any{}
	msgs := []provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolName: "bash", ToolCallID: "1"},
			{Type: provider.PartToolCall, ToolName: "read", ToolCallID: "2"},
			{Type: provider.PartToolCall, ToolName: "bash", ToolCallID: "3"},
		}},
	}
	ensureToolConfigForHistory(body, msgs)
	tools := body["toolConfig"].(map[string]any)["tools"].([]map[string]any)
	if len(tools) != 2 {
		t.Fatalf("expected 2 unique tools, got %d", len(tools))
	}
}
