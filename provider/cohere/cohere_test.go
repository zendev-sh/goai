package cohere

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"Hello\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
		if r.URL.Path != "/chat" {
			t.Errorf("path = %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["stream"] != false {
			t.Error("expected stream=false")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":5}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"message":"Rate limited"}`)
	}))
	defer server.Close()

	model := Chat("model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNoTokenSource(t *testing.T) {
	model := Chat("model")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithAPIKey("key"), WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithAPIKey("k"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("command-r-plus", WithAPIKey("k"))
	if model.ModelID() != "command-r-plus" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
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

func TestEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			t.Errorf("path = %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_document" {
			t.Errorf("input_type = %v", req["input_type"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2,0.3],[0.4,0.5,0.6]]},"meta":{"billed_units":{"input_tokens":10}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Errorf("embeddings count = %d", len(result.Embeddings))
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", result.Usage.InputTokens)
	}
}

func TestEmbeddingModelID(t *testing.T) {
	model := Embedding("embed-v4.0", WithAPIKey("k"))
	if model.ModelID() != "embed-v4.0" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestEmbeddingMaxValues(t *testing.T) {
	model := Embedding("embed-v4.0", WithAPIKey("k"))
	if model.MaxValuesPerCall() != 96 {
		t.Errorf("MaxValuesPerCall() = %d", model.MaxValuesPerCall())
	}
}

func TestEmbeddingNoTokenSource(t *testing.T) {
	model := Embedding("embed-v4.0")
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dynamic" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("dynamic")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChatWithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, _ := req["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools = %d, want 1", len(tools))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[],"tool_calls":[{"id":"tc1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":10,"output_tokens":20}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get weather", InputSchema: []byte(`{"type":"object"}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "get_weather" {
		t.Errorf("ToolName = %q", result.ToolCalls[0].Name)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q", result.FinishReason)
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
		WithAPIKey("k"),
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

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestStream_ToolCallEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"city\\\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\": \\\"Paris\\\"}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":20}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotToolStart bool
	var gotToolCall bool
	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCallStreamStart {
			gotToolStart = true
			if chunk.ToolName != "get_weather" {
				t.Errorf("ToolName = %q", chunk.ToolName)
			}
			if chunk.ToolCallID != "tc1" {
				t.Errorf("ToolCallID = %q", chunk.ToolCallID)
			}
		}
		if chunk.Type == provider.ChunkToolCall {
			gotToolCall = true
			if chunk.ToolCallID != "tc1" {
				t.Errorf("ToolCallID = %q", chunk.ToolCallID)
			}
			if chunk.ToolName != "get_weather" {
				t.Errorf("ToolName = %q", chunk.ToolName)
			}
			if chunk.ToolInput != `{"city": "Paris"}` {
				t.Errorf("ToolInput = %q, want %q", chunk.ToolInput, `{"city": "Paris"}`)
			}
		}
		if chunk.Type == provider.ChunkFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishToolCalls {
				t.Errorf("FinishReason = %q", chunk.FinishReason)
			}
		}
	}
	if !gotToolStart {
		t.Error("no tool call start chunk")
	}
	if !gotToolCall {
		t.Error("no tool call chunk with accumulated arguments")
	}
	if !gotFinish {
		t.Error("no finish chunk")
	}
}

func TestStream_ContextCancellation(t *testing.T) {
	// Test context cancellation - with TrySend, the goroutine exits cleanly
	// instead of sending an error chunk (which could block if buffer is full).
	ctx, cancel := context.WithCancel(t.Context())

	out := make(chan provider.StreamChunk, 64)
	data := "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n"
	cancel() // cancel before parsing

	done := make(chan struct{})
	go func() {
		parseChatStream(ctx, strings.NewReader(data), out)
		close(done)
	}()
	<-done

	// Goroutine exited without blocking - drain any chunks.
	for range out {
	}
}

func TestStream_ScannerError(t *testing.T) {
	out := make(chan provider.StreamChunk, 64)
	go parseChatStream(t.Context(), errReader{}, out)

	var gotError bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			gotError = true
			if !strings.Contains(chunk.Error.Error(), "reading stream") {
				t.Errorf("unexpected error: %s", chunk.Error)
			}
		}
	}
	if !gotError {
		t.Error("expected error chunk from scanner error")
	}
}

func TestStream_MaxTokensFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"truncated\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"MAX_TOKENS\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":100}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishLength {
				t.Errorf("FinishReason = %q, want length", chunk.FinishReason)
			}
		}
	}
	if !gotFinish {
		t.Error("no finish chunk")
	}
}

func TestStream_UnknownFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"UNKNOWN_REASON\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.FinishReason != provider.FinishOther {
				t.Errorf("FinishReason = %q, want other", chunk.FinishReason)
			}
		}
	}
}

func TestStream_BilledUnitsFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// No "tokens" field -- should fall back to billed_units.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"billed_units\":{\"input_tokens\":15,\"output_tokens\":25}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.Usage.InputTokens != 15 {
				t.Errorf("InputTokens = %d, want 15", chunk.Usage.InputTokens)
			}
			if chunk.Usage.OutputTokens != 25 {
				t.Errorf("OutputTokens = %d, want 25", chunk.Usage.OutputTokens)
			}
		}
	}
}

func TestStream_DONE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		// This should not be reached after [DONE].
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"after done\"}}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
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
	if len(texts) != 1 || texts[0] != "hi" {
		t.Errorf("texts = %v, want [hi]", texts)
	}
}

func TestStream_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {invalid json}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"after\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Invalid JSON lines should be skipped; valid ones should still work.
	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "after" {
		t.Errorf("texts = %v, want [after]", texts)
	}
}

func TestStream_NonDataLines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: ping\n\n")
		_, _ = fmt.Fprint(w, ": comment\n\n")
		_, _ = fmt.Fprint(w, "\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"ok\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
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
	if len(texts) != 1 || texts[0] != "ok" {
		t.Errorf("texts = %v, want [ok]", texts)
	}
}

func TestGenerate_ToolCallFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[],"tool_calls":[{"id":"tc1","type":"function","function":{"name":"calc","arguments":"{\"x\":1}"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":5,"output_tokens":10}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
}

func TestGenerate_MaxTokensFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"truncated"}]},"finish_reason":"MAX_TOKENS","usage":{"tokens":{"input_tokens":5,"output_tokens":100}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishLength {
		t.Errorf("FinishReason = %q, want length", result.FinishReason)
	}
}

func TestGenerate_UnknownFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"UNKNOWN","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishOther {
		t.Errorf("FinishReason = %q, want other", result.FinishReason)
	}
}

func TestGenerate_BilledUnitsFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// tokens is zero, should fall back to billed_units.
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":15,"output_tokens":25},"tokens":{"input_tokens":0,"output_tokens":0}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokens != 15 {
		t.Errorf("InputTokens = %d, want 15", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 25 {
		t.Errorf("OutputTokens = %d, want 25", result.Usage.OutputTokens)
	}
}

func TestGenerate_MultipleContentBlocks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
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
}

func TestGenerate_EmptyContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":0}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
}

func TestBuildChatRequest_ToolResultMessages(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
			{Role: provider.RoleAssistant, Content: []provider.Part{
				{Type: provider.PartToolCall, ToolCallID: "tc1", ToolName: "calc", ToolInput: json.RawMessage(`{"x":1}`)},
			}},
			{Role: "tool", Content: []provider.Part{
				{Type: provider.PartToolResult, ToolCallID: "tc1", ToolOutput: "42"},
			}},
		},
	}
	body := buildChatRequest(params, "model", false)
	msgs, _ := body["messages"].([]map[string]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3", len(msgs))
	}
	// Third message should be tool result.
	if msgs[2]["role"] != "tool" {
		t.Errorf("role = %v", msgs[2]["role"])
	}
	if msgs[2]["tool_call_id"] != "tc1" {
		t.Errorf("tool_call_id = %v", msgs[2]["tool_call_id"])
	}
	if msgs[2]["content"] != "42" {
		t.Errorf("content = %v", msgs[2]["content"])
	}
}

func TestBuildChatRequest_ToolCallInAssistant(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleAssistant, Content: []provider.Part{
				{Type: provider.PartToolCall, ToolCallID: "tc1", ToolName: "calc", ToolInput: json.RawMessage(`{"x":1}`)},
			}},
		},
	}
	body := buildChatRequest(params, "model", false)
	msgs, _ := body["messages"].([]map[string]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %d, want 1", len(msgs))
	}
	tcs, _ := msgs[0]["tool_calls"].([]map[string]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls = %d, want 1", len(tcs))
	}
	// When there are tool calls but no text, content should be empty string.
	if msgs[0]["content"] != "" {
		t.Errorf("content = %v, want empty string", msgs[0]["content"])
	}
}

func TestBuildChatRequest_SystemMessage(t *testing.T) {
	params := provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}
	body := buildChatRequest(params, "model", false)
	msgs, _ := body["messages"].([]map[string]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0]["role"] != "system" {
		t.Errorf("first msg role = %v", msgs[0]["role"])
	}
}

func TestBuildChatRequest_WithParams(t *testing.T) {
	temp := 0.5
	topP := 0.9
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		MaxOutputTokens: 100,
		Temperature:     &temp,
		TopP:            &topP,
	}
	body := buildChatRequest(params, "model", false)
	if body["max_tokens"] != 100 {
		t.Errorf("max_tokens = %v", body["max_tokens"])
	}
	if body["temperature"] != 0.5 {
		t.Errorf("temperature = %v", body["temperature"])
	}
	if body["p"] != 0.9 {
		t.Errorf("p = %v", body["p"])
	}
}

func TestBuildChatRequest_InvalidToolSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "t", Description: "d", InputSchema: []byte(`{invalid}`)},
		},
	}
	body := buildChatRequest(params, "model", false)
	tools, _ := body["tools"].([]map[string]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(tools))
	}
	// Invalid schema should fallback to empty map.
	fn, _ := tools[0]["function"].(map[string]any)
	schema, ok := fn["parameters"].(map[string]any)
	if !ok || len(schema) != 0 {
		t.Errorf("parameters = %v, want empty map", fn["parameters"])
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"message":"bad request"}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedding_ReadError(t *testing.T) {
	model := Embedding("embed-v4.0",
		WithAPIKey("k"),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ParseError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{invalid json}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ConnectionError(t *testing.T) {
	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Embedding("embed-v4.0", WithAPIKey("k"), WithHTTPClient(c))
	em := model.(*embeddingModel)
	if em.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestEmbedding_WithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- BF.16: Tool call delta accumulation ---

func TestStream_ToolCallDeltaAccumulation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Start with initial args fragment.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"x\\\"\"}}}}}\n\n")
		// Delta fragments.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\": 1, \"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"\\\"y\\\": 2}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":10}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolInput string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolInput = chunk.ToolInput
		}
	}
	if toolInput != `{"x": 1, "y": 2}` {
		t.Errorf("accumulated ToolInput = %q, want %q", toolInput, `{"x": 1, "y": 2}`)
	}
}

func TestStream_ToolCallNullArgs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"noop\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"null\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "noop"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			if chunk.ToolInput != "{}" {
				t.Errorf("ToolInput = %q, want %q (null should become empty object)", chunk.ToolInput, "{}")
			}
		}
	}
}

func TestStream_MultipleToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// First tool call.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"add\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"a\\\": 1}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		// Second tool call.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc2\",\"type\":\"function\",\"function\":{\"name\":\"mul\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"b\\\": 2}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":10}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}
	if len(toolCalls) != 2 {
		t.Fatalf("got %d tool calls, want 2", len(toolCalls))
	}
	if toolCalls[0].ToolCallID != "tc1" || toolCalls[0].ToolName != "add" {
		t.Errorf("first tool call: id=%q name=%q", toolCalls[0].ToolCallID, toolCalls[0].ToolName)
	}
	if toolCalls[1].ToolCallID != "tc2" || toolCalls[1].ToolName != "mul" {
		t.Errorf("second tool call: id=%q name=%q", toolCalls[1].ToolCallID, toolCalls[1].ToolName)
	}
}

// --- BF.16: Thinking / reasoning support ---

func TestBuildChatRequest_Thinking(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type":        "enabled",
				"budgetTokens": 4096,
			},
		},
	}
	body := buildChatRequest(params, "command-a-reasoning", false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not set in request body")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v", thinking["type"])
	}
	if thinking["token_budget"] != 4096 {
		t.Errorf("thinking.token_budget = %v", thinking["token_budget"])
	}
}

func TestBuildChatRequest_ThinkingDefaultType(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"budgetTokens": 2048,
			},
		},
	}
	body := buildChatRequest(params, "command-a-reasoning", false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not set in request body")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want 'enabled' (default)", thinking["type"])
	}
}

func TestBuildChatRequest_ThinkingDisabled(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type": "disabled",
			},
		},
	}
	body := buildChatRequest(params, "model", false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not set in request body")
	}
	if thinking["type"] != "disabled" {
		t.Errorf("thinking.type = %v", thinking["type"])
	}
}

func TestBuildChatRequest_NoThinking(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}
	body := buildChatRequest(params, "model", false)
	if _, ok := body["thinking"]; ok {
		t.Error("thinking should not be set when not in ProviderOptions")
	}
}

func TestGenerate_ThinkingContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"thinking","thinking":"Let me reason about this..."},{"type":"text","text":"The answer is 42."}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":20}}}`)
	}))
	defer server.Close()

	model := Chat("command-a-reasoning", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "The answer is 42." {
		t.Errorf("Text = %q", result.Text)
	}
	// Reasoning should be in ProviderMetadata.
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	reasoning, ok := result.ProviderMetadata["cohere"]["reasoning"].(string)
	if !ok || reasoning != "Let me reason about this..." {
		t.Errorf("reasoning = %q", reasoning)
	}
}

func TestStream_ThinkingContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Thinking content block.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-start\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"thinking\",\"thinking\":\"\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"thinking\":\"Reasoning...\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-end\",\"index\":0}\n\n")
		// Text content block.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-start\",\"index\":1,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"index\":1,\"delta\":{\"message\":{\"content\":{\"text\":\"Answer\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-end\",\"index\":1}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":20}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotReasoning bool
	var gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning {
			gotReasoning = true
			if chunk.Text != "Reasoning..." {
				t.Errorf("reasoning text = %q", chunk.Text)
			}
		}
		if chunk.Type == provider.ChunkText {
			gotText = true
			if chunk.Text != "Answer" {
				t.Errorf("text = %q", chunk.Text)
			}
		}
	}
	if !gotReasoning {
		t.Error("no reasoning chunk")
	}
	if !gotText {
		t.Error("no text chunk")
	}
}

// --- BF.16: Embedding input_type and truncate ---

func TestEmbedding_InputType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_query" {
			t.Errorf("input_type = %v, want search_query", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":5}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"query"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "search_query",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("embeddings = %d", len(result.Embeddings))
	}
}

func TestEmbedding_InputTypeClassification(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "classification" {
			t.Errorf("input_type = %v, want classification", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "classification",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_InputTypeClustering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "clustering" {
			t.Errorf("input_type = %v, want clustering", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "clustering",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_DefaultInputType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_document" {
			t.Errorf("input_type = %v, want search_document (default)", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_Truncate(t *testing.T) {
	for _, tc := range []string{"NONE", "START", "END"} {
		t.Run(tc, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var req map[string]any
				_ = json.Unmarshal(body, &req)
				if req["truncate"] != tc {
					t.Errorf("truncate = %v, want %q", req["truncate"], tc)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
			}))
			defer server.Close()

			model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
			_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
				ProviderOptions: map[string]any{
					"truncate": tc,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEmbedding_NoTruncateByDefault(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if _, ok := req["truncate"]; ok {
			t.Errorf("truncate should not be set by default, got %v", req["truncate"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_InputTypeAndTruncateCombined(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_query" {
			t.Errorf("input_type = %v", req["input_type"])
		}
		if req["truncate"] != "END" {
			t.Errorf("truncate = %v", req["truncate"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":5}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "search_query",
			"truncate":  "END",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- BF.16: Streaming message-end with delta.finish_reason ---

func TestStream_MessageEndDeltaFormat(t *testing.T) {
	// Vercel reference uses delta.finish_reason and delta.usage.tokens for message-end.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":8,\"output_tokens\":3}}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.FinishReason != provider.FinishStop {
				t.Errorf("FinishReason = %q, want stop", chunk.FinishReason)
			}
			if chunk.Usage.InputTokens != 8 {
				t.Errorf("InputTokens = %d, want 8", chunk.Usage.InputTokens)
			}
			if chunk.Usage.OutputTokens != 3 {
				t.Errorf("OutputTokens = %d, want 3", chunk.Usage.OutputTokens)
			}
		}
	}
}

func TestChat_Generate_UnknownFinishReason(t *testing.T) {
	// When finish_reason is missing/empty, parseChatResponse should fall back to FinishOther.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// No finish_reason field → defaults to empty string → mapFinishReason("") returns ""
		_, _ = fmt.Fprint(w, `{"id":"resp-1","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishOther {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishOther)
	}
}

func TestChat_Stream_UnknownFinishReason(t *testing.T) {
	// When message-end has no finish_reason (or empty) at both event.FinishReason
	// and event.Delta.FinishReason, parseChatStream should fall back to FinishOther.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"ok\"}}}}\n\n")
		// message-end with no finish_reason at either level
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"delta\":{\"usage\":{\"tokens\":{\"input_tokens\":3,\"output_tokens\":1}}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkStepFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishOther {
				t.Errorf("FinishReason = %q, want %q", chunk.FinishReason, provider.FinishOther)
			}
		}
	}
	if !gotFinish {
		t.Error("no step_finish chunk received")
	}
}

func TestChat_PerRequestHeaders(t *testing.T) {
	// Verify that per-request headers (params.Headers) are extracted from the body
	// as _headers and applied to the HTTP request.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify custom header was applied.
		if got := r.Header.Get("X-Custom-Header"); got != "custom-value" {
			t.Errorf("X-Custom-Header = %q, want %q", got, "custom-value")
		}
		if got := r.Header.Get("X-Another"); got != "another-value" {
			t.Errorf("X-Another = %q, want %q", got, "another-value")
		}

		// Verify _headers is NOT in the JSON body (should be stripped).
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]any
		_ = json.Unmarshal(body, &reqBody)
		if _, ok := reqBody["_headers"]; ok {
			t.Error("_headers should be stripped from the request body")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"test","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{
			"X-Custom-Header": "custom-value",
			"X-Another":       "another-value",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestBuildChatRequest_HeadersInjection(t *testing.T) {
	// Verify that buildChatRequest adds _headers to the body when params.Headers is set.
	body := buildChatRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{
			"X-Test": "value",
		},
	}, "command-r-plus", false)

	headers, ok := body["_headers"].(map[string]string)
	if !ok {
		t.Fatal("_headers not found in request body")
	}
	if headers["X-Test"] != "value" {
		t.Errorf("_headers[X-Test] = %q, want %q", headers["X-Test"], "value")
	}
}

func TestBuildChatRequest_ResponseFormatWithSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "person",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	}
	body := buildChatRequest(params, "command-r-plus", false)
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format not set")
	}
	if rf["type"] != "json_object" {
		t.Errorf("type = %v, want json_object", rf["type"])
	}
	js, ok := rf["json_schema"].(map[string]any)
	if !ok {
		t.Fatal("json_schema not set")
	}
	if js["name"] != "person" {
		t.Errorf("name = %v, want person", js["name"])
	}
}

func TestBuildChatRequest_ResponseFormatNoSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{},
	}
	body := buildChatRequest(params, "command-r-plus", false)
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format not set")
	}
	if rf["type"] != "json_object" {
		t.Errorf("type = %v, want json_object", rf["type"])
	}
	if _, ok := rf["json_schema"]; ok {
		t.Error("json_schema should not be set when Schema is empty")
	}
}

func TestBuildChatRequest_NoHeadersWhenEmpty(t *testing.T) {
	// Verify that buildChatRequest does NOT add _headers when params.Headers is empty.
	body := buildChatRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}, "command-r-plus", false)

	if _, ok := body["_headers"]; ok {
		t.Error("_headers should not be present when Headers is empty")
	}
}

// --- Citation tests ---

func TestChat_Generate_Citations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "resp-1",
			"message": {
				"role": "assistant",
				"content": [{"type": "text", "text": "The answer is documented."}],
				"citations": [
					{
						"start": 0,
						"end": 10,
						"text": "The answer",
						"sources": [
							{"document": {"id": "doc-1", "title": "Reference Doc", "text": "full text"}}
						]
					},
					{
						"start": 11,
						"end": 24,
						"text": "is documented",
						"sources": [
							{"document": {"id": "doc-2", "title": "Another Doc", "text": "more text"}}
						]
					}
				]
			},
			"finish_reason": "COMPLETE",
			"usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}}
		}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Sources) != 2 {
		t.Fatalf("len(Sources) = %d, want 2", len(result.Sources))
	}

	s0 := result.Sources[0]
	if s0.Type != "document" {
		t.Errorf("Sources[0].Type = %q, want document", s0.Type)
	}
	if s0.ID != "doc-1" {
		t.Errorf("Sources[0].ID = %q, want doc-1", s0.ID)
	}
	if s0.Title != "Reference Doc" {
		t.Errorf("Sources[0].Title = %q", s0.Title)
	}
	if s0.StartIndex != 0 || s0.EndIndex != 10 {
		t.Errorf("Sources[0] indices = (%d, %d), want (0, 10)", s0.StartIndex, s0.EndIndex)
	}
	if s0.ProviderMetadata == nil {
		t.Fatal("Sources[0].ProviderMetadata is nil")
	}
	if cohere, ok := s0.ProviderMetadata["cohere"].(map[string]any); ok {
		if cohere["text"] != "The answer" {
			t.Errorf("ProviderMetadata text = %v", cohere["text"])
		}
	} else {
		t.Error("Sources[0].ProviderMetadata missing cohere key")
	}

	s1 := result.Sources[1]
	if s1.ID != "doc-2" || s1.Title != "Another Doc" {
		t.Errorf("Sources[1] = %+v", s1)
	}
}

func TestChat_Generate_NoCitations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "resp-2",
			"message": {
				"role": "assistant",
				"content": [{"type": "text", "text": "No citations here."}]
			},
			"finish_reason": "COMPLETE",
			"usage": {"tokens": {"input_tokens": 5, "output_tokens": 3}}
		}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 0 {
		t.Errorf("len(Sources) = %d, want 0", len(result.Sources))
	}
}

func TestChat_Stream_Citations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"Hello\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"citation-start\",\"delta\":{\"message\":{\"citations\":{\"start\":0,\"end\":5,\"text\":\"Hello\",\"sources\":[{\"document\":{\"id\":\"d1\",\"title\":\"Doc1\",\"text\":\"hello text\"}}]}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var finishChunk provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}

	if finishChunk.Metadata == nil {
		t.Fatal("finish chunk has no metadata")
	}
	sources, ok := finishChunk.Metadata["sources"].([]provider.Source)
	if !ok {
		t.Fatal("finish chunk metadata missing 'sources'")
	}
	if len(sources) != 1 {
		t.Fatalf("len(sources) = %d, want 1", len(sources))
	}
	if sources[0].Type != "document" || sources[0].ID != "d1" {
		t.Errorf("sources[0] = %+v", sources[0])
	}
	if sources[0].Title != "Doc1" {
		t.Errorf("sources[0].Title = %q", sources[0].Title)
	}
}

func TestResolveEnv_APIKey(t *testing.T) {
	t.Setenv("COHERE_API_KEY", "env-key")
	m := Chat("command-r-plus")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from COHERE_API_KEY")
	}
}

func TestResolveEnv_BaseURL(t *testing.T) {
	t.Setenv("COHERE_API_KEY", "env-key")
	t.Setenv("COHERE_BASE_URL", "https://custom.cohere.com/v2")
	m := Chat("command-r-plus")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.cohere.com/v2" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestResolveEnv_NotOverrideExplicit(t *testing.T) {
	t.Setenv("COHERE_BASE_URL", "https://env.url")
	m := Chat("command-r-plus", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestParseChatStream_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in parseChatStream.

	tests := []struct {
		name  string
		input string
	}{
		{
			// content-delta thinking (line 662)
			name:  "thinking_delta",
			input: "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"thinking\":\"hmm\"}}}}\n",
		},
		{
			// content-delta text (line 669)
			name:  "text_delta",
			input: "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hello\"}}}}\n",
		},
		{
			// tool-call-start (line 691)
			name:  "tool_call_start",
			input: "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"t1\",\"function\":{\"name\":\"fn\"}}}}}\n",
		},
		{
			// message-end StepFinish (line 759)
			name:  "message_end",
			input: "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}}\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			out := make(chan provider.StreamChunk) // unbuffered
			done := make(chan struct{})
			go func() {
				parseChatStream(ctx, strings.NewReader(tc.input), out)
				close(done)
			}()
			<-done
			for range out {
			}
		})
	}

	// Nested: tool-call-end (line 710) requires tool-call-start TrySend to succeed first.
	t.Run("tool_call_end_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"t1\",\"function\":{\"name\":\"fn\"}}}}}\n" +
			"data: {\"type\":\"tool-call-end\"}\n"

		done := make(chan struct{})
		go func() {
			parseChatStream(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // tool start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: message-end ChunkFinish (line 774) requires StepFinish TrySend to succeed first.
	t.Run("message_end_finish_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}}\n"

		done := make(chan struct{})
		go func() {
			parseChatStream(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // step finish
		cancel()
		<-done
		for range out {
		}
	})
}
