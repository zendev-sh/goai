package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func ptr[T any](v T) *T { return &v }

func TestConvertMessage_Roles(t *testing.T) {
	// System.
	sys, err := convertMessage(provider.Message{
		Role:    provider.RoleSystem,
		Content: []provider.Part{{Type: provider.PartText, Text: "sys"}},
	})
	if err != nil {
		t.Fatalf("system: unexpected error: %v", err)
	}
	if len(sys) != 1 || sys[0].Role != "system" || sys[0].Content != "sys" {
		t.Errorf("system message = %+v", sys)
	}

	// User.
	usr, err := convertMessage(provider.Message{
		Role:    provider.RoleUser,
		Content: []provider.Part{{Type: provider.PartText, Text: "hi"}},
	})
	if err != nil {
		t.Fatalf("user: unexpected error: %v", err)
	}
	if len(usr) != 1 || usr[0].Role != "user" || usr[0].Content != "hi" {
		t.Errorf("user message = %+v", usr)
	}

	// Assistant: text + reasoning concatenate; tool call with nil input defaults to {}.
	asst, err := convertMessage(provider.Message{
		Role: provider.RoleAssistant,
		Content: []provider.Part{
			{Type: provider.PartText, Text: "ans"},
			{Type: provider.PartReasoning, Text: "why"},
			{Type: provider.PartToolCall, ToolName: "f", ToolInput: nil},
			{Type: provider.PartToolCall, ToolName: "g", ToolInput: json.RawMessage(`{"a":1}`)},
		},
	})
	if err != nil {
		t.Fatalf("assistant: unexpected error: %v", err)
	}
	if len(asst) != 1 {
		t.Fatalf("assistant produced %d messages, want 1", len(asst))
	}
	if asst[0].Content != "ans" {
		t.Errorf("assistant content = %q, want %q", asst[0].Content, "ans")
	}
	if asst[0].Thinking != "why" {
		t.Errorf("assistant thinking = %q, want %q", asst[0].Thinking, "why")
	}
	if len(asst[0].ToolCalls) != 2 {
		t.Fatalf("assistant tool calls = %d, want 2", len(asst[0].ToolCalls))
	}
	if got := string(asst[0].ToolCalls[0].Function.Arguments); got != "{}" {
		t.Errorf("nil tool input = %q, want {}", got)
	}
	if asst[0].ToolCalls[1].Function.Name != "g" {
		t.Errorf("tool call name = %q, want g", asst[0].ToolCalls[1].Function.Name)
	}

	// Tool: one Ollama message per result part.
	tool, err := convertMessage(provider.Message{
		Role: provider.RoleTool,
		Content: []provider.Part{
			{Type: provider.PartToolResult, ToolCallID: "id1", ToolOutput: "out1"},
			{Type: provider.PartToolResult, ToolCallID: "id2", ToolOutput: "out2"},
		},
	})
	if err != nil {
		t.Fatalf("tool: unexpected error: %v", err)
	}
	if len(tool) != 2 {
		t.Fatalf("tool produced %d messages, want 2", len(tool))
	}
	if tool[0].Role != "tool" || tool[0].ToolCallID != "id1" || tool[0].Content != "out1" {
		t.Errorf("tool message[0] = %+v", tool[0])
	}

	// Unsupported role.
	if _, err := convertMessage(provider.Message{Role: provider.Role("nope")}); err == nil {
		t.Error("expected error for unsupported role, got nil")
	}
}

func TestConvertTools(t *testing.T) {
	if tools, err := convertTools(nil); err != nil || tools != nil {
		t.Errorf("nil tools = (%v, %v), want (nil, nil)", tools, err)
	}

	tools, err := convertTools([]provider.ToolDefinition{
		{Name: "f", Description: "does f", InputSchema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tools) != 1 || tools[0].Type != "function" || tools[0].Function.Name != "f" {
		t.Errorf("tools = %+v", tools)
	}
	if tools[0].Function.Description != "does f" {
		t.Errorf("description = %q", tools[0].Function.Description)
	}

	if _, err := convertTools([]provider.ToolDefinition{
		{Name: "bad", InputSchema: json.RawMessage(`{not json`)},
	}); err == nil {
		t.Error("expected error for invalid input schema, got nil")
	}
}

func TestBuildOllamaOptions(t *testing.T) {
	if opts := buildOllamaOptions(provider.GenerateParams{}); opts != nil {
		t.Errorf("empty params = %v, want nil", opts)
	}

	opts := buildOllamaOptions(provider.GenerateParams{
		Temperature:     ptr(0.7),
		TopP:            ptr(0.9),
		TopK:            ptr(40),
		Seed:            ptr(7),
		StopSequences:   []string{"STOP"},
		MaxOutputTokens: 128,
	})
	for _, key := range []string{"temperature", "top_p", "top_k", "seed", "stop", "num_predict"} {
		if _, ok := opts[key]; !ok {
			t.Errorf("missing option %q in %v", key, opts)
		}
	}
	if opts["num_predict"] != 128 {
		t.Errorf("num_predict = %v, want 128", opts["num_predict"])
	}
}

func TestExtractThink(t *testing.T) {
	if extractThink(nil) {
		t.Error("nil options = true, want false")
	}
	if !extractThink(map[string]any{"think": true}) {
		t.Error("think:true = false, want true")
	}
	if extractThink(map[string]any{"think": "yes"}) {
		t.Error("non-bool think = true, want false")
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]provider.FinishReason{
		"stop":       provider.FinishStop,
		"tool_calls": provider.FinishToolCalls,
		"length":     provider.FinishOther,
		"":           provider.FinishOther,
	}
	for in, want := range cases {
		if got := mapFinishReason(in); got != want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDoGenerate_ToolCallsAndOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body ollamaChatRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if len(body.Tools) != 1 || body.Tools[0].Function.Name != "get_weather" {
			t.Errorf("tools in request = %+v", body.Tools)
		}
		if body.Options["temperature"] == nil {
			t.Errorf("temperature option missing: %v", body.Options)
		}
		chatNDJSON(w,
			`{"model":"llama3","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Hanoi"}}}]},"done":true,"done_reason":"tool_calls","prompt_eval_count":7,"eval_count":2}`,
		)
	}))
	defer server.Close()

	model := Chat("llama3", WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:    []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather?"}}}},
		Temperature: ptr(0.5),
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "weather", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishToolCalls)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("ToolCalls = %+v", result.ToolCalls)
	}
	if string(result.ToolCalls[0].Input) != `{"city":"Hanoi"}` {
		t.Errorf("tool input = %s", result.ToolCalls[0].Input)
	}
}

func TestDoGenerate_SystemPrompt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body ollamaChatRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if len(body.Messages) < 1 || body.Messages[0].Role != "system" || body.Messages[0].Content != "be brief" {
			t.Errorf("system message not prepended: %+v", body.Messages)
		}
		chatNDJSON(w, `{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		System:   "be brief",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDoGenerate_BuildRequestError(t *testing.T) {
	model := Chat("m", WithBaseURL("http://localhost:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools:    []provider.ToolDefinition{{Name: "bad", InputSchema: json.RawMessage(`{nope`)}},
	})
	if err == nil {
		t.Fatal("expected build request error, got nil")
	}
}

func TestDoGenerate_UnsupportedRole(t *testing.T) {
	model := Chat("m", WithBaseURL("http://localhost:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.Role("ghost")}},
	})
	if err == nil {
		t.Fatal("expected unsupported role error, got nil")
	}
}

func TestDoGenerate_DecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w, `{not valid json`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected decode error, got nil")
	}
}

func TestDoGenerate_CreateRequestError(t *testing.T) {
	// A base URL containing a control character makes http.NewRequestWithContext fail.
	model := Chat("m", WithBaseURL("http://\x7f"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected create request error, got nil")
	}
}

func TestDoStream_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w,
			`{"model":"m","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"x":1}}}]},"done":false}`,
			`{"model":"m","message":{"role":"assistant","content":""},"done":true,"done_reason":"tool_calls","prompt_eval_count":3,"eval_count":1}`,
		)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var sawStart, sawCall bool
	var input string
	var finish provider.FinishReason
	for chunk := range result.Stream {
		switch chunk.Type {
		case provider.ChunkToolCallStreamStart:
			sawStart = true
		case provider.ChunkToolCall:
			sawCall = true
			input = chunk.ToolInput
		case provider.ChunkFinish:
			finish = chunk.FinishReason
		}
	}
	if !sawStart || !sawCall {
		t.Errorf("tool call chunks: start=%v call=%v, want both true", sawStart, sawCall)
	}
	if input != `{"x":1}` {
		t.Errorf("tool input = %q", input)
	}
	if finish != provider.FinishToolCalls {
		t.Errorf("finish = %q, want %q", finish, provider.FinishToolCalls)
	}
}

func TestDoStream_DecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w, `{bad json`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var sawError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			sawError = true
		}
	}
	if !sawError {
		t.Error("expected ChunkError for invalid JSON")
	}
}

func TestDoStream_BuildRequestError(t *testing.T) {
	model := Chat("m", WithBaseURL("http://localhost:1"))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools:    []provider.ToolDefinition{{Name: "bad", InputSchema: json.RawMessage(`{nope`)}},
	})
	if err == nil {
		t.Fatal("expected build request error, got nil")
	}
}

func TestDoStream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected HTTP error, got nil")
	}
}

func TestDoEmbed_MultipleValues(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("path = %s, want /api/embed", r.URL.Path)
		}
		var body ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if len(body.Input) != 2 {
			t.Errorf("input len = %d, want 2", len(body.Input))
		}
		embedJSON(w, `{"model":"e","embeddings":[[0.1,0.2],[0.3,0.4]],"prompt_eval_count":4}`)
	}))
	defer server.Close()

	model := Embedding("e", WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("embeddings = %d, want 2", len(result.Embeddings))
	}
	if result.Usage.InputTokens != 4 {
		t.Errorf("InputTokens = %d, want 4", result.Usage.InputTokens)
	}
}

func TestDoEmbed_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad"}`))
	}))
	defer server.Close()

	model := Embedding("e", WithBaseURL(server.URL))
	if _, err := model.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{}); err == nil {
		t.Fatal("expected HTTP error, got nil")
	}
}

func TestDoEmbed_DecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedJSON(w, `{not json`)
	}))
	defer server.Close()

	model := Embedding("e", WithBaseURL(server.URL))
	if _, err := model.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{}); err == nil {
		t.Fatal("expected decode error, got nil")
	}
}

func TestDoEmbed_CreateRequestError(t *testing.T) {
	model := Embedding("e", WithBaseURL("http://\x7f"))
	if _, err := model.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{}); err == nil {
		t.Fatal("expected create request error, got nil")
	}
}

func TestDoEmbed_SendError(t *testing.T) {
	// Closed server → connection refused at client.Do.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := server.URL
	server.Close()

	model := Embedding("e", WithBaseURL(url))
	if _, err := model.DoEmbed(context.Background(), []string{"a"}, provider.EmbedParams{}); err == nil {
		t.Fatal("expected send error, got nil")
	}
}

func TestWithHeaders_CopiesMap(t *testing.T) {
	src := map[string]string{"X-A": "1"}
	model := Chat("m", WithHeaders(src))
	// Mutating the caller's map after construction must not affect the model.
	src["X-A"] = "mutated"
	src["X-B"] = "added"
	if model.headers["X-A"] != "1" {
		t.Errorf("header X-A = %q, want 1 (external mutation leaked in)", model.headers["X-A"])
	}
	if _, ok := model.headers["X-B"]; ok {
		t.Error("externally added key leaked into model headers")
	}
}

func TestWithHeaders_Empty(t *testing.T) {
	if model := Chat("m", WithHeaders(nil)); model.headers != nil {
		t.Errorf("nil headers = %v, want nil", model.headers)
	}
}

func TestBuildFormat(t *testing.T) {
	if f := buildFormat(nil); f != nil {
		t.Errorf("nil response format = %s, want nil", f)
	}
	// No schema -> generic JSON mode.
	if f := buildFormat(&provider.ResponseFormat{}); string(f) != `"json"` {
		t.Errorf("empty schema = %s, want \"json\"", f)
	}
	// Schema -> constrained output carrying the schema verbatim.
	schema := json.RawMessage(`{"type":"object"}`)
	if f := buildFormat(&provider.ResponseFormat{Schema: schema}); string(f) != string(schema) {
		t.Errorf("schema format = %s, want %s", f, schema)
	}
}

func TestDoGenerate_ResponseFormatSchema(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body ollamaChatRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if string(body.Format) != `{"type":"object"}` {
			t.Errorf("format in request = %s, want the schema", body.Format)
		}
		chatNDJSON(w, `{"model":"m","message":{"role":"assistant","content":"{}"},"done":true,"done_reason":"stop"}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:       []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestConvertMessage_AssistantReasoningToThinking(t *testing.T) {
	msgs, err := convertMessage(provider.Message{
		Role: provider.RoleAssistant,
		Content: []provider.Part{
			{Type: provider.PartText, Text: "answer"},
			{Type: provider.PartReasoning, Text: "because"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	if msgs[0].Content != "answer" {
		t.Errorf("content = %q, want %q", msgs[0].Content, "answer")
	}
	if msgs[0].Thinking != "because" {
		t.Errorf("thinking = %q, want %q (reasoning must not merge into content)", msgs[0].Thinking, "because")
	}
}
