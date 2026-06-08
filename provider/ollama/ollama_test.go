package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// chatNDJSON writes one NDJSON line (one JSON object) per response chunk as
// expected by the Ollama native /api/chat endpoint.
func chatNDJSON(w http.ResponseWriter, lines ...string) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	for _, line := range lines {
		_, _ = fmt.Fprintln(w, line)
	}
}

// embedJSON writes a JSON response as expected by /api/embed.
func embedJSON(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, body)
}

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path = %s, want /api/chat", r.URL.Path)
		}
		// Native Ollama API does not require Authorization.
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("unexpected Authorization header: %q", auth)
		}
		chatNDJSON(w,
			`{"model":"llama3","message":{"role":"assistant","content":"Hello from Ollama"},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":3}`,
		)
	}))
	defer server.Close()

	model := Chat("llama3", WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello from Ollama" {
		t.Errorf("Text = %q, want %q", result.Text, "Hello from Ollama")
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
}

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w,
			`{"model":"llama3","message":{"role":"assistant","content":"Hi"},"done":false}`,
			`{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":1}`,
		)
	}))
	defer server.Close()

	model := Chat("llama3", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	var gotFinish bool
	for chunk := range result.Stream {
		switch chunk.Type {
		case provider.ChunkText:
			texts = append(texts, chunk.Text)
		case provider.ChunkFinish:
			gotFinish = true
		}
	}
	if len(texts) != 1 || texts[0] != "Hi" {
		t.Errorf("texts = %v, want [Hi]", texts)
	}
	if !gotFinish {
		t.Error("expected ChunkFinish")
	}
}

func TestChat_DefaultBaseURL(t *testing.T) {
	// Verify the model is created without panicking when no URL option is given.
	model := Chat("llama3")
	if model.ModelID() != "llama3" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "llama3")
	}
}

func TestChat_WithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header X-Custom: val")
		}
		chatNDJSON(w,
			`{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`,
		)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_WithHTTPClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w,
			`{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`,
		)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL), WithHTTPClient(&http.Client{}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":"server error"}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m")
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Errorf("unexpected capabilities: Temperature=%v ToolCall=%v", caps.Temperature, caps.ToolCall)
	}
}

func TestModelID(t *testing.T) {
	model := Chat("llama3")
	if model.ModelID() != "llama3" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "llama3")
	}
}

// --- Think mode tests ---

func TestChat_ThinkMode_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the request body includes think:true.
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		thinkRaw, ok := body["think"]
		if !ok {
			t.Error("think field missing from request")
		}
		if string(thinkRaw) != "true" {
			t.Errorf("think = %s, want true", thinkRaw)
		}

		chatNDJSON(w,
			`{"model":"qwen3","message":{"role":"assistant","thinking":"let me think","content":"answer"},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}`,
		)
	}))
	defer server.Close()

	model := Chat("qwen3", WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
		ProviderOptions: map[string]any{"think": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "answer" {
		t.Errorf("Text = %q, want %q", result.Text, "answer")
	}
	if result.Reasoning != "let me think" {
		t.Errorf("Reasoning = %q, want %q", result.Reasoning, "let me think")
	}
}

func TestChat_ThinkMode_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w,
			`{"model":"qwen3","message":{"role":"assistant","thinking":"reasoning chunk","content":""},"done":false}`,
			`{"model":"qwen3","message":{"role":"assistant","thinking":"","content":"answer chunk"},"done":false}`,
			`{"model":"qwen3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}`,
		)
	}))
	defer server.Close()

	model := Chat("qwen3", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
		ProviderOptions: map[string]any{"think": true},
	})
	if err != nil {
		t.Fatal(err)
	}

	var textChunks []string
	var reasonChunks []string
	for chunk := range result.Stream {
		switch chunk.Type {
		case provider.ChunkText:
			textChunks = append(textChunks, chunk.Text)
		case provider.ChunkReasoning:
			reasonChunks = append(reasonChunks, chunk.Text)
		}
	}

	if len(reasonChunks) != 1 || reasonChunks[0] != "reasoning chunk" {
		t.Errorf("reasonChunks = %v, want [reasoning chunk]", reasonChunks)
	}
	if len(textChunks) != 1 || textChunks[0] != "answer chunk" {
		t.Errorf("textChunks = %v, want [answer chunk]", textChunks)
	}
}

func TestChat_ThinkMode_DefaultFalse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		// think should be present and false (we always send it).
		thinkRaw, ok := body["think"]
		if !ok {
			t.Error("think field missing from request")
		}
		if string(thinkRaw) != "false" {
			t.Errorf("think = %s, want false", thinkRaw)
		}
		chatNDJSON(w,
			`{"model":"m","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop"}`,
		)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- Embedding tests ---

func TestEmbedding_SingleValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("expected /api/embed, got %s", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("unexpected Authorization header: %q", auth)
		}
		embedJSON(w, `{"model":"nomic-embed-text","embeddings":[[0.1,0.2]],"prompt_eval_count":3}`)
	}))
	defer srv.Close()

	model := Embedding("nomic-embed-text", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
	// float32→float64 conversion introduces rounding; check within epsilon.
	const epsilon = 1e-6
	if diff := result.Embeddings[0][0] - 0.1; diff > epsilon || diff < -epsilon {
		t.Errorf("first embedding value = %v, want ~0.1", result.Embeddings[0][0])
	}
}

func TestEmbedding_DefaultBaseURL(t *testing.T) {
	model := Embedding("nomic-embed-text")
	if model.ModelID() != "nomic-embed-text" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "nomic-embed-text")
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("m")
	if got := model.MaxValuesPerCall(); got != defaultMaxValuesPerCall {
		t.Errorf("MaxValuesPerCall = %d, want %d", got, defaultMaxValuesPerCall)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("nomic-embed-text")
	if model.ModelID() != "nomic-embed-text" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "nomic-embed-text")
	}
}

func TestChat_StreamToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatNDJSON(w,
			`{"model":"llama3","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"Paris"}}}]},"done":false}`,
			`{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"tool_calls","prompt_eval_count":8,"eval_count":4}`,
		)
	}))
	defer server.Close()

	model := Chat("llama3", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather?"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotStreamStart bool
	var gotToolCall bool
	var toolName string
	var toolInput string
	for chunk := range result.Stream {
		switch chunk.Type {
		case provider.ChunkToolCallStreamStart:
			gotStreamStart = true
			toolName = chunk.ToolName
		case provider.ChunkToolCall:
			gotToolCall = true
			toolInput = chunk.ToolInput
		}
	}

	if !gotStreamStart {
		t.Error("expected ChunkToolCallStreamStart")
	}
	if !gotToolCall {
		t.Error("expected ChunkToolCall")
	}
	if toolName != "get_weather" {
		t.Errorf("ToolName = %q, want %q", toolName, "get_weather")
	}
	if toolInput == "" {
		t.Error("ToolInput should not be empty")
	}
}
