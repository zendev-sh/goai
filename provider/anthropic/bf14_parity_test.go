package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// --- BF.14 Item 1: firstDelta JSON wrapping for code execution tools ---

func TestStream_FirstDelta_CodeExecutionWrapping(t *testing.T) {
	// When server_tool_use with name=bash_code_execution streams input_json_delta,
	// the first delta should be wrapped with {"type": "bash_code_execution",...}.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"tc1","name":"bash_code_execution"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\": \"ls\""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "run ls"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolDeltas []string
	var toolCallInput string
	for chunk := range result.Stream {
		switch chunk.Type {
		case provider.ChunkToolCallDelta:
			toolDeltas = append(toolDeltas, chunk.ToolInput)
		case provider.ChunkToolCall:
			toolCallInput = chunk.ToolInput
		}
	}

	// First delta should have been wrapped with type field.
	if len(toolDeltas) != 2 {
		t.Fatalf("expected 2 tool deltas, got %d", len(toolDeltas))
	}
	if !strings.Contains(toolDeltas[0], `"type": "bash_code_execution"`) {
		t.Errorf("first delta should contain type wrapping, got: %s", toolDeltas[0])
	}

	// Complete input should parse as valid JSON with type field.
	var input map[string]any
	if err := json.Unmarshal([]byte(toolCallInput), &input); err != nil {
		t.Fatalf("tool call input is not valid JSON: %v, got: %s", err, toolCallInput)
	}
	if input["type"] != "bash_code_execution" {
		t.Errorf("type = %v, want bash_code_execution", input["type"])
	}
}

// --- BF.14 Item 2: Iterations in usage ---

func TestDoGenerate_Iterations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "Hello"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 100,
				"output_tokens": 50,
				"iterations": [
					{"type": "message", "input_tokens": 60, "output_tokens": 30},
					{"type": "compaction", "input_tokens": 40, "output_tokens": 20}
				]
			}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Usage should sum across iterations.
	if result.Usage.InputTokens != 100 { // 60+40
		t.Errorf("InputTokens = %d, want 100", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 50 { // 30+20
		t.Errorf("OutputTokens = %d, want 50", result.Usage.OutputTokens)
	}

	// Provider metadata should contain iterations.
	iters, ok := result.Response.ProviderMetadata["iterations"].([]map[string]any)
	if !ok {
		t.Fatalf("iterations not found in provider metadata")
	}
	if len(iters) != 2 {
		t.Fatalf("expected 2 iterations, got %d", len(iters))
	}
	if iters[0]["type"] != "message" {
		t.Errorf("iterations[0].type = %v, want message", iters[0]["type"])
	}
	if iters[1]["type"] != "compaction" {
		t.Errorf("iterations[1].type = %v, want compaction", iters[1]["type"])
	}
}

func TestStream_Iterations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":100}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50,"iterations":[{"type":"message","input_tokens":60,"output_tokens":30},{"type":"compaction","input_tokens":40,"output_tokens":20}]}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
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

	// Usage should sum across iterations.
	if finishChunk.Usage.InputTokens != 100 { // 60+40
		t.Errorf("InputTokens = %d, want 100", finishChunk.Usage.InputTokens)
	}
	if finishChunk.Usage.OutputTokens != 50 { // 30+20
		t.Errorf("OutputTokens = %d, want 50", finishChunk.Usage.OutputTokens)
	}

	// Metadata should contain iterations.
	iters, ok := finishChunk.Metadata["iterations"].([]map[string]any)
	if !ok {
		t.Fatalf("iterations not found in finish metadata")
	}
	if len(iters) != 2 {
		t.Fatalf("expected 2 iterations, got %d", len(iters))
	}
}

// --- BF.14 Item 3: Context management metadata ---

func TestDoGenerate_ContextManagement(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "Done"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5},
			"context_management": {
				"applied_edits": [
					{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 3, "cleared_input_tokens": 500},
					{"type": "compact_20260112"}
				]
			}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	cm, ok := result.Response.ProviderMetadata["contextManagement"].(map[string]any)
	if !ok {
		t.Fatal("contextManagement not found in provider metadata")
	}
	edits, ok := cm["appliedEdits"].([]map[string]any)
	if !ok {
		t.Fatal("appliedEdits not found")
	}
	if len(edits) != 2 {
		t.Fatalf("expected 2 edits, got %d", len(edits))
	}
	if edits[0]["type"] != "clear_tool_uses_20250919" {
		t.Errorf("edit[0].type = %v", edits[0]["type"])
	}
}

func TestStream_ContextManagement(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5},"context_management":{"applied_edits":[{"type":"clear_thinking_20251015","cleared_thinking_turns":2,"cleared_input_tokens":300}]}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
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

	cm, ok := finishChunk.Metadata["contextManagement"].(map[string]any)
	if !ok {
		t.Fatal("contextManagement not found in finish metadata")
	}
	if cm == nil {
		t.Fatal("contextManagement is nil")
	}
}

// --- BF.14 Item 4: Reasoning tokens in usage ---

func TestDoGenerate_ReasoningTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "Hi"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 10,
				"output_tokens": 50,
				"output_tokens_details": {"thinking_tokens": 35}
			}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Usage.ReasoningTokens != 35 {
		t.Errorf("ReasoningTokens = %d, want 35", result.Usage.ReasoningTokens)
	}
}

func TestStream_ReasoningTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50,"output_tokens_details":{"thinking_tokens":35}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
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

	if finishChunk.Usage.ReasoningTokens != 35 {
		t.Errorf("ReasoningTokens = %d, want 35", finishChunk.Usage.ReasoningTokens)
	}
}

// --- BF.14 Item 5: Native output_format (structuredOutputMode) ---

func TestDoGenerate_NativeOutputFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify output_format is set and no synthetic tool trick.
		if _, ok := req["output_format"]; !ok {
			t.Error("output_format not present in request")
		}
		of := req["output_format"].(map[string]any)
		if of["type"] != "json_schema" {
			t.Errorf("output_format.type = %v, want json_schema", of["type"])
		}
		// Should NOT have the json_response synthetic tool.
		if tools, ok := req["tools"]; ok {
			if toolList, ok := tools.([]any); ok {
				for _, tool := range toolList {
					if tm, ok := tool.(map[string]any); ok {
						if tm["name"] == responseFormatToolName {
							t.Error("synthetic json_response tool should not be present in outputFormat mode")
						}
					}
				}
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-sonnet-4-6-20260310",
			"content": [{"type": "text", "text": "{\"name\": \"test\"}"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`)
	}))
	defer server.Close()

	// Using claude-sonnet-4-6 to match supportsNativeOutputFormat.
	model := Chat("claude-sonnet-4-6-20260310", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
		ProviderOptions: map[string]any{
			"structuredOutputMode": "outputFormat",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != `{"name": "test"}` {
		t.Errorf("Text = %q, want json", result.Text)
	}
}

func TestDoGenerate_NativeOutputFormat_Auto_Unsupported(t *testing.T) {
	// With "auto" mode and an older model, should fall back to tool trick.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// output_format should NOT be present (fallback to tool trick).
		if _, ok := req["output_format"]; ok {
			t.Error("output_format should not be present for unsupported model in auto mode")
		}
		// Synthetic tool should be present.
		if tools, ok := req["tools"]; ok {
			toolList := tools.([]any)
			found := false
			for _, tool := range toolList {
				if tm, ok := tool.(map[string]any); ok {
					if tm["name"] == responseFormatToolName {
						found = true
					}
				}
			}
			if !found {
				t.Error("synthetic json_response tool should be present in fallback mode")
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-3-5-sonnet",
			"content": [{"type": "tool_use", "id": "tc1", "name": "json_response", "input": {"name": "test"}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-3-5-sonnet", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
		ProviderOptions: map[string]any{
			"structuredOutputMode": "auto",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Should have extracted the tool call result as text.
	if result.Text == "" {
		t.Error("expected text from tool trick extraction")
	}
}

// --- BF.14 Item 6: disableParallelToolUse ---

func TestBuildRequest_DisableParallelToolUse(t *testing.T) {
	model := &chatModel{id: "claude-sonnet-4-20250514"}

	t.Run("with existing tool_choice", func(t *testing.T) {
		body := model.buildRequest(provider.GenerateParams{
			Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
			Tools: []provider.ToolDefinition{
				{Name: "test", Description: "test tool", InputSchema: json.RawMessage(`{}`)},
			},
			ToolChoice: "auto",
			ProviderOptions: map[string]any{
				"disableParallelToolUse": true,
			},
		}, false)

		tc, ok := body["tool_choice"].(map[string]any)
		if !ok {
			t.Fatal("tool_choice not found")
		}
		if tc["type"] != "auto" {
			t.Errorf("tool_choice.type = %v, want auto", tc["type"])
		}
		if tc["disable_parallel_tool_use"] != true {
			t.Error("disable_parallel_tool_use not set to true")
		}
	})

	t.Run("without tool_choice defaults to auto", func(t *testing.T) {
		body := model.buildRequest(provider.GenerateParams{
			Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
			Tools: []provider.ToolDefinition{
				{Name: "test", Description: "test tool", InputSchema: json.RawMessage(`{}`)},
			},
			ProviderOptions: map[string]any{
				"disableParallelToolUse": true,
			},
		}, false)

		tc, ok := body["tool_choice"].(map[string]any)
		if !ok {
			t.Fatal("tool_choice not found")
		}
		if tc["type"] != "auto" {
			t.Errorf("tool_choice.type = %v, want auto", tc["type"])
		}
		if tc["disable_parallel_tool_use"] != true {
			t.Error("disable_parallel_tool_use not set")
		}
	})

	t.Run("with required tool_choice", func(t *testing.T) {
		body := model.buildRequest(provider.GenerateParams{
			Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
			Tools: []provider.ToolDefinition{
				{Name: "test", Description: "test tool", InputSchema: json.RawMessage(`{}`)},
			},
			ToolChoice: "required",
			ProviderOptions: map[string]any{
				"disableParallelToolUse": true,
			},
		}, false)

		tc, ok := body["tool_choice"].(map[string]any)
		if !ok {
			t.Fatal("tool_choice not found")
		}
		if tc["type"] != "any" { // required → any in Anthropic
			t.Errorf("tool_choice.type = %v, want any", tc["type"])
		}
		if tc["disable_parallel_tool_use"] != true {
			t.Error("disable_parallel_tool_use not set")
		}
	})
}

// --- BF.14 Item 7: Message-level cache control ---

func TestConvertMessages_MessageLevelCacheControl(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{
			Role: provider.RoleUser,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "first"},
				{Type: provider.PartText, Text: "second"},
			},
			ProviderOptions: map[string]any{
				"anthropic": map[string]any{
					"cacheControl": map[string]any{"type": "ephemeral"},
				},
			},
		},
	})

	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(content))
	}

	// First part should NOT have cache control (only last part gets message-level).
	if _, ok := content[0]["cache_control"]; ok {
		t.Error("first part should not have cache_control")
	}

	// Last part should have message-level cache control.
	cc, ok := content[1]["cache_control"].(map[string]any)
	if !ok {
		t.Fatal("last part should have cache_control")
	}
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v, want ephemeral", cc["type"])
	}
}

func TestConvertMessages_PartLevelOverridesMessageLevel(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "response", CacheControl: "ephemeral"},
			},
			ProviderOptions: map[string]any{
				"anthropic": map[string]any{
					"cacheControl": map[string]any{"type": "some_other"},
				},
			},
		},
	})

	content := msgs[0]["content"].([]map[string]any)
	cc := content[0]["cache_control"].(map[string]any)
	// Part-level should take precedence.
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v, want ephemeral (part-level)", cc["type"])
	}
}

// --- BF.14 Item 8: Container specification ---

func TestBuildRequest_Container(t *testing.T) {
	model := &chatModel{id: "claude-sonnet-4-20250514"}
	body := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"container": map[string]any{
				"id": "container-123",
				"skills": []any{
					map[string]any{
						"type":    "anthropic",
						"skillId": "data_analysis",
						"version": "1.0",
					},
				},
			},
		},
	}, false)

	container, ok := body["container"].(map[string]any)
	if !ok {
		t.Fatal("container not present in request")
	}
	if container["id"] != "container-123" {
		t.Errorf("container.id = %v, want container-123", container["id"])
	}
	skills, ok := container["skills"].([]map[string]any)
	if !ok {
		t.Fatal("skills not present")
	}
	if len(skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(skills))
	}
	if skills[0]["skill_id"] != "data_analysis" {
		t.Errorf("skill_id = %v, want data_analysis", skills[0]["skill_id"])
	}
}

func TestDoGenerate_ContainerResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "Done"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5},
			"container": {
				"expires_at": "2026-03-10T20:00:00Z",
				"id": "container-123",
				"skills": [{"type": "anthropic", "skill_id": "data_analysis", "version": "1.0"}]
			}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	container, ok := result.Response.ProviderMetadata["container"].(map[string]any)
	if !ok {
		t.Fatal("container not in provider metadata")
	}
	if container["id"] != "container-123" {
		t.Errorf("container.id = %v", container["id"])
	}
	if container["expiresAt"] != "2026-03-10T20:00:00Z" {
		t.Errorf("container.expiresAt = %v", container["expiresAt"])
	}
}

// --- BF.14 Item 9: contextManagement options ---

func TestBuildRequest_ContextManagement(t *testing.T) {
	model := &chatModel{id: "claude-sonnet-4-20250514"}
	body := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"contextManagement": map[string]any{
				"edits": []any{
					map[string]any{
						"type":    "clear_tool_uses_20250919",
						"trigger": map[string]any{"type": "input_tokens", "value": float64(100000)},
					},
					map[string]any{
						"type":             "compact_20260112",
						"pauseAfterCompaction": true,
						"instructions":     "Keep key facts",
					},
				},
			},
		},
	}, false)

	cm, ok := body["context_management"].(map[string]any)
	if !ok {
		t.Fatal("context_management not present in request")
	}
	edits, ok := cm["edits"].([]map[string]any)
	if !ok {
		t.Fatal("edits not present")
	}
	if len(edits) != 2 {
		t.Fatalf("expected 2 edits, got %d", len(edits))
	}
	if edits[0]["type"] != "clear_tool_uses_20250919" {
		t.Errorf("edit[0].type = %v", edits[0]["type"])
	}
	if edits[1]["pause_after_compaction"] != true {
		t.Error("pause_after_compaction not set")
	}
	if edits[1]["instructions"] != "Keep key facts" {
		t.Error("instructions not set")
	}
}

// --- BF.14 Item 10: effort + speed options ---

func TestBuildRequest_Effort(t *testing.T) {
	model := &chatModel{id: "claude-sonnet-4-20250514"}
	body := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"effort": "low",
		},
	}, false)

	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatal("output_config not present")
	}
	if oc["effort"] != "low" {
		t.Errorf("effort = %v, want low", oc["effort"])
	}
}

func TestBuildRequest_Speed(t *testing.T) {
	model := &chatModel{id: "claude-opus-4-6-20260310"}
	body := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"speed": "fast",
		},
	}, false)

	if body["speed"] != "fast" {
		t.Errorf("speed = %v, want fast", body["speed"])
	}
}

// --- BF.14 Item 11: Schema-validate ProviderOptions passthrough ---

func TestBuildRequest_HandledKeysNotPassthrough(t *testing.T) {
	model := &chatModel{id: "claude-sonnet-4-20250514"}
	body := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"thinking":              map[string]any{"type": "enabled", "budgetTokens": 1024},
			"disableParallelToolUse": true,
			"effort":                "high",
			"speed":                 "fast",
			"container":             map[string]any{"id": "c1"},
			"contextManagement":     map[string]any{"edits": []any{}},
			"structuredOutputMode":  "jsonTool",
			"sendReasoning":         true,
			"cacheControl":          map[string]any{"type": "ephemeral"},
			"customPassthrough":     "should-appear",
		},
	}, false)

	// Handled keys should be processed, not pass through as raw.
	// "thinking" → body["thinking"]
	if _, ok := body["thinking"]; !ok {
		t.Error("thinking should be set in body")
	}
	// "customPassthrough" should pass through.
	if body["customPassthrough"] != "should-appear" {
		t.Errorf("customPassthrough = %v, want should-appear", body["customPassthrough"])
	}
	// "structuredOutputMode" should NOT appear as raw key in body.
	if _, ok := body["structuredOutputMode"]; ok {
		t.Error("structuredOutputMode should not pass through to body")
	}
	// "sendReasoning" should NOT appear as raw key in body.
	if _, ok := body["sendReasoning"]; ok {
		t.Error("sendReasoning should not pass through to body")
	}
}

// --- BF.14 Item 12: Citations ---

func TestDoGenerate_Citations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"type": "message",
			"id": "msg_1",
			"model": "claude-sonnet-4-20250514",
			"content": [{
				"type": "text",
				"text": "According to the source...",
				"citations": [
					{
						"type": "web_search_result_location",
						"cited_text": "some fact",
						"url": "https://example.com",
						"title": "Example",
						"encrypted_index": "abc123"
					},
					{
						"type": "page_location",
						"cited_text": "another fact",
						"document_index": 0,
						"document_title": "Doc A",
						"start_page_number": 1,
						"end_page_number": 3
					}
				]
			}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	citations, ok := result.Response.ProviderMetadata["citations"].([]map[string]any)
	if !ok {
		t.Fatal("citations not found in provider metadata")
	}
	if len(citations) != 2 {
		t.Fatalf("expected 2 citations, got %d", len(citations))
	}

	// Web search result location.
	if citations[0]["type"] != "web_search_result_location" {
		t.Errorf("citation[0].type = %v", citations[0]["type"])
	}
	if citations[0]["url"] != "https://example.com" {
		t.Errorf("citation[0].url = %v", citations[0]["url"])
	}
	if citations[0]["citedText"] != "some fact" {
		t.Errorf("citation[0].citedText = %v", citations[0]["citedText"])
	}

	// Page location.
	if citations[1]["type"] != "page_location" {
		t.Errorf("citation[1].type = %v", citations[1]["type"])
	}
	if citations[1]["documentTitle"] != "Doc A" {
		t.Errorf("citation[1].documentTitle = %v", citations[1]["documentTitle"])
	}
}

func TestStream_CitationsDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Found"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"fact","url":"https://ex.com","title":"Ex","encrypted_index":"enc1"}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var citationChunks []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Metadata != nil {
			if _, ok := chunk.Metadata["citation"]; ok {
				citationChunks = append(citationChunks, chunk)
			}
		}
	}

	if len(citationChunks) != 1 {
		t.Fatalf("expected 1 citation chunk, got %d", len(citationChunks))
	}
	cit := citationChunks[0].Metadata["citation"].(map[string]any)
	if cit["type"] != "web_search_result_location" {
		t.Errorf("citation.type = %v", cit["type"])
	}
	if cit["url"] != "https://ex.com" {
		t.Errorf("citation.url = %v", cit["url"])
	}
}

// --- BF.14 Item: Signature delta in streaming ---

func TestStream_SignatureDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc123"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":1}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var sigChunks []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning && chunk.Metadata != nil {
			if _, ok := chunk.Metadata["signature"]; ok {
				sigChunks = append(sigChunks, chunk)
			}
		}
	}

	if len(sigChunks) != 1 {
		t.Fatalf("expected 1 signature chunk, got %d", len(sigChunks))
	}
	if sigChunks[0].Metadata["signature"] != "sig_abc123" {
		t.Errorf("signature = %v, want sig_abc123", sigChunks[0].Metadata["signature"])
	}
}

// --- BF.14 Item: Container in streaming message_delta ---

func TestStream_ContainerInMessageDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","container":{"expires_at":"2026-03-10T20:00:00Z","id":"ctr-1"}},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
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

	container, ok := finishChunk.Metadata["container"].(map[string]any)
	if !ok {
		t.Fatal("container not in finish metadata")
	}
	if container["id"] != "ctr-1" {
		t.Errorf("container.id = %v, want ctr-1", container["id"])
	}
}

// --- BF.14: Empty tool args become "{}" ---

func TestStream_EmptyToolArgs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc1","name":"no_args_tool"}}`+"\n\n")
		// No input_json_delta events -- empty input.
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			if chunk.ToolInput != "{}" {
				t.Errorf("empty tool input = %q, want {}", chunk.ToolInput)
			}
			return
		}
	}
	t.Fatal("no tool call chunk received")
}

// --- BF.14: text_editor_code_execution firstDelta wrapping ---

func TestStream_FirstDelta_TextEditorWrapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"tc1","name":"text_editor_code_execution"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\": \"view\""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":", \"path\": \"/tmp/f.py\"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCallInput string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCallInput = chunk.ToolInput
		}
	}

	var input map[string]any
	if err := json.Unmarshal([]byte(toolCallInput), &input); err != nil {
		t.Fatalf("invalid JSON: %v, got: %s", err, toolCallInput)
	}
	if input["type"] != "text_editor_code_execution" {
		t.Errorf("type = %v, want text_editor_code_execution", input["type"])
	}
}

// --- Regression: normal tool_use should NOT get firstDelta wrapping ---

func TestStream_NormalToolUse_NoFirstDeltaWrapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc1","name":"read_file"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"/tmp\""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCallInput string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCallInput = chunk.ToolInput
		}
	}

	// Should NOT have the type wrapping.
	var input map[string]any
	if err := json.Unmarshal([]byte(toolCallInput), &input); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if _, hasType := input["type"]; hasType {
		t.Error("normal tool_use should not have type wrapping from firstDelta")
	}
	if input["path"] != "/tmp" {
		t.Errorf("path = %v, want /tmp", input["path"])
	}
}
