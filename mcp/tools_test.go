package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestConvertTools_Empty(t *testing.T) {
	c := NewClient("test", "1.0")
	tools := ConvertTools(c, nil)
	if len(tools) != 0 {
		t.Errorf("len = %d, want 0", len(tools))
	}

	tools = ConvertTools(c, []Tool{})
	if len(tools) != 0 {
		t.Errorf("len = %d, want 0", len(tools))
	}
}

func TestConvertTools_SingleTool(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"number"}}}`)
	mcpTools := []Tool{
		{
			Name:        "calculator",
			Description: "Does math",
			InputSchema: schema,
		},
	}

	goaiTools := ConvertTools(c, mcpTools)
	if len(goaiTools) != 1 {
		t.Fatalf("len = %d, want 1", len(goaiTools))
	}
	if goaiTools[0].Name != "calculator" {
		t.Errorf("name = %q, want %q", goaiTools[0].Name, "calculator")
	}
	if goaiTools[0].Description != "Does math" {
		t.Errorf("description = %q, want %q", goaiTools[0].Description, "Does math")
	}
	if string(goaiTools[0].InputSchema) != string(schema) {
		t.Errorf("schema = %s, want %s", goaiTools[0].InputSchema, schema)
	}
}

func TestConvertTools_ExecuteCallsTool(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mcpTools := []Tool{
		{Name: "greet", Description: "Say hello"},
	}

	goaiTools := ConvertTools(c, mcpTools)

	// Override sendFunc to capture the call
	var calledName string
	var calledArgs map[string]any
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/call" {
			var p struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			json.Unmarshal(msg.Params, &p)
			calledName = p.Name
			calledArgs = p.Arguments

			result, _ := json.Marshal(CallToolResult{
				Content: []ContentBlock{
					json.RawMessage(`{"type":"text","text":"Hello, World!"}`),
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	input := json.RawMessage(`{"name":"World"}`)
	output, err := goaiTools[0].Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}

	if calledName != "greet" {
		t.Errorf("called tool = %q, want %q", calledName, "greet")
	}
	if calledArgs["name"] != "World" {
		t.Errorf("args = %v", calledArgs)
	}
	if output != "Hello, World!" {
		t.Errorf("output = %q, want %q", output, "Hello, World!")
	}
}

func TestConvertTools_MultipleDontShareClosure(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mcpTools := []Tool{
		{Name: "tool-a", Description: "A"},
		{Name: "tool-b", Description: "B"},
		{Name: "tool-c", Description: "C"},
	}

	goaiTools := ConvertTools(c, mcpTools)

	// Verify each tool has its own name in the closure
	var calledNames []string
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/call" {
			var p struct {
				Name string `json:"name"`
			}
			json.Unmarshal(msg.Params, &p)
			calledNames = append(calledNames, p.Name)

			result, _ := json.Marshal(CallToolResult{
				Content: []ContentBlock{
					json.RawMessage(`{"type":"text","text":"ok"}`),
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	for _, tool := range goaiTools {
		tool.Execute(context.Background(), json.RawMessage(`{}`))
	}

	if len(calledNames) != 3 {
		t.Fatalf("called %d tools, want 3", len(calledNames))
	}
	// Check each tool called with its own name (not the last one)
	expected := []string{"tool-a", "tool-b", "tool-c"}
	for i, name := range calledNames {
		if name != expected[i] {
			t.Errorf("call %d: name = %q, want %q", i, name, expected[i])
		}
	}
}

func TestConvertTools_EmptyInput(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mcpTools := []Tool{{Name: "noop"}}
	goaiTools := ConvertTools(c, mcpTools)

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/call" {
			result, _ := json.Marshal(CallToolResult{
				Content: []ContentBlock{json.RawMessage(`{"type":"text","text":"done"}`)},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	// Call with nil/empty input
	output, err := goaiTools[0].Execute(context.Background(), nil)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if output != "done" {
		t.Errorf("output = %q", output)
	}
}

// --- FormatContent tests ---

func TestFormatContent_Empty(t *testing.T) {
	result := FormatContent(nil, false)
	if result != "" {
		t.Errorf("result = %q, want empty", result)
	}
}

func TestFormatContent_SingleTextBlock(t *testing.T) {
	content := []ContentBlock{
		json.RawMessage(`{"type":"text","text":"Hello"}`),
	}
	result := FormatContent(content, false)
	if result != "Hello" {
		t.Errorf("result = %q, want %q", result, "Hello")
	}
}

func TestFormatContent_MultipleTextBlocks(t *testing.T) {
	content := []ContentBlock{
		json.RawMessage(`{"type":"text","text":"Line 1"}`),
		json.RawMessage(`{"type":"text","text":"Line 2"}`),
	}
	result := FormatContent(content, false)
	if result != "Line 1\nLine 2" {
		t.Errorf("result = %q, want %q", result, "Line 1\nLine 2")
	}
}

func TestFormatContent_NonTextBlocksSkipped(t *testing.T) {
	content := []ContentBlock{
		json.RawMessage(`{"type":"text","text":"Hello"}`),
		json.RawMessage(`{"type":"image","data":"base64data","mimeType":"image/png"}`),
		json.RawMessage(`{"type":"text","text":"World"}`),
	}
	result := FormatContent(content, false)
	if result != "Hello\nWorld" {
		t.Errorf("result = %q, want %q", result, "Hello\nWorld")
	}
}

func TestFormatContent_IsError(t *testing.T) {
	content := []ContentBlock{
		json.RawMessage(`{"type":"text","text":"something failed"}`),
	}
	result := FormatContent(content, true)
	if !strings.HasPrefix(result, "Error:") {
		t.Errorf("result = %q, want Error: prefix", result)
	}
	if !strings.Contains(result, "something failed") {
		t.Errorf("result = %q, should contain error message", result)
	}
}

func TestFormatContent_InvalidJSON(t *testing.T) {
	content := []ContentBlock{
		json.RawMessage(`{invalid json}`),
		json.RawMessage(`{"type":"text","text":"ok"}`),
	}
	result := FormatContent(content, false)
	// Invalid block should be skipped, valid one included
	if result != "ok" {
		t.Errorf("result = %q, want %q", result, "ok")
	}
}

func TestFormatContent_IsError_NoContent(t *testing.T) {
	result := FormatContent(nil, true)
	if result != "Error:" {
		t.Errorf("result = %q, want %q", result, "Error:")
	}
}

func TestParseTextContent_Valid(t *testing.T) {
	block := json.RawMessage(`{"type":"text","text":"hello"}`)
	tc, ok := ParseTextContent(block)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if tc.Text != "hello" {
		t.Errorf("text = %q", tc.Text)
	}
	if tc.Type != "text" {
		t.Errorf("type = %q", tc.Type)
	}
}

func TestParseTextContent_NonText(t *testing.T) {
	block := json.RawMessage(`{"type":"image","data":"abc"}`)
	_, ok := ParseTextContent(block)
	if ok {
		t.Error("expected ok=false for non-text block")
	}
}

func TestParseTextContent_InvalidJSON(t *testing.T) {
	block := json.RawMessage(`not json`)
	_, ok := ParseTextContent(block)
	if ok {
		t.Error("expected ok=false for invalid JSON")
	}
}

func TestConvertTools_ExecuteIsError(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(t.Context()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mcpTools := []Tool{{Name: "failing-tool"}}
	goaiTools := ConvertTools(c, mcpTools)

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/call" {
			result, _ := json.Marshal(CallToolResult{
				IsError: true,
				Content: []ContentBlock{
					json.RawMessage(`{"type":"text","text":"something went wrong"}`),
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	output, err := goaiTools[0].Execute(t.Context(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.HasPrefix(output, "Error:") {
		t.Errorf("output = %q, want prefix %q", output, "Error:")
	}
	if !strings.Contains(output, "something went wrong") {
		t.Errorf("output = %q, should contain error message text", output)
	}
}

func TestConvertTools_ExecuteInvalidInput(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mcpTools := []Tool{{Name: "test-tool"}}
	goaiTools := ConvertTools(c, mcpTools)

	// Pass invalid JSON as input
	_, err := goaiTools[0].Execute(context.Background(), json.RawMessage(`{invalid}`))
	if err == nil {
		t.Error("expected error for invalid JSON input")
	}
}
