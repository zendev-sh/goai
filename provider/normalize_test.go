package provider

import (
	"testing"
)

// --- NormalizeToolMessages ---

func TestNormalizeToolMessages_NoToolCalls(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "hello"}}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "hi"}}},
	}
	result := NormalizeToolMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	if result[0].Role != RoleUser || result[1].Role != RoleAssistant {
		t.Errorf("roles = %s/%s, want user/assistant", result[0].Role, result[1].Role)
	}
}

func TestNormalizeToolMessages_MatchedPairs(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "weather?"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "weather"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "sunny"},
		}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "It is sunny"}}},
	}
	result := NormalizeToolMessages(msgs)
	// Should merge tool+assistant but no synthetic results needed.
	// Verify no "Tool execution aborted" anywhere.
	for _, m := range result {
		for _, p := range m.Content {
			if p.ToolOutput == "Tool execution aborted" {
				t.Error("unexpected synthetic tool result for matched pair")
			}
		}
	}
}

func TestNormalizeToolMessages_OrphanedToolCall(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "do stuff"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "do_thing"},
		}},
		// No tool result follows - orphan.
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "never mind"}}},
	}
	result := ensureToolResultPairing(msgs)
	// The user message after assistant should now contain a synthetic result.
	found := false
	for _, m := range result {
		for _, p := range m.Content {
			if p.Type == PartToolResult && p.ToolCallID == "tc1" && p.ToolOutput == "Tool execution aborted" {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected synthetic tool result for orphaned tc1")
	}
}

func TestNormalizeToolMessages_PartialMatch(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "multi"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "a"},
			{Type: PartToolCall, ToolCallID: "tc2", ToolName: "b"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "result-a"},
			// tc2 is orphaned
		}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "done"}}},
	}
	result := ensureToolResultPairing(msgs)
	// tc2 should get a synthetic result injected into the tool message.
	foundTc2 := false
	for _, m := range result {
		for _, p := range m.Content {
			if p.Type == PartToolResult && p.ToolCallID == "tc2" && p.ToolOutput == "Tool execution aborted" {
				foundTc2 = true
			}
		}
	}
	if !foundTc2 {
		t.Error("expected synthetic tool result for orphaned tc2")
	}
}

func TestNormalizeToolMessages_EndOfConversationOrphan(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "go"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "action"},
		}},
		// End of conversation - no following message at all.
	}
	result := ensureToolResultPairing(msgs)
	// Should insert a new tool message after assistant.
	if len(result) != 3 {
		t.Fatalf("len = %d, want 3 (user, assistant, injected tool)", len(result))
	}
	injected := result[2]
	if injected.Role != RoleTool {
		t.Errorf("injected message role = %s, want tool", injected.Role)
	}
	if len(injected.Content) != 1 {
		t.Fatalf("injected content len = %d, want 1", len(injected.Content))
	}
	p := injected.Content[0]
	if p.Type != PartToolResult || p.ToolCallID != "tc1" || p.ToolOutput != "Tool execution aborted" {
		t.Errorf("injected part = %+v", p)
	}
}

func TestNormalizeToolMessages_MultipleConsecutiveToolMessages(t *testing.T) {
	// GoAI buildToolMessages pattern: multiple tool messages after one assistant.
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "run tools"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "a"},
			{Type: PartToolCall, ToolCallID: "tc2", ToolName: "b"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "r1"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc2", ToolOutput: "r2"},
		}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "done"}}},
	}
	result := NormalizeToolMessages(msgs)
	// Both tool results are matched - no synthetic results.
	for _, m := range result {
		for _, p := range m.Content {
			if p.ToolOutput == "Tool execution aborted" {
				t.Error("unexpected synthetic result - both tool calls are matched")
			}
		}
	}
}

// --- mergeConsecutiveRoles ---

func TestMergeConsecutiveRoles_SameRole(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "a"}}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "b"}}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "c"}}},
	}
	result := mergeConsecutiveRoles(msgs)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	if len(result[0].Content) != 2 {
		t.Fatalf("merged content len = %d, want 2", len(result[0].Content))
	}
	if result[0].Content[0].Text != "a" || result[0].Content[1].Text != "b" {
		t.Errorf("merged texts = %q, %q", result[0].Content[0].Text, result[0].Content[1].Text)
	}
}

func TestMergeConsecutiveRoles_ToolAndUser(t *testing.T) {
	// Tool + user should merge (tool treated as user), with tool-result parts first.
	msgs := []Message{
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "result"},
		}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "thanks"}}},
	}
	result := mergeConsecutiveRoles(msgs)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1 (merged)", len(result))
	}
	// Tool-result parts should come before text parts.
	if result[0].Content[0].Type != PartToolResult {
		t.Errorf("first part type = %s, want tool-result", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != PartText {
		t.Errorf("second part type = %s, want text", result[0].Content[1].Type)
	}
}

func TestMergeConsecutiveRoles_Empty(t *testing.T) {
	result := mergeConsecutiveRoles(nil)
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
}

// --- ReorderAssistantParts ---

func TestReorderAssistantParts_TextBeforeToolCall(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "fn"},
			{Type: PartText, Text: "thinking..."},
		}},
	}
	result := ReorderAssistantParts(msgs)
	if result[0].Content[0].Type != PartText {
		t.Errorf("first part = %s, want text", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != PartToolCall {
		t.Errorf("second part = %s, want tool-call", result[0].Content[1].Type)
	}
}

func TestReorderAssistantParts_AlreadyOrdered(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []Part{
			{Type: PartText, Text: "let me check"},
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "fn"},
		}},
	}
	result := ReorderAssistantParts(msgs)
	if result[0].Content[0].Type != PartText {
		t.Errorf("first part = %s, want text (unchanged)", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != PartToolCall {
		t.Errorf("second part = %s, want tool-call (unchanged)", result[0].Content[1].Type)
	}
}

func TestReorderAssistantParts_NonAssistantUntouched(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{
			{Type: PartText, Text: "hello"},
		}},
	}
	result := ReorderAssistantParts(msgs)
	if len(result) != 1 || result[0].Content[0].Text != "hello" {
		t.Error("user message should be untouched")
	}
}

func TestReorderAssistantParts_Empty(t *testing.T) {
	result := ReorderAssistantParts(nil)
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
}

func TestNormalizeToolMessages_EmptyMessages(t *testing.T) {
	result := NormalizeToolMessages(nil)
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
	result = NormalizeToolMessages([]Message{})
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
}
