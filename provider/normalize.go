package provider

// NormalizeToolMessages prepares messages for providers that require:
// 1. Every assistant tool-call has a matching tool-result (orphan fix)
// 2. Alternating user/assistant roles (merge consecutive same-role)
//
// Call this before provider-specific message conversion.
func NormalizeToolMessages(msgs []Message) []Message {
	msgs = ensureToolResultPairing(msgs)
	msgs = mergeConsecutiveRoles(msgs)
	return msgs
}

// ensureToolResultPairing ensures every assistant message with tool-call parts
// has matching tool-result parts in following messages. Injects synthetic
// "Tool execution aborted" results for orphaned tool-calls.
func ensureToolResultPairing(msgs []Message) []Message {
	for i := 0; i < len(msgs); i++ {
		if msgs[i].Role != RoleAssistant {
			continue
		}
		var callIDs []string
		for _, p := range msgs[i].Content {
			if p.Type == PartToolCall && p.ToolCallID != "" {
				callIDs = append(callIDs, p.ToolCallID)
			}
		}
		if len(callIDs) == 0 {
			continue
		}
		// Scan all consecutive tool/user messages after assistant
		resultIDs := make(map[string]bool)
		for j := i + 1; j < len(msgs); j++ {
			r := msgs[j].Role
			if r != RoleTool && r != RoleUser {
				break
			}
			for _, p := range msgs[j].Content {
				if p.Type == PartToolResult && p.ToolCallID != "" {
					resultIDs[p.ToolCallID] = true
				}
			}
		}
		var orphans []Part
		for _, id := range callIDs {
			if !resultIDs[id] {
				orphans = append(orphans, Part{
					Type:       PartToolResult,
					ToolCallID: id,
					ToolOutput: "Tool execution aborted",
				})
			}
		}
		if len(orphans) == 0 {
			continue
		}
		if i+1 < len(msgs) && (msgs[i+1].Role == RoleTool || msgs[i+1].Role == RoleUser) {
			msgs[i+1].Content = append(msgs[i+1].Content, orphans...)
		} else {
			toolMsg := Message{Role: RoleTool, Content: orphans}
			msgs = append(msgs, Message{})
			copy(msgs[i+2:], msgs[i+1:])
			msgs[i+1] = toolMsg
		}
	}
	return msgs
}

// mergeConsecutiveRoles merges consecutive messages with the same role.
// Tool-role messages are treated as user-role for merging purposes.
// When merging, tool-result parts are placed before text parts
// (providers require tool-result immediately after tool-use).
func mergeConsecutiveRoles(msgs []Message) []Message {
	if len(msgs) == 0 {
		return msgs
	}
	var result []Message
	for _, msg := range msgs {
		effectiveRole := msg.Role
		if effectiveRole == RoleTool {
			effectiveRole = RoleUser
		}

		if len(result) > 0 {
			lastRole := result[len(result)-1].Role
			if lastRole == RoleTool {
				lastRole = RoleUser
			}
			if lastRole == effectiveRole {
				// Merge: tool-result parts first, then others
				merged := append(result[len(result)-1].Content, msg.Content...)
				var toolResults, others []Part
				for _, p := range merged {
					if p.Type == PartToolResult {
						toolResults = append(toolResults, p)
					} else {
						others = append(others, p)
					}
				}
				if len(toolResults) > 0 {
					merged = append(toolResults, others...)
				}
				result[len(result)-1].Content = merged
				continue
			}
		}
		result = append(result, msg)
	}
	return result
}

// ReorderAssistantParts sorts assistant message parts so text/reasoning
// come before tool-call parts. Anthropic/Bedrock require this ordering.
func ReorderAssistantParts(msgs []Message) []Message {
	for i := range msgs {
		if msgs[i].Role != RoleAssistant {
			continue
		}
		var textParts, toolCallParts []Part
		for _, p := range msgs[i].Content {
			if p.Type == PartToolCall {
				toolCallParts = append(toolCallParts, p)
			} else {
				textParts = append(textParts, p)
			}
		}
		if len(toolCallParts) > 0 && len(textParts) > 0 {
			msgs[i].Content = append(textParts, toolCallParts...)
		}
	}
	return msgs
}
