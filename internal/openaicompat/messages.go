package openaicompat

import (
	"strings"

	"github.com/zendev-sh/goai/provider"
)

// ConvertMessages converts provider.Message slice to OpenAI wire format.
// The system prompt is prepended as the first message if non-empty.
func ConvertMessages(msgs []provider.Message, system string) []map[string]any {
	result := make([]map[string]any, 0, len(msgs)+1)

	if system != "" {
		result = append(result, map[string]any{
			"role":    "system",
			"content": system,
		})
	}

	for _, msg := range msgs {
		if msg.Role == provider.RoleSystem {
			result = append(result, map[string]any{
				"role":    "system",
				"content": partsToText(msg.Content),
			})
			continue
		}

		if msg.Role == provider.RoleTool {
			for _, part := range msg.Content {
				if part.Type == provider.PartToolResult {
					result = append(result, map[string]any{
						"role":         "tool",
						"tool_call_id": part.ToolCallID,
						"content":      part.ToolOutput,
					})
				}
			}
			continue
		}

		m := map[string]any{"role": string(msg.Role)}

		var toolCalls []map[string]any
		var textParts []string
		var hasImage bool

		for _, part := range msg.Content {
			switch part.Type {
			case provider.PartText:
				textParts = append(textParts, part.Text)
			case provider.PartImage:
				hasImage = true
			case provider.PartToolCall:
				// Use raw ToolInput bytes directly -- they are already JSON.
				args := string(part.ToolInput)
				toolCalls = append(toolCalls, map[string]any{
					"id":   part.ToolCallID,
					"type": "function",
					"function": map[string]any{
						"name":      part.ToolName,
						"arguments": args,
					},
				})
			}
		}

		// If message has images, use content array format.
		if hasImage && msg.Role == provider.RoleUser {
			var contentArr []map[string]any
			for _, part := range msg.Content {
				switch part.Type {
				case provider.PartText:
					if part.Text != "" {
						contentArr = append(contentArr, map[string]any{
							"type": "text",
							"text": part.Text,
						})
					}
				case provider.PartImage:
					imgURL := map[string]any{
						"url": part.URL,
					}
					// Item 6: add detail field if specified ("low"/"high"/"auto").
					if part.Detail != "" {
						imgURL["detail"] = part.Detail
					}
					contentArr = append(contentArr, map[string]any{
						"type":      "image_url",
						"image_url": imgURL,
					})
				}
			}
			m["content"] = contentArr
			result = append(result, m)
			continue
		}

		if len(textParts) > 0 {
			m["content"] = joinText(textParts)
		}
		if len(toolCalls) > 0 {
			m["tool_calls"] = toolCalls
			// OpenAI spec requires content to be null (not empty string) when
			// only tool_calls are present. Omit content entirely when absent.
		}

		result = append(result, m)
	}

	return result
}

func partsToText(parts []provider.Part) string {
	var texts []string
	for _, p := range parts {
		if p.Type == provider.PartText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return joinText(texts)
}

func joinText(parts []string) string {
	return strings.Join(parts, "\n")
}
