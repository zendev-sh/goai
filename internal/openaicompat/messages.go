package openaicompat

import (
	"encoding/base64"
	"strings"

	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// MessagesConfig carries per-request serialization knobs for ConvertMessages.
type MessagesConfig struct {
	// IncludeReasoningContent controls serialization of the non-standard
	// reasoning_content field on assistant messages.
	IncludeReasoningContent bool

	// FlatInputFile makes PDF parts serialize as the flat
	// {"type":"input_file","file_data":...} shape (Requesty) instead of the
	// nested {"type":"file","file":{...}} shape (item 59).
	FlatInputFile bool
}

// ConvertMessagesWithConfig converts provider.Message slice to OpenAI wire
// format using the supplied MessagesConfig.
func ConvertMessagesWithConfig(msgs []provider.Message, system string, cfg MessagesConfig) []map[string]any {
	includeReasoning := cfg.IncludeReasoningContent
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
		var reasoningParts []string
		var hasImage bool
		var hasFile bool

		for _, part := range msg.Content {
			switch part.Type {
			case provider.PartText:
				textParts = append(textParts, part.Text)
			case provider.PartReasoning:
				if part.Text != "" {
					reasoningParts = append(reasoningParts, part.Text)
				}
			case provider.PartImage:
				hasImage = true
			case provider.PartFile:
				hasFile = true
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

		// If message has images or files, use content array format.
		if (hasImage || hasFile) && msg.Role == provider.RoleUser {
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
				case provider.PartFile:
					if item, ok := filePartToContent(part, cfg.FlatInputFile); ok {
						contentArr = append(contentArr, item)
					}
				}
			}
			if len(contentArr) == 0 {
				// Every part was dropped: keep the message valid.
				m["content"] = ""
			} else {
				m["content"] = contentArr
			}
			result = append(result, m)
			continue
		}

		if len(textParts) > 0 {
			m["content"] = joinText(textParts)
		}
		if includeReasoning && len(reasoningParts) > 0 {
			m["reasoning_content"] = joinText(reasoningParts)
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

// filePartToContent converts a PartFile to an OpenAI content item using the
// chat completions shapes that OpenAI and compat gateways (OpenRouter et al.)
// accept: PDFs become a "file" part (inline data URL, or the plain URL for
// remote files) and audio becomes an "input_audio" part. Anything else has no
// wire representation and is omitted (ok=false) — inlining raw base64 as text
// only feeds the model garbage.
func filePartToContent(part provider.Part, flatInputFile bool) (map[string]any, bool) {
	mediaType := part.MediaType
	fileData := "" // what goes in file.file_data: a data URL or a plain URL
	payload := ""  // bare base64 for input_audio

	switch {
	case part.RemoteRef != nil && len(part.RemoteRef.Data) > 0:
		if mediaType == "" {
			mediaType = part.RemoteRef.MediaType
		}
		// RemoteRef.Data holds raw file bytes; base64-encode them for the
		// data URL (mirrors provider/google).
		payload = base64.StdEncoding.EncodeToString(part.RemoteRef.Data)
		fileData = "data:" + mediaType + ";base64," + payload
	case strings.HasPrefix(part.URL, "data:"):
		if mt, data, ok := httpc.ParseDataURL(part.URL); ok {
			if mediaType == "" {
				mediaType = mt
			}
			payload = data
			fileData = "data:" + mediaType + ";base64," + payload
		}
		// An invalid data URL (no ";base64,") is not inlined: it must not
		// degrade into an empty base64 payload, so leave fileData empty.
	case part.URL != "":
		// Remote URL: usable for the "file" part (gateways fetch it), never
		// for input_audio (audio must be base64 inline).
		fileData = part.URL
	}

	if mediaType == "application/pdf" && fileData != "" {
		filename := part.Filename
		if filename == "" {
			filename = "document.pdf"
		}
		if flatInputFile {
			// Requesty uses the flat {"type":"input_file","file_data":...} shape
			// instead of the nested {"type":"file","file":{...}} shape (item 59).
			return map[string]any{
				"type":      "input_file",
				"file_data": fileData,
			}, true
		}
		return map[string]any{
			"type": "file",
			"file": map[string]any{
				"filename":  filename,
				"file_data": fileData,
			},
		}, true
	}
	if format, ok := audioFormat(mediaType); ok && payload != "" {
		return map[string]any{
			"type": "input_audio",
			"input_audio": map[string]any{
				"data":   payload,
				"format": format,
			},
		}, true
	}
	return nil, false
}

// audioFormat maps an audio media type to the input_audio format
// identifier (wav, mp3, aiff, aac, ogg, flac, m4a).
func audioFormat(mediaType string) (string, bool) {
	switch mediaType {
	case "audio/wav", "audio/wave", "audio/x-wav":
		return "wav", true
	case "audio/mp3", "audio/mpeg":
		return "mp3", true
	case "audio/aiff", "audio/x-aiff":
		return "aiff", true
	case "audio/aac":
		return "aac", true
	case "audio/ogg", "application/ogg":
		return "ogg", true
	case "audio/flac", "audio/x-flac":
		return "flac", true
	case "audio/mp4", "audio/x-m4a":
		return "m4a", true
	}
	return "", false
}
