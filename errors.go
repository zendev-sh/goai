package goai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"slices"
)

// ContextOverflowError indicates the prompt exceeded the model's context window.
type ContextOverflowError struct {
	Message      string
	ResponseBody string
}

func (e *ContextOverflowError) Error() string {
	return e.Message
}

// APIError represents a non-overflow API error.
type APIError struct {
	Message         string
	StatusCode      int
	IsRetryable     bool
	ResponseBody    string
	ResponseHeaders map[string]string
}

func (e *APIError) Error() string {
	return e.Message
}

// Compiled overflow detection patterns -- ported from opencode's error.ts.
var overflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)prompt is too long`),                       // Anthropic
	regexp.MustCompile(`(?i)input is too long for requested model`),    // Amazon Bedrock
	regexp.MustCompile(`(?i)exceeds the context window`),               // OpenAI
	regexp.MustCompile(`(?i)input token count.*exceeds the maximum`),   // Google Gemini
	regexp.MustCompile(`(?i)maximum prompt length is \d+`),             // xAI (Grok)
	regexp.MustCompile(`(?i)reduce the length of the messages`),        // Groq
	regexp.MustCompile(`(?i)maximum context length is \d+ tokens`),     // OpenRouter, DeepSeek
	regexp.MustCompile(`(?i)exceeds the limit of \d+`),                 // GitHub Copilot
	regexp.MustCompile(`(?i)exceeds the available context size`),       // llama.cpp server
	regexp.MustCompile(`(?i)greater than the context length`),          // LM Studio
	regexp.MustCompile(`(?i)context window exceeds limit`),             // MiniMax
	regexp.MustCompile(`(?i)exceeded model token limit`),               // Kimi, Moonshot
	regexp.MustCompile(`(?i)context[_ ]length[_ ]exceeded`),            // Generic fallback
	regexp.MustCompile(`(?i)^4(00|13)\s*(status code)?\s*\(no body\)`), // Cerebras, Mistral
}

// IsOverflow checks if an error message indicates a context overflow.
func IsOverflow(message string) bool {
	return slices.ContainsFunc(overflowPatterns, func(p *regexp.Regexp) bool {
		return p.MatchString(message)
	})
}

// StreamErrorType classifies parsed stream errors.
type StreamErrorType string

const (
	StreamErrorContextOverflow StreamErrorType = "context_overflow"
	StreamErrorAPI             StreamErrorType = "api_error"
)

// ParsedStreamError represents a parsed error from an SSE stream.
type ParsedStreamError struct {
	Type         StreamErrorType
	Message      string
	IsRetryable  bool
	ResponseBody string
}

// ParseStreamError parses a stream error event (used by Anthropic/OpenAI error events).
func ParseStreamError(body []byte) *ParsedStreamError {
	var obj struct {
		Type  string `json:"type"`
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil
	}
	if obj.Type != "error" {
		return nil
	}

	switch obj.Error.Code {
	case "context_length_exceeded":
		return &ParsedStreamError{
			Type:         StreamErrorContextOverflow,
			Message:      "Input exceeds context window of this model",
			ResponseBody: string(body),
		}
	case "insufficient_quota":
		return &ParsedStreamError{
			Type:         StreamErrorAPI,
			Message:      "Quota exceeded. Check your plan and billing details.",
			IsRetryable:  false,
			ResponseBody: string(body),
		}
	case "usage_not_included":
		return &ParsedStreamError{
			Type:         StreamErrorAPI,
			Message:      "To use Codex with your ChatGPT plan, upgrade to Plus.",
			IsRetryable:  false,
			ResponseBody: string(body),
		}
	case "invalid_prompt":
		msg := "Invalid prompt."
		if obj.Error.Message != "" {
			msg = obj.Error.Message
		}
		return &ParsedStreamError{
			Type:         StreamErrorAPI,
			Message:      msg,
			IsRetryable:  false,
			ResponseBody: string(body),
		}
	}

	return nil
}

// ClassifyStreamError parses a stream error event and returns the appropriate
// typed error (*ContextOverflowError or *APIError), or nil if the data is not
// a recognized error event.
func ClassifyStreamError(body []byte) error {
	parsed := ParseStreamError(body)
	if parsed == nil {
		return nil
	}
	if parsed.Type == StreamErrorContextOverflow {
		return &ContextOverflowError{Message: parsed.Message, ResponseBody: parsed.ResponseBody}
	}
	return &APIError{Message: parsed.Message, IsRetryable: parsed.IsRetryable}
}

// ParseHTTPError classifies an HTTP error response.
func ParseHTTPError(providerID string, statusCode int, body []byte) error {
	return ParseHTTPErrorWithHeaders(providerID, statusCode, body, nil)
}

// ParseHTTPErrorWithHeaders parses an HTTP error response, preserving retry-related headers.
func ParseHTTPErrorWithHeaders(providerID string, statusCode int, body []byte, headers http.Header) error {
	message := extractErrorMessage(statusCode, body)

	if IsOverflow(message) {
		return &ContextOverflowError{
			Message:      message,
			ResponseBody: string(body),
		}
	}

	isRetryable := statusCode == http.StatusTooManyRequests ||
		statusCode == http.StatusServiceUnavailable ||
		statusCode >= 500

	// OpenAI sometimes returns 404 for models that are actually available.
	if providerID == "openai" && statusCode == http.StatusNotFound {
		isRetryable = true
	}

	// Extract retry-related headers for backoff logic.
	var respHeaders map[string]string
	if headers != nil {
		for _, key := range []string{"retry-after", "retry-after-ms"} {
			if v := headers.Get(key); v != "" {
				if respHeaders == nil {
					respHeaders = make(map[string]string)
				}
				respHeaders[key] = v
			}
		}
	}

	return &APIError{
		Message:         message,
		StatusCode:      statusCode,
		IsRetryable:     isRetryable,
		ResponseBody:    string(body),
		ResponseHeaders: respHeaders,
	}
}

// extractErrorMessage returns a human-readable error from the response body.
// The raw body is truncated to 200 chars to limit information disclosure.
// Handles both Chat Completions format ({error: {message}}) and
// Responses API format ({message, code, type}).
func extractErrorMessage(statusCode int, body []byte) string {
	if len(body) == 0 {
		return fmt.Sprintf("%d (no body)", statusCode)
	}

	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err == nil {
		// Try error.message (Chat Completions format: Anthropic, OpenAI)
		if errObj, ok := obj["error"].(map[string]any); ok {
			if msg, ok := errObj["message"].(string); ok && msg != "" {
				return msg
			}
		}
		// Try top-level message (Responses API error format)
		if msg, ok := obj["message"].(string); ok && msg != "" {
			return msg
		}
		// Try error as string
		if msg, ok := obj["error"].(string); ok && msg != "" {
			return msg
		}
	}

	// Fallback: include truncated body for debugging.
	// body is non-empty here (len(body)==0 returned early above).
	statusText := http.StatusText(statusCode)
	bodyStr := string(body)
	if len(bodyStr) > 200 {
		bodyStr = bodyStr[:200] + "..."
	}
	return fmt.Sprintf("%d %s: %s", statusCode, statusText, bodyStr)
}
