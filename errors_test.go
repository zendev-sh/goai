package goai

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestIsOverflow(t *testing.T) {
	tests := []struct {
		message string
		want    bool
	}{
		{"prompt is too long", true},
		{"Input is too long for requested model", true},
		{"Request exceeds the context window", true},
		{"input token count of 500000 exceeds the maximum of 200000", true},
		{"maximum prompt length is 128000", true},
		{"reduce the length of the messages", true},
		{"maximum context length is 32768 tokens", true},
		{"exceeds the limit of 100000", true},
		{"exceeds the available context size", true},
		{"greater than the context length", true},
		{"context window exceeds limit", true},
		{"exceeded model token limit", true},
		{"context_length_exceeded", true},
		{"context length exceeded", true},
		{"400 (no body)", true},
		{"413 (no body)", true},
		{"413 status code (no body)", true},
		// Negative cases
		{"normal error message", false},
		{"rate limit exceeded", false},
		{"unauthorized", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.message, func(t *testing.T) {
			got := IsOverflow(tt.message)
			if got != tt.want {
				t.Errorf("IsOverflow(%q) = %v, want %v", tt.message, got, tt.want)
			}
		})
	}
}

func TestParseHTTPError_Overflow(t *testing.T) {
	body := []byte(`{"error":{"message":"prompt is too long","type":"invalid_request_error"}}`)
	err := ParseHTTPError("anthropic", 400, body)

	var overflow *ContextOverflowError
	if !errors.As(err, &overflow) {
		t.Fatalf("expected ContextOverflowError, got %T: %v", err, err)
	}
	if overflow.Message != "prompt is too long" {
		t.Errorf("expected 'prompt is too long', got %q", overflow.Message)
	}
}

func TestParseHTTPError_APIError(t *testing.T) {
	body := []byte(`{"error":{"message":"rate limit exceeded"}}`)
	err := ParseHTTPError("openai", 429, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T: %v", err, err)
	}
	if !apiErr.IsRetryable {
		t.Error("expected 429 to be retryable")
	}
}

func TestParseHTTPError_OpenAI404(t *testing.T) {
	body := []byte(`{"error":{"message":"model not found"}}`)
	err := ParseHTTPError("openai", 404, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	if !apiErr.IsRetryable {
		t.Error("expected OpenAI 404 to be retryable")
	}
}

func TestParseHTTPError_NonOpenAI404(t *testing.T) {
	body := []byte(`{"error":{"message":"not found"}}`)
	err := ParseHTTPError("anthropic", 404, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	if apiErr.IsRetryable {
		t.Error("expected non-OpenAI 404 to not be retryable")
	}
}

func TestParseHTTPError_EmptyBody(t *testing.T) {
	err := ParseHTTPError("anthropic", 400, nil)

	var overflow *ContextOverflowError
	if !errors.As(err, &overflow) {
		t.Fatalf("expected ContextOverflowError for '400 (no body)', got %T: %v", err, err)
	}
}

func TestParseHTTPError_500(t *testing.T) {
	body := []byte(`{"error":"internal server error"}`)
	err := ParseHTTPError("anthropic", 500, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	if !apiErr.IsRetryable {
		t.Error("expected 500 to be retryable")
	}
	if apiErr.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", apiErr.StatusCode)
	}
}

func TestParseStreamError(t *testing.T) {
	tests := []struct {
		name string
		body string
		want *ParsedStreamError
	}{
		{
			"context overflow",
			`{"type":"error","error":{"code":"context_length_exceeded","message":"too long"}}`,
			&ParsedStreamError{Type: "context_overflow", Message: "Input exceeds context window of this model"},
		},
		{
			"insufficient quota",
			`{"type":"error","error":{"code":"insufficient_quota","message":"quota exceeded"}}`,
			&ParsedStreamError{Type: "api_error", Message: "Quota exceeded. Check your plan and billing details."},
		},
		{
			"usage not included",
			`{"type":"error","error":{"code":"usage_not_included","message":"upgrade"}}`,
			&ParsedStreamError{Type: "api_error", Message: "To use Codex with your ChatGPT plan, upgrade to Plus."},
		},
		{
			"invalid prompt with message",
			`{"type":"error","error":{"code":"invalid_prompt","message":"bad input"}}`,
			&ParsedStreamError{Type: "api_error", Message: "bad input"},
		},
		{
			"invalid prompt without message",
			`{"type":"error","error":{"code":"invalid_prompt","message":""}}`,
			&ParsedStreamError{Type: "api_error", Message: "Invalid prompt."},
		},
		{
			"not an error",
			`{"type":"message_start"}`,
			nil,
		},
		{
			"invalid json",
			`not json`,
			nil,
		},
		{
			"unknown error code",
			`{"type":"error","error":{"code":"unknown_code","message":"something"}}`,
			nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseStreamError([]byte(tt.body))
			if tt.want == nil {
				if got != nil {
					t.Errorf("expected nil, got %+v", got)
				}
				return
			}
			if got == nil {
				t.Fatal("expected non-nil result")
			}
			if got.Type != tt.want.Type {
				t.Errorf("type: got %q, want %q", got.Type, tt.want.Type)
			}
			if got.Message != tt.want.Message {
				t.Errorf("message: got %q, want %q", got.Message, tt.want.Message)
			}
		})
	}
}

func TestContextOverflowError_Error(t *testing.T) {
	err := &ContextOverflowError{Message: "too long"}
	if err.Error() != "too long" {
		t.Errorf("Error() = %q, want %q", err.Error(), "too long")
	}
}

func TestAPIError_Error(t *testing.T) {
	err := &APIError{Message: "rate limited", StatusCode: 429}
	if err.Error() != "rate limited" {
		t.Errorf("Error() = %q, want %q", err.Error(), "rate limited")
	}
}

func TestParseHTTPError_TopLevelMessage(t *testing.T) {
	// Responses API format: top-level message field
	body := []byte(`{"message":"model not available","code":"model_not_found"}`)
	err := ParseHTTPError("openai", 404, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	if apiErr.Message != "model not available" {
		t.Errorf("Message = %q", apiErr.Message)
	}
}

func TestParseHTTPError_ErrorAsString(t *testing.T) {
	body := []byte(`{"error":"something went wrong"}`)
	err := ParseHTTPError("anthropic", 500, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	if apiErr.Message != "something went wrong" {
		t.Errorf("Message = %q", apiErr.Message)
	}
}

func TestParseHTTPError_InvalidJSON(t *testing.T) {
	body := []byte(`not json at all`)
	err := ParseHTTPError("openai", 502, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	// Should fall back to status text + body for debugging
	if apiErr.Message != "502 Bad Gateway: not json at all" {
		t.Errorf("Message = %q, want '502 Bad Gateway: not json at all'", apiErr.Message)
	}
}

func TestParseHTTPError_EmptyJSONObject(t *testing.T) {
	body := []byte(`{}`)
	err := ParseHTTPError("openai", 503, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	// Empty JSON object falls back to status text + body for debugging
	if apiErr.Message != "503 Service Unavailable: {}" {
		t.Errorf("Message = %q", apiErr.Message)
	}
}

func TestParseHTTPError_UnknownStatusCode(t *testing.T) {
	// Status code 999 has no standard text
	body := []byte(`not json`)
	err := ParseHTTPError("openai", 999, body)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	// Status code 999 has no standard text, but body is included
	if apiErr.Message != "999 : not json" {
		t.Errorf("Message = %q, want '999 : not json'", apiErr.Message)
	}
}

func TestExtractErrorMessage_BodyTruncation(t *testing.T) {
	// Body > 200 chars should be truncated with "..."
	longBody := make([]byte, 300)
	for i := range longBody {
		longBody[i] = 'x'
	}
	err := ParseHTTPError("test", 502, longBody)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	// Should end with "..."
	if len(apiErr.Message) > 220 && !strings.HasSuffix(apiErr.Message, "...") {
		t.Errorf("expected truncated body with ..., got len=%d", len(apiErr.Message))
	}
	if !strings.Contains(apiErr.Message, "...") {
		t.Errorf("expected ... in truncated message, got %q", apiErr.Message)
	}
}

func TestExtractErrorMessage_MultiByte(t *testing.T) {
	// Build a body of 201 Japanese characters (each is 3 bytes in UTF-8).
	// Byte-based truncation at 200 would split a multi-byte sequence, producing invalid UTF-8.
	// Rune-based truncation must produce valid UTF-8 and exactly 200 runes before the "...".
	rune200 := []rune("あ") // single 3-byte rune
	body := make([]rune, 201)
	for i := range body {
		body[i] = rune200[0]
	}
	bodyBytes := []byte(string(body))

	err := ParseHTTPError("test", 502, bodyBytes)

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	// The message format is "502 Bad Gateway: <truncated_body>..."
	// Verify it ends with "..." (was truncated).
	if !strings.HasSuffix(apiErr.Message, "...") {
		t.Errorf("expected message to end with '...', got %q", apiErr.Message)
	}
	// Verify the result is valid UTF-8 (byte truncation would produce invalid UTF-8 here).
	if !utf8.ValidString(apiErr.Message) {
		t.Error("message contains invalid UTF-8 after truncation")
	}
}

func TestExtractErrorMessage_EmptyBodyWithStatusText(t *testing.T) {
	// 999 with no body → "999 (no body)" which does not match overflow patterns.
	err := ParseHTTPError("test", 999, nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T: %v", err, err)
	}
	if apiErr.Message == "" {
		t.Error("expected non-empty message for status 999 no body")
	}
}

func TestExtractErrorMessage_StatusTextOnly(t *testing.T) {
	// Status code 0 with empty body: body length == 0 triggers early return path
	// "0 (no body)". Verify we get an error (not panic) and the message is non-empty.
	err := ParseHTTPError("test", 0, []byte{})
	if err == nil {
		t.Fatal("expected error for status 0 empty body, got nil")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		// status 0 empty body → "0 (no body)"; if it doesn't match overflow it's an APIError
		t.Fatalf("expected APIError, got %T: %v", err, err)
	}
	if apiErr.Message == "" {
		t.Error("expected non-empty message for status 0 empty body")
	}
}

func TestClassifyStreamError(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		wantNil  bool
		wantType string // "overflow" or "api"
		wantMsg  string
	}{
		{
			name:    "nil for invalid json",
			body:    "not json",
			wantNil: true,
		},
		{
			name:    "nil for non-error type",
			body:    `{"type":"message_start"}`,
			wantNil: true,
		},
		{
			name:     "context overflow",
			body:     `{"type":"error","error":{"code":"context_length_exceeded","message":"too long"}}`,
			wantType: "overflow",
			wantMsg:  "Input exceeds context window of this model",
		},
		{
			name:     "api error - insufficient quota",
			body:     `{"type":"error","error":{"code":"insufficient_quota","message":"quota exceeded"}}`,
			wantType: "api",
			wantMsg:  "Quota exceeded. Check your plan and billing details.",
		},
		{
			name:     "api error - usage not included",
			body:     `{"type":"error","error":{"code":"usage_not_included","message":"upgrade"}}`,
			wantType: "api",
			wantMsg:  "To use Codex with your ChatGPT plan, upgrade to Plus.",
		},
		{
			name:    "nil for unknown code",
			body:    `{"type":"error","error":{"code":"unknown","message":"nope"}}`,
			wantNil: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ClassifyStreamError([]byte(tt.body))
			if tt.wantNil {
				if err != nil {
					t.Fatalf("expected nil, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected non-nil error")
			}
			switch tt.wantType {
			case "overflow":
				var overflow *ContextOverflowError
				if !errors.As(err, &overflow) {
					t.Fatalf("expected ContextOverflowError, got %T", err)
				}
				if overflow.Message != tt.wantMsg {
					t.Errorf("Message = %q, want %q", overflow.Message, tt.wantMsg)
				}
			case "api":
				var apiErr *APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("expected APIError, got %T", err)
				}
				if apiErr.Message != tt.wantMsg {
					t.Errorf("Message = %q, want %q", apiErr.Message, tt.wantMsg)
				}
			}
		})
	}
}

func TestParseHTTPErrorWithHeaders_RetryAfter(t *testing.T) {
	tests := []struct {
		name       string
		headers    map[string]string
		wantHeader string
		wantValue  string
	}{
		{
			name:       "retry-after header",
			headers:    map[string]string{"Retry-After": "5"},
			wantHeader: "retry-after",
			wantValue:  "5",
		},
		{
			name:       "retry-after-ms header",
			headers:    map[string]string{"Retry-After-Ms": "500"},
			wantHeader: "retry-after-ms",
			wantValue:  "500",
		},
		{
			name:       "both headers",
			headers:    map[string]string{"Retry-After": "5", "Retry-After-Ms": "500"},
			wantHeader: "retry-after",
			wantValue:  "5",
		},
		{
			name: "no retry headers",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var h http.Header
			if tt.headers != nil {
				h = make(http.Header)
				for k, v := range tt.headers {
					h.Set(k, v)
				}
			}
			err := ParseHTTPErrorWithHeaders("test", 429, []byte(`{"error":"rate limited"}`), h)
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("expected APIError, got %T", err)
			}
			if tt.wantHeader != "" {
				if apiErr.ResponseHeaders == nil {
					t.Fatal("expected ResponseHeaders to be non-nil")
				}
				if apiErr.ResponseHeaders[tt.wantHeader] != tt.wantValue {
					t.Errorf("ResponseHeaders[%q] = %q, want %q", tt.wantHeader, apiErr.ResponseHeaders[tt.wantHeader], tt.wantValue)
				}
			} else if tt.headers == nil {
				if apiErr.ResponseHeaders != nil {
					t.Errorf("expected nil ResponseHeaders, got %v", apiErr.ResponseHeaders)
				}
			}
		})
	}
}
