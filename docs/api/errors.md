---
title: Errors
description: "API reference for GoAI error types. Covers APIError, ContextOverflowError, ParseHTTPError, ParseStreamError, and error handling patterns."
---

# Errors

GoAI defines typed error values for API failures. Use `errors.As` to inspect error details and decide how to handle them.

Import: `github.com/zendev-sh/goai`

---

## APIError

Represents a non-overflow API error from the provider.

```go
type APIError struct {
    Message         string            // Human-readable error message.
    StatusCode      int               // HTTP status code.
    IsRetryable     bool              // Whether this error can be retried.
    ResponseBody    string            // Raw response body for debugging.
    ResponseHeaders map[string]string // Selected response headers.
}
```

`APIError` implements the `error` interface via its `Error()` method, which returns the `Message` field.

### Retryable Status Codes

The following HTTP status codes produce retryable errors:

| Status Code       | Meaning                          |
| ----------------- | -------------------------------- |
| 429               | Too Many Requests (rate limited) |
| 503               | Service Unavailable              |
| 500+              | Any server error                 |
| 404 (OpenAI only) | Intermittent model availability  |

GoAI's built-in retry logic (controlled by `WithMaxRetries`, default 2) automatically retries these errors with exponential backoff. Non-retryable errors are returned immediately.

---

## ContextOverflowError

Indicates the prompt exceeded the model's context window. This error is never retryable - the request must be modified (e.g., by shortening the conversation history).

```go
type ContextOverflowError struct {
    Message      string // Human-readable error message.
    ResponseBody string // Raw response body.
}
```

GoAI detects context overflow from error messages across all supported providers, including:

- Anthropic ("prompt is too long")
- Amazon Bedrock ("input is too long for requested model")
- OpenAI ("exceeds the context window")
- Google Gemini ("input token count exceeds the maximum")
- xAI / Grok ("maximum prompt length is N")
- Groq ("reduce the length of the messages")
- OpenRouter / DeepSeek ("maximum context length is N tokens")
- GitHub Copilot ("exceeds the limit of N")
- llama.cpp / LM Studio ("exceeds the available context size", "greater than the context length")
- MiniMax ("context window exceeds limit")
- Kimi / Moonshot ("exceeded model token limit")
- Cerebras / Mistral (400/413 status with no body)
- Generic fallback (regex: `(?i)context[_ ]length[_ ]exceeded`)

---

## IsOverflow

A helper function that checks if an error message indicates a context overflow. Useful when you have a raw error string and need to classify it.

```go
func IsOverflow(message string) bool
```

Returns `true` if the message matches any known overflow pattern.

---

## ParseHTTPError

Classifies an HTTP error response into either `*ContextOverflowError` or `*APIError`. Called internally by provider implementations.

```go
func ParseHTTPError(providerID string, statusCode int, body []byte) error
```

**Parameters:**

| Name         | Type     | Description                                                   |
| ------------ | -------- | ------------------------------------------------------------- |
| `providerID` | `string` | Provider identifier (e.g., "openai"). Affects retry behavior. |
| `statusCode` | `int`    | HTTP status code from the response.                           |
| `body`       | `[]byte` | Raw response body.                                            |

**Returns:** `*ContextOverflowError` if the error message matches an overflow pattern, otherwise `*APIError`.

The function extracts human-readable error messages from two common API formats:

- Chat Completions format: `{"error": {"message": "..."}}`
- Responses API format: `{"message": "...", "code": "...", "type": "..."}` — goai extracts only the `message` field.

## ParseHTTPErrorWithHeaders

Like `ParseHTTPError` but preserves retry-related response headers (`retry-after`, `retry-after-ms`) on the returned `*APIError`. Used by providers to support header-based backoff.

```go
func ParseHTTPErrorWithHeaders(providerID string, statusCode int, body []byte, headers http.Header) error
```

---

## ParseStreamError

Parses error events from SSE streams (used by Anthropic and OpenAI streaming).

```go
func ParseStreamError(body []byte) *ParsedStreamError
```

Returns `nil` if the body does not contain a recognized error event. When non-nil, the `ParsedStreamError` contains:

```go
type StreamErrorType string

const (
    StreamErrorContextOverflow StreamErrorType = "context_overflow"
    StreamErrorAPI             StreamErrorType = "api_error"
)

type ParsedStreamError struct {
    Type         StreamErrorType // "context_overflow" or "api_error".
    Message      string          // Human-readable error message.
    IsRetryable  bool            // Whether this error can be retried.
    ResponseBody string          // Raw event body.
}
```

Recognized stream error codes:

| Code                      | Type               | Retryable | Message                                        |
| ------------------------- | ------------------ | --------- | ---------------------------------------------- |
| `context_length_exceeded` | `context_overflow` | No        | `"Input exceeds context window of this model"` |
| `insufficient_quota`      | `api_error`        | No        |
| `usage_not_included`      | `api_error`        | No        |
| `invalid_prompt`          | `api_error`        | No        |

---

## ClassifyStreamError

Higher-level variant of `ParseStreamError` that returns typed errors directly. Returns `*ContextOverflowError` or `*APIError`, or `nil` if the data is not a recognized error event.

```go
func ClassifyStreamError(body []byte) error
```

---

## ErrUnknownTool

Sentinel error set on `ToolCallInfo.Error` when a tool call references a tool not in the tool map during auto tool loop execution. Defined in `generate.go`.

```go
var ErrUnknownTool = errors.New("goai: unknown tool")
```

---

## Usage Patterns

### Checking for a specific error type

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("..."),
)
if err != nil {
    var overflow *goai.ContextOverflowError
    if errors.As(err, &overflow) {
        // Prompt too long - truncate conversation and retry.
        fmt.Println("Context overflow:", overflow.Message)
        return
    }

    var apiErr *goai.APIError
    if errors.As(err, &apiErr) {
        fmt.Printf("API error %d: %s (retryable: %v)\n",
            apiErr.StatusCode, apiErr.Message, apiErr.IsRetryable)
        return
    }

    // Other errors (network, context cancelled, etc.)
    log.Fatal(err)
}
```

### Handling overflow in a conversation loop

```go
for {
    result, err := goai.GenerateText(ctx, model,
        goai.WithMessages(messages...),
    )

    var overflow *goai.ContextOverflowError
    if errors.As(err, &overflow) {
        // Remove oldest messages and retry.
        messages = messages[2:]
        continue
    }
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result.Text)
    break
}
```

### Accessing error details in telemetry hooks

```go
goai.WithOnResponse(func(info goai.ResponseInfo) {
    if info.Error != nil {
        fmt.Printf("Request failed (status %d): %v\n", info.StatusCode, info.Error)
    } else {
        fmt.Printf("OK - %d tokens in %.2fs\n", info.Usage.TotalTokens, info.Latency.Seconds())
    }
})
```
