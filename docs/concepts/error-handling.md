---
title: Error Handling
description: "Handle API errors and context overflow in GoAI. Learn about automatic retries with exponential backoff, error types, and retry configuration."
---

# Error Handling

GoAI generation calls primarily return `*APIError` and `*ContextOverflowError`. Use `errors.As` to inspect them. Note that when retries are enabled (`WithMaxRetries > 0`), a retryable `*APIError` may be wrapped when the retry budget is exhausted.

## APIError

Returned for HTTP errors from the provider (rate limits, auth failures, server errors).

```go
type APIError struct {
    Message         string
    StatusCode      int
    IsRetryable     bool
    ResponseBody    string
    ResponseHeaders map[string]string
}
```

```go
result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    var apiErr *goai.APIError
    if errors.As(err, &apiErr) {
        fmt.Printf("HTTP %d: %s (retryable: %v)\n",
            apiErr.StatusCode, apiErr.Message, apiErr.IsRetryable)
    }
}
```

## ContextOverflowError

Returned when the prompt exceeds the model's context window. GoAI detects this from provider error messages across all supported providers.

```go
type ContextOverflowError struct {
    Message      string
    ResponseBody string
}
```

```go
var overflow *goai.ContextOverflowError
if errors.As(err, &overflow) {
    // Truncate messages and retry, or switch to a model with a larger context.
    fmt.Println("Context overflow:", overflow.Message)
}
```

## IsOverflow

`IsOverflow` checks an arbitrary error message string against known overflow patterns from all major providers. Useful if you handle raw HTTP responses yourself.

```go
if goai.IsOverflow(someErrorMessage) {
    // Handle context overflow
}
```

## ErrUnknownTool

Sentinel error returned when a tool call references a tool not in the tool map during auto tool loop execution.

```go
var overflow *goai.ContextOverflowError
if errors.As(err, &overflow) {
    // Truncate messages and retry, or switch to a model with a larger context.
    fmt.Println("Context overflow:", overflow.Message)
}

// Check for unknown tool
if errors.Is(err, goai.ErrUnknownTool) {
    // Tool was called that wasn't registered
}
```

## Stream Errors

GoAI provides types for handling streaming errors:

```go
// StreamErrorType classifies the type of stream error
type StreamErrorType string

const (
    StreamErrorContextOverflow StreamErrorType = "context_overflow"
    StreamErrorAPI             StreamErrorType = "api_error"
)

// ParsedStreamError represents a parsed streaming error
type ParsedStreamError struct {
    Type         StreamErrorType
    Message      string
    IsRetryable  bool
    ResponseBody string
}

// Parse a stream error from SSE data
parsed := goai.ParseStreamError(sseData)
if parsed != nil {
    if parsed.Type == goai.StreamErrorContextOverflow {
        // Handle context overflow mid-stream
    }
}

// Classify a raw error for handling
err := goai.ClassifyStreamError(body)
```

## HTTP Error Parsing

GoAI provides helper functions for parsing HTTP errors:

```go
// Parse a raw HTTP error response
err := goai.ParseHTTPError("openai", statusCode, body)

// Parse with response headers (for retry-after handling)
err := goai.ParseHTTPErrorWithHeaders("openai", statusCode, body, headers)

// Classify a stream error from SSE data
err := goai.ClassifyStreamError(sseBody)
```

## Retry Behavior

GoAI automatically retries on transient errors. Control this with `WithMaxRetries` (default: 2).

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Hello"),
    goai.WithMaxRetries(3),
)
```

### Which errors are retried

An error is retried only if `APIError.IsRetryable` is true. This includes:

- HTTP 429 (Too Many Requests)
- HTTP 503 (Service Unavailable)
- HTTP 5xx (Server Errors)
- HTTP 404 from OpenAI (intermittent model availability)

These errors are never retried:

- HTTP 400 (Bad Request)
- HTTP 401 (Unauthorized)
- HTTP 403 (Forbidden)
- `ContextOverflowError`
- Any non-API error (network timeout, context cancellation)

### Backoff strategy

Retries use exponential backoff with jitter:

- Base delay: 2s, 4s, 8s, 16s, ... capped at 60s
- Jitter: 0.5x to 1.5x of the base delay (max ~90s)
- Respects `retry-after` / `retry-after-ms` headers (capped at 60s)
- Respects context cancellation during the wait

### Disabling retries

```go
goai.WithMaxRetries(0) // No retries, fail on first error
```

## Streaming Error Detection

When using `StreamText` or `StreamObject`, errors that occur mid-stream (network failures, malformed SSE data, provider errors) are not returned inline. Instead, call `Err()` after consuming the stream to check for errors:

```go
stream, err := goai.StreamText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err) // Connection-level failure.
}

for text := range stream.TextStream() {
    fmt.Print(text)
}

if err := stream.Err(); err != nil {
    log.Fatal("streaming error:", err) // Mid-stream failure.
}
```

`Err()` blocks until the stream is fully consumed, then returns any error encountered during streaming. Always check `Err()` after draining the stream channel.
