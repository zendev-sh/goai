---
title: Error Handling
description: "Handle API errors and context overflow in GoAI. Learn about automatic retries with exponential backoff, error types, and retry configuration."
---

# Error Handling

GoAI returns two error types from generation calls. Use `errors.As` to inspect them.

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
- Jitter: 0.5x to 1.5x of the base delay
- Respects `Retry-After` / `Retry-After-ms` headers when present
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
