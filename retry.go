package goai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"time"
)

// retryable checks if an error should be retried.
func retryable(err error) bool {
	if err == nil {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.IsRetryable
	}
	return false
}

// backoffDuration returns the delay for attempt n (0-indexed) with jitter.
// Base delay: 2s, 4s, 8s, 16s, ... capped at 60s.
// Matches Vercel AI SDK's initial delay of 2000ms with backoff factor 2.
func backoffDuration(attempt int) time.Duration {
	base := 2 * math.Pow(2, float64(attempt))
	if base > 60 {
		base = 60
	}
	// Add jitter: 0.5x to 1.5x of base.
	jitter := 0.5 + rand.Float64()
	return time.Duration(base*jitter*1000) * time.Millisecond
}

// sleep waits for d or until ctx is cancelled.
func sleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// retryAfterDuration extracts a Retry-After delay from an APIError's response headers.
// Supports retry-after-ms (OpenAI) and retry-after (seconds). Returns 0 if not present
// or exceeds 60s (to avoid absurdly long waits).
func retryAfterDuration(err error) time.Duration {
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.ResponseHeaders == nil {
		return 0
	}
	// OpenAI uses retry-after-ms (milliseconds).
	if ms, ok := apiErr.ResponseHeaders["retry-after-ms"]; ok {
		if v, parseErr := strconv.ParseInt(ms, 10, 64); parseErr == nil && v > 0 {
			if v > 60000 {
				v = 60000
			}
			return time.Duration(v) * time.Millisecond
		}
	}
	// Standard Retry-After (seconds).
	if secs, ok := apiErr.ResponseHeaders["retry-after"]; ok {
		if v, parseErr := strconv.ParseInt(secs, 10, 64); parseErr == nil && v > 0 {
			if v > 60 {
				v = 60
			}
			return time.Duration(v) * time.Second
		}
	}
	return 0
}

// withRetry executes fn up to maxRetries+1 times, retrying on retryable errors.
// Respects Retry-After/Retry-After-ms headers when present; falls back to
// exponential backoff otherwise.
func withRetry[T any](ctx context.Context, maxRetries int, fn func() (T, error)) (T, error) {
	result, err := fn()
	attempt := 0
	for ; err != nil && retryable(err) && attempt < maxRetries; attempt++ {
		delay := retryAfterDuration(err)
		if delay == 0 {
			delay = backoffDuration(attempt)
		}
		if sleepErr := sleep(ctx, delay); sleepErr != nil {
			var zero T
			return zero, sleepErr
		}
		result, err = fn()
	}
	// Wrap the error only when all retries were consumed and the error is still retryable.
	// Non-retryable errors and errors returned when maxRetries==0 are returned unwrapped
	// so callers can compare them directly (e.g. errors.Is / identity checks).
	if err != nil && attempt == maxRetries && maxRetries > 0 {
		return result, fmt.Errorf("goai: %d retries exhausted: %w", maxRetries, err)
	}
	return result, err
}
