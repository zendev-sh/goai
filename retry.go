package goai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"net"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// retryable checks if an error should be retried.
// API errors use the IsRetryable flag. Network-level errors (connection reset,
// timeout, DNS failure) are always retryable since they indicate transient
// infrastructure issues, not permanent request problems.
func retryable(err error) bool {
	if err == nil {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.IsRetryable
	}
	return isNetworkError(err)
}

// isNetworkError reports whether err is a transient network error (connection
// reset, refused, DNS, TLS handshake timeout). Context errors are excluded
// because they represent caller-initiated cancellation.
func isNetworkError(err error) bool {
	if err == nil {
		return false
	}
	// Context errors are intentional cancellations, not network errors.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return false
	}
	// net.Error covers timeouts, DNS failures, connection resets, refused
	// connections, and broken pipes (net.OpError and syscall.Errno both
	// implement net.Error).
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	// Fallback: string match for common network error messages that may be
	// wrapped in ways that don't implement net.Error.
	msg := err.Error()
	return strings.Contains(msg, "connection reset by peer") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "TLS handshake timeout") ||
		strings.Contains(msg, "no such host") ||
		strings.Contains(msg, "i/o timeout")
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

// retryAfterDuration extracts a Retry-After delay from an APIError.
// Checks (in order): retry-after-ms header (OpenAI), retry-after header (standard),
// then falls back to parsing "wait N seconds" from the error message body (Azure).
//
// Following the Vercel AI SDK pattern: the server-requested delay is only used
// when it is "reasonable" - under 60s or less than the caller's exponential
// backoff. Long delays (≥60s) are ignored so the caller can fail fast and let
// the exponential schedule take over. The caller is responsible for applying
// this threshold via retryDelay().
func retryAfterDuration(err error) time.Duration {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return 0
	}
	// OpenAI uses retry-after-ms (milliseconds).
	if apiErr.ResponseHeaders != nil {
		if ms, ok := apiErr.ResponseHeaders["retry-after-ms"]; ok {
			if v, parseErr := strconv.ParseInt(ms, 10, 64); parseErr == nil && v > 0 {
				return time.Duration(v) * time.Millisecond
			}
		}
		// Standard Retry-After (seconds).
		if secs, ok := apiErr.ResponseHeaders["retry-after"]; ok {
			if v, parseErr := strconv.ParseInt(secs, 10, 64); parseErr == nil && v > 0 {
				return time.Duration(v) * time.Second
			}
		}
	}
	// Fallback: parse "wait N seconds" or "retry after N seconds" from error body.
	// Azure AI Services embeds the retry hint in the error message rather than headers.
	if d := parseRetryFromBody(apiErr.Message); d > 0 {
		return d
	}
	return 0
}

// retryDelay returns the delay to use before the next retry attempt.
// Follows the Vercel AI SDK pattern: the server-requested delay (from headers
// or error body) is used only when it is under 60s or less than the calculated
// exponential backoff. Otherwise the exponential backoff is used. This ensures
// fast failure for long rate-limit windows instead of blocking for minutes.
func retryDelay(err error, attempt int) time.Duration {
	serverDelay := retryAfterDuration(err)
	expDelay := backoffDuration(attempt)
	if serverDelay > 0 && (serverDelay < 60*time.Second || serverDelay < expDelay) {
		return serverDelay
	}
	return expDelay
}

// waitSecondsRe matches patterns like "wait 60 seconds", "retry after 30 seconds",
// "Please wait 60 seconds before retrying".
var waitSecondsRe = regexp.MustCompile(`(?i)(?:wait|retry after)\s+(\d+)\s+seconds?`)

// parseRetryFromBody extracts a retry delay from an error message body.
func parseRetryFromBody(msg string) time.Duration {
	m := waitSecondsRe.FindStringSubmatch(msg)
	if m == nil {
		return 0
	}
	v, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil || v <= 0 {
		return 0
	}
	return time.Duration(v) * time.Second
}

// withRetry executes fn up to maxRetries+1 times, retrying on retryable errors.
// Respects Retry-After/Retry-After-ms headers when present; falls back to
// exponential backoff otherwise.
func withRetry[T any](ctx context.Context, maxRetries int, fn func() (T, error)) (T, error) {
	result, err := fn()
	attempt := 0
	for ; err != nil && retryable(err) && attempt < maxRetries; attempt++ {
		delay := retryDelay(err, attempt)
		if sleepErr := sleep(ctx, delay); sleepErr != nil {
			var zero T
			return zero, sleepErr
		}
		result, err = fn()
	}
	// Wrap the error only when all retries were consumed AND the error is still retryable.
	// If the final error is non-retryable (e.g. a 400 returned on the last retry), it is
	// returned unwrapped ;  "retries exhausted" would be misleading since the loop stopped
	// due to non-retryability, not exhaustion of the retry budget.
	// Non-retryable errors and errors returned when maxRetries==0 are returned unwrapped
	// so callers can compare them directly (e.g. errors.Is / identity checks).
	if err != nil && attempt == maxRetries && maxRetries > 0 && retryable(err) {
		return result, fmt.Errorf("goai: %d retries exhausted: %w", maxRetries, err)
	}
	return result, err
}
