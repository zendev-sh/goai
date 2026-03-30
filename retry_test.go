package goai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

func TestRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"non-API", fmt.Errorf("random"), false},
		{"retryable", &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true}, true},
		{"not retryable", &APIError{StatusCode: http.StatusBadRequest, IsRetryable: false}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := retryable(tt.err); got != tt.want {
				t.Errorf("retryable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestBackoffDuration(t *testing.T) {
	// Attempt 0: ~2s (1s-3s), Attempt 1: ~4s (2s-6s), etc.
	for i := 0; i < 5; i++ {
		d := backoffDuration(i)
		if d <= 0 {
			t.Errorf("backoffDuration(%d) = %v, want > 0", i, d)
		}
	}
	// High attempt should be capped at 60s base * 1.5 jitter = 90s max.
	d := backoffDuration(100)
	if d > 90*time.Second {
		t.Errorf("backoffDuration(100) = %v, should be capped", d)
	}
}

func TestWithRetry_Success(t *testing.T) {
	calls := 0
	result, err := withRetry(t.Context(), 2, func() (string, error) {
		calls++
		return "ok", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != "ok" {
		t.Errorf("result = %q", result)
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
}

func TestWithRetry_RetryThenSuccess(t *testing.T) {
	calls := 0
	result, err := withRetry(t.Context(), 2, func() (string, error) {
		calls++
		if calls < 3 {
			return "", &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true, Message: "rate limited"}
		}
		return "ok", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != "ok" {
		t.Errorf("result = %q", result)
	}
	if calls != 3 {
		t.Errorf("calls = %d, want 3", calls)
	}
}

func TestWithRetry_AllFail(t *testing.T) {
	calls := 0
	_, err := withRetry(t.Context(), 2, func() (string, error) {
		calls++
		return "", &APIError{StatusCode: http.StatusServiceUnavailable, IsRetryable: true, Message: "unavailable"}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 3 {
		t.Errorf("calls = %d, want 3 (1 + 2 retries)", calls)
	}
}

func TestWithRetry_NonRetryable(t *testing.T) {
	calls := 0
	_, err := withRetry(t.Context(), 2, func() (string, error) {
		calls++
		return "", &APIError{StatusCode: http.StatusBadRequest, IsRetryable: false, Message: "bad request"}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1 (no retry)", calls)
	}
}

func TestWithRetry_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	calls := 0
	cancel() // Cancel immediately.
	_, err := withRetry(ctx, 5, func() (string, error) {
		calls++
		return "", &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true, Message: "rate limited"}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	// With a pre-cancelled context, only 1 call should happen (initial attempt).
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
}

func TestSleep_ContextDone(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err := sleep(ctx, 10*time.Second)
	if err == nil {
		t.Error("expected context error")
	}
}

func TestSleep_NormalWait(t *testing.T) {
	start := time.Now()
	err := sleep(t.Context(), 10*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 5*time.Millisecond {
		t.Errorf("elapsed = %v, expected >= 5ms", elapsed)
	}
}

func TestRetryAfterDuration(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want time.Duration
	}{
		{
			name: "nil error",
			err:  nil,
			want: 0,
		},
		{
			name: "non-API error",
			err:  fmt.Errorf("random"),
			want: 0,
		},
		{
			name: "API error without headers",
			err:  &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true},
			want: 0,
		},
		{
			name: "API error with nil headers map",
			err:  &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true, ResponseHeaders: nil},
			want: 0,
		},
		{
			name: "retry-after-ms valid",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "500"},
			},
			want: 500 * time.Millisecond,
		},
		{
			name: "retry-after-ms too large (>60s)",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "120000"},
			},
			want: 60000 * time.Millisecond,
		},
		{
			name: "retry-after-ms zero",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "0"},
			},
			want: 0,
		},
		{
			name: "retry-after-ms negative",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "-100"},
			},
			want: 0,
		},
		{
			name: "retry-after-ms invalid string",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "not-a-number"},
			},
			want: 0,
		},
		{
			name: "retry-after seconds valid",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after": "5"},
			},
			want: 5 * time.Second,
		},
		{
			name: "retry-after seconds too large (>60)",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after": "120"},
			},
			want: 60 * time.Second,
		},
		{
			name: "retry-after seconds zero",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after": "0"},
			},
			want: 0,
		},
		{
			name: "retry-after seconds invalid",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after": "abc"},
			},
			want: 0,
		},
		{
			name: "retry-after-ms takes priority over retry-after",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "200", "retry-after": "10"},
			},
			want: 200 * time.Millisecond,
		},
		{
			name: "fallback to retry-after when retry-after-ms is invalid",
			err: &APIError{
				StatusCode:      http.StatusTooManyRequests,
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "bad", "retry-after": "3"},
			},
			want: 3 * time.Second,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := retryAfterDuration(tt.err)
			if got != tt.want {
				t.Errorf("retryAfterDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGenerateText_WithRetry(t *testing.T) {
	calls := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			calls++
			if calls == 1 {
				return nil, &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true, Message: "rate limited"}
			}
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("hi"), WithMaxRetries(2))
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q", result.Text)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}
}

func TestRetryAfterDuration_WrappedError(t *testing.T) {
	// Wrap an *APIError with fmt.Errorf so it is no longer a direct *APIError.
	// retryAfterDuration must use errors.As internally to unwrap and find it.
	apiErr := &APIError{
		StatusCode:      http.StatusTooManyRequests,
		IsRetryable:     true,
		ResponseHeaders: map[string]string{"retry-after-ms": "750"},
	}
	wrapped := fmt.Errorf("wrapped: %w", apiErr)

	// Sanity-check: errors.As should find the inner *APIError.
	var extracted *APIError
	if !errors.As(wrapped, &extracted) {
		t.Fatal("errors.As could not unwrap *APIError from wrapped error")
	}

	got := retryAfterDuration(wrapped)
	want := 750 * time.Millisecond
	if got != want {
		t.Errorf("retryAfterDuration(wrapped) = %v, want %v", got, want)
	}
}

func TestStreamText_WithRetry(t *testing.T) {
	calls := 0
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			calls++
			if calls == 1 {
				return nil, &APIError{StatusCode: http.StatusServiceUnavailable, IsRetryable: true, Message: "unavailable"}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "ok"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"), WithMaxRetries(2))
	if err != nil {
		t.Fatal(err)
	}
	result := stream.Result()
	if result.Text != "ok" {
		t.Errorf("Text = %q", result.Text)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}
}

func TestGenerateText_WithTimeout(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(ctx context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				return &provider.GenerateResult{Text: "late"}, nil
			}
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithTimeout(50*time.Millisecond),
		WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestStreamText_WithTimeout(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(ctx context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "late"},
				), nil
			}
		},
	}

	_, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithTimeout(50*time.Millisecond),
		WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestWithRetry_ZeroRetries(t *testing.T) {
	calls := 0
	_, err := withRetry(t.Context(), 0, func() (string, error) {
		calls++
		return "", &APIError{StatusCode: http.StatusTooManyRequests, IsRetryable: true, Message: "rate limited"}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1 (no retries)", calls)
	}
}
