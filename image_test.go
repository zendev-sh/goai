package goai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

type mockImageModel struct {
	generateFn func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error)
}

func (m *mockImageModel) ModelID() string { return "test-model" }
func (m *mockImageModel) DoGenerate(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
	return m.generateFn(ctx, params)
}

func TestGenerateImage_NilModel(t *testing.T) {
	_, err := GenerateImage(t.Context(), nil, WithImagePrompt("test"))
	if err == nil {
		t.Fatal("expected error for nil model, got nil")
	}
	if err.Error() != "goai: model must not be nil" {
		t.Errorf("error = %q, want %q", err.Error(), "goai: model must not be nil")
	}
}

func TestGenerateImage(t *testing.T) {
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			if params.Prompt != "a cat" {
				t.Errorf("prompt = %q", params.Prompt)
			}
			if params.N != 1 {
				t.Errorf("n = %d", params.N)
			}
			return &provider.ImageResult{
				Images: []provider.ImageData{
					{Data: []byte("fake-image"), MediaType: "image/png"},
				},
			}, nil
		},
	}

	result, err := GenerateImage(t.Context(), model, WithImagePrompt("a cat"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 {
		t.Errorf("images = %d", len(result.Images))
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("mediaType = %q", result.Images[0].MediaType)
	}
}

func TestGenerateImage_Options(t *testing.T) {
	called := false
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			called = true
			if params.N != 3 {
				t.Errorf("n = %d, want 3", params.N)
			}
			if params.Size != "1024x1024" {
				t.Errorf("size = %q", params.Size)
			}
			if params.AspectRatio != "16:9" {
				t.Errorf("aspect = %q", params.AspectRatio)
			}
			return &provider.ImageResult{Images: []provider.ImageData{{Data: []byte("x"), MediaType: "image/png"}}}, nil
		},
	}

	result, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageCount(3),
		WithImageSize("1024x1024"),
		WithAspectRatio("16:9"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("generateFn was never called")
	}
	if len(result.Images) != 1 {
		t.Errorf("images = %d, want 1", len(result.Images))
	}
}

func TestGenerateImage_WithProviderOptions(t *testing.T) {
	called := false
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			called = true
			if params.ProviderOptions["quality"] != "hd" {
				t.Errorf("quality = %v, want hd", params.ProviderOptions["quality"])
			}
			return &provider.ImageResult{Images: []provider.ImageData{{Data: []byte("x"), MediaType: "image/png"}}}, nil
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageProviderOptions(map[string]any{"quality": "hd"}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("generateFn was never called")
	}
}

func TestWithImageProviderOptions_ValidValues(t *testing.T) {
	// Should not panic with JSON-serializable values.
	// Validation fires at construction time ;  no need to call GenerateImage.
	WithImageProviderOptions(map[string]any{
		"quality": "hd",
		"count":   2,
		"nested":  map[string]any{"key": "val"},
	})
}

func TestWithImageProviderOptions_NonSerializable(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for non-serializable image provider option")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value is not a string: %v", r)
		}
		if !strings.Contains(msg, "WithImageProviderOptions") {
			t.Errorf("panic message %q should contain caller name WithImageProviderOptions", msg)
		}
	}()
	// validateProviderOptions fires at option construction time, before GenerateImage is called.
	WithImageProviderOptions(map[string]any{"fn": func() {}})
}

func TestGenerateImage_Error(t *testing.T) {
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			return nil, context.DeadlineExceeded
		},
	}

	_, err := GenerateImage(t.Context(), model, WithImagePrompt("test"))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestGenerateImage_WithTimeout(t *testing.T) {
	// Verify the timeout actually fires when the model blocks longer than the deadline.
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			select {
			case <-time.After(2 * time.Second):
				return &provider.ImageResult{Images: []provider.ImageData{{Data: []byte("x"), MediaType: "image/png"}}}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageTimeout(50*time.Millisecond),
	)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %v, want context.DeadlineExceeded", err)
	}
}

func TestGenerateImage_WithMaxRetries(t *testing.T) {
	var calls atomic.Int32
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			n := calls.Add(1)
			if n < 3 {
				// retry-after-ms: 1 drives 1ms backoff instead of the 1-3s exponential default.
				return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: true, ResponseHeaders: map[string]string{"retry-after-ms": "1"}}
			}
			return &provider.ImageResult{Images: []provider.ImageData{{Data: []byte("x"), MediaType: "image/png"}}}, nil
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageMaxRetries(3),
	)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 3 {
		t.Errorf("expected 3 calls, got %d", calls.Load())
	}
}

func TestGenerateImage_RetryExhausted(t *testing.T) {
	// Verifies that withRetry wraps the error as "retries exhausted" for GenerateImage
	// when all retries are consumed and the final error is still retryable.
	var calls atomic.Int32
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			calls.Add(1)
			return nil, &APIError{
				StatusCode:      429,
				Message:         "rate limited",
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "1"},
			}
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageMaxRetries(1),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// MaxRetries=1: 1 initial + 1 retry = 2 total calls.
	if calls.Load() != 2 {
		t.Errorf("expected 2 calls (1 initial + 1 retry), got %d", calls.Load())
	}
	// withRetry wraps when all retries exhausted and final error is retryable.
	if !strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("expected 'retries exhausted' wrapper, got: %v", err)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError in error chain, got %T", err)
	}
}

func TestWithImageMaxRetries_NegativeClamped(t *testing.T) {
	var calls atomic.Int32
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			calls.Add(1)
			return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: true}
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageMaxRetries(-5),
	)
	// With maxRetries=0 (clamped from -5), only 1 attempt is made, then error returned.
	if err == nil {
		t.Fatal("expected error")
	}
	if calls.Load() != 1 {
		t.Errorf("expected 1 call (no retries), got %d", calls.Load())
	}
	// Error must be returned unwrapped ;  maxRetries==0 means no retry budget, so
	// "retries exhausted" wrapping must NOT occur. Caller must get the raw *APIError.
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("maxRetries=0 must not produce 'retries exhausted' wrapper, got: %v", err)
	}
}

func TestWithImageCount_NonPositiveClamped(t *testing.T) {
	for _, input := range []int{-5, 0, 1} {
		input := input
		t.Run(fmt.Sprintf("input=%d", input), func(t *testing.T) {
			var gotN int
			model := &mockImageModel{
				generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
					gotN = params.N
					return &provider.ImageResult{Images: []provider.ImageData{{Data: []byte("x"), MediaType: "image/png"}}}, nil
				},
			}
			_, err := GenerateImage(t.Context(), model,
				WithImagePrompt("test"),
				WithImageCount(input),
			)
			if err != nil {
				t.Fatal(err)
			}
			if gotN != 1 {
				t.Errorf("WithImageCount(%d): params.N = %d, want 1 (clamped to minimum)", input, gotN)
			}
		})
	}
}

func TestGenerateImage_MaxRetriesZero(t *testing.T) {
	// Verifies that WithImageMaxRetries(0) returns the raw error without wrapping.
	// This is the explicit-zero case (not clamped-from-negative).
	// The withRetry condition requires maxRetries > 0 to produce "retries exhausted",
	// so callers opting out of retries always receive the raw provider error directly.
	var calls atomic.Int32
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			calls.Add(1)
			return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: true}
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected error")
	}
	// Exactly 1 call: no retry budget.
	if calls.Load() != 1 {
		t.Errorf("expected 1 call (no retries), got %d", calls.Load())
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("maxRetries=0 must not produce 'retries exhausted' wrapper, got: %v", err)
	}
}
