package goai

import (
	"context"
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
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			// Verify context has a deadline from WithImageTimeout.
			if _, ok := ctx.Deadline(); !ok {
				t.Error("expected context to have deadline from WithImageTimeout")
			}
			return &provider.ImageResult{Images: []provider.ImageData{{Data: []byte("x"), MediaType: "image/png"}}}, nil
		},
	}

	_, err := GenerateImage(t.Context(), model,
		WithImagePrompt("test"),
		WithImageTimeout(5*time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImage_WithMaxRetries(t *testing.T) {
	calls := 0
	model := &mockImageModel{
		generateFn: func(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
			calls++
			if calls < 3 {
				return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: true}
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
	if calls != 3 {
		t.Errorf("expected 3 calls, got %d", calls)
	}
}
