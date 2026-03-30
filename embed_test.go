package goai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// mockEmbeddingModel implements provider.EmbeddingModel for testing.
type mockEmbeddingModel struct {
	mu              sync.Mutex
	modelID         string
	maxPerCall      int
	embedFn         func(ctx context.Context, values []string) (*provider.EmbedResult, error)
	callCount       atomic.Int32
	callArgs        [][]string // recorded arguments for each call
	concurrentPeak  atomic.Int32
	concurrentCount atomic.Int32
	lastParams      provider.EmbedParams // last EmbedParams received
}

func (m *mockEmbeddingModel) ModelID() string { return m.modelID }

func (m *mockEmbeddingModel) MaxValuesPerCall() int { return m.maxPerCall }

func (m *mockEmbeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	m.callCount.Add(1)
	cur := m.concurrentCount.Add(1)
	for {
		peak := m.concurrentPeak.Load()
		if cur <= peak || m.concurrentPeak.CompareAndSwap(peak, cur) {
			break
		}
	}
	defer m.concurrentCount.Add(-1)

	m.mu.Lock()
	m.callArgs = append(m.callArgs, values)
	m.lastParams = params
	m.mu.Unlock()

	return m.embedFn(ctx, values)
}



// --- Embed tests ---

func TestEmbed_SingleEmbedding(t *testing.T) {
	expected := []float64{0.1, 0.2, 0.3}
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = expected
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: 5},
			}, nil
		},
	}

	result, err := Embed(t.Context(), model, "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embedding) != 3 {
		t.Fatalf("expected 3 floats, got %d", len(result.Embedding))
	}
	for i, v := range expected {
		if result.Embedding[i] != v {
			t.Errorf("embedding[%d] = %f, want %f", i, result.Embedding[i], v)
		}
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("usage.InputTokens = %d, want 5", result.Usage.InputTokens)
	}
	if model.callCount.Load() != 1 {
		t.Errorf("expected 1 call, got %d", model.callCount.Load())
	}
}

func TestEmbed_EmptyEmbeddings(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{},
			}, nil
		},
	}

	_, err := Embed(t.Context(), model, "hello")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "goai: no embedding returned" {
		t.Errorf("error = %q, want %q", err.Error(), "goai: no embedding returned")
	}
}

func TestEmbed_ModelError(t *testing.T) {
	modelErr := fmt.Errorf("model exploded")
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return nil, modelErr
		},
	}

	_, err := Embed(t.Context(), model, "hello")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err != modelErr {
		t.Errorf("error = %v, want %v", err, modelErr)
	}
}

func TestEmbed_RetryOnTransientError(t *testing.T) {
	var attempt atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			n := attempt.Add(1)
			if n <= 2 {
				return nil, &APIError{
					Message:     "rate limited",
					StatusCode:  429,
					IsRetryable: true,
				}
			}
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0, 2.0}},
				Usage:      provider.Usage{InputTokens: 3},
			}, nil
		},
	}

	result, err := Embed(t.Context(), model, "hello", WithMaxRetries(3))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Embedding[0] != 1.0 || result.Embedding[1] != 2.0 {
		t.Errorf("unexpected embedding: %v", result.Embedding)
	}
	if attempt.Load() != 3 {
		t.Errorf("expected 3 attempts, got %d", attempt.Load())
	}
}

func TestEmbed_RetryExhausted(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return nil, &APIError{
				Message:     "rate limited",
				StatusCode:  429,
				IsRetryable: true,
			}
		},
	}

	_, err := Embed(t.Context(), model, "hello", WithMaxRetries(1))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if !apiErr.IsRetryable {
		t.Error("expected IsRetryable = true")
	}
}

// --- EmbedMany tests ---

func TestEmbedMany_SingleBatch(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{float64(i + 1)}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: len(values) * 2},
			}, nil
		},
	}

	values := []string{"a", "b", "c"}
	result, err := EmbedMany(t.Context(), model, values)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Embeddings))
	}
	for i, emb := range result.Embeddings {
		if emb[0] != float64(i+1) {
			t.Errorf("embedding[%d] = %v, want [%d]", i, emb, i+1)
		}
	}
	if result.Usage.InputTokens != 6 {
		t.Errorf("usage.InputTokens = %d, want 6", result.Usage.InputTokens)
	}
	if model.callCount.Load() != 1 {
		t.Errorf("expected 1 call, got %d", model.callCount.Load())
	}
}

func TestEmbedMany_AutoChunking(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 2,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			// Small delay to allow parallelism observation.
			time.Sleep(10 * time.Millisecond)
			embeddings := make([][]float64, len(values))
			for i, v := range values {
				embeddings[i] = []float64{float64(len(v))}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage: provider.Usage{
					InputTokens:  len(values) * 3,
					OutputTokens: len(values),
				},
			}, nil
		},
	}

	values := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	result, err := EmbedMany(t.Context(), model, values)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 5 values with maxPerCall=2 => chunks: [a,bb], [ccc,dddd], [eeeee] => 3 calls
	if model.callCount.Load() != 3 {
		t.Errorf("expected 3 calls, got %d", model.callCount.Load())
	}

	// Verify embeddings are in correct order.
	if len(result.Embeddings) != 5 {
		t.Fatalf("expected 5 embeddings, got %d", len(result.Embeddings))
	}
	expected := []float64{1, 2, 3, 4, 5}
	for i, emb := range result.Embeddings {
		if emb[0] != expected[i] {
			t.Errorf("embedding[%d] = %v, want [%f]", i, emb, expected[i])
		}
	}
}

func TestEmbedMany_UnlimitedMaxPerCall(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 0, // unlimited
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{1.0}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: len(values)},
			}, nil
		},
	}

	values := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	result, err := EmbedMany(t.Context(), model, values)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 10 {
		t.Fatalf("expected 10 embeddings, got %d", len(result.Embeddings))
	}
	// Should be a single call since maxPerCall=0 means no chunking.
	if model.callCount.Load() != 1 {
		t.Errorf("expected 1 call (no chunking), got %d", model.callCount.Load())
	}
}

func TestEmbedMany_MaxParallelCalls(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			// Hold for a bit so concurrency can build up.
			time.Sleep(50 * time.Millisecond)
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{1.0}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: 1},
			}, nil
		},
	}

	values := []string{"a", "b", "c", "d", "e", "f"}
	result, err := EmbedMany(t.Context(), model, values, WithMaxParallelCalls(2))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 6 {
		t.Fatalf("expected 6 embeddings, got %d", len(result.Embeddings))
	}
	// Peak concurrency should not exceed 2.
	peak := model.concurrentPeak.Load()
	if peak > 2 {
		t.Errorf("peak concurrency = %d, want <= 2", peak)
	}
	if peak < 2 {
		t.Errorf("expected parallelism, peak=%d", peak)
	}
	if model.callCount.Load() != 6 {
		t.Errorf("expected 6 calls, got %d", model.callCount.Load())
	}
}

func TestEmbedMany_ChunkError(t *testing.T) {
	chunkErr := fmt.Errorf("chunk failed")
	var callNum atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 2,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			n := callNum.Add(1)
			// Fail on the second chunk.
			if n == 2 {
				return nil, chunkErr
			}
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{1.0}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: 1},
			}, nil
		},
	}

	values := []string{"a", "b", "c", "d", "e"}
	_, err := EmbedMany(t.Context(), model, values)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// The error should be the chunk error (non-retryable, so no retry).
	if err != chunkErr {
		t.Errorf("error = %v, want %v", err, chunkErr)
	}
}

func TestEmbedMany_UsageAggregated(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 2,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{1.0}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage: provider.Usage{
					InputTokens:      10,
					OutputTokens:     5,
					ReasoningTokens:  2,
					CacheReadTokens:  1,
					CacheWriteTokens: 3,
				},
			}, nil
		},
	}

	// 5 values, maxPerCall=2 => 3 chunks
	values := []string{"a", "b", "c", "d", "e"}
	result, err := EmbedMany(t.Context(), model, values)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 3 chunks * each usage value
	if result.Usage.InputTokens != 30 {
		t.Errorf("InputTokens = %d, want 30", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 15 {
		t.Errorf("OutputTokens = %d, want 15", result.Usage.OutputTokens)
	}
	if result.Usage.ReasoningTokens != 6 {
		t.Errorf("ReasoningTokens = %d, want 6", result.Usage.ReasoningTokens)
	}
	if result.Usage.CacheReadTokens != 3 {
		t.Errorf("CacheReadTokens = %d, want 3", result.Usage.CacheReadTokens)
	}
	if result.Usage.CacheWriteTokens != 9 {
		t.Errorf("CacheWriteTokens = %d, want 9", result.Usage.CacheWriteTokens)
	}
}

func TestEmbedMany_WithTimeout(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(ctx context.Context, _ []string) (*provider.EmbedResult, error) {
			select {
			case <-time.After(2 * time.Second):
				return &provider.EmbedResult{
					Embeddings: [][]float64{{1.0}},
				}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	values := []string{"a"}
	_, err := EmbedMany(t.Context(), model, values, WithTimeout(50*time.Millisecond))
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if err != context.DeadlineExceeded {
		t.Errorf("error = %v, want context.DeadlineExceeded", err)
	}
}

func TestEmbed_WithTimeout(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(ctx context.Context, _ []string) (*provider.EmbedResult, error) {
			select {
			case <-time.After(2 * time.Second):
				return &provider.EmbedResult{
					Embeddings: [][]float64{{1.0}},
				}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	_, err := Embed(t.Context(), model, "hello", WithTimeout(50*time.Millisecond))
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if err != context.DeadlineExceeded {
		t.Errorf("error = %v, want context.DeadlineExceeded", err)
	}
}

func TestEmbedMany_ChunkingPreservesOrder(t *testing.T) {
	// Verify that even with parallel execution, results maintain input order.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			// Vary sleep to encourage out-of-order completion.
			if values[0] == "first" {
				time.Sleep(30 * time.Millisecond)
			}
			embeddings := make([][]float64, len(values))
			for i, v := range values {
				embeddings[i] = []float64{float64(len(v))}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: 1},
			}, nil
		},
	}

	values := []string{"first", "ab", "abc"}
	result, err := EmbedMany(t.Context(), model, values)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// "first"=5, "ab"=2, "abc"=3 -- order must be preserved.
	expected := []float64{5, 2, 3}
	for i, emb := range result.Embeddings {
		if emb[0] != expected[i] {
			t.Errorf("embedding[%d] = %v, want [%f]", i, emb, expected[i])
		}
	}
}

func TestEmbedMany_EmptyValues(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{},
				Usage:      provider.Usage{},
			}, nil
		},
	}

	result, err := EmbedMany(t.Context(), model, []string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 0 {
		t.Errorf("expected 0 embeddings, got %d", len(result.Embeddings))
	}
	if model.callCount.Load() != 0 {
		t.Errorf("expected DoEmbed not to be called for empty input, got %d calls", model.callCount.Load())
	}
}

func TestEmbed_NilEmbeddings(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: nil,
			}, nil
		},
	}

	_, err := Embed(t.Context(), model, "hello")
	if err == nil {
		t.Fatal("expected error for nil embeddings, got nil")
	}
	if err.Error() != "goai: no embedding returned" {
		t.Errorf("error = %q, want %q", err.Error(), "goai: no embedding returned")
	}
}

func TestEmbed_ProviderOptionsPassedThrough(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0}},
				Usage:      provider.Usage{InputTokens: 1},
			}, nil
		},
	}

	provOpts := map[string]any{
		"input_type": "search_query",
		"truncate":   "END",
	}
	_, err := Embed(t.Context(), model, "hello", WithEmbeddingProviderOptions(provOpts))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	model.mu.Lock()
	got := model.lastParams
	model.mu.Unlock()

	if got.ProviderOptions == nil {
		t.Fatal("expected ProviderOptions to be set, got nil")
	}
	if got.ProviderOptions["input_type"] != "search_query" {
		t.Errorf("ProviderOptions[input_type] = %v, want search_query", got.ProviderOptions["input_type"])
	}
	if got.ProviderOptions["truncate"] != "END" {
		t.Errorf("ProviderOptions[truncate] = %v, want END", got.ProviderOptions["truncate"])
	}
}

func TestEmbedMany_ProviderOptionsPassedThrough(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 2,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{1.0}
			}
			return &provider.EmbedResult{
				Embeddings: embeddings,
				Usage:      provider.Usage{InputTokens: 1},
			}, nil
		},
	}

	provOpts := map[string]any{
		"dimensions": 256,
	}
	// 3 values with maxPerCall=2 forces chunking -- verify options reach all chunks.
	_, err := EmbedMany(t.Context(), model, []string{"a", "b", "c"}, WithEmbeddingProviderOptions(provOpts))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	model.mu.Lock()
	got := model.lastParams
	model.mu.Unlock()

	if got.ProviderOptions == nil {
		t.Fatal("expected ProviderOptions to be set, got nil")
	}
	if got.ProviderOptions["dimensions"] != 256 {
		t.Errorf("ProviderOptions[dimensions] = %v, want 256", got.ProviderOptions["dimensions"])
	}
}

func TestEmbedMany_ContextCancelledDuringSemaphore(t *testing.T) {
	// Test that goroutines waiting on the semaphore exit when ctx is cancelled.
	// Use maxParallelCalls=1 with many chunks so goroutines must queue on the semaphore.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(ctx context.Context, values []string) (*provider.EmbedResult, error) {
			// Block long enough for other goroutines to queue on the semaphore.
			select {
			case <-time.After(500 * time.Millisecond):
				return &provider.EmbedResult{
					Embeddings: [][]float64{{1.0}},
					Usage:      provider.Usage{InputTokens: 1},
				}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	ctx, cancel := context.WithCancel(t.Context())
	// Cancel quickly so goroutines waiting on the semaphore hit the ctx.Done() case.
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	// 10 values with maxPerCall=1, maxParallel=1: 9 goroutines will queue on semaphore.
	values := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	_, err := EmbedMany(ctx, model, values, WithMaxParallelCalls(1))
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	// The error should be context.Canceled from goroutines that couldn't acquire the semaphore.
	if err != context.Canceled {
		t.Errorf("error = %v, want context.Canceled", err)
	}
}

func TestEmbed_NilProviderOptionsDefault(t *testing.T) {
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0}},
				Usage:      provider.Usage{InputTokens: 1},
			}, nil
		},
	}

	// No WithEmbeddingProviderOptions -- ProviderOptions should be nil.
	_, err := Embed(t.Context(), model, "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	model.mu.Lock()
	got := model.lastParams
	model.mu.Unlock()

	if got.ProviderOptions != nil {
		t.Errorf("expected nil ProviderOptions when not set, got %v", got.ProviderOptions)
	}
}

func TestEmbed_NilModel(t *testing.T) {
	_, err := Embed(t.Context(), nil, "text")
	if err == nil {
		t.Fatal("expected error for nil model, got nil")
	}
	want := "model must not be nil"
	if !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want it to contain %q", err.Error(), want)
	}
}

func TestEmbedMany_NilModel(t *testing.T) {
	_, err := EmbedMany(t.Context(), nil, []string{"text"})
	if err == nil {
		t.Fatal("expected error for nil model, got nil")
	}
	want := "model must not be nil"
	if !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want it to contain %q", err.Error(), want)
	}
}
