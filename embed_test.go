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
					Message:         "rate limited",
					StatusCode:      429,
					IsRetryable:     true,
					ResponseHeaders: map[string]string{"retry-after-ms": "1"},
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
				// retry-after-ms: 1 drives retryAfterDuration to 1ms, bypassing the
				// 1–3s exponential backoff so the test completes fast.
				ResponseHeaders: map[string]string{"retry-after-ms": "1"},
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
	// MaxRetries=1 means 1 retry after the initial attempt → 2 total calls.
	if model.callCount.Load() != 2 {
		t.Errorf("expected 2 calls (1 initial + 1 retry), got %d", model.callCount.Load())
	}
	// withRetry wraps the error when retries are exhausted; verify the wrapping is present.
	if !strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("expected error to mention 'retries exhausted', got: %v", err)
	}
}

func TestEmbed_RetryTransitionToNonRetryable(t *testing.T) {
	// Verifies withRetry behavior when the first call returns a retryable error but
	// the retry returns a non-retryable error. withRetry only wraps the error as
	// "retries exhausted" when the final error is still retryable ;  non-retryable
	// final errors are returned unwrapped so callers can compare them directly.
	var call atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			n := call.Add(1)
			if n == 1 {
				// First call: retryable ;  use retry-after-ms:1 to avoid slow backoff.
				return nil, &APIError{
					StatusCode:      429,
					Message:         "rate limited",
					IsRetryable:     true,
					ResponseHeaders: map[string]string{"retry-after-ms": "1"},
				}
			}
			// Retry: non-retryable.
			return nil, &APIError{StatusCode: 400, Message: "bad request", IsRetryable: false}
		},
	}

	_, err := Embed(t.Context(), model, "hello", WithMaxRetries(1))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Non-retryable final error is returned unwrapped ;  the *APIError is the error itself.
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("expected status 400, got %d", apiErr.StatusCode)
	}
	// Must NOT be wrapped with "retries exhausted" ;  the loop exited due to non-retryability.
	if strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("non-retryable error must not be wrapped as 'retries exhausted', got: %v", err)
	}
	if call.Load() != 2 {
		t.Errorf("expected 2 calls, got %d", call.Load())
	}
}

func TestEmbed_MaxRetriesZero(t *testing.T) {
	// Verifies that WithMaxRetries(0) returns the error unwrapped, even for retryable errors.
	// The withRetry condition requires maxRetries > 0 to produce the "retries exhausted" wrapper,
	// so callers opting out of retries always receive the raw provider error directly.
	var call atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			call.Add(1)
			return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: true}
		},
	}

	_, err := Embed(t.Context(), model, "hello", WithMaxRetries(0))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Exactly 1 call: no retry budget.
	if call.Load() != 1 {
		t.Errorf("expected 1 call, got %d", call.Load())
	}
	// Error must be the raw *APIError, NOT wrapped with "retries exhausted".
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", apiErr.StatusCode)
	}
	if strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("maxRetries=0 must not produce 'retries exhausted' wrapper, got: %v", err)
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
	// gateCh is closed once at least 2 goroutines have entered the mock concurrently.
	// This acts as a start barrier: once 2 goroutines are confirmed concurrent, all
	// goroutines can proceed. This guarantees peak >= 2 without relying on sleep timing.
	gateCh := make(chan struct{})
	var gateOnce sync.Once
	var entered atomic.Int32

	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(ctx context.Context, values []string) (*provider.EmbedResult, error) {
			// Signal entry; once 2 goroutines are inside, open the gate for all.
			if entered.Add(1) >= 2 {
				gateOnce.Do(func() { close(gateCh) })
			}
			select {
			case <-gateCh:
			case <-ctx.Done():
				return nil, ctx.Err()
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

	values := []string{"a", "b", "c", "d", "e", "f"}
	result, err := EmbedMany(t.Context(), model, values, WithMaxParallelCalls(2))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 6 {
		t.Fatalf("expected 6 embeddings, got %d", len(result.Embeddings))
	}
	peak := model.concurrentPeak.Load()
	if peak > 2 {
		t.Errorf("peak concurrency = %d, want <= 2", peak)
	}
	if peak < 2 {
		t.Errorf("expected peak concurrency >= 2, got %d; barrier guarantees 2 goroutines entered concurrently", peak)
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
			// Fail on whichever goroutine is second to execute (atomic order, not slice index order).
			// The test verifies that a single chunk error propagates through EmbedMany.
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
	if !errors.Is(err, chunkErr) {
		t.Errorf("error = %v, want %v", err, chunkErr)
	}
}

func TestEmbedMany_RetryExhausted(t *testing.T) {
	// Verifies that withRetry in EmbedMany's goroutine-based chunked path wraps errors
	// as "retries exhausted" when all retries are consumed and the final error is retryable.
	// Uses 2 values with maxPerCall=1 (len(values) > maxPerCall) to force the chunked goroutine
	// path. EmbedMany's wg.Wait() waits for all goroutines to complete before returning,
	// so call counts are deterministic: 2 chunks x 2 calls each (initial + 1 retry) = 4 total.
	var callCount atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			callCount.Add(1)
			return nil, &APIError{
				StatusCode:      429,
				Message:         "rate limited",
				IsRetryable:     true,
				ResponseHeaders: map[string]string{"retry-after-ms": "1"},
			}
		},
	}

	_, err := EmbedMany(t.Context(), model, []string{"a", "b"}, WithMaxRetries(1))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// 2 chunks x (1 initial + 1 retry) = 4 total calls; wg.Wait() guarantees all complete.
	if callCount.Load() != 4 {
		t.Errorf("expected 4 calls (2 chunks x 2 attempts), got %d", callCount.Load())
	}
	// Error is wrapped with chunk context.
	if !strings.Contains(err.Error(), "chunk") {
		t.Errorf("expected chunk context in error, got: %v", err)
	}
	// withRetry wraps when all retries exhausted and final error is retryable.
	if !strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("expected 'retries exhausted' wrapper, got: %v", err)
	}
	// errors.As still works through the chunk wrapping.
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError in error chain, got %T", err)
	}
}

func TestEmbedMany_MaxRetriesZero(t *testing.T) {
	// Verifies that WithMaxRetries(0) in EmbedMany returns the raw *APIError unwrapped.
	// Uses 2 values with maxPerCall=1 to force the goroutine-based chunked path.
	// With maxRetries=0 each chunk makes exactly 1 attempt; wg.Wait() ensures both complete,
	// so total calls = 2 (one per chunk).
	var callCount atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			callCount.Add(1)
			return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: true}
		},
	}

	_, err := EmbedMany(t.Context(), model, []string{"a", "b"}, WithMaxRetries(0))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// 2 chunks × 1 call each (no retries) = 2 total; wg.Wait() guarantees both complete.
	if callCount.Load() != 2 {
		t.Errorf("expected 2 calls (1 per chunk, no retries), got %d", callCount.Load())
	}
	// Error must be the raw *APIError ;  maxRetries=0 never triggers the "retries exhausted" wrapper.
	// EmbedMany iterates results in slice order and returns the first cr.err != nil it encounters;
	// the other chunk's error is silently discarded. This is the first-error-in-slice-order contract.
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", apiErr.StatusCode)
	}
	if strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("maxRetries=0 must not produce 'retries exhausted' wrapper, got: %v", err)
	}
}

func TestEmbedMany_UsageAggregated(t *testing.T) {
	// TotalTokens is treated as an independent provider-reported field and summed directly
	// across chunks (same as InputTokens, OutputTokens, etc.). The code does not re-derive
	// TotalTokens from InputTokens+OutputTokens after aggregation. The mock sets
	// TotalTokens = InputTokens + OutputTokens (15 = 10 + 5) per call, but this is a
	// test convenience, not a contract. The test verifies addUsage's arithmetic is correct.
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
					TotalTokens:      15, // = InputTokens + OutputTokens (per provider convention)
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
	if result.Usage.TotalTokens != 45 {
		t.Errorf("TotalTokens = %d, want 45", result.Usage.TotalTokens)
	}
	if result.Usage.CacheWriteTokens != 9 {
		t.Errorf("CacheWriteTokens = %d, want 9", result.Usage.CacheWriteTokens)
	}
}

func TestEmbedMany_ProviderMetadataMerged(t *testing.T) {
	// Use the input value to assign a deterministic chunk index regardless of goroutine scheduling order.
	// maxPerCall=1 guarantees goroutine i receives exactly values[i]: "a"→0, "b"→1, "c"→2.
	// mergeProviderMetadata iterates results[] in slice-index order (0→1→2), so results[2] wins on
	// conflicting keys. "Last chunk wins" refers to slice-index order, not goroutine completion order.
	chunkByValue := map[string]int{"a": 0, "b": 1, "c": 2}
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			idx, ok := chunkByValue[values[0]]
			if !ok {
				return nil, fmt.Errorf("unexpected chunk value %q", values[0])
			}
			meta := map[string]map[string]any{
				"test": {"chunkIndex": idx, "model": "v1"},
			}
			return &provider.EmbedResult{
				Embeddings:       [][]float64{{float64(idx)}},
				ProviderMetadata: meta,
			}, nil
		},
	}

	// 3 values, maxPerCall=1 => 3 chunks merged in index order; last chunk (index 2) wins conflicts.
	result, err := EmbedMany(t.Context(), model, []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata should not be nil")
	}
	testMeta, ok := result.ProviderMetadata["test"]
	if !ok {
		t.Fatal("ProviderMetadata missing 'test' namespace")
	}
	// "model" is the same across all chunks ;  any value is fine.
	if testMeta["model"] != "v1" {
		t.Errorf("model = %v, want v1", testMeta["model"])
	}
	// "chunkIndex" last-writer-wins: results are merged in slice order, so index 2 wins.
	if testMeta["chunkIndex"] != 2 {
		t.Errorf("chunkIndex = %v, want 2 (last chunk wins)", testMeta["chunkIndex"])
	}
}

func TestEmbedMany_ProviderMetadataNonConflicting(t *testing.T) {
	// Verify that non-conflicting keys from different chunks are all preserved.
	// mergeProviderMetadata merges at the key level within each namespace ; 
	// each chunk's unique keys survive alongside keys from other chunks.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			meta := map[string]map[string]any{}
			switch values[0] {
			case "a":
				meta["ns"] = map[string]any{"keyA": "valA"}
			case "b":
				meta["ns"] = map[string]any{"keyB": "valB"}
			}
			return &provider.EmbedResult{
				Embeddings:       [][]float64{{1.0}},
				ProviderMetadata: meta,
			}, nil
		},
	}

	result, err := EmbedMany(t.Context(), model, []string{"a", "b"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata should not be nil")
	}
	ns := result.ProviderMetadata["ns"]
	if ns == nil {
		t.Fatal("missing 'ns' namespace in merged ProviderMetadata")
	}
	if ns["keyA"] != "valA" {
		t.Errorf("keyA = %v, want valA (key from chunk 'a' should survive)", ns["keyA"])
	}
	if ns["keyB"] != "valB" {
		t.Errorf("keyB = %v, want valB (key from chunk 'b' should survive)", ns["keyB"])
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
	if !errors.Is(err, context.DeadlineExceeded) {
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
	if !errors.Is(err, context.DeadlineExceeded) {
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

func TestEmbedMany_ContextCancelDuringSemaphore(t *testing.T) {
	// Covers embed.go:172-174: the select case <-ctx.Done() while waiting for the semaphore.
	// We use maxPerCall=1 to produce many chunks and MaxParallelCalls=1 so all but one goroutine
	// block on the semaphore. The first chunk holds the semaphore while we cancel the context,
	// forcing the blocked goroutines to take the ctx.Done() path.
	started := make(chan struct{})
	unblock := make(chan struct{})

	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(ctx context.Context, values []string) (*provider.EmbedResult, error) {
			// Signal that this goroutine entered DoEmbed, then block until told to proceed.
			select {
			case started <- struct{}{}:
			default:
			}
			select {
			case <-unblock:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			embeddings := make([][]float64, len(values))
			for i := range values {
				embeddings[i] = []float64{1.0}
			}
			return &provider.EmbedResult{Embeddings: embeddings}, nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())

	// 5 values, maxPerCall=1 => 5 chunks; MaxParallelCalls=1 means only 1 runs at a time.
	// The first goroutine acquires the semaphore and blocks inside embedFn.
	// The remaining 4 goroutines block on the semaphore select (the path we want to cover).
	errCh := make(chan error, 1)
	go func() {
		_, err := EmbedMany(ctx, model, []string{"a", "b", "c", "d", "e"}, WithMaxParallelCalls(1))
		errCh <- err
	}()

	// Wait for the first goroutine to enter embedFn, then cancel the context.
	<-started
	cancel()
	// Unblock the first goroutine so it exits cleanly.
	close(unblock)

	err := <-errCh
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context.Canceled", err)
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

	// WithTimeout cancels after 20ms; the mock blocks for 500ms, so the cancellation
	// fires reliably without relying on goroutine scheduling timing.
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()

	// 10 values with maxPerCall=1, maxParallel=1: the one goroutine holding the semaphore
	// is blocked inside DoEmbed; the remaining 9 wait on the semaphore select. When the
	// deadline fires, ctx.Done() unblocks all paths ;  both DoEmbed's select and the
	// semaphore-wait select return ctx.Err(). The test verifies that context cancellation
	// propagates correctly through EmbedMany regardless of which path a goroutine is on.
	values := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	_, err := EmbedMany(ctx, model, values, WithMaxParallelCalls(1))
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	// context.WithTimeout produces context.DeadlineExceeded (not context.Canceled).
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %v, want context.DeadlineExceeded", err)
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

func TestEmbedMany_CountMismatch_SingleCall(t *testing.T) {
	// Exercises the embedding count mismatch check in the single-call fast path
	// (len(values) <= maxPerCall). A misbehaving provider returns fewer embeddings
	// than the number of values requested.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{Embeddings: [][]float64{{1.0}}}, nil
		},
	}
	_, err := EmbedMany(t.Context(), model, []string{"a", "b"})
	if err == nil {
		t.Fatal("expected count mismatch error")
	}
	if !strings.Contains(err.Error(), "count mismatch") {
		t.Errorf("error = %q, want count mismatch", err.Error())
	}
}

func TestEmbedMany_CountMismatch_ChunkedPath(t *testing.T) {
	// Exercises the per-chunk embedding count mismatch check in the goroutine-based
	// chunked path. maxPerCall=1 with 2 values forces the goroutine chunked path (2
	// chunks of 1 value each). One chunk returns an empty Embeddings slice (under-delivers),
	// triggering the per-chunk check: len(cr.result.Embeddings) != len(chunks[i]).
	var callNum atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			n := callNum.Add(1)
			if n == 1 {
				return &provider.EmbedResult{Embeddings: [][]float64{{1.0}}}, nil
			}
			return &provider.EmbedResult{Embeddings: [][]float64{}}, nil
		},
	}
	_, err := EmbedMany(t.Context(), model, []string{"a", "b"})
	if err == nil {
		t.Fatal("expected count mismatch error")
	}
	if !strings.Contains(err.Error(), "count mismatch") {
		t.Errorf("error = %q, want count mismatch", err.Error())
	}
}

func TestEmbedMany_CountMismatch_Compensating(t *testing.T) {
	// Exercises the production gap where compensating chunk mismatches would bypass
	// the aggregate check. One chunk over-delivers (2 embeddings for 1 input) and
	// the other under-delivers (0 embeddings for 1 input). Which chunk gets which
	// behavior is determined by atomic callNum order (non-deterministic), but the
	// observable outcome is always an error: whichever chunk has a count mismatch
	// is caught by the per-chunk check. Without per-chunk validation, allEmbeddings
	// total = 2 = len(values), so no error fires and the caller gets silently wrong
	// embeddings. The per-chunk check catches this immediately.
	var callNum atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			n := callNum.Add(1)
			if n == 1 {
				// over-delivers: 2 embeddings for 1 input value
				return &provider.EmbedResult{Embeddings: [][]float64{{1.0}, {2.0}}}, nil
			}
			// under-delivers: 0 embeddings for 1 input value
			return &provider.EmbedResult{Embeddings: [][]float64{}}, nil
		},
	}
	_, err := EmbedMany(t.Context(), model, []string{"a", "b"})
	if err == nil {
		t.Fatal("expected count mismatch error for compensating chunks")
	}
	if !strings.Contains(err.Error(), "count mismatch") {
		t.Errorf("error = %q, want count mismatch", err.Error())
	}
}

func TestEmbed_OverDelivery(t *testing.T) {
	// Documents Embed's behavior when a provider returns more embeddings than requested.
	// Embed sends a single value; if the provider returns multiple embeddings,
	// Embeddings[0] is used (the correct embedding for the single input) and extras
	// are silently ignored. This is consistent with the single-value contract.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0, 2.0}, {3.0, 4.0}}, // 2 embeddings for 1 input
			}, nil
		},
	}
	result, err := Embed(t.Context(), model, "hello")
	if err != nil {
		t.Fatal(err)
	}
	// Embeddings[0] is returned; the extra embedding is ignored.
	if len(result.Embedding) != 2 || result.Embedding[0] != 1.0 || result.Embedding[1] != 2.0 {
		t.Errorf("Embedding = %v, want [1.0 2.0]", result.Embedding)
	}
}

func TestEmbedMany_ProviderMetadataShallowCopy(t *testing.T) {
	// Documents mergeProviderMetadata's shallow-copy semantics: the outer map and each
	// namespace's inner map are freshly allocated, but reference-type values within the
	// inner maps are shared with the original provider result (not deep-copied).
	sharedSlice := []float64{1.0, 2.0}
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0}},
				ProviderMetadata: map[string]map[string]any{
					"ns": {"vec": sharedSlice},
				},
			}, nil
		},
	}
	result, err := EmbedMany(t.Context(), model, []string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	got, ok := result.ProviderMetadata["ns"]["vec"].([]float64)
	if !ok {
		t.Fatal("expected []float64 in ProviderMetadata[ns][vec]")
	}
	// Mutating the original slice is reflected in the returned metadata (shared reference).
	sharedSlice[0] = 99.0
	if got[0] != 99.0 {
		t.Errorf("expected shallow copy: got[0] = %v, want 99.0 (shared reference)", got[0])
	}
}

func TestEmbedMany_OverDelivery_SingleCall(t *testing.T) {
	// EmbedMany strictly rejects over-delivery on the single-call fast path ; 
	// unlike Embed which silently uses Embeddings[0]. If a provider returns more
	// embeddings than values requested, the count mismatch check fires.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0}, {2.0}}, // 2 embeddings for 1 input
			}, nil
		},
	}
	_, err := EmbedMany(t.Context(), model, []string{"a"})
	if err == nil {
		t.Fatal("expected count mismatch error for over-delivery")
	}
	if !strings.Contains(err.Error(), "count mismatch") {
		t.Errorf("error = %q, want count mismatch", err.Error())
	}
}

func TestEmbedMany_ProviderMetadataShallowCopy_ChunkedPath(t *testing.T) {
	// Documents mergeProviderMetadata's shallow-copy semantics via the goroutine-based
	// chunked path (maxPerCall=1, 2 values). Both chunks return the same sharedSlice under
	// key "vec"; mergeProviderMetadata last-write-wins overwrites the key, so the returned
	// value is always sharedSlice. The test verifies shallow-copy: if mergeProviderMetadata
	// deep-copied, `got` would be a new slice and mutating sharedSlice[0] would NOT change
	// got[0] → the test would fail. This confirms the reference is shared, not copied.
	sharedSlice := []float64{1.0, 2.0}
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, values []string) (*provider.EmbedResult, error) {
			embs := make([][]float64, len(values))
			for i := range embs {
				embs[i] = []float64{float64(i)}
			}
			return &provider.EmbedResult{
				Embeddings: embs,
				ProviderMetadata: map[string]map[string]any{
					"ns": {"vec": sharedSlice},
				},
			}, nil
		},
	}
	result, err := EmbedMany(t.Context(), model, []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	got, ok := result.ProviderMetadata["ns"]["vec"].([]float64)
	if !ok {
		t.Fatal("expected []float64 in ProviderMetadata[ns][vec]")
	}
	// Mutating the original slice is reflected in the returned metadata (shared reference).
	sharedSlice[0] = 99.0
	if got[0] != 99.0 {
		t.Errorf("expected shallow copy: got[0] = %v, want 99.0 (shared reference)", got[0])
	}
}

func TestEmbedMany_ProviderMetadataNilMix(t *testing.T) {
	// Verifies mergeProviderMetadata correctly handles a mix of nil and non-nil
	// ProviderMetadata across chunks in the goroutine-based chunked path.
	// One goroutine returns nil metadata (n==1), the other returns populated metadata (n==2).
	// The callNum atomic counter determines which goroutine gets which n value ;  this is
	// non-deterministic with respect to chunk index. However, results[] is indexed by chunk
	// (not atomic-call order), and mergeProviderMetadata processes results[] in slice-index
	// order after wg.Wait(). Chunks with nil ProviderMetadata are silently skipped, so the
	// populated chunk's metadata always appears in the merged result regardless of scheduling.
	var callNum atomic.Int32
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 1,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			n := callNum.Add(1)
			if n == 1 {
				// First atomic call: no metadata (nil ProviderMetadata).
				return &provider.EmbedResult{Embeddings: [][]float64{{1.0}}}, nil
			}
			// Second atomic call: has metadata.
			return &provider.EmbedResult{
				Embeddings: [][]float64{{2.0}},
				ProviderMetadata: map[string]map[string]any{
					"ns": {"key": "value"},
				},
			}, nil
		},
	}
	result, err := EmbedMany(t.Context(), model, []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	// Exactly one chunk has metadata; it is always present in the merged result
	// regardless of which goroutine executes first.
	if result.ProviderMetadata == nil {
		t.Fatal("expected non-nil ProviderMetadata from the populated chunk")
	}
	if result.ProviderMetadata["ns"] == nil {
		t.Fatalf("expected 'ns' namespace, got %v", result.ProviderMetadata)
	}
	if result.ProviderMetadata["ns"]["key"] != "value" {
		t.Errorf("ns.key = %v, want value", result.ProviderMetadata["ns"]["key"])
	}
}

func TestEmbed_ProviderMetadata(t *testing.T) {
	// Verifies that Embed correctly propagates non-nil ProviderMetadata from the provider.
	// Exercises the mergeProviderMetadata call on Embed's return path (embed.go line 72)
	// with a populated result ;  complementing EmbedMany tests that cover the same function.
	model := &mockEmbeddingModel{
		modelID:    "test-embed",
		maxPerCall: 10,
		embedFn: func(_ context.Context, _ []string) (*provider.EmbedResult, error) {
			return &provider.EmbedResult{
				Embeddings: [][]float64{{1.0, 2.0}},
				ProviderMetadata: map[string]map[string]any{
					"test": {"model": "v2", "usage": 42},
				},
			}, nil
		},
	}

	result, err := Embed(t.Context(), model, "hello")
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderMetadata == nil {
		t.Fatal("expected non-nil ProviderMetadata")
	}
	testMeta, ok := result.ProviderMetadata["test"]
	if !ok {
		t.Fatal("expected 'test' namespace in ProviderMetadata")
	}
	if testMeta["model"] != "v2" {
		t.Errorf("model = %v, want v2", testMeta["model"])
	}
	if testMeta["usage"] != 42 {
		t.Errorf("usage = %v, want 42", testMeta["usage"])
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
