package goai

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/zendev-sh/goai/provider"
)

// EmbedResult is the result of a single embedding generation.
type EmbedResult struct {
	// Embedding is the generated vector.
	Embedding []float64

	// Usage tracks token consumption.
	Usage provider.Usage

	// ProviderMetadata contains provider-specific response data.
	ProviderMetadata map[string]map[string]any
}

// EmbedManyResult is the result of multiple embedding generations.
type EmbedManyResult struct {
	// Embeddings contains the generated vectors (one per input value).
	Embeddings [][]float64

	// Usage is the aggregated token consumption.
	Usage provider.Usage

	// ProviderMetadata contains provider-specific response data.
	ProviderMetadata map[string]map[string]any
}

// Embed generates an embedding vector for a single value.
func Embed(ctx context.Context, model provider.EmbeddingModel, value string, opts ...Option) (*EmbedResult, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.Timeout)
		defer cancel()
	}

	embedParams := provider.EmbedParams{
		ProviderOptions: o.EmbeddingProviderOptions,
	}

	result, err := withRetry(ctx, o.MaxRetries, func() (*provider.EmbedResult, error) {
		return model.DoEmbed(ctx, []string{value}, embedParams)
	})
	if err != nil {
		return nil, err
	}

	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("goai: no embedding returned")
	}

	// Use Embeddings[0] for the single requested value; any extras are silently ignored.
	// Embed's single-value contract does not require strict count equality.
	// EmbedMany enforces strict equality (len != len(values) → error).
	return &EmbedResult{
		Embedding:        result.Embeddings[0],
		Usage:            result.Usage,
		ProviderMetadata: mergeProviderMetadata([]embedChunkResult{{result: result}}),
	}, nil
}

// EmbedMany generates embedding vectors for multiple values.
// Auto-chunks when values exceed the model's MaxValuesPerCall limit
// and processes chunks in parallel (controlled by WithMaxParallelCalls).
func EmbedMany(ctx context.Context, model provider.EmbeddingModel, values []string, opts ...Option) (*EmbedManyResult, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.Timeout)
		defer cancel()
	}

	embedParams := provider.EmbedParams{
		ProviderOptions: o.EmbeddingProviderOptions,
	}

	// Short-circuit for empty input - no API call needed.
	if len(values) == 0 {
		return &EmbedManyResult{}, nil
	}

	maxPerCall := model.MaxValuesPerCall()

	// Single call when no chunking needed.
	if maxPerCall <= 0 || len(values) <= maxPerCall {
		result, err := withRetry(ctx, o.MaxRetries, func() (*provider.EmbedResult, error) {
			return model.DoEmbed(ctx, values, embedParams)
		})
		if err != nil {
			return nil, err
		}
		if len(result.Embeddings) != len(values) {
			return nil, fmt.Errorf("goai: embedding count mismatch: got %d, expected %d", len(result.Embeddings), len(values))
		}
		return &EmbedManyResult{
			Embeddings:       result.Embeddings,
			Usage:            result.Usage,
			ProviderMetadata: mergeProviderMetadata([]embedChunkResult{{result: result}}),
		}, nil
	}

	// Split into chunks.
	var chunks [][]string
	for chunk := range slices.Chunk(values, maxPerCall) {
		chunks = append(chunks, chunk)
	}

	// Process chunks with bounded parallelism.
	maxParallel := o.MaxParallelCalls
	if maxParallel <= 0 {
		maxParallel = 4
	}

	results := make([]embedChunkResult, len(chunks))
	sem := make(chan struct{}, maxParallel)
	var wg sync.WaitGroup

	for i, chunk := range chunks {
		wg.Add(1)
		go func(i int, chunk []string) {
			defer wg.Done()
			// Use select to avoid blocking forever if ctx is cancelled
			// while waiting for the semaphore.
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				results[i] = embedChunkResult{err: ctx.Err()}
				return
			}

			r, err := withRetry(ctx, o.MaxRetries, func() (*provider.EmbedResult, error) {
				return model.DoEmbed(ctx, chunk, embedParams)
			})
			results[i] = embedChunkResult{result: r, err: err}
		}(i, chunk)
	}
	wg.Wait()

	// Combine results in order, validating each chunk's embedding count.
	// Per-chunk validation catches both under-delivery and compensating mismatches
	// (over-delivery on one chunk paired with under-delivery on another), which would
	// produce the correct aggregate count while silently corrupting the input→embedding mapping.
	var allEmbeddings [][]float64
	var totalUsage provider.Usage
	for i, cr := range results {
		if cr.err != nil {
			return nil, cr.err
		}
		if len(cr.result.Embeddings) != len(chunks[i]) {
			return nil, fmt.Errorf("goai: embedding count mismatch in chunk %d: got %d, expected %d", i, len(cr.result.Embeddings), len(chunks[i]))
		}
		allEmbeddings = append(allEmbeddings, cr.result.Embeddings...)
		totalUsage = addUsage(totalUsage, cr.result.Usage)
	}
	// Note: no aggregate check needed here ;  per-chunk validation guarantees
	// len(allEmbeddings) == sum(len(chunks[i])) == len(values) by induction.

	// At this point all cr.err == nil: the loop above returns on the first error,
	// so mergeProviderMetadata only receives successfully-completed chunk results.
	providerMeta := mergeProviderMetadata(results)

	return &EmbedManyResult{
		Embeddings:       allEmbeddings,
		Usage:            totalUsage,
		ProviderMetadata: providerMeta,
	}, nil
}

type embedChunkResult struct {
	result *provider.EmbedResult
	err    error
}

// mergeProviderMetadata merges ProviderMetadata from all chunk results.
// When multiple chunks set the same namespace key, later chunks (by slice index) win ; 
// this is intentional last-write-wins semantics, not accumulation.
// The returned outer map and each namespace's inner map are newly allocated.
// Values stored within the inner maps are not deep-copied: if a value is a reference
// type (map, slice, pointer), it shares underlying data with the original provider result.
func mergeProviderMetadata(results []embedChunkResult) map[string]map[string]any {
	var merged map[string]map[string]any
	for _, cr := range results {
		if cr.result == nil || cr.result.ProviderMetadata == nil {
			continue
		}
		if merged == nil {
			merged = make(map[string]map[string]any, len(cr.result.ProviderMetadata))
		}
		for ns, kv := range cr.result.ProviderMetadata {
			if merged[ns] == nil {
				merged[ns] = make(map[string]any, len(kv))
			}
			for k, v := range kv {
				merged[ns][k] = v
			}
		}
	}
	return merged
}
