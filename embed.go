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

	return &EmbedResult{
		Embedding:        result.Embeddings[0],
		Usage:            result.Usage,
		ProviderMetadata: result.ProviderMetadata,
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
			ProviderMetadata: result.ProviderMetadata,
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

	type chunkResult struct {
		result *provider.EmbedResult
		err    error
	}

	results := make([]chunkResult, len(chunks))
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
				results[i] = chunkResult{err: ctx.Err()}
				return
			}

			r, err := withRetry(ctx, o.MaxRetries, func() (*provider.EmbedResult, error) {
				return model.DoEmbed(ctx, chunk, embedParams)
			})
			results[i] = chunkResult{result: r, err: err}
		}(i, chunk)
	}
	wg.Wait()

	// Combine results in order.
	var allEmbeddings [][]float64
	var totalUsage provider.Usage
	for _, cr := range results {
		if cr.err != nil {
			return nil, cr.err
		}
		allEmbeddings = append(allEmbeddings, cr.result.Embeddings...)
		totalUsage = addUsage(totalUsage, cr.result.Usage)
	}

	if len(allEmbeddings) != len(values) {
		return nil, fmt.Errorf("goai: embedding count mismatch: got %d, expected %d", len(allEmbeddings), len(values))
	}

	var providerMeta map[string]map[string]any
	if results[0].result != nil {
		providerMeta = results[0].result.ProviderMetadata
	}

	return &EmbedManyResult{
		Embeddings:       allEmbeddings,
		Usage:            totalUsage,
		ProviderMetadata: providerMeta,
	}, nil
}
