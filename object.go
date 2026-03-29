package goai

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// ObjectResult is the final result of a structured output generation.
type ObjectResult[T any] struct {
	// Object is the parsed structured output.
	Object T

	// Usage tracks token consumption.
	Usage provider.Usage

	// FinishReason indicates why generation stopped.
	FinishReason provider.FinishReason

	// Response contains provider metadata (ID, Model).
	Response provider.ResponseMetadata

	// ProviderMetadata contains provider-specific response data.
	ProviderMetadata map[string]map[string]any
}

// ObjectStream is a streaming structured output response.
type ObjectStream[T any] struct {
	ctx           context.Context
	source        <-chan provider.StreamChunk
	consumeOnce   sync.Once
	doneCh        chan struct{}
	timeoutCancel context.CancelFunc

	// Channel returned by the first PartialObjectStream() call.
	partialCh <-chan *T

	// Hook support.
	onResponse func(ResponseInfo)
	startTime  time.Time

	// Accumulated state.
	text             strings.Builder
	finishReason     provider.FinishReason
	usage            provider.Usage
	response         provider.ResponseMetadata
	providerMetadata map[string]map[string]any
	finalObject      *T
	parseErr         error
	streamErr        error
}

func newObjectStream[T any](ctx context.Context, source <-chan provider.StreamChunk) *ObjectStream[T] {
	return &ObjectStream[T]{
		ctx:    ctx,
		source: source,
		doneCh: make(chan struct{}),
	}
}

// PartialObjectStream returns a channel that emits partial objects as JSON accumulates.
// Each emitted value has progressively more fields populated.
// Mutually exclusive with Result() -- only call one consumption method first.
func (os *ObjectStream[T]) PartialObjectStream() <-chan *T {
	ch := make(chan *T, 64)
	os.consumeOnce.Do(func() {
		os.partialCh = ch
		go os.consume(ch)
	})
	if os.partialCh != nil {
		return os.partialCh
	}
	// Called after Result() consumed the source -- return closed channel.
	close(ch)
	return ch
}

// Result blocks until the stream completes and returns the final validated object.
// Returns an error if JSON parsing of the accumulated text fails.
func (os *ObjectStream[T]) Result() (*ObjectResult[T], error) {
	os.consumeOnce.Do(func() {
		go os.consume(nil)
	})
	<-os.doneCh

	if os.streamErr != nil {
		return nil, os.streamErr
	}

	if os.parseErr != nil {
		return nil, fmt.Errorf("parsing structured output: %w (raw: %s)", os.parseErr, truncate(os.text.String(), 200))
	}

	if os.finalObject == nil {
		return &ObjectResult[T]{
			Usage:            os.usage,
			FinishReason:     os.finishReason,
			Response:         os.response,
			ProviderMetadata: os.providerMetadata,
		}, nil
	}
	return &ObjectResult[T]{
		Object:           *os.finalObject,
		Usage:            os.usage,
		FinishReason:     os.finishReason,
		Response:         os.response,
		ProviderMetadata: os.providerMetadata,
	}, nil
}

// Err returns the first stream error encountered, or nil.
// Must be called after the stream is fully consumed (after Result(),
// or after the PartialObjectStream() channel is drained).
// Follows the bufio.Scanner.Err() pattern.
func (os *ObjectStream[T]) Err() error {
	<-os.doneCh
	return os.streamErr
}

func (os *ObjectStream[T]) consume(partialOut chan<- *T) {
	defer close(os.doneCh)
	if os.timeoutCancel != nil {
		defer os.timeoutCancel()
	}
	if partialOut != nil {
		defer close(partialOut)
	}

	// Call OnResponse hook when consume finishes (after all chunks processed).
	if os.onResponse != nil {
		defer func() {
			os.onResponse(ResponseInfo{
				Latency:      time.Since(os.startTime),
				Usage:        os.usage,
				FinishReason: os.finishReason,
				Error:        os.streamErr,
			})
		}()
	}

	for chunk := range os.source {
		switch chunk.Type {
		case provider.ChunkText:
			os.text.WriteString(chunk.Text)

			// Try to parse partial JSON.
			if partialOut != nil {
				if partial, err := parsePartialJSON[T](os.text.String()); err == nil {
					select {
					case partialOut <- partial:
					case <-os.ctx.Done():
						return
					}
				}
			}

		case provider.ChunkFinish:
			os.finishReason = chunk.FinishReason
			os.usage = chunk.Usage
			os.response = chunk.Response
			// Extract provider metadata embedded in the finish chunk.
			// openaicompat encodes it as Metadata["providerMetadata"] = map[string]map[string]any.
			// The type assertion safely returns false for any other type (no panic).
			if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
				os.providerMetadata = pm
			}
			// Copy remaining flat metadata keys into Response.ProviderMetadata.
			// Providers use this for per-response data: Anthropic ("iterations",
			// "contextManagement", "container"), Bedrock ("cacheWriteInputTokens").
			// Skip "providerMetadata" (handled above) and "sources" (ObjectResult has no Sources field).
			for k, v := range chunk.Metadata {
				if k == "providerMetadata" || k == "sources" {
					continue
				}
				if os.response.ProviderMetadata == nil {
					os.response.ProviderMetadata = map[string]any{}
				}
				os.response.ProviderMetadata[k] = v
			}
		case provider.ChunkError:
			if os.streamErr == nil {
				os.streamErr = chunk.Error
			}
		}
		if partialOut == nil {
			if os.ctx.Err() != nil {
				return
			}
		}
	}

	// Parse final result.
	text := os.text.String()
	if text != "" {
		var obj T
		if err := json.Unmarshal([]byte(text), &obj); err != nil {
			os.parseErr = err
		} else {
			os.finalObject = &obj
		}
	}
}

// GenerateObject performs a non-streaming structured output generation.
// The schema is auto-generated from T, or can be overridden with WithExplicitSchema.
func GenerateObject[T any](ctx context.Context, model provider.LanguageModel, opts ...Option) (*ObjectResult[T], error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.Timeout)
		defer cancel()
	}

	// Resolve schema.
	schema := o.ExplicitSchema
	if schema == nil {
		schema = SchemaFrom[T]()
	}

	schemaName := cmp.Or(o.SchemaName, "response")

	params := buildParams(o)
	params.ResponseFormat = &provider.ResponseFormat{
		Name:   schemaName,
		Schema: schema,
	}

	if o.OnRequest != nil {
		o.OnRequest(RequestInfo{
			Model:        model.ModelID(),
			MessageCount: len(params.Messages),
			ToolCount:    len(params.Tools),
			Timestamp:    time.Now(),
		})
	}

	start := time.Now()
	result, err := withRetry(ctx, o.MaxRetries, func() (*provider.GenerateResult, error) {
		return model.DoGenerate(ctx, params)
	})

	if o.OnResponse != nil {
		info := ResponseInfo{Latency: time.Since(start), Error: err}
		if err == nil {
			info.Usage = result.Usage
			info.FinishReason = result.FinishReason
		}
		var apiErr *APIError
		if errors.As(err, &apiErr) {
			info.StatusCode = apiErr.StatusCode
		}
		o.OnResponse(info)
	}

	if err != nil {
		return nil, err
	}

	// Parse the JSON response into T.
	var obj T
	text := result.Text
	if text == "" {
		return nil, fmt.Errorf("goai: empty response from model")
	}
	if err := json.Unmarshal([]byte(text), &obj); err != nil {
		return nil, fmt.Errorf("parsing structured output: %w (raw: %s)", err, truncate(text, 200))
	}

	return &ObjectResult[T]{
		Object:           obj,
		Usage:            result.Usage,
		FinishReason:     result.FinishReason,
		Response:         result.Response,
		ProviderMetadata: result.ProviderMetadata,
	}, nil
}

// StreamObject performs a streaming structured output generation.
// Returns an ObjectStream that emits progressively populated partial objects.
func StreamObject[T any](ctx context.Context, model provider.LanguageModel, opts ...Option) (*ObjectStream[T], error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	var timeoutCancel context.CancelFunc
	if o.Timeout > 0 {
		ctx, timeoutCancel = context.WithTimeout(ctx, o.Timeout)
	}

	// Resolve schema.
	schema := o.ExplicitSchema
	if schema == nil {
		schema = SchemaFrom[T]()
	}

	schemaName := cmp.Or(o.SchemaName, "response")

	params := buildParams(o)
	params.ResponseFormat = &provider.ResponseFormat{
		Name:   schemaName,
		Schema: schema,
	}

	if o.OnRequest != nil {
		o.OnRequest(RequestInfo{
			Model:        model.ModelID(),
			MessageCount: len(params.Messages),
			ToolCount:    len(params.Tools),
			Timestamp:    time.Now(),
		})
	}

	start := time.Now()
	result, err := withRetry(ctx, o.MaxRetries, func() (*provider.StreamResult, error) {
		return model.DoStream(ctx, params)
	})
	if err != nil {
		if timeoutCancel != nil {
			timeoutCancel()
		}
		if o.OnResponse != nil {
			info := ResponseInfo{Latency: time.Since(start), Error: err}
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				info.StatusCode = apiErr.StatusCode
			}
			o.OnResponse(info)
		}
		return nil, err
	}

	os := newObjectStream[T](ctx, result.Stream)
	os.timeoutCancel = timeoutCancel
	os.onResponse = o.OnResponse
	os.startTime = start
	return os, nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	// Truncate at rune boundary to avoid splitting multi-byte UTF-8.
	truncated := []rune(s)
	if len(truncated) <= max {
		return s
	}
	return string(truncated[:max]) + "..."
}
