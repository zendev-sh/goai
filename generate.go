package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// ErrUnknownTool is returned when a tool call references a tool not in the tool map.
var ErrUnknownTool = errors.New("goai: unknown tool")

// TextResult is the final result of a text generation call.
type TextResult struct {
	// Text is the accumulated generated text across all steps.
	// For StreamText, this includes reasoning tokens (ChunkReasoning) for backward
	// compatibility. Use Steps[n].Text for text-only content excluding reasoning.
	Text string

	// ToolCalls requested by the model in the final step.
	ToolCalls []provider.ToolCall

	// Steps contains results from each generation step (for multi-step tool loops).
	Steps []StepResult

	// TotalUsage is the aggregated token usage across all steps.
	TotalUsage provider.Usage

	// FinishReason indicates why generation stopped.
	FinishReason provider.FinishReason

	// Response contains provider metadata from the last step (ID, Model).
	Response provider.ResponseMetadata

	// ProviderMetadata contains provider-specific response data from the last step
	// (e.g. logprobs, prediction tokens).
	ProviderMetadata map[string]map[string]any

	// Sources contains citations/references extracted from the response.
	Sources []provider.Source
}

// StepResult is the result of a single generation step in a tool loop.
type StepResult struct {
	// Number is the 1-based step index.
	Number int

	// Text generated in this step (excludes reasoning tokens).
	// For StreamText, reasoning is included in TextResult.Text but excluded here.
	Text string

	// ToolCalls requested in this step.
	ToolCalls []provider.ToolCall

	// FinishReason for this step.
	FinishReason provider.FinishReason

	// Usage for this step.
	Usage provider.Usage

	// Response contains provider metadata for this step (ID, Model).
	Response provider.ResponseMetadata

	// ProviderMetadata contains provider-specific response data for this step.
	ProviderMetadata map[string]map[string]any

	// Sources contains citations/references from this step.
	Sources []provider.Source
}

// TextStream is a streaming text generation response.
//
// Callers must consume the stream (via Stream, TextStream, or Result) or cancel
// the context. Discarding a TextStream without consuming leaks goroutines.
//
// It provides three consumption modes (Stream, TextStream, Result).
// Stream() and TextStream() are mutually exclusive -- only call one.
// Result() can always be called, including after Stream() or TextStream(),
// to get the accumulated final result.
type TextStream struct {
	ctx           context.Context
	source        <-chan provider.StreamChunk
	consumeOnce   sync.Once
	doneCh        chan struct{}
	timeoutCancel context.CancelFunc

	// Channels returned by the first Stream()/TextStream() call.
	// Subsequent calls return the same channel instead of a dead one.
	rawCh  <-chan provider.StreamChunk
	textCh <-chan string

	// Hook support.
	onResponse   func(ResponseInfo)
	onStepFinish func(StepResult)
	startTime    time.Time

	// Accumulated state (written by consume goroutine, read after doneCh closes).
	text             strings.Builder
	toolCalls        []provider.ToolCall
	sources          []provider.Source
	finishReason     provider.FinishReason
	usage            provider.Usage
	response         provider.ResponseMetadata
	providerMetadata map[string]map[string]any
	streamErr        error

	// Multi-step accumulation (written by consume goroutine).
	steps         []StepResult
	currentStep   int
	stepText      strings.Builder
	stepToolCalls []provider.ToolCall
	stepSources   []provider.Source
}

func newTextStream(ctx context.Context, source <-chan provider.StreamChunk) *TextStream {
	return &TextStream{
		ctx:    ctx,
		source: source,
		doneCh: make(chan struct{}),
	}
}

// Stream returns a channel that emits raw StreamChunks from the provider.
// Mutually exclusive with TextStream() -- only call one streaming method.
func (ts *TextStream) Stream() <-chan provider.StreamChunk {
	ch := make(chan provider.StreamChunk, 64)
	ts.consumeOnce.Do(func() {
		ts.rawCh = ch
		go ts.consume(ch, nil)
	})
	if ts.rawCh != nil {
		return ts.rawCh
	}
	// Called after TextStream() consumed the source -- return closed channel.
	close(ch)
	return ch
}

// TextStream returns the underlying channel of text chunks.
// Note: this method has the same name as the containing type (TextStream);
// call it as stream.TextStream() to receive the channel.
// Mutually exclusive with Stream() -- only call one streaming method.
func (ts *TextStream) TextStream() <-chan string {
	ch := make(chan string, 64)
	ts.consumeOnce.Do(func() {
		ts.textCh = ch
		go ts.consume(nil, ch)
	})
	if ts.textCh != nil {
		return ts.textCh
	}
	// Called after Stream() consumed the source -- return closed channel.
	close(ch)
	return ch
}

// Result blocks until the stream completes and returns the accumulated result.
// Check Err() after Result() to detect stream errors - Result does not surface
// errors directly (use Err or check result.Steps for partial data).
// Can be called after Stream() or TextStream() to get accumulated data.
// Note: unlike ObjectStream.Result(), this method does not return an error.
// Call Err() after Result() to check for stream errors.
func (ts *TextStream) Result() *TextResult {
	ts.consumeOnce.Do(func() {
		go ts.consume(nil, nil)
	})
	<-ts.doneCh
	return ts.buildResult()
}

// Err returns the first stream error encountered, or nil.
// Must be called after the stream is fully consumed (after Result(),
// or after the Stream()/TextStream() channel is drained).
// Follows the bufio.Scanner.Err() pattern.
func (ts *TextStream) Err() error {
	<-ts.doneCh
	return ts.streamErr
}

func (ts *TextStream) consume(rawOut chan<- provider.StreamChunk, textOut chan<- string) {
	defer close(ts.doneCh)
	if ts.timeoutCancel != nil {
		defer ts.timeoutCancel()
	}
	if rawOut != nil {
		defer close(rawOut)
	}
	if textOut != nil {
		defer close(textOut)
	}

	// Call OnStepFinish hook when consume finishes (single-step streaming only).
	// For multi-step streaming (streamWithToolLoop), OnStepFinish fires inline per
	// step and ts.onStepFinish is nil, so this block is skipped.
	// Deferred BEFORE OnResponse so it runs AFTER it (LIFO order), matching
	// GenerateText's OnResponse → OnStepFinish sequence.
	if ts.onStepFinish != nil {
		defer func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
				}
			}()
			ts.onStepFinish(StepResult{
				Number:           1,
				Text:             ts.stepText.String(),
				ToolCalls:        ts.toolCalls,
				FinishReason:     ts.finishReason,
				Usage:            ts.usage,
				Response:         ts.response,
				Sources:          ts.sources,
				ProviderMetadata: ts.providerMetadata,
			})
		}()
	}

	// Call OnResponse hook when consume finishes (after all chunks processed).
	if ts.onResponse != nil {
		defer func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
				}
			}()
			info := ResponseInfo{
				Latency:      time.Since(ts.startTime),
				Usage:        ts.usage,
				FinishReason: ts.finishReason,
				Error:        ts.streamErr,
			}
			var apiErr *APIError
			if errors.As(ts.streamErr, &apiErr) {
				info.StatusCode = apiErr.StatusCode
			}
			ts.onResponse(info)
		}()
	}

	for chunk := range ts.source {
		switch chunk.Type {
		case provider.ChunkText:
			ts.text.WriteString(chunk.Text)     // global accumulator (existing, includes reasoning)
			ts.stepText.WriteString(chunk.Text) // per-step text-only accumulator (new)
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				ts.sources = append(ts.sources, s)         // global (existing)
				ts.stepSources = append(ts.stepSources, s) // per-step (new)
			}

		case provider.ChunkReasoning:
			ts.text.WriteString(chunk.Text) // global accumulator (existing, includes reasoning)
			// NOTE: reasoning is NOT written to ts.stepText. This matches drainStep behavior
			// where text and reasoning are separated. StepResult.Text contains text-only,
			// consistent between OnStepFinish hook (from drainStep) and Result().Steps (from consume).
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				ts.sources = append(ts.sources, s)         // global (preserve existing behavior)
				ts.stepSources = append(ts.stepSources, s) // per-step
			}

		case provider.ChunkToolCall:
			tc := provider.ToolCall{ID: chunk.ToolCallID, Name: chunk.ToolName, Input: json.RawMessage(chunk.ToolInput)}
			ts.toolCalls = append(ts.toolCalls, tc)
			ts.stepToolCalls = append(ts.stepToolCalls, tc)

		case provider.ChunkStepFinish:
			// GoAI-emitted step boundaries: build per-step StepResult.
			if stepSource, _ := chunk.Metadata["stepSource"].(string); stepSource == "goai" {
				ts.currentStep++
				// Response is set directly on the chunk by the step loop.
				// ProviderMetadata is embedded in Metadata (no dedicated StreamChunk field).
				var stepProviderMeta map[string]map[string]any
				if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
					stepProviderMeta = pm
				}
				ts.steps = append(ts.steps, StepResult{
					Number:           ts.currentStep,
					Text:             ts.stepText.String(),
					ToolCalls:        ts.stepToolCalls,
					FinishReason:     chunk.FinishReason,
					Usage:            chunk.Usage,
					Sources:          ts.stepSources,
					Response:         chunk.Response,
					ProviderMetadata: stepProviderMeta,
				})
				// Accumulate per-step usage and finishReason for resilience:
				// if stream terminates before ChunkFinish (e.g., context cancel between
				// ChunkStepFinish and ChunkFinish sends), ts.usage still reflects completed steps.
				// ChunkFinish will overwrite with the authoritative totalUsage when it arrives.
				ts.usage = addUsage(ts.usage, chunk.Usage)
				ts.finishReason = chunk.FinishReason
				// Update ts.response for the overall TextResult (last step wins).
				ts.response = chunk.Response
				ts.providerMetadata = stepProviderMeta
				// Reset per-step accumulators.
				ts.stepText.Reset()
				ts.stepToolCalls = nil
				ts.stepSources = nil
			} else {
				// Provider-internal step boundary (e.g., Anthropic extended thinking).
				// Preserve existing behavior: extract response, metadata, sources.
				// This ensures single-step streaming continues to work correctly.
				// NOTE: Do NOT accumulate usage here with addUsage. Provider-internal
				// ChunkStepFinish usage is already included in the provider's ChunkFinish
				// (which uses direct assignment), and the GoAI ChunkStepFinish (which uses
				// addUsage via drainStep). Accumulating here would double-count.
				ts.finishReason = chunk.FinishReason
				ts.response = chunk.Response
				if sources, ok := chunk.Metadata["sources"].([]provider.Source); ok {
					ts.sources = append(ts.sources, sources...)
					ts.stepSources = append(ts.stepSources, sources...)
				}
				if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
					ts.providerMetadata = pm
				}
				// Copy flat metadata keys to Response.ProviderMetadata (existing behavior).
				for k, v := range chunk.Metadata {
					if k == "providerMetadata" || k == "sources" {
						continue
					}
					if ts.response.ProviderMetadata == nil {
						ts.response.ProviderMetadata = map[string]any{}
					}
					ts.response.ProviderMetadata[k] = v
				}
			}

		case provider.ChunkFinish:
			// Direct assignment (not addUsage): ChunkFinish carries authoritative total usage.
			ts.usage = chunk.Usage
			ts.finishReason = chunk.FinishReason
			// Preserve existing single-step behavior: extract response, metadata, sources
			// from the provider's ChunkFinish. For multi-step, the goai-emitted ChunkFinish
			// does not carry these (they are embedded in ChunkStepFinish metadata instead).
			if chunk.Response.ID != "" || chunk.Response.Model != "" {
				ts.response = chunk.Response
			}
			if sources, ok := chunk.Metadata["sources"].([]provider.Source); ok {
				ts.sources = append(ts.sources, sources...)
			}
			if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
				ts.providerMetadata = pm
			}
			// Copy flat metadata keys to Response.ProviderMetadata (existing behavior).
			for k, v := range chunk.Metadata {
				if k == "providerMetadata" || k == "sources" {
					continue
				}
				if ts.response.ProviderMetadata == nil {
					ts.response.ProviderMetadata = map[string]any{}
				}
				ts.response.ProviderMetadata[k] = v
			}

		case provider.ChunkError:
			if ts.streamErr == nil {
				ts.streamErr = chunk.Error
			}
		}

		if rawOut != nil {
			select {
			case rawOut <- chunk:
			case <-ts.ctx.Done():
				ts.streamErr = ts.ctx.Err()
				return
			}
		}
		if textOut != nil && chunk.Type == provider.ChunkText {
			select {
			case textOut <- chunk.Text:
			case <-ts.ctx.Done():
				ts.streamErr = ts.ctx.Err()
				return
			}
		}
		if rawOut == nil && textOut == nil {
			if ts.ctx.Err() != nil {
				ts.streamErr = ts.ctx.Err()
				return
			}
		}
	}
}

func (ts *TextStream) buildResult() *TextResult {
	text := ts.text.String() // full accumulated text across all steps
	result := &TextResult{
		Text:             text,
		ToolCalls:        ts.toolCalls,
		FinishReason:     ts.finishReason,
		TotalUsage:       ts.usage,
		Response:         ts.response,
		Sources:          ts.sources,
		ProviderMetadata: ts.providerMetadata,
	}
	if len(ts.steps) > 0 {
		result.Steps = ts.steps
		// Match GenerateText: ToolCalls is the LAST step's tool calls, not all steps'.
		result.ToolCalls = ts.steps[len(ts.steps)-1].ToolCalls
	} else if text != "" || len(ts.toolCalls) > 0 || ts.finishReason != "" {
		// Single-step fallback (no multi-step ChunkStepFinish received, but data exists).
		result.Steps = []StepResult{{
			Number:           1,
			Text:             ts.stepText.String(),
			ToolCalls:        ts.toolCalls,
			FinishReason:     ts.finishReason,
			Usage:            ts.usage,
			Response:         ts.response,
			Sources:          ts.sources,
			ProviderMetadata: ts.providerMetadata,
		}}
	}
	// No data: Steps is nil.
	return result
}

// buildParams converts options to provider.GenerateParams.
func buildParams(opts options) provider.GenerateParams {
	var tools []provider.ToolDefinition
	for _, t := range opts.Tools {
		tools = append(tools, provider.ToolDefinition{
			Name:                   t.Name,
			Description:            t.Description,
			InputSchema:            t.InputSchema,
			ProviderDefinedType:    t.ProviderDefinedType,
			ProviderDefinedOptions: t.ProviderDefinedOptions,
		})
	}

	msgs := opts.Messages
	if opts.Prompt != "" {
		msgs = append([]provider.Message{UserMessage(opts.Prompt)}, msgs...)
	} else {
		// Always copy so tool-loop appends never mutate the caller's slice.
		msgs = slices.Clone(msgs)
	}

	if opts.PromptCaching {
		msgs = applyCaching(msgs)
	}

	return provider.GenerateParams{
		Messages:         msgs,
		System:           opts.System,
		Tools:            tools,
		MaxOutputTokens:  opts.MaxOutputTokens,
		Temperature:      opts.Temperature,
		TopP:             opts.TopP,
		TopK:             opts.TopK,
		FrequencyPenalty: opts.FrequencyPenalty,
		PresencePenalty:  opts.PresencePenalty,
		Seed:             opts.Seed,
		StopSequences:    slices.Clone(opts.StopSequences),
		Headers:          maps.Clone(opts.Headers),
		ProviderOptions:  maps.Clone(opts.ProviderOptions),
		PromptCaching:    opts.PromptCaching,
		ToolChoice:       opts.ToolChoice,
	}
}

func streamWithToolLoop(ctx context.Context, model provider.LanguageModel, o options, toolMap map[string]Tool) (*TextStream, error) {
	params := buildParams(o)

	var timeoutCancel context.CancelFunc
	if o.Timeout > 0 {
		ctx, timeoutCancel = context.WithTimeout(ctx, o.Timeout)
	}

	// --- Step 1 DoStream: synchronous (preserves (nil, error) contract) ---
	// This ensures StreamText ALWAYS returns (nil, error) when the first DoStream
	// fails, regardless of MaxSteps. Eliminates the split error contract.
	if o.OnRequest != nil {
		o.OnRequest(RequestInfo{
			Model:        model.ModelID(),
			MessageCount: len(params.Messages),
			ToolCount:    len(params.Tools),
			Timestamp:    time.Now(),
			Messages:     requestMessages(params.System, params.Messages),
		})
	}

	start := time.Now()
	firstResult, err := withRetry(ctx, o.MaxRetries, func() (*provider.StreamResult, error) {
		return model.DoStream(ctx, params)
	})
	if err != nil {
		if timeoutCancel != nil {
			timeoutCancel()
		}
		// OnRequest/OnResponse: not recover-wrapped (caller's goroutine).
		// OnStepFinish: always recover-wrapped (prevents losing accumulated results).
		// Inside goroutines: all hooks recover-wrapped.
		if o.OnResponse != nil {
			info := ResponseInfo{Latency: time.Since(start), Error: err}
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				info.StatusCode = apiErr.StatusCode
			}
			o.OnResponse(info)
		}
		return nil, err // SAME error contract as single-step StreamText
	}

	// Step 1 succeeded. Goroutine-local copy of start time avoids closure capture.
	step1Start := start
	out := make(chan provider.StreamChunk, 64)

	go func() {
		defer close(out)
		if timeoutCancel != nil {
			defer timeoutCancel()
		}

		var totalUsage provider.Usage
		var lastFinishReason provider.FinishReason
		var lastResponse provider.ResponseMetadata
		firstStep := true       // true only for step 1 (already have firstResult)
		stepStart := step1Start // goroutine-local start time per step

		for step := 1; step <= o.MaxSteps; step++ {
			var result *provider.StreamResult

			if firstStep {
				// Step 1: use the already-obtained firstResult.
				result = firstResult
				firstStep = false
			} else {
				// Steps 2+: DoStream inside goroutine.
				if o.OnRequest != nil {
					func() {
						defer func() {
							if r := recover(); r != nil {
								fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
							}
						}()
						o.OnRequest(RequestInfo{
							Model:        model.ModelID(),
							MessageCount: len(params.Messages),
							ToolCount:    len(params.Tools),
							Timestamp:    time.Now(),
							Messages:     requestMessages(params.System, params.Messages),
						})
					}()
				}

				stepStart = time.Now()
				var err error
				result, err = withRetry(ctx, o.MaxRetries, func() (*provider.StreamResult, error) {
					return model.DoStream(ctx, params)
				})
				if err != nil {
					// Fire OnResponse on error (recover-wrapped).
					if o.OnResponse != nil {
						func() {
							defer func() {
								if r := recover(); r != nil {
									fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
								}
							}()
							info := ResponseInfo{Latency: time.Since(stepStart), Error: err}
							var apiErr *APIError
							if errors.As(err, &apiErr) {
								info.StatusCode = apiErr.StatusCode
							}
							o.OnResponse(info)
						}()
					}
					provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: err})
					provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: lastFinishReason, Usage: totalUsage})
					return
				}
			}

			ds := drainStep(ctx, result.Stream, out)
			if ds.err != nil {
				// Context cancelled during drain.
				provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: ds.err})
				provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkFinish, Usage: totalUsage})
				return
			}

			// Guard: skip empty step (provider closed channel without sending
			// any meaningful chunks, e.g., after a ChunkError). Prevents emitting
			// a phantom empty StepResult and ChunkStepFinish.
			if ds.text == "" && len(ds.toolCalls) == 0 && ds.finishReason == "" {
				break
			}

			// OnResponse: Error is NOT set (call succeeded). Mid-stream errors use stream.Err().
			if o.OnResponse != nil {
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					o.OnResponse(ResponseInfo{
						Latency:      time.Since(stepStart),
						Usage:        ds.usage,
						FinishReason: ds.finishReason,
					})
				}()
			}

			// --- Build StepResult, fire OnStepFinish ---
			stepResult := StepResult{
				Number:           step,
				Text:             ds.text,
				ToolCalls:        ds.toolCalls,
				FinishReason:     ds.finishReason,
				Usage:            ds.usage,
				Sources:          ds.sources,
				Response:         ds.response,
				ProviderMetadata: ds.providerMetadata,
			}
			totalUsage = addUsage(totalUsage, ds.usage)
			lastFinishReason = ds.finishReason
			lastResponse = ds.response

			// OnStepFinish (recover-wrapped).
			if o.OnStepFinish != nil {
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					o.OnStepFinish(stepResult)
				}()
			}

			// --- Emit ChunkStepFinish ---
			// Set Response directly on the chunk (StreamChunk.Response is a plain struct
			// field with no restriction on who sets it). ProviderMetadata goes in Metadata
			// since there is no dedicated field for it on StreamChunk.
			provider.TrySend(ctx, out, provider.StreamChunk{
				Type:         provider.ChunkStepFinish,
				FinishReason: ds.finishReason,
				Usage:        ds.usage,
				Response:     ds.response,
				Metadata: map[string]any{
					"stepSource":       "goai",
					"providerMetadata": ds.providerMetadata,
				},
			})

			// --- Exit conditions (same as GenerateText, generate.go:464) ---
			if ds.finishReason != provider.FinishToolCalls || len(ds.toolCalls) == 0 || len(toolMap) == 0 {
				break
			}

			// --- Execute tools in parallel ---
			toolMsgs := executeToolsParallel(ctx, ds.toolCalls, toolMap, step, o.OnToolCallStart, o.OnToolCall)

			// --- Append messages for next step ---
			params.Messages = appendToolRoundTrip(params.Messages, ds.text, ds.toolCalls, toolMsgs)
			// Clear ToolChoice so model can freely respond on subsequent steps (matches generate.go:471).
			// Set on every iteration for simplicity; idempotent after step 1.
			params.ToolChoice = ""
		}

		// Emit final ChunkFinish with total usage and last step Response metadata.
		provider.TrySend(ctx, out, provider.StreamChunk{
			Type:         provider.ChunkFinish,
			FinishReason: lastFinishReason,
			Usage:        totalUsage,
			Response:     lastResponse,
		})
	}()

	ts := newTextStream(ctx, out)
	// OnResponse handled per-step inside the goroutine (ts.onResponse not set).
	return ts, nil
}

// StreamText performs a streaming text generation.
// When MaxSteps > 1 and executable tools are provided, StreamText runs an automatic
// tool loop. The initial DoStream failure still returns (nil, error). Subsequent step
// errors flow through the stream as ChunkError chunks; check stream.Err() after consuming.
func StreamText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextStream, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.Prompt == "" && len(o.Messages) == 0 {
		return nil, errors.New("goai: prompt or messages must not be empty")
	}

	toolMap := buildToolMap(o.Tools)

	if o.MaxSteps > 1 && len(toolMap) > 0 {
		return streamWithToolLoop(ctx, model, o, toolMap)
	}

	var timeoutCancel context.CancelFunc
	if o.Timeout > 0 {
		ctx, timeoutCancel = context.WithTimeout(ctx, o.Timeout)
	}

	params := buildParams(o)

	if o.OnRequest != nil {
		o.OnRequest(RequestInfo{
			Model:        model.ModelID(),
			MessageCount: len(params.Messages),
			ToolCount:    len(params.Tools),
			Timestamp:    time.Now(),
			Messages:     requestMessages(params.System, params.Messages),
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

	ts := newTextStream(ctx, result.Stream)
	ts.timeoutCancel = timeoutCancel
	ts.onResponse = o.OnResponse
	ts.onStepFinish = o.OnStepFinish
	ts.startTime = start
	return ts, nil
}

// GenerateText performs a non-streaming text generation.
// When tools with Execute functions are provided and MaxSteps > 1,
// it automatically runs a tool loop: generate → execute tools → re-generate.
func GenerateText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextResult, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.Prompt == "" && len(o.Messages) == 0 {
		return nil, errors.New("goai: prompt or messages must not be empty")
	}

	if o.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.Timeout)
		defer cancel()
	}

	params := buildParams(o)

	// Build tool lookup for auto loop.
	toolMap := buildToolMap(o.Tools)

	var steps []StepResult
	var totalUsage provider.Usage

	for step := 1; step <= o.MaxSteps; step++ {
		if o.OnRequest != nil {
			o.OnRequest(RequestInfo{
				Model:        model.ModelID(),
				MessageCount: len(params.Messages),
				ToolCount:    len(params.Tools),
				Timestamp:    time.Now(),
				Messages:     requestMessages(params.System, params.Messages),
			})
		}

		start := time.Now()
		result, err := withRetry(ctx, o.MaxRetries, func() (*provider.GenerateResult, error) {
			return model.DoGenerate(ctx, params)
		})

		if o.OnResponse != nil {
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
					}
				}()
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
			}()
		}

		if err != nil {
			return nil, err
		}

		stepResult := StepResult{
			Number:           step,
			Text:             result.Text,
			ToolCalls:        result.ToolCalls,
			FinishReason:     result.FinishReason,
			Usage:            result.Usage,
			Response:         result.Response,
			ProviderMetadata: result.ProviderMetadata,
			Sources:          result.Sources,
		}
		steps = append(steps, stepResult)
		totalUsage = addUsage(totalUsage, result.Usage)

		if o.OnStepFinish != nil {
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
					}
				}()
				o.OnStepFinish(stepResult)
			}()
		}

		// If no tools have Execute functions, skip the tool loop regardless of MaxSteps.
		// This allows callers to provide tool definitions for the model's awareness
		// without requiring executable tools.
		// No empty-step guard needed (unlike streaming): DoGenerate returns content or error.
		if result.FinishReason != provider.FinishToolCalls || len(result.ToolCalls) == 0 || len(toolMap) == 0 {
			return buildTextResult(steps, totalUsage), nil
		}

		// Execute tools and build continuation messages.
		// Clear tool_choice after the first tool step so the model can freely
		// produce a text response on subsequent steps.
		params.ToolChoice = ""
		toolMessages := executeToolsParallel(ctx, result.ToolCalls, toolMap, step, o.OnToolCallStart, o.OnToolCall)

		// Append assistant message with tool calls + tool result messages.
		params.Messages = appendToolRoundTrip(params.Messages, result.Text, result.ToolCalls, toolMessages)
	}

	// MaxSteps reached.
	return buildTextResult(steps, totalUsage), nil
}

// requestMessages returns msgs with a system message prepended when system is non-empty.
// Always allocates a new slice so hooks cannot mutate the caller's message state.
func requestMessages(system string, msgs []provider.Message) []provider.Message {
	if system == "" {
		out := make([]provider.Message, len(msgs))
		copy(out, msgs)
		return out
	}
	out := make([]provider.Message, 0, len(msgs)+1)
	out = append(out, SystemMessage(system))
	return append(out, msgs...)
}

// buildToolMap creates a name→Tool lookup from the options.
func buildToolMap(tools []Tool) map[string]Tool {
	if len(tools) == 0 {
		return nil
	}
	m := make(map[string]Tool, len(tools))
	for _, t := range tools {
		if t.Execute != nil {
			if t.Name == "" {
				fmt.Fprintf(os.Stderr, "goai: tool with empty name skipped\n")
				continue
			}
			m[t.Name] = t
		}
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// toolOutput holds the result of a single tool execution (package-level type
// shared between executeToolsParallel and buildToolMessages).
type toolOutput struct {
	index  int
	result string
	err    error
}

func executeToolsParallel(
	ctx context.Context,
	calls []provider.ToolCall,
	toolMap map[string]Tool,
	step int,
	onToolCallStart func(ToolCallStartInfo),
	onToolCall func(ToolCallInfo),
) []provider.Message {

	results := make([]toolOutput, len(calls))
	var wg sync.WaitGroup

	for i, tc := range calls {
		tool, ok := toolMap[tc.Name]
		if !ok {
			// Unknown tool: fire OnToolCallStart + OnToolCall with ErrUnknownTool.
			// Each hook is independently recover-wrapped so a panic in OnToolCallStart
			// does not prevent OnToolCall from firing.
			//
			// Asymmetry with known-tool path (below): for known tools, OnToolCallStart
			// panic prevents Execute from running (the tool should not execute if the
			// pre-hook crashed). For unknown tools, Execute never runs anyway, so both
			// hooks fire independently for observability completeness.
			results[i] = toolOutput{index: i, err: ErrUnknownTool}
			if onToolCallStart != nil {
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					onToolCallStart(ToolCallStartInfo{ToolCallID: tc.ID, ToolName: tc.Name, Step: step, Input: tc.Input})
				}()
			}
			if onToolCall != nil {
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					onToolCall(ToolCallInfo{
						ToolCallID: tc.ID,
						ToolName:   tc.Name,
						Step:       step,
						Input:      tc.Input,
						Error:      ErrUnknownTool,
					})
				}()
			}
			continue
		}

		wg.Add(1)
		go func(i int, tc provider.ToolCall, tool Tool) {
			defer wg.Done()
			var hookFired bool // true after OnToolCallStart completes (before Execute)
			var executed bool  // tracks whether Execute ran (for panic recovery)
			defer func() {
				if r := recover(); r != nil {
					if !executed {
						panicStr := fmt.Sprintf("%v", r)
						runes := []rune(panicStr)
						if len(runes) > 500 {
							panicStr = string(runes[:500]) + "..."
						}
						// Distinguish OnToolCallStart panic from Execute panic.
						if !hookFired {
							results[i] = toolOutput{index: i, err: fmt.Errorf("goai: OnToolCallStart hook for tool %q panicked: %s", tc.Name, panicStr)}
						} else {
							results[i] = toolOutput{index: i, err: fmt.Errorf("goai: tool %q panicked: %s", tc.Name, panicStr)}
						}
					}
					// executed==true: Execute succeeded, results[i] already set.
					// OnToolCall panic after Execute is swallowed (preserve result).
				}
			}()

			// OnToolCallStart: pre-execution.
			if onToolCallStart != nil {
				onToolCallStart(ToolCallStartInfo{
					ToolCallID: tc.ID,
					ToolName:   tc.Name,
					Step:       step,
					Input:      tc.Input,
				})
			}
			hookFired = true

			start := time.Now()
			output, err := tool.Execute(ctx, tc.Input)
			executed = true
			results[i] = toolOutput{index: i, result: output, err: err}

			// OnToolCall: post-execution.
			if onToolCall != nil {
				info := ToolCallInfo{
					ToolCallID: tc.ID,
					ToolName:   tc.Name,
					Step:       step,
					Input:      tc.Input,
					Output:     output,
					StartTime:  start,
					Duration:   time.Since(start),
					Error:      err,
				}
				var parsed any
				if err == nil && json.Unmarshal([]byte(output), &parsed) == nil {
					info.OutputObject = parsed
				}
				onToolCall(info)
			}
		}(i, tc, tool)
	}

	wg.Wait()
	return buildToolMessages(calls, results)
}

// buildToolMessages converts tool call results to provider messages.
// Note: toolOutput is defined as a package-level type (not function-scoped)
// so both executeToolsParallel and buildToolMessages can reference it.
func buildToolMessages(calls []provider.ToolCall, results []toolOutput) []provider.Message {
	msgs := make([]provider.Message, 0, len(calls))
	for i, tc := range calls {
		r := results[i]
		if r.err != nil {
			errStr := r.err.Error()
			runes := []rune(errStr)
			if len(runes) > 500 {
				errStr = string(runes[:500]) + "..."
			}
			msgs = append(msgs, ToolMessage(tc.ID, tc.Name, "error: "+errStr))
		} else {
			msgs = append(msgs, ToolMessage(tc.ID, tc.Name, r.result))
		}
	}
	return msgs
}

// appendToolRoundTrip appends an assistant message (with tool_use parts)
// and tool result messages for the streaming tool loop.
func appendToolRoundTrip(
	msgs []provider.Message,
	text string,
	toolCalls []provider.ToolCall,
	toolMsgs []provider.Message,
) []provider.Message {
	var parts []provider.Part
	if text != "" {
		parts = append(parts, provider.Part{Type: provider.PartText, Text: text})
	}
	for _, tc := range toolCalls {
		parts = append(parts, provider.Part{
			Type:       provider.PartToolCall,
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			ToolInput:  tc.Input,
		})
	}
	msgs = append(msgs, provider.Message{Role: provider.RoleAssistant, Content: parts})
	msgs = append(msgs, toolMsgs...)
	return msgs
}

// buildTextResult constructs a TextResult from accumulated steps.
// Caller must ensure steps is non-empty.
func buildTextResult(steps []StepResult, totalUsage provider.Usage) *TextResult {
	if len(steps) == 0 {
		return &TextResult{TotalUsage: totalUsage}
	}
	last := steps[len(steps)-1]
	// Accumulate text from all steps.
	var allText strings.Builder
	for _, s := range steps {
		allText.WriteString(s.Text)
	}
	// Collect sources from all steps.
	var allSources []provider.Source
	for _, s := range steps {
		allSources = append(allSources, s.Sources...)
	}

	return &TextResult{
		Text:             allText.String(),
		ToolCalls:        last.ToolCalls,
		Steps:            steps,
		TotalUsage:       totalUsage,
		FinishReason:     last.FinishReason,
		Response:         last.Response,
		ProviderMetadata: last.ProviderMetadata,
		Sources:          allSources,
	}
}

type drainResult struct {
	text string // text-only (ChunkText), used for appendToolRoundTrip
	// Note: ChunkReasoning is forwarded to consumer but NOT accumulated here.
	// Reasoning is visible via Stream()/TextStream() but not echoed back to the model.
	toolCalls        []provider.ToolCall
	usage            provider.Usage
	finishReason     provider.FinishReason
	sources          []provider.Source
	response         provider.ResponseMetadata
	providerMetadata map[string]map[string]any
	err              error // non-nil if context cancelled during drain
}

func drainStep(
	ctx context.Context,
	source <-chan provider.StreamChunk,
	out chan<- provider.StreamChunk,
) drainResult {
	var (
		textBuf strings.Builder // ChunkText only (reasoning excluded)
		dr      drainResult
	)

	for chunk := range source {
		// Forward chunk to consumer. Suppress these types (handled explicitly below):
		// - ChunkFinish: step loop emits its own ChunkFinish with totalUsage
		// - ChunkError: forwarded explicitly in the switch to avoid double-send
		// Provider ChunkStepFinish IS forwarded (not suppressed) so Stream() consumers
		// can see provider-internal boundaries (e.g., Anthropic thinking steps). These
		// do NOT carry Metadata["stepSource"]="goai", so consume() distinguishes them.
		if chunk.Type != provider.ChunkFinish && chunk.Type != provider.ChunkError {
			if !provider.TrySend(ctx, out, chunk) {
				// Drain source to unblock provider.
				drainRemaining(source)
				dr.err = ctx.Err()
				return dr
			}
		}

		// Accumulate state (same logic as TextStream.consume, generate.go:198-242).
		// Note: ChunkToolCallDelta, ChunkToolCallStreamStart, and ChunkToolResult are
		// forwarded to the consumer (not suppressed) but NOT accumulated here. drainStep
		// only captures complete ChunkToolCall chunks. Providers always emit a final
		// ChunkToolCall with complete data after all deltas.
		switch chunk.Type {
		case provider.ChunkText:
			textBuf.WriteString(chunk.Text)
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				dr.sources = append(dr.sources, s)
			}
		case provider.ChunkReasoning:
			// Forwarded but not accumulated in text (reasoning excluded from tool round-trip).
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				dr.sources = append(dr.sources, s)
			}
		case provider.ChunkToolCall:
			dr.toolCalls = append(dr.toolCalls, provider.ToolCall{
				ID:    chunk.ToolCallID,
				Name:  chunk.ToolName,
				Input: json.RawMessage(chunk.ToolInput),
			})
		case provider.ChunkStepFinish:
			// Provider-internal step boundary (e.g., Anthropic extended thinking).
			// Use direct assignment (last value wins), matching ChunkFinish below.
			// This is correct for both zero-usage providers (Anthropic: 0 overwrites 0)
			// and running-total providers (Google: last total is authoritative).
			dr.finishReason = chunk.FinishReason
			dr.usage = chunk.Usage
			dr.response = chunk.Response
			if sources, ok := chunk.Metadata["sources"].([]provider.Source); ok {
				dr.sources = append(dr.sources, sources...)
			}
			if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
				dr.providerMetadata = pm
			}
			for k, v := range chunk.Metadata {
				if k == "providerMetadata" || k == "sources" {
					continue
				}
				if dr.response.ProviderMetadata == nil {
					dr.response.ProviderMetadata = map[string]any{}
				}
				dr.response.ProviderMetadata[k] = v
			}
		case provider.ChunkFinish:
			// Terminal chunk. Use direct assignment for usage (not addUsage) to avoid
			// double-counting when providers emit both ChunkStepFinish and ChunkFinish
			// with the same accumulated usage (e.g., Google).
			dr.finishReason = chunk.FinishReason
			dr.usage = chunk.Usage
			dr.response = chunk.Response
			if sources, ok := chunk.Metadata["sources"].([]provider.Source); ok {
				dr.sources = append(dr.sources, sources...)
			}
			if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
				dr.providerMetadata = pm
			}
			// Copy flat metadata keys to Response.ProviderMetadata (same as consume(),
			// generate.go:229-237). Providers use this for per-response data: Anthropic
			// ("iterations", "contextManagement"), Bedrock ("cacheWriteInputTokens").
			for k, v := range chunk.Metadata {
				if k == "providerMetadata" || k == "sources" {
					continue
				}
				if dr.response.ProviderMetadata == nil {
					dr.response.ProviderMetadata = map[string]any{}
				}
				dr.response.ProviderMetadata[k] = v
			}
		case provider.ChunkError:
			// Forward error chunks to consumer. Mid-stream errors flow through
			// ChunkError chunks to the consumer; OnResponse does not report them.
			if !provider.TrySend(ctx, out, chunk) {
				drainRemaining(source)
				dr.err = ctx.Err()
				return dr
			}
		}
	}

	dr.text = textBuf.String()
	return dr
}

// drainRemaining reads and discards all remaining chunks from source.
// This unblocks the provider's write-side goroutine on context cancellation,
// preventing goroutine leaks. Note: this blocks until the provider closes its
// channel. A misbehaving provider that never closes will cause this to hang.
// All current providers use defer close(out) so this is safe in practice.
func drainRemaining(source <-chan provider.StreamChunk) {
	for range source {
		// discard
	}
}

// addUsage adds b's counts to a and returns the result.
func addUsage(a, b provider.Usage) provider.Usage {
	return provider.Usage{
		InputTokens:      a.InputTokens + b.InputTokens,
		OutputTokens:     a.OutputTokens + b.OutputTokens,
		TotalTokens:      a.TotalTokens + b.TotalTokens,
		ReasoningTokens:  a.ReasoningTokens + b.ReasoningTokens,
		CacheReadTokens:  a.CacheReadTokens + b.CacheReadTokens,
		CacheWriteTokens: a.CacheWriteTokens + b.CacheWriteTokens,
	}
}
