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

// toolCallIDKey is the context key for the current tool call ID.
type toolCallIDKey struct{}

// ToolCallIDFromContext returns the tool call ID from the context.
// This is available inside a Tool's Execute function to identify which
// tool call is being executed.
func ToolCallIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(toolCallIDKey{}).(string); ok {
		return v
	}
	return ""
}

// TextResult is the final result of a text generation call.
type TextResult struct {
	// Text is the accumulated generated text across all steps.
	// For StreamText, this includes reasoning tokens (ChunkReasoning) for backward
	// compatibility. Use Steps[n].Text for text-only content excluding reasoning.
	Text string

	// Reasoning is the model's accumulated thinking/reasoning text
	// across all steps (PartReasoning). Populated for both GenerateText
	// and StreamText when the provider returns reasoning content (e.g.
	// Anthropic extended thinking on Bedrock). Empty when reasoning is
	// disabled or unsupported.
	Reasoning string

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

	// StepsExhausted is true when the tool loop terminated because MaxSteps was reached
	// while the model still requested tool calls. This distinguishes "model finished
	// naturally" (StepsExhausted=false) from "loop was cut short" (StepsExhausted=true).
	StepsExhausted bool

	// ResponseMessages contains the assistant and tool messages from all generation steps.
	// For multi-turn conversations, append these to your message history:
	//   messages = append(messages, result.ResponseMessages...)
	//
	// Nil when the response has no content (empty text and no tool calls).
	// For StreamText, check Err() before using , on stream errors, ResponseMessages
	// may be partial (intermediate tool round-trips lost) or reflect only completed
	// steps. Do not use ResponseMessages for conversation continuation when Err() != nil.
	// Reasoning parts (PartReasoning) are included for StreamText (both single-step
	// and multi-step) but not for GenerateText (which does not expose reasoning).
	// Reasoning chunks are consolidated into a single PartReasoning part with merged
	// metadata (e.g. Anthropic/Bedrock signatures).
	ResponseMessages []provider.Message
}

// StepResult is the result of a single generation step in a tool loop.
type StepResult struct {
	// Number is the 1-based step index.
	Number int

	// Text generated in this step (excludes reasoning tokens).
	// For StreamText, reasoning is included in TextResult.Text but excluded here.
	Text string

	// Reasoning is the consolidated thinking/reasoning text for this
	// step (PartReasoning, signature stripped). Populated for both
	// GenerateText and StreamText when the provider returns reasoning.
	Reasoning string

	// ToolCalls requested in this step.
	ToolCalls []provider.ToolCall

	// ToolResults contains one entry per completed ToolCall in this step,
	// populated AFTER executeToolsParallel returns and BEFORE WithStopWhen
	// is evaluated. Ordering matches ToolCalls element-for-element.
	//
	// Empty when the step had no tool calls or when the loop exits before
	// executing tools (e.g. MaxSteps reached with pending tool calls that
	// never ran, or StopCauseNoExecutableTools).
	//
	// Mirrors Vercel AI SDK's DefaultStepResult.toolResults so predicates
	// passed to WithStopWhen can inspect tool outputs (matching the
	// placement documented on StopCondition).
	//
	// Streaming visibility: consumers using StreamText who read raw
	// chunks via stream.Stream() cannot observe per-step ToolResults in
	// real time. The ChunkStepFinish chunk with stepSource="goai" is
	// emitted BEFORE tools execute (so ToolResults is empty at that
	// point); the subsequent goai-internal "goai-tool-results" chunk
	// that backfills ToolResults is consumed by the stream reducer and
	// NOT re-emitted to the raw chunk channel. To observe ToolResults
	// per step from a streaming call, use one of:
	//   - stream.Result() after the stream closes (Steps[].ToolResults
	//     is fully populated).
	//   - OnToolCall hook (fires synchronously after each tool Execute
	//     returns with per-call detail).
	//   - OnAfterToolExecute hook (same timing as OnToolCall, richer
	//     metadata).
	// The OnStepFinish hook always receives a StepResult with an EMPTY
	// ToolResults slice because tools execute AFTER the hook fires.
	ToolResults []provider.ToolResult

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
// Stream() and TextStream() are mutually exclusive - only call one.
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
	onResponse   []func(ResponseInfo)
	onStepFinish []func(StepResult)
	onFinish     []func(FinishInfo)
	startTime    time.Time

	// stateRef, when non-nil, is transitioned to StepIdle when the consume
	// goroutine returns. Only set by the single-shot StreamText path
	// (streamWithToolLoop owns its own StateRef lifecycle inline). A nil
	// stateRef is a no-op; AgentState.set handles nil receiver safely.
	// See FIX 34.
	stateRef *AgentState

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
	reasoningBuf  strings.Builder // consolidated reasoning text (matches drainStep)
	reasoningMeta map[string]any  // merged reasoning metadata (e.g. Anthropic signature)

	// responseMessages is set by the streamWithToolLoop goroutine before doneCh closes.
	responseMessages []provider.Message
	// stepsExhausted is set by the streamWithToolLoop goroutine when MaxSteps was reached.
	stepsExhausted bool
}

func newTextStream(ctx context.Context, source <-chan provider.StreamChunk) *TextStream {
	return &TextStream{
		ctx:    ctx,
		source: source,
		doneCh: make(chan struct{}),
	}
}

// Stream returns a channel that emits raw StreamChunks from the provider.
// Mutually exclusive with TextStream() - only call one streaming method.
func (ts *TextStream) Stream() <-chan provider.StreamChunk {
	ch := make(chan provider.StreamChunk, 64)
	ts.consumeOnce.Do(func() {
		ts.rawCh = ch
		go ts.consume(ch, nil)
	})
	if ts.rawCh != nil {
		return ts.rawCh
	}
	// Called after TextStream() consumed the source - return closed channel.
	close(ch)
	return ch
}

// TextStream returns the underlying channel of text chunks.
// Note: this method has the same name as the containing type (TextStream);
// call it as stream.TextStream() to receive the channel.
// Mutually exclusive with Stream() - only call one streaming method.
func (ts *TextStream) TextStream() <-chan string {
	ch := make(chan string, 64)
	ts.consumeOnce.Do(func() {
		ts.textCh = ch
		go ts.consume(nil, ch)
	})
	if ts.textCh != nil {
		return ts.textCh
	}
	// Called after Stream() consumed the source - return closed channel.
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
	// FIX 34: transition StateRef to StepIdle as the consume goroutine
	// exits. Single-shot StreamText wires this (stateRef is nil for the
	// multi-step streamWithToolLoop path, which owns its own StateRef
	// transitions inline). Deferred second from the top so it runs just
	// before close(doneCh) - after all user hooks (OnResponse, OnStepFinish,
	// OnFinish) have fired, so a poller that observes StepIdle can assume
	// all hooks have already returned. Step count is 1 for a single-shot
	// stream (the one DoStream call), regardless of whether the provider
	// emitted any chunks.
	if ts.stateRef != nil {
		defer func() { ts.stateRef.set(StepIdle, 1) }()
	}
	if ts.timeoutCancel != nil {
		defer ts.timeoutCancel()
	}
	if rawOut != nil {
		defer close(rawOut)
	}
	if textOut != nil {
		defer close(textOut)
	}

	// Call OnFinish hook when consume finishes (single-step streaming only).
	// For multi-step streaming (streamWithToolLoop), OnFinish fires inline in the
	// goroutine and ts.onFinish is nil, so this block is skipped.
	// Deferred BEFORE OnStepFinish so it runs AFTER it (LIFO order).
	if len(ts.onFinish) > 0 {
		defer func() {
			fireOnFinish(ts.onFinish, FinishInfo{
				TotalSteps:   1,
				TotalUsage:   ts.usage,
				FinishReason: ts.finishReason,
				StoppedBy:    provider.StopCauseNatural,
			})
		}()
	}

	// Call OnStepFinish hook when consume finishes (single-step streaming only).
	// For multi-step streaming (streamWithToolLoop), OnStepFinish fires inline per
	// step and ts.onStepFinish is nil, so this block is skipped.
	// Deferred BEFORE OnResponse so it runs AFTER it (LIFO order), matching
	// GenerateText's OnResponse → OnStepFinish sequence.
	if len(ts.onStepFinish) > 0 {
		defer func() {
			stepResult := StepResult{
				Number:           1,
				Text:             ts.stepText.String(),
				ToolCalls:        ts.toolCalls,
				FinishReason:     ts.finishReason,
				Usage:            ts.usage,
				Response:         ts.response,
				Sources:          ts.sources,
				ProviderMetadata: ts.providerMetadata,
			}
			for _, fn := range ts.onStepFinish {
				func(f func(StepResult)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(stepResult)
				}(fn)
			}
		}()
	}

	// Call OnResponse hook when consume finishes (after all chunks processed).
	if len(ts.onResponse) > 0 {
		defer func() {
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
			for _, fn := range ts.onResponse {
				func(f func(ResponseInfo)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(info)
				}(fn)
			}
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
			// Consolidate reasoning fragments into one Part (matching drainStep behavior).
			// Text is accumulated; metadata is merged (last chunk carries the signature).
			if chunk.Text != "" {
				ts.reasoningBuf.WriteString(chunk.Text)
			}
			if chunk.Metadata != nil {
				if ts.reasoningMeta == nil {
					ts.reasoningMeta = make(map[string]any)
				}
				for k, v := range chunk.Metadata {
					ts.reasoningMeta[k] = v
				}
			}
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				ts.sources = append(ts.sources, s)         // global (preserve existing behavior)
				ts.stepSources = append(ts.stepSources, s) // per-step
			}

		case provider.ChunkToolCall:
			tc := provider.ToolCall{ID: chunk.ToolCallID, Name: chunk.ToolName, Input: json.RawMessage(chunk.ToolInput), Metadata: chunk.Metadata}
			ts.toolCalls = append(ts.toolCalls, tc)
			ts.stepToolCalls = append(ts.stepToolCalls, tc)

		case provider.ChunkStepFinish:
			// GoAI-emitted tool-results backfill: update the last completed
			// step's ToolResults (FIX 7 streaming parity). Emitted after
			// executeToolsParallel returns so ts.steps exposes the same data
			// the sync path's StepResult.ToolResults carries.
			if stepSource, _ := chunk.Metadata["stepSource"].(string); stepSource == "goai-tool-results" {
				if trs, ok := chunk.Metadata["toolResults"].([]provider.ToolResult); ok && len(ts.steps) > 0 {
					ts.steps[len(ts.steps)-1].ToolResults = trs
				}
				continue
			}
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
					Reasoning:        ts.reasoningBuf.String(),
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
				ts.reasoningBuf.Reset()
				ts.reasoningMeta = nil
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
		// Aggregate per-step reasoning so streaming TextResult exposes
		// the same .Reasoning field as GenerateText.
		var reasoningAll strings.Builder
		for _, s := range ts.steps {
			reasoningAll.WriteString(s.Reasoning)
		}
		result.Reasoning = reasoningAll.String()
	} else if text != "" || len(ts.toolCalls) > 0 || ts.finishReason != "" {
		// Single-step fallback (no multi-step ChunkStepFinish received, but data exists).
		stepReasoning := ts.reasoningBuf.String()
		result.Steps = []StepResult{{
			Number:           1,
			Text:             ts.stepText.String(),
			Reasoning:        stepReasoning,
			ToolCalls:        ts.toolCalls,
			FinishReason:     ts.finishReason,
			Usage:            ts.usage,
			Response:         ts.response,
			Sources:          ts.sources,
			ProviderMetadata: ts.providerMetadata,
		}}
		result.Reasoning = stepReasoning
	}
	// No data: Steps is nil.

	// Set StepsExhausted from streamWithToolLoop goroutine.
	result.StepsExhausted = ts.stepsExhausted

	// Populate ResponseMessages.
	if ts.responseMessages != nil {
		// Multi-step: set by streamWithToolLoop goroutine.
		result.ResponseMessages = ts.responseMessages
	} else if text != "" || len(result.ToolCalls) > 0 {
		// Single-step: build a simple assistant message from the result.
		// Use stepText (text-only, excludes reasoning) for ResponseMessages so reasoning
		// doesn't get baked into PartText. Pass consolidated reasoning part separately
		// (matching drainStep: one Part with merged metadata including signatures).
		var reasoning []provider.Part
		if ts.reasoningBuf.Len() > 0 || len(ts.reasoningMeta) > 0 {
			reasoning = []provider.Part{{
				Type:            provider.PartReasoning,
				Text:            ts.reasoningBuf.String(),
				ProviderOptions: ts.reasoningMeta,
			}}
		}
		result.ResponseMessages = buildFinalAssistantMessages(ts.stepText.String(), result.ToolCalls, reasoning)
	}
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
	originalLen := len(params.Messages)

	var timeoutCancel context.CancelFunc
	if o.Timeout > 0 {
		ctx, timeoutCancel = context.WithTimeout(ctx, o.Timeout)
	}

	// AgentState: initial. o.StateRef may be nil (set is a no-op).
	o.StateRef.set(StepStarting, 0)

	// --- Step 1 DoStream: synchronous (preserves (nil, error) contract) ---
	// This ensures StreamText ALWAYS returns (nil, error) when the first DoStream
	// fails, regardless of MaxSteps. Eliminates the split error contract.
	o.StateRef.set(StepLLMInFlight, 1)
	for _, fn := range o.OnRequest {
		fn(RequestInfo{
			Ctx:          ctx,
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
		// FIX 47: preserve step=1 on error - monotonicity. The store above
		// already moved the step counter to 1 (StepLLMInFlight, 1); a poller
		// observing between the two stores must not see step regress to 0.
		o.StateRef.set(StepIdle, 1)
		// OnRequest/OnResponse: not recover-wrapped (caller's goroutine).
		// OnStepFinish: always recover-wrapped (prevents losing accumulated results).
		// Inside goroutines: all hooks recover-wrapped.
		for _, fn := range o.OnResponse {
			info := ResponseInfo{Latency: time.Since(start), Error: err}
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				info.StatusCode = apiErr.StatusCode
			}
			fn(info)
		}
		return nil, err // SAME error contract as single-step StreamText
	}

	// Step 1 succeeded. Goroutine-local copy of start time avoids closure capture.
	step1Start := start
	out := make(chan provider.StreamChunk, 64)
	ts := newTextStream(ctx, out)

	go func() {
		defer close(out)
		if timeoutCancel != nil {
			defer timeoutCancel()
		}
		var totalUsage provider.Usage
		var lastFinishReason provider.FinishReason
		var lastResponse provider.ResponseMetadata
		var lastReasoning []provider.Part
		var steps []StepResult
		var stepsExhausted bool
		var hookStopped bool              // true iff WithStopWhen or OnBeforeStep.Stop broke the loop
		var stopCause provider.StopCause  // classifies how the loop exited (FIX 5)
		firstStep := true                 // true only for step 1 (already have firstResult)
		stepStart := step1Start           // goroutine-local start time per step

		// highestInflightStep tracks the maximum step counter announced via
		// o.StateRef.set(StepLLMInFlight, ...). The deferred StepIdle publish
		// uses max(len(steps), highestInflightStep) so the observable step
		// value never regresses (FIX 47 monotonicity: if a mid-loop step
		// errors before being appended to `steps`, len(steps) lags behind
		// highestInflightStep; the defer must publish the larger value).
		//
		// There are TWO writes to highestInflightStep in this function, and
		// both are intentional (FIX 54 + FIX 55):
		//
		//   1. The `= 1` assignment below (pre-loop): mirrors the
		//      o.StateRef.set(StepLLMInFlight, 1) call that happens BEFORE
		//      this goroutine starts (see the pre-goroutine set call
		//      earlier in streamWithToolLoop). This guarantees the deferred
		//      StepIdle publish has something >= 1 to report even if the
		//      goroutine exits before entering the loop body (e.g. panic
		//      recovery, pre-loop early exit).
		//   2. The `= step` assignment in the firstStep branch (loop body):
		//      refactor-safety for future changes that move the step-1
		//      StepLLMInFlight announcement inside the loop. If such a
		//      refactor happens, write #1 should be deleted; write #2
		//      continues to maintain the invariant from inside the loop.
		//
		// Both writes together ensure the invariant "highestInflightStep
		// reflects the latest StepLLMInFlight announcement" holds on every
		// path the defer can fire from.
		highestInflightStep := 0
		highestInflightStep = 1
		// Ensure any exit path (break, return, panic-recover-above, natural
		// termination) leaves the observable state as Idle. Use closure-captured
		// steps so the final step count is visible to pollers.
		defer func() {
			finalStep := len(steps)
			if highestInflightStep > finalStep {
				finalStep = highestInflightStep
			}
			o.StateRef.set(StepIdle, finalStep)
		}()

		for step := 1; step <= o.MaxSteps; step++ {
			var result *provider.StreamResult

			if firstStep {
				// Step 1: use the already-obtained firstResult.
				result = firstResult
				firstStep = false
				// FIX 54: record the step-1 announcement inside the loop body
				// so the invariant "every StepLLMInFlight has a matching
				// highestInflightStep write" holds from step 1. The actual
				// atomic store happened before the goroutine; this is the
				// in-loop bookkeeping companion.
				highestInflightStep = step
			} else {
				// Steps 2+: OnBeforeStep hook (can inject messages or stop loop).
				if o.OnBeforeStep != nil {
					var bsr BeforeStepResult
					func() {
						defer func() {
							if r := recover(); r != nil {
								fmt.Fprintf(os.Stderr, "goai: recovered panic in OnBeforeStep hook: %v\n", r)
							}
						}()
						bsr = o.OnBeforeStep(BeforeStepInfo{
							Ctx:      ctx,
							Step:     step,
							Messages: slices.Clone(params.Messages),
						})
					}()
					if bsr.Stop {
						// Semantic parity with WithStopWhen: mark as hookStopped.
						hookStopped = true
						stopCause = provider.StopCauseBeforeStep
						break
					}
					if len(bsr.ExtraMessages) > 0 {
						params.Messages = append(params.Messages, bsr.ExtraMessages...)
					}
				}

				// Steps 2+: DoStream inside goroutine.
				for _, fn := range o.OnRequest {
					func(f func(RequestInfo)) {
						defer func() {
							if r := recover(); r != nil {
								fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
							}
						}()
						f(RequestInfo{
							Ctx:          ctx,
							Model:        model.ModelID(),
							MessageCount: len(params.Messages),
							ToolCount:    len(params.Tools),
							Timestamp:    time.Now(),
							Messages:     requestMessages(params.System, params.Messages),
						})
					}(fn)
				}

				stepStart = time.Now()
				o.StateRef.set(StepLLMInFlight, step)
				highestInflightStep = step
				var err error
				result, err = withRetry(ctx, o.MaxRetries, func() (*provider.StreamResult, error) {
					return model.DoStream(ctx, params)
				})
				if err != nil {
					// Fire OnResponse on error (recover-wrapped).
					for _, fn := range o.OnResponse {
						func(f func(ResponseInfo)) {
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
							f(info)
						}(fn)
					}
					// responseMessages intentionally not set on error , buildResult falls back to
					// a minimal assistant message from accumulated text. Intermediate tool round-trip
					// messages are lost. Callers should check Err() and not rely on ResponseMessages
					// when the stream has errors.
					provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: err})
					provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: lastFinishReason, Usage: totalUsage, StoppedBy: provider.StopCauseAbort})
					// Fire OnFinish so observability hooks can close spans/flush traces.
					lastFinish := provider.FinishReason("")
					if len(steps) > 0 {
						lastFinish = steps[len(steps)-1].FinishReason
					}
					fireOnFinish(o.OnFinish, FinishInfo{
						TotalSteps:   len(steps),
						TotalUsage:   totalUsage,
						FinishReason: lastFinish,
						StoppedBy:    provider.StopCauseAbort,
					})
					return
				}
			}

			ds := drainStep(ctx, result.Stream, out)
			if ds.err != nil {
				// responseMessages intentionally not set on error - buildResult falls back to
				// a minimal assistant message from accumulated text. Intermediate tool round-trip
				// messages are lost. Callers should check Err() and not rely on ResponseMessages
				// when the stream has errors.
				provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: ds.err})
				provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkFinish, Usage: totalUsage, StoppedBy: provider.StopCauseAbort})
				// Fire OnFinish so observability hooks can close spans/flush traces.
				// Without this, OTel root spans leak and Langfuse traces are lost.
				lastFinish := provider.FinishReason("")
				if len(steps) > 0 {
					lastFinish = steps[len(steps)-1].FinishReason
				}
				fireOnFinish(o.OnFinish, FinishInfo{
					TotalSteps:   len(steps),
					TotalUsage:   totalUsage,
					FinishReason: lastFinish,
					StoppedBy:    provider.StopCauseAbort,
				})
				return
			}

			// Guard: skip empty step (provider closed channel without sending
			// any meaningful chunks, e.g., after a ChunkError). Prevents emitting
			// a phantom empty StepResult and ChunkStepFinish. This is not an
			// error path (a separate ChunkError path covers real errors) - use
			// StopCauseEmpty so consumers can distinguish a no-op response from
			// an abort.
			if ds.text == "" && len(ds.toolCalls) == 0 && ds.finishReason == "" {
				stopCause = provider.StopCauseEmpty
				break
			}

			// OnResponse: Error is NOT set (call succeeded). Mid-stream errors use stream.Err().
			for _, fn := range o.OnResponse {
				func(f func(ResponseInfo)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(ResponseInfo{
						Latency:      time.Since(stepStart),
						Usage:        ds.usage,
						FinishReason: ds.finishReason,
					})
				}(fn)
			}

			// --- Build StepResult, fire OnStepFinish ---
			stepResult := StepResult{
				Number:           step,
				Text:             ds.text,
				Reasoning:        ds.reasoningText,
				ToolCalls:        ds.toolCalls,
				FinishReason:     ds.finishReason,
				Usage:            ds.usage,
				Sources:          ds.sources,
				Response:         ds.response,
				ProviderMetadata: ds.providerMetadata,
			}
			steps = append(steps, stepResult)
			totalUsage = addUsage(totalUsage, ds.usage)
			lastResponse = ds.response
			lastReasoning = ds.reasoning

			// AgentState: the step's stream has fully drained; tool exec and
			// stop-predicate evaluation have not started yet. Pollers observing
			// in this window must see StepStepFinished, not StepLLMInFlight.
			o.StateRef.set(StepStepFinished, step)

			// OnStepFinish (recover-wrapped).
			for _, fn := range o.OnStepFinish {
				func(f func(StepResult)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(stepResult)
				}(fn)
			}

			// Normalize: providers that send empty/wrong finish_reason with tool calls
			// (MiniMax, Azure MaaS deepseek, etc.) would cause the loop to exit early.
			// The presence of tool calls is authoritative. Must run BEFORE capturing
			// lastFinishReason so the final ChunkFinish carries a non-empty reason
			// even when providers (gemini, bedrock-sonnet, azure-sonnet) emit empty
			// finish_reason alongside tool calls.
			if len(ds.toolCalls) > 0 && ds.finishReason != provider.FinishToolCalls {
				ds.finishReason = provider.FinishToolCalls
			}
			lastFinishReason = ds.finishReason

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

			// --- Exit conditions (same as GenerateText) ---
			// Note: streamWithToolLoop is only entered when len(toolMap) > 0
			// (guarded at StreamText entry - see generate.go:992), and toolMap
			// is immutable for the lifetime of the call. The "no executable
			// tools" exit therefore cannot be reached here; StopCauseNoExecutableTools
			// is a sync-only cause (GenerateText). See StopCauseNoExecutableTools
			// godoc in provider/types.go.
			if ds.finishReason != provider.FinishToolCalls || len(ds.toolCalls) == 0 {
				stopCause = provider.StopCauseNatural
				break
			}

			// --- Execute tools in parallel ---
			o.StateRef.set(StepToolExecuting, step)
			toolMsgs, toolResults := executeToolsParallel(ctx, ds.toolCalls, toolMap, step, toolHooks{
				sequential:      o.SequentialTools,
				onToolCallStart: o.OnToolCallStart,
				onToolCall:      o.OnToolCall,
				onBeforeExecute: o.OnBeforeToolExecute,
				onAfterExecute:  o.OnAfterToolExecute,
			})
			// Attach ToolResults to the step BEFORE the stop predicate sees it
			// (FIX 7 / Vercel DefaultStepResult parity). steps[-1] is this step.
			steps[len(steps)-1].ToolResults = toolResults

			// Notify the consumer (TextStream.consume) that this step now has
			// tool results so ts.steps[len-1].ToolResults can be backfilled.
			// We reuse ChunkStepFinish with a distinguishing stepSource tag so
			// existing provider-internal ChunkStepFinish handling is untouched.
			provider.TrySend(ctx, out, provider.StreamChunk{
				Type: provider.ChunkStepFinish,
				Metadata: map[string]any{
					"stepSource":  "goai-tool-results",
					"toolResults": toolResults,
				},
			})

			// --- Append messages for next step ---
			params.Messages = appendToolRoundTrip(params.Messages, ds.text, ds.reasoning, ds.toolCalls, toolMsgs)
			// Clear ToolChoice so model can freely respond on subsequent steps.
			// Set on every iteration for simplicity; idempotent after step 1.
			params.ToolChoice = ""

			// WithStopWhen (Vercel parity): evaluated AFTER this step's tool
			// executions complete and the tool-result messages have been
			// folded into params.Messages. ResponseMessages built on break
			// here is a valid replay transcript (matching tool_use /
			// tool_result pairs). Matches
			// vercel-ai/packages/ai/src/generate-text/generate-text.ts.
			if o.StopWhen != nil && stopSafe(o.StopWhen, steps) {
				hookStopped = true
				stopCause = provider.StopCausePredicate
				break
			}
		}

		stopCause, stepsExhausted = finalizeStopCause(hookStopped, stopCause, steps, o.MaxSteps)

		// Set responseMessages before emitting ChunkFinish (safe: ts.buildResult reads
		// responseMessages only after doneCh closes, which happens after out is closed).
		ts.stepsExhausted = stepsExhausted
		ts.responseMessages = buildResponseMessages(params.Messages[originalLen:], steps, lastReasoning)

		fireOnFinish(o.OnFinish, FinishInfo{
			StepsExhausted: stepsExhausted,
			TotalSteps:     len(steps),
			TotalUsage:     totalUsage,
			FinishReason:   lastFinishReason,
			StoppedBy:      stopCause,
		})

		// Emit final ChunkFinish with total usage and last step Response metadata.
		provider.TrySend(ctx, out, provider.StreamChunk{
			Type:         provider.ChunkFinish,
			FinishReason: lastFinishReason,
			Usage:        totalUsage,
			Response:     lastResponse,
			StoppedBy:    stopCause,
		})
	}()

	// OnResponse handled per-step inside the goroutine (ts.onResponse not set).
	return ts, nil
}

// StreamText performs a streaming text generation.
// When MaxSteps > 1 and executable tools are provided, StreamText runs an automatic
// tool loop. The initial DoStream failure still returns (nil, error). Subsequent step
// errors flow through the stream as ChunkError chunks; check stream.Err() after consuming.
func StreamText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextStream, error) {
	// Apply options FIRST so o.StateRef is populated before any early return.
	// This guarantees pollers waiting for StepIdle do not deadlock when we
	// return (nil, err) before the streaming goroutine starts (e.g. nil model,
	// empty prompt, initial DoStream failure).
	o := applyOptions(opts...)

	if model == nil {
		// Transition StepStarting→StepIdle for any observer so pollers do not deadlock.
		o.StateRef.set(StepStarting, 0)
		o.StateRef.set(StepIdle, 0)
		return nil, errors.New("goai: model must not be nil")
	}

	if o.Prompt == "" && len(o.Messages) == 0 {
		// Pre-loop validation error must still transition any observer to
		// StepIdle so pollers waiting on it do not deadlock.
		o.StateRef.set(StepStarting, 0)
		o.StateRef.set(StepIdle, 0)
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

	// FIX 34: single-shot StreamText never touched StateRef. Pollers using
	// WithStateRef(&state) + WithMaxSteps(1) (or no executable tools) got
	// stuck at the zero value (StepStarting, 0) forever. Transition through
	// StepStarting → StepLLMInFlight here; StepIdle is deferred in consume
	// (see ts.stateRef assignment below).
	o.StateRef.set(StepStarting, 0)
	o.StateRef.set(StepLLMInFlight, 1)

	for _, fn := range o.OnRequest {
		fn(RequestInfo{
			Ctx:          ctx,
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
		// FIX 34: DoStream failed before the consume goroutine could be
		// started, so the consume-based StepIdle defer will never run.
		// Transition to StepIdle inline so pollers waiting for it do not
		// deadlock. FIX 47: preserve step=1 (the step we just set to
		// LLMInFlight) instead of regressing to 0 - pollers observing
		// between the StepLLMInFlight store above and this store must
		// see a monotonically non-decreasing step counter.
		o.StateRef.set(StepIdle, 1)
		for _, fn := range o.OnResponse {
			info := ResponseInfo{Latency: time.Since(start), Error: err}
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				info.StatusCode = apiErr.StatusCode
			}
			fn(info)
		}
		return nil, err
	}

	ts := newTextStream(ctx, result.Stream)
	ts.timeoutCancel = timeoutCancel
	ts.onResponse = o.OnResponse
	ts.onStepFinish = o.OnStepFinish
	ts.onFinish = o.OnFinish
	ts.startTime = start
	// FIX 34: hand StateRef ownership to the consume goroutine; it will
	// transition to StepIdle when the stream drains / errors. Only set on
	// the single-shot path; streamWithToolLoop manages StateRef inline.
	ts.stateRef = o.StateRef
	return ts, nil
}

// GenerateText performs a non-streaming text generation.
// When tools with Execute functions are provided and MaxSteps > 1,
// it automatically runs a tool loop: generate → execute tools → re-generate.
func GenerateText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextResult, error) {
	// Apply options FIRST so o.StateRef is populated before any early return.
	// Registered BEFORE nil-model / prompt validation so pre-loop error returns
	// also transition observers to StepIdle (otherwise pollers waiting for
	// StepIdle would deadlock on validation errors).
	o := applyOptions(opts...)

	var steps []StepResult
	// highestInflightStep tracks the largest step index we have announced
	// via StepLLMInFlight. Used by the StepIdle defer to enforce
	// monotonicity (FIX 47): if step N's DoGenerate errors before the
	// step is appended to `steps`, len(steps) is N-1 but we already
	// advertised StepLLMInFlight at N, so the final StepIdle must carry
	// max(len(steps), highestInflightStep) to avoid a step-counter
	// regression visible to pollers.
	var highestInflightStep int
	// AgentState: initial (StepStarting, 0). set() is a no-op when o.StateRef is nil.
	o.StateRef.set(StepStarting, 0)
	defer func() {
		finalStep := len(steps)
		if highestInflightStep > finalStep {
			finalStep = highestInflightStep
		}
		o.StateRef.set(StepIdle, finalStep)
	}()

	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	if o.Prompt == "" && len(o.Messages) == 0 {
		return nil, errors.New("goai: prompt or messages must not be empty")
	}

	if o.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.Timeout)
		defer cancel()
	}

	params := buildParams(o)
	originalLen := len(params.Messages)

	// Build tool lookup for auto loop.
	toolMap := buildToolMap(o.Tools)

	var totalUsage provider.Usage
	var hookStopped bool             // true iff WithStopWhen or OnBeforeStep.Stop broke the loop
	var stopCause provider.StopCause // classifies how the loop exited (FIX 5)

	for step := 1; step <= o.MaxSteps; step++ {
		// OnBeforeStep: step 2+ only (step 1 has no prior tool results to act on).
		if step > 1 && o.OnBeforeStep != nil {
			var bsr BeforeStepResult
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Fprintf(os.Stderr, "goai: recovered panic in OnBeforeStep hook: %v\n", r)
					}
				}()
				bsr = o.OnBeforeStep(BeforeStepInfo{
					Ctx:      ctx,
					Step:     step,
					Messages: slices.Clone(params.Messages),
				})
			}()
			if bsr.Stop {
				// Semantic parity with WithStopWhen: mark as hookStopped so
				// post-loop StepsExhausted derivation does not mistake a
				// hook-driven break at the MaxSteps boundary for natural exhaustion.
				hookStopped = true
				stopCause = provider.StopCauseBeforeStep
				break
			}
			if len(bsr.ExtraMessages) > 0 {
				params.Messages = append(params.Messages, bsr.ExtraMessages...)
			}
		}

		for _, fn := range o.OnRequest {
			fn(RequestInfo{
				Ctx:          ctx,
				Model:        model.ModelID(),
				MessageCount: len(params.Messages),
				ToolCount:    len(params.Tools),
				Timestamp:    time.Now(),
				Messages:     requestMessages(params.System, params.Messages),
			})
		}

		start := time.Now()
		o.StateRef.set(StepLLMInFlight, step)
		highestInflightStep = step
		result, err := withRetry(ctx, o.MaxRetries, func() (*provider.GenerateResult, error) {
			return model.DoGenerate(ctx, params)
		})

		for _, fn := range o.OnResponse {
			func(f func(ResponseInfo)) {
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
				f(info)
			}(fn)
		}

		if err != nil {
			// Fire OnFinish so user hooks can observe error termination.
			// Observability hooks (OTel/Langfuse) already handle this via OnResponse
			// error -> end(), but user-registered OnFinish hooks need this signal.
			lastFinish := provider.FinishReason("")
			if len(steps) > 0 {
				lastFinish = steps[len(steps)-1].FinishReason
			}
			fireOnFinish(o.OnFinish, FinishInfo{
				TotalSteps:   len(steps),
				TotalUsage:   totalUsage,
				FinishReason: lastFinish,
				StoppedBy:    provider.StopCauseAbort,
			})
			return nil, err
		}

		stepResult := StepResult{
			Number:           step,
			Text:             result.Text,
			Reasoning:        result.Reasoning,
			ToolCalls:        result.ToolCalls,
			FinishReason:     result.FinishReason,
			Usage:            result.Usage,
			Response:         result.Response,
			ProviderMetadata: result.ProviderMetadata,
			Sources:          result.Sources,
		}
		steps = append(steps, stepResult)
		totalUsage = addUsage(totalUsage, result.Usage)

		// AgentState: LLM call for this step is complete; tool exec and
		// stop-predicate evaluation have not started yet. Pollers observing
		// in this window must see StepStepFinished, not StepLLMInFlight.
		o.StateRef.set(StepStepFinished, step)

		for _, fn := range o.OnStepFinish {
			func(f func(StepResult)) {
				defer func() {
					if r := recover(); r != nil {
						fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
					}
				}()
				f(stepResult)
			}(fn)
		}

		// If no tools have Execute functions, skip the tool loop regardless of MaxSteps.
		// This allows callers to provide tool definitions for the model's awareness
		// without requiring executable tools.
		// No empty-step guard needed (unlike streaming): DoGenerate returns content or error.
		// Normalize: providers that send empty/wrong finish_reason with tool calls
		// (MiniMax, Azure MaaS deepseek, etc.) would cause the loop to exit early.
		// The presence of tool calls is authoritative.
		if len(result.ToolCalls) > 0 && result.FinishReason != provider.FinishToolCalls {
			result.FinishReason = provider.FinishToolCalls
		}

		if result.FinishReason != provider.FinishToolCalls || len(result.ToolCalls) == 0 || len(toolMap) == 0 {
			tr := buildTextResult(steps, totalUsage)
			tr.ResponseMessages = buildResponseMessages(params.Messages[originalLen:], steps, nil)
			// Distinguish "model stopped on its own" from "model wants more
			// tool calls but no tool has Execute" (FIX 11). Both exit cleanly
			// but mean very different things to consumers.
			cause := provider.StopCauseNatural
			if len(result.ToolCalls) > 0 && len(toolMap) == 0 {
				cause = provider.StopCauseNoExecutableTools
			}
			fireOnFinish(o.OnFinish, FinishInfo{
				TotalSteps:   len(steps),
				TotalUsage:   totalUsage,
				FinishReason: tr.FinishReason,
				StoppedBy:    cause,
			})
			return tr, nil
		}

		// Execute tools and build continuation messages.
		// Clear tool_choice after the first tool step so the model can freely
		// produce a text response on subsequent steps.
		params.ToolChoice = ""
		o.StateRef.set(StepToolExecuting, step)
		toolMessages, toolResults := executeToolsParallel(ctx, result.ToolCalls, toolMap, step, toolHooks{
			sequential:      o.SequentialTools,
			onToolCallStart: o.OnToolCallStart,
			onToolCall:      o.OnToolCall,
			onBeforeExecute: o.OnBeforeToolExecute,
			onAfterExecute:  o.OnAfterToolExecute,
		})
		// Attach ToolResults to the step BEFORE the stop predicate sees it
		// (FIX 7 / Vercel DefaultStepResult parity). steps[-1] is this step.
		steps[len(steps)-1].ToolResults = toolResults

		// Append assistant message with tool calls + tool result messages.
		params.Messages = appendToolRoundTrip(params.Messages, result.Text, nil, result.ToolCalls, toolMessages)

		// WithStopWhen (Vercel parity): evaluated AFTER this step's LLM call
		// AND its tool executions complete. The tool-result messages are
		// already folded into params.Messages, so ResponseMessages produced
		// when the loop breaks here is a valid replay transcript (assistant
		// tool_use paired with matching tool_result). Matches
		// vercel-ai/packages/ai/src/generate-text/generate-text.ts where
		// stopWhen() gates the next iteration only after tools have run.
		if o.StopWhen != nil && stopSafe(o.StopWhen, steps) {
			hookStopped = true
			stopCause = provider.StopCausePredicate
			break
		}
	}

	// Post-loop: reachable when MaxSteps was exhausted OR when OnBeforeStep.Stop=true
	// caused a break. Only set StepsExhausted when MaxSteps was actually reached AND
	// the last step still had tool calls pending (model wanted to continue but was cut
	// short). This matches StreamText's conditional logic and correctly distinguishes
	// "hook stopped" (StepsExhausted=false) from "max steps" (StepsExhausted=true).
	tr := buildTextResult(steps, totalUsage)
	// In the sync path every early-exit cause (Natural / NoExecutableTools /
	// Abort) returns immediately inside the loop, so the only way we reach
	// here with stopCause != "" is via a hookStopped break (BeforeStep /
	// Predicate). The hookStopped guard alone is therefore sufficient; the
	// former `stopCause == ""` check was a redundant belt-and-suspenders
	// that is intentionally dropped here for clarity. Streaming keeps the
	// extra guard because its loop has a Natural-case break that sets
	// stopCause without setting hookStopped - see streamWithToolLoop.
	var stepsExhausted bool
	stopCause, stepsExhausted = finalizeStopCause(hookStopped, stopCause, steps, o.MaxSteps)
	tr.StepsExhausted = stepsExhausted
	tr.ResponseMessages = buildResponseMessages(params.Messages[originalLen:], steps, nil)
	fireOnFinish(o.OnFinish, FinishInfo{
		StepsExhausted: tr.StepsExhausted,
		TotalSteps:     len(steps),
		TotalUsage:     totalUsage,
		FinishReason:   tr.FinishReason,
		StoppedBy:      stopCause,
	})
	return tr, nil
}

// fireOnFinish calls all OnFinish hooks with individual panic recovery.
// stopSafe evaluates a StopCondition with recover. A panicking predicate is
// treated as "do not stop" and logged to stderr (consistent with how
// OnBeforeStep / OnStepFinish handle panics).
// finalizeStopCause classifies the terminal StopCause when the tool loop
// exits by natural termination (no break set a cause). It encapsulates the
// post-loop MaxSteps-exhaustion guard and the StopCauseNatural default so
// both GenerateText and streamWithToolLoop share one implementation.
//
// Returns the resolved StopCause and whether MaxSteps was exhausted.
func finalizeStopCause(hookStopped bool, current provider.StopCause, steps []StepResult, maxSteps int) (provider.StopCause, bool) {
	stepsExhausted := false
	if !hookStopped && current == "" && len(steps) >= maxSteps && len(steps) > 0 && len(steps[len(steps)-1].ToolCalls) > 0 {
		stepsExhausted = true
		current = provider.StopCauseMaxSteps
	}
	if current == "" {
		current = provider.StopCauseNatural
	}
	return current, stepsExhausted
}

func stopSafe(pred StopCondition, steps []StepResult) (stop bool) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "goai: recovered panic in StopWhen predicate: %v\n", r)
			stop = false
		}
	}()
	// Pass a shallow defensive copy so predicates cannot re-order or truncate
	// the internal slice. NOTE: this is a TOP-LEVEL copy only - nested slices
	// (StepResult.ToolCalls, StepResult.ToolResults, StepResult.Content) are
	// ALIASED into the caller's view. Predicates MUST treat the StepResult
	// contents as read-only; mutating nested slices corrupts goai internal
	// state and is a contract violation. Deep-clone would be prohibitively
	// expensive per-step for a feature (predicate side-effects) that is not
	// a supported use case.
	return pred(slices.Clone(steps))
}

func fireOnFinish(hooks []func(FinishInfo), info FinishInfo) {
	for _, fn := range hooks {
		func(f func(FinishInfo)) {
			defer func() {
				if r := recover(); r != nil {
					fmt.Fprintf(os.Stderr, "goai: recovered panic in OnFinish hook: %v\n", r)
				}
			}()
			f(info)
		}(fn)
	}
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

// toolHooks bundles the hook functions and options passed to executeToolsParallel.
type toolHooks struct {
	sequential         bool // when true, execute tools one at a time
	onToolCallStart    []func(ToolCallStartInfo)
	onToolCall         []func(ToolCallInfo)
	onBeforeExecute    func(BeforeToolExecuteInfo) BeforeToolExecuteResult
	onAfterExecute     func(AfterToolExecuteInfo) AfterToolExecuteResult
}

func executeToolsParallel(
	ctx context.Context,
	calls []provider.ToolCall,
	toolMap map[string]Tool,
	step int,
	hooks toolHooks,
) ([]provider.Message, []provider.ToolResult) {

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
			for _, fn := range hooks.onToolCallStart {
				func(f func(ToolCallStartInfo)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(ToolCallStartInfo{ToolCallID: tc.ID, ToolName: tc.Name, Step: step, Input: tc.Input})
				}(fn)
			}
			for _, fn := range hooks.onToolCall {
				func(f func(ToolCallInfo)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(ToolCallInfo{
						ToolCallID: tc.ID,
						ToolName:   tc.Name,
						Step:       step,
						Input:      tc.Input,
						Error:      ErrUnknownTool,
					})
				}(fn)
			}
			continue
		}

		toolFn := func(i int, tc provider.ToolCall, tool Tool) {
			if !hooks.sequential {
				defer wg.Done()
			}
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

			// OnToolCallStart: pre-execution (each independently recover-wrapped).
			var hookPanicked bool
			for _, fn := range hooks.onToolCallStart {
				func(f func(ToolCallStartInfo)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
							hookPanicked = true
						}
					}()
					f(ToolCallStartInfo{
						ToolCallID: tc.ID,
						ToolName:   tc.Name,
						Step:       step,
						Input:      tc.Input,
					})
				}(fn)
			}
			hookFired = true
			if hookPanicked {
				panicStr := fmt.Sprintf("goai: OnToolCallStart hook for tool %q panicked", tc.Name)
				results[i] = toolOutput{index: i, err: fmt.Errorf("%s", panicStr)}
				return
			}

			// Create tool context early so both hooks and Execute share it.
			toolCtx := context.WithValue(ctx, toolCallIDKey{}, tc.ID)

			// OnBeforeToolExecute: can skip execution (permission, doom loop, etc.).
			if hooks.onBeforeExecute != nil {
				var beforeResult BeforeToolExecuteResult
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in OnBeforeToolExecute hook: %v\n", r)
							beforeResult = BeforeToolExecuteResult{
								Skip:  true,
								Error: fmt.Errorf("goai: OnBeforeToolExecute hook panicked: %v", r),
							}
						}
					}()
					beforeResult = hooks.onBeforeExecute(BeforeToolExecuteInfo{
						Ctx:        toolCtx,
						ToolCallID: tc.ID,
						ToolName:   tc.Name,
						Step:       step,
						Input:      tc.Input,
					})
				}()
				if beforeResult.Skip {
					executed = true // prevent outer panic handler from overwriting
					if beforeResult.Error != nil {
						results[i] = toolOutput{index: i, result: beforeResult.Result, err: beforeResult.Error}
					} else {
						results[i] = toolOutput{index: i, result: beforeResult.Result}
					}
					// Fire OnToolCall with skipped result for observability.
					for _, fn := range hooks.onToolCall {
						func(f func(ToolCallInfo)) {
							defer func() {
								if r := recover(); r != nil {
									fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
								}
							}()
							f(ToolCallInfo{
								ToolCallID: tc.ID,
								ToolName:   tc.Name,
								Step:       step,
								Input:      tc.Input,
								Output:     beforeResult.Result,
								StartTime:  time.Now(),
								Skipped:    true,
								Error:      beforeResult.Error,
							})
						}(fn)
					}
					return
				}
				// Apply hook overrides for non-skipped tools.
				if beforeResult.Ctx != nil {
					toolCtx = beforeResult.Ctx
				}
				if beforeResult.Input != nil {
					tc.Input = beforeResult.Input
				}
			}

			start := time.Now()
			output, err := tool.Execute(toolCtx, tc.Input)
			executed = true

			// OnAfterToolExecute: can modify output (secret scanning, truncation, etc.).
			var afterMetadata map[string]any
			if hooks.onAfterExecute != nil {
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in OnAfterToolExecute hook: %v\n", r)
							// Preserve original result on panic.
						}
					}()
					afterResult := hooks.onAfterExecute(AfterToolExecuteInfo{
						Ctx:        toolCtx,
						ToolCallID: tc.ID,
						ToolName:   tc.Name,
						Step:       step,
						Input:      tc.Input,
						Output:     output,
						Error:      err,
					})
					if afterResult.Output != "" {
						output = afterResult.Output
					}
					if afterResult.Error != nil {
						err = afterResult.Error
					}
					afterMetadata = afterResult.Metadata
				}()
			}

			results[i] = toolOutput{index: i, result: output, err: err}

			// OnToolCall: post-execution (each independently recover-wrapped).
			for _, fn := range hooks.onToolCall {
				func(f func(ToolCallInfo)) {
					defer func() {
						if r := recover(); r != nil {
							fmt.Fprintf(os.Stderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					info := ToolCallInfo{
						ToolCallID: tc.ID,
						ToolName:   tc.Name,
						Step:       step,
						Input:      tc.Input,
						Output:     output,
						StartTime:  start,
						Duration:   time.Since(start),
						Error:      err,
						Metadata:   afterMetadata,
					}
					var parsed any
					if err == nil && json.Unmarshal([]byte(output), &parsed) == nil {
						info.OutputObject = parsed
					}
					f(info)
				}(fn)
			}
		}
		if hooks.sequential {
			toolFn(i, tc, tool)
		} else {
			wg.Add(1)
			go toolFn(i, tc, tool)
		}
	}

	if !hooks.sequential {
		wg.Wait()
	}
	return buildToolMessages(calls, results), buildToolResults(calls, results)
}

// buildToolResults converts raw tool outputs into structured provider.ToolResult
// values (one per call, in call order). The Output string mirrors exactly what
// buildToolMessages places in the tool-result message content so consumers can
// correlate predicate input with the on-wire transcript.
func buildToolResults(calls []provider.ToolCall, results []toolOutput) []provider.ToolResult {
	if len(calls) == 0 {
		return nil
	}
	out := make([]provider.ToolResult, len(calls))
	for i, tc := range calls {
		r := results[i]
		tr := provider.ToolResult{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Error:      r.err,
			IsError:    r.err != nil,
		}
		if r.err != nil {
			errStr := r.err.Error()
			runes := []rune(errStr)
			if len(runes) > 500 {
				errStr = string(runes[:500]) + "..."
			}
			tr.Output = "error: " + errStr
		} else {
			tr.Output = r.result
		}
		out[i] = tr
	}
	return out
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
// reasoning parts are placed first so providers that require thinking blocks
// (e.g. Bedrock with extended thinking) see them before tool_use content.
func appendToolRoundTrip(
	msgs []provider.Message,
	text string,
	reasoning []provider.Part,
	toolCalls []provider.ToolCall,
	toolMsgs []provider.Message,
) []provider.Message {
	var parts []provider.Part
	// Reasoning first (before text and tool_use). Clone ProviderOptions to avoid
	// aliasing between params.Messages and ResponseMessages.
	for _, r := range reasoning {
		parts = append(parts, provider.Part{
			Type:            r.Type,
			Text:            r.Text,
			ProviderOptions: maps.Clone(r.ProviderOptions),
		})
	}
	if text != "" {
		parts = append(parts, provider.Part{Type: provider.PartText, Text: text})
	}
	for _, tc := range toolCalls {
		parts = append(parts, provider.Part{
			Type:            provider.PartToolCall,
			ToolCallID:      tc.ID,
			ToolName:        tc.Name,
			ToolInput:       append(json.RawMessage(nil), tc.Input...), // clone byte slice
			ProviderOptions: maps.Clone(tc.Metadata),                  // shallow clone (matches buildFinalAssistantMessages)
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
	// Accumulate reasoning text from all steps. Concatenated as-is so
	// callers see the same boundaries the steps had; consumers wanting
	// per-step reasoning can iterate Steps directly.
	var allReasoning strings.Builder
	for _, s := range steps {
		allReasoning.WriteString(s.Reasoning)
	}
	// Collect sources from all steps.
	var allSources []provider.Source
	for _, s := range steps {
		allSources = append(allSources, s.Sources...)
	}

	return &TextResult{
		Text:             allText.String(),
		Reasoning:        allReasoning.String(),
		ToolCalls:        last.ToolCalls,
		Steps:            steps,
		TotalUsage:       totalUsage,
		FinishReason:     last.FinishReason,
		Response:         last.Response,
		ProviderMetadata: last.ProviderMetadata,
		Sources:          allSources,
	}
}

// buildResponseMessages constructs the full ResponseMessages from the tool round-trip
// messages (delta between original and final params.Messages) and the steps.
//
// With Vercel-parity StopWhen placement (evaluated AFTER tool execution and
// appendToolRoundTrip), the delta always contains the assistant + tool-result
// messages for every completed step whose LLM response carried tool calls --
// including the last step on a StopWhen break and at MaxSteps exhaustion. The
// only case the delta is missing the last step's assistant message is a
// natural termination (last step produced text with no tool calls): that
// message is appended here.
//
// Consecutive tool messages are merged into a single message because some
// providers require parallel tool results in a single message (not split
// across multiple messages).
//
// The reasoning parameter provides thinking/reasoning parts for the final
// assistant message (streaming path only; GenerateText passes nil). It is
// only applied when the delta is empty or the last step had no tool calls
// (i.e. when we are actually building the final assistant message here).
func buildResponseMessages(roundTripDelta []provider.Message, steps []StepResult, reasoning []provider.Part) []provider.Message {
	if len(steps) == 0 {
		return nil
	}
	last := steps[len(steps)-1]
	if len(roundTripDelta) == 0 {
		return buildFinalAssistantMessages(last.Text, last.ToolCalls, reasoning)
	}
	msgs := mergeToolMessages(roundTripDelta)
	if len(last.ToolCalls) > 0 {
		// Delta already contains this step's assistant + tool-result messages
		// (appendToolRoundTrip ran before loop break). Avoid duplication.
		return msgs
	}
	// Natural termination: last step produced text with no tool calls - its
	// assistant message is NOT yet in the delta.
	finalMsg := buildFinalAssistantMessages(last.Text, last.ToolCalls, reasoning)
	return append(msgs, finalMsg...)
}

// mergeToolMessages merges consecutive tool-role messages into single messages.
// The internal tool loop creates one message per tool call, but callers expect
// parallel tool results grouped in a single message per round-trip.
func mergeToolMessages(msgs []provider.Message) []provider.Message {
	out := make([]provider.Message, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == provider.RoleTool && len(out) > 0 && out[len(out)-1].Role == provider.RoleTool {
			// Merge parts into the previous tool message.
			out[len(out)-1].Content = append(out[len(out)-1].Content, m.Content...)
		} else {
			// Clone the message to avoid mutating the original.
			out = append(out, provider.Message{
				Role:    m.Role,
				Content: slices.Clone(m.Content),
			})
		}
	}
	return out
}

// buildFinalAssistantMessages builds a single assistant message from text, tool calls,
// and/or reasoning parts. Returns nil when all inputs are empty.
// Reasoning parts are placed first so providers that require thinking blocks
// (e.g. Bedrock with extended thinking) see them before text/tool_use content.
func buildFinalAssistantMessages(text string, toolCalls []provider.ToolCall, reasoning []provider.Part) []provider.Message {
	var parts []provider.Part
	parts = append(parts, reasoning...)
	if text != "" {
		parts = append(parts, provider.Part{Type: provider.PartText, Text: text})
	}
	for _, tc := range toolCalls {
		parts = append(parts, provider.Part{
			Type:       provider.PartToolCall,
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			ToolInput:  append(json.RawMessage(nil), tc.Input...),
			// Shallow copy of Metadata , matches the existing appendToolRoundTrip pattern.
			// Nested map values are shared, not deep-cloned.
			ProviderOptions: maps.Clone(tc.Metadata),
		})
	}
	if len(parts) == 0 {
		return nil
	}
	return []provider.Message{{Role: provider.RoleAssistant, Content: parts}}
}

type drainResult struct {
	text          string          // text-only (ChunkText), used for appendToolRoundTrip
	reasoning     []provider.Part // reasoning/thinking parts, echoed back for providers that require it (e.g. Bedrock)
	reasoningText string          // consolidated reasoning text (PartReasoning), surfaced via StepResult.Reasoning
	toolCalls     []provider.ToolCall
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
		textBuf      strings.Builder // ChunkText only (reasoning excluded)
		reasoningBuf strings.Builder // accumulated reasoning text
		reasoningMeta map[string]any // last metadata (contains signature)
		dr           drainResult
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
			// Accumulate into a single buffer. The final chunk carries the
			// signature (text="", metadata={"signature":"..."}); earlier chunks
			// carry text fragments. Consolidating produces one complete part.
			if chunk.Text != "" {
				reasoningBuf.WriteString(chunk.Text)
			}
			if chunk.Metadata != nil {
				if reasoningMeta == nil {
					reasoningMeta = make(map[string]any)
				}
				for k, v := range chunk.Metadata {
					reasoningMeta[k] = v
				}
			}
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				dr.sources = append(dr.sources, s)
			}
		case provider.ChunkToolCall:
			dr.toolCalls = append(dr.toolCalls, provider.ToolCall{
				ID:       chunk.ToolCallID,
				Name:     chunk.ToolName,
				Input:    json.RawMessage(chunk.ToolInput),
				Metadata: chunk.Metadata,
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
	dr.reasoningText = reasoningBuf.String()

	// Consolidate reasoning fragments into one Part (text + signature metadata)
	// so the codec serializes a single complete block.
	if reasoningBuf.Len() > 0 || len(reasoningMeta) > 0 {
		dr.reasoning = []provider.Part{{
			Type:            provider.PartReasoning,
			Text:            reasoningBuf.String(),
			ProviderOptions: reasoningMeta,
		}}
	}

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
