package goai

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// osStderr holds os.Stderr at package level so ObjectStream methods can write to
// stderr despite their receiver being named "os", which shadows the os package.
var osStderr = os.Stderr

// Per-function once flags for GenerateObject and StreamObject warnings
// (FIX 12). A single package-global once would fire only once total across
// both functions, silencing the warning for whichever function is called
// second - even though the caller passed WithStopWhen to a different entry
// point. Each function now emits its warning at most once per process.
//
// Implementation (FIX 21): uses atomic.Bool CAS instead of sync.Once so that
// tests can reset the flag cleanly without the "reassign a sync.Once" hazard
// (which is technically unsound if observed concurrently by another goroutine).
// atomic.Bool.Store(false) is a well-defined reset. The warn function itself
// is also swappable via warnStopWhenIgnoredForObject for tests that want to
// inject a capture instead of stderr plumbing.
//
// FIX 33 - reset caveat: unlike sync.Once, a CAS-guarded atomic.Bool can be
// reset with Store(false). This is safe ONLY when no other goroutine may
// concurrently call defaultWarnStopWhenIgnoredForObject (i.e. the reset site
// externally serializes with all warn callers). Tests in this package call
// Store(false) from the test goroutine while the warning code path is
// quiescent; they do NOT use t.Parallel() for exactly this reason. DO NOT
// add t.Parallel() to any test that resets these flags, and DO NOT reset
// these flags from production code.
var (
	generateObjectStopWhenWarned atomic.Bool
	streamObjectStopWhenWarned   atomic.Bool
	generateObjectStateRefWarned atomic.Bool
	streamObjectStateRefWarned   atomic.Bool
)

// warnStopWhenIgnoredForObject is a package-level function variable so that
// tests can swap in a capture implementation if needed. Production code path
// uses defaultWarnStopWhenIgnoredForObject.
var warnStopWhenIgnoredForObject = defaultWarnStopWhenIgnoredForObject

// warnStateRefIgnoredForObject mirrors warnStopWhenIgnoredForObject: a
// package-level function variable swappable in tests. Production path uses
// defaultWarnStateRefIgnoredForObject. See FIX 35.
var warnStateRefIgnoredForObject = defaultWarnStateRefIgnoredForObject

func defaultWarnStopWhenIgnoredForObject(fn string) {
	msg := fmt.Sprintf("goai: WithStopWhen is not supported in %s (ignored)\n", fn)
	switch fn {
	case "GenerateObject":
		if generateObjectStopWhenWarned.CompareAndSwap(false, true) {
			_, _ = fmt.Fprint(osStderr, msg)
		}
	case "StreamObject":
		if streamObjectStopWhenWarned.CompareAndSwap(false, true) {
			_, _ = fmt.Fprint(osStderr, msg)
		}
	default:
		// Unknown call site: print once per call (should never happen - all
		// call sites pass a hard-coded literal).
		_, _ = fmt.Fprint(osStderr, msg)
	}
}

// defaultWarnStateRefIgnoredForObject is invoked from GenerateObject and
// StreamObject when the caller passed WithStateRef. Those functions do not
// run a multi-step tool loop worth polling, so the AgentState would be
// stuck at its zero value (StepStarting, 0) forever - a subtle deadlock for
// pollers. One-shot stderr warning matches the WithStopWhen-in-object
// convention above (FIX 35). The FIX 33 reset caveat applies identically.
func defaultWarnStateRefIgnoredForObject(fn string) {
	msg := fmt.Sprintf("goai: WithStateRef is not supported in %s (ignored - AgentState will not advance)\n", fn)
	switch fn {
	case "GenerateObject":
		if generateObjectStateRefWarned.CompareAndSwap(false, true) {
			_, _ = fmt.Fprint(osStderr, msg)
		}
	case "StreamObject":
		if streamObjectStateRefWarned.CompareAndSwap(false, true) {
			_, _ = fmt.Fprint(osStderr, msg)
		}
	default:
		_, _ = fmt.Fprint(osStderr, msg)
	}
}

// ObjectResult is the final result of a structured output generation.
type ObjectResult[T any] struct {
	// Object is the parsed structured output.
	Object T

	// Usage tracks total token consumption across all steps.
	Usage provider.Usage

	// FinishReason indicates why generation stopped.
	FinishReason provider.FinishReason

	// Response contains provider metadata (ID, Model).
	Response provider.ResponseMetadata

	// ProviderMetadata contains provider-specific response data.
	ProviderMetadata map[string]map[string]any

	// ResponseMessages contains the assistant and tool messages from all generation steps.
	// For multi-turn conversations, append these to your message history:
	//   messages = append(messages, result.ResponseMessages...)
	//
	// Nil when the response has no content.
	// Reasoning parts are not included (GenerateObject uses non-streaming DoGenerate,
	// and StreamObject does not capture reasoning chunks).
	ResponseMessages []provider.Message

	// Steps contains results from each generation step (for multi-step tool loops).
	// The final step is always the structured output step.
	Steps []StepResult
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
	onResponse   []func(ResponseInfo)
	onStepFinish []func(StepResult)
	onFinish     []func(FinishInfo)
	startTime    time.Time

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
// Mutually exclusive with Result() - only call one consumption method first.
func (os *ObjectStream[T]) PartialObjectStream() <-chan *T {
	ch := make(chan *T, 64)
	os.consumeOnce.Do(func() {
		os.partialCh = ch
		go os.consume(ch)
	})
	if os.partialCh != nil {
		return os.partialCh
	}
	// Called after Result() consumed the source - return closed channel.
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
		// Raw model output is truncated to 200 chars to limit information disclosure.
		return nil, fmt.Errorf("parsing structured output: %w (raw: %s)", os.parseErr, truncate(os.text.String(), 200))
	}

	text := os.text.String()
	var responseMessages []provider.Message
	if text != "" {
		responseMessages = buildFinalAssistantMessages(text, nil, nil)
	}

	if os.finalObject == nil {
		return &ObjectResult[T]{
			Usage:            os.usage,
			FinishReason:     os.finishReason,
			Response:         os.response,
			ProviderMetadata: os.providerMetadata,
			ResponseMessages: responseMessages,
		}, nil
	}
	return &ObjectResult[T]{
		Object:           *os.finalObject,
		Usage:            os.usage,
		FinishReason:     os.finishReason,
		Response:         os.response,
		ProviderMetadata: os.providerMetadata,
		ResponseMessages: responseMessages,
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

	// Call OnFinish hook when consume finishes (single-step streaming).
	// Deferred BEFORE OnStepFinish so it runs AFTER it (LIFO order).
	if len(os.onFinish) > 0 {
		defer func() {
			fireOnFinish(os.onFinish, FinishInfo{
				TotalSteps:   1,
				TotalUsage:   os.usage,
				FinishReason: os.finishReason,
				StoppedBy:    provider.StopCauseNatural,
			})
		}()
	}

	// Call OnStepFinish hook when consume finishes (single-step streaming).
	// Deferred BEFORE OnResponse so it runs AFTER it (LIFO order).
	if len(os.onStepFinish) > 0 {
		defer func() {
			stepResult := StepResult{
				Number:       1,
				Text:         os.text.String(),
				FinishReason: os.finishReason,
				Usage:        os.usage,
				Response:     os.response,
				ProviderMetadata: os.providerMetadata,
			}
			for _, fn := range os.onStepFinish {
				func(f func(StepResult)) {
					defer func() {
						if r := recover(); r != nil {
							_, _ = fmt.Fprintf(osStderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					f(stepResult)
				}(fn)
			}
		}()
	}

	// Call OnResponse hook when consume finishes (after all chunks processed).
	if len(os.onResponse) > 0 {
		defer func() {
			info := ResponseInfo{
				Latency:      time.Since(os.startTime),
				Usage:        os.usage,
				FinishReason: os.finishReason,
				Error:        os.streamErr,
			}
			var apiErr *APIError
			if errors.As(os.streamErr, &apiErr) {
				info.StatusCode = apiErr.StatusCode
			}
			for _, fn := range os.onResponse {
				func() {
					defer func() {
						if r := recover(); r != nil {
							_, _ = fmt.Fprintf(osStderr, "goai: recovered panic in hook: %v\n", r)
						}
					}()
					fn(info)
				}()
			}
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
						os.streamErr = os.ctx.Err()
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
				os.streamErr = os.ctx.Err()
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
//
// When tools with Execute functions are provided and MaxSteps > 1, GenerateObject
// runs a tool loop (identical to GenerateText) with ResponseFormat set on every
// step. The model decides when to call tools vs produce the final JSON output.
// Structured output is parsed from whichever step returns finishReason "stop".
//
// If MaxSteps is exhausted before a "stop" step occurs, an error is returned.
// This differs slightly from Vercel AI SDK, which returns a partial result with
// a nil output field - in Go, returning an error is the idiomatic equivalent.
func GenerateObject[T any](ctx context.Context, model provider.LanguageModel, opts ...Option) (*ObjectResult[T], error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.StopWhen != nil {
		warnStopWhenIgnoredForObject("GenerateObject")
	}
	if o.StateRef != nil {
		warnStateRefIgnoredForObject("GenerateObject")
	}

	if o.Prompt == "" && len(o.Messages) == 0 {
		return nil, errors.New("goai: prompt or messages must not be empty")
	}

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
	originalLen := len(params.Messages)
	// ResponseFormat is set upfront and sent on every step - the model decides
	// when to call tools vs return structured JSON. This mirrors Vercel AI SDK
	// where output parsing happens only on the step with finishReason "stop".
	params.ResponseFormat = &provider.ResponseFormat{
		Name:   schemaName,
		Schema: schema,
	}

	toolMap := buildToolMap(o.Tools)

	var steps []StepResult
	var totalUsage provider.Usage

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
		result, err := withRetry(ctx, o.MaxRetries, func() (*provider.GenerateResult, error) {
			return model.DoGenerate(ctx, params)
		})

		for _, fn := range o.OnResponse {
			func(f func(ResponseInfo)) {
				defer func() {
					if r := recover(); r != nil {
						_, _ = fmt.Fprintf(osStderr, "goai: recovered panic in hook: %v\n", r)
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
			ToolCalls:        result.ToolCalls,
			FinishReason:     result.FinishReason,
			Usage:            result.Usage,
			Response:         result.Response,
			ProviderMetadata: result.ProviderMetadata,
			Sources:          result.Sources,
		}
		steps = append(steps, stepResult)
		totalUsage = addUsage(totalUsage, result.Usage)

		for _, fn := range o.OnStepFinish {
			func(f func(StepResult)) {
				defer func() {
					if r := recover(); r != nil {
						_, _ = fmt.Fprintf(osStderr, "goai: recovered panic in hook: %v\n", r)
					}
				}()
				f(stepResult)
			}(fn)
		}

		// If no tools have Execute functions, skip the tool loop regardless of MaxSteps.
		// This allows callers to provide tool definitions for the model's awareness
		// without requiring executable tools.
		// Parse structured output only when the model finished with "stop"
		// (not tool calls). This is identical to Vercel's behaviour.
		if result.FinishReason != provider.FinishToolCalls || len(result.ToolCalls) == 0 || len(toolMap) == 0 {
			text := result.Text
			if text == "" {
				return nil, fmt.Errorf("goai: empty response from model")
			}
			var obj T
			if err := json.Unmarshal([]byte(text), &obj); err != nil {
				// Raw model output is truncated to 200 chars to limit information disclosure.
				return nil, fmt.Errorf("parsing structured output: %w (raw: %s)", err, truncate(text, 200))
			}
			fireOnFinish(o.OnFinish, FinishInfo{
				TotalSteps:   len(steps),
				TotalUsage:   totalUsage,
				FinishReason: result.FinishReason,
				StoppedBy:    provider.StopCauseNatural,
			})
			return &ObjectResult[T]{
				Object:           obj,
				Usage:            totalUsage,
				FinishReason:     result.FinishReason,
				Response:         result.Response,
				ProviderMetadata: result.ProviderMetadata,
				ResponseMessages: buildResponseMessages(params.Messages[originalLen:], steps, nil),
				Steps:            steps,
			}, nil
		}

		// Model requested tool calls - execute them and continue.
		// Clear tool_choice after the first tool step so the model can freely
		// produce structured output on subsequent steps.
		params.ToolChoice = ""
		toolMessages, _ := executeToolsParallel(ctx, result.ToolCalls, toolMap, step, toolHooks{
			sequential:      o.SequentialTools,
			onToolCallStart: o.OnToolCallStart,
			onToolCall:      o.OnToolCall,
			onBeforeExecute: o.OnBeforeToolExecute,
			onAfterExecute:  o.OnAfterToolExecute,
		})
		params.Messages = appendToolRoundTrip(params.Messages, result.Text, nil, result.ToolCalls, toolMessages)
	}

	// MaxSteps exhausted with tool calls still pending - no structured output produced.
	lastFinish := provider.FinishReason("")
	if len(steps) > 0 {
		lastFinish = steps[len(steps)-1].FinishReason
	}
	fireOnFinish(o.OnFinish, FinishInfo{
		StepsExhausted: true,
		TotalSteps:     len(steps),
		TotalUsage:     totalUsage,
		FinishReason:   lastFinish,
		StoppedBy:      provider.StopCauseMaxSteps,
	})
	return nil, fmt.Errorf("goai: max steps (%d) reached without producing structured output", o.MaxSteps)
}

// StreamObject performs a streaming structured output generation.
// Returns an ObjectStream that emits progressively populated partial objects.
//
// Unlike GenerateObject, StreamObject is intentionally single-step: it initiates
// one streaming request and returns immediately. Tool loops are not supported
// because the caller consumes the stream asynchronously; use GenerateObject when
// you need tools and multi-step behaviour.
func StreamObject[T any](ctx context.Context, model provider.LanguageModel, opts ...Option) (*ObjectStream[T], error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

	if o.StopWhen != nil {
		warnStopWhenIgnoredForObject("StreamObject")
	}
	if o.StateRef != nil {
		warnStateRefIgnoredForObject("StreamObject")
	}

	if o.Prompt == "" && len(o.Messages) == 0 {
		return nil, errors.New("goai: prompt or messages must not be empty")
	}

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
		for _, fn := range o.OnResponse {
			func(f func(ResponseInfo)) {
				defer func() {
					if r := recover(); r != nil {
						_, _ = fmt.Fprintf(osStderr, "goai: recovered panic in hook: %v\n", r)
					}
				}()
				info := ResponseInfo{Latency: time.Since(start), Error: err}
				var apiErr *APIError
				if errors.As(err, &apiErr) {
					info.StatusCode = apiErr.StatusCode
				}
				f(info)
			}(fn)
		}
		return nil, err
	}

	os := newObjectStream[T](ctx, result.Stream)
	os.timeoutCancel = timeoutCancel
	os.onResponse = o.OnResponse
	os.onStepFinish = o.OnStepFinish
	os.onFinish = o.OnFinish
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
