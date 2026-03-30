package goai

import (
	"context"
	"encoding/json"
	"errors"
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
	// Text is the accumulated generated text.
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

	// Text generated in this step.
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
	onResponse func(ResponseInfo)
	startTime  time.Time

	// Accumulated state (written by consume goroutine, read after doneCh closes).
	text             strings.Builder
	toolCalls        []provider.ToolCall
	sources          []provider.Source
	finishReason     provider.FinishReason
	usage            provider.Usage
	response         provider.ResponseMetadata
	providerMetadata map[string]map[string]any
	streamErr        error
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

// TextStream returns a channel that emits only text content strings.
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

// Result blocks until the stream completes and returns the final result.
// Can be called after Stream() or TextStream() to get accumulated data.
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

	// Call OnResponse hook when consume finishes (after all chunks processed).
	if ts.onResponse != nil {
		defer func() {
			ts.onResponse(ResponseInfo{
				Latency:      time.Since(ts.startTime),
				Usage:        ts.usage,
				FinishReason: ts.finishReason,
				Error:        ts.streamErr,
			})
		}()
	}

	for chunk := range ts.source {
		switch chunk.Type {
		case provider.ChunkText, provider.ChunkReasoning:
			ts.text.WriteString(chunk.Text)
			// Capture per-annotation sources (e.g. url_citation).
			if s, ok := chunk.Metadata["source"].(provider.Source); ok {
				ts.sources = append(ts.sources, s)
			}
		case provider.ChunkToolCall:
			ts.toolCalls = append(ts.toolCalls, provider.ToolCall{
				ID:    chunk.ToolCallID,
				Name:  chunk.ToolName,
				Input: json.RawMessage(chunk.ToolInput),
			})
		case provider.ChunkFinish, provider.ChunkStepFinish:
			ts.finishReason = chunk.FinishReason
			ts.usage = addUsage(ts.usage, chunk.Usage)
			ts.response = chunk.Response
			// Capture top-level citations (xAI, Perplexity).
			if sources, ok := chunk.Metadata["sources"].([]provider.Source); ok {
				ts.sources = append(ts.sources, sources...)
			}
			// Extract provider metadata embedded in the finish chunk.
			// openaicompat encodes it as Metadata["providerMetadata"] = map[string]map[string]any.
			// The type assertion safely returns false for any other type (no panic).
			if pm, ok := chunk.Metadata["providerMetadata"].(map[string]map[string]any); ok {
				ts.providerMetadata = pm
			}
			// Copy remaining flat metadata keys into Response.ProviderMetadata.
			// Providers use this for per-response data: Anthropic ("iterations",
			// "contextManagement", "container"), Bedrock ("cacheWriteInputTokens").
			// Skip "providerMetadata" (handled above) and "sources" (handled separately).
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
				return
			}
		}
		if textOut != nil && chunk.Type == provider.ChunkText {
			select {
			case textOut <- chunk.Text:
			case <-ts.ctx.Done():
				return
			}
		}
		if rawOut == nil && textOut == nil {
			if ts.ctx.Err() != nil {
				return
			}
		}
	}
}

func (ts *TextStream) buildResult() *TextResult {
	text := ts.text.String()
	return &TextResult{
		Text:             text,
		ToolCalls:        ts.toolCalls,
		FinishReason:     ts.finishReason,
		TotalUsage:       ts.usage,
		Response:         ts.response,
		Sources:          ts.sources,
		ProviderMetadata: ts.providerMetadata,
		Steps: []StepResult{{
			Number:           1,
			Text:             text,
			ToolCalls:        ts.toolCalls,
			FinishReason:     ts.finishReason,
			Usage:            ts.usage,
			Response:         ts.response,
			Sources:          ts.sources,
			ProviderMetadata: ts.providerMetadata,
		}},
	}
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
		StopSequences:    opts.StopSequences,
		Headers:          opts.Headers,
		ProviderOptions:  opts.ProviderOptions,
		PromptCaching:    opts.PromptCaching,
		ToolChoice:       opts.ToolChoice,
	}
}

// StreamText performs a streaming text generation.
func StreamText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextStream, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := applyOptions(opts...)

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
			o.OnStepFinish(stepResult)
		}

		// If no tool calls or no executable tools, we're done.
		if result.FinishReason != provider.FinishToolCalls || len(result.ToolCalls) == 0 || len(toolMap) == 0 {
			return buildTextResult(steps, totalUsage), nil
		}

		// Execute tools and build continuation messages.
		toolMessages := executeTools(ctx, result.ToolCalls, toolMap, step, o.OnToolCall)

		// Append assistant message with tool calls + tool result messages.
		params.Messages = appendToolRoundTrip(params.Messages, result, toolMessages)
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
			m[t.Name] = t
		}
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// executeTools runs each tool call and returns the tool result messages.
func executeTools(ctx context.Context, calls []provider.ToolCall, toolMap map[string]Tool, step int, onToolCall func(ToolCallInfo)) []provider.Message {
	var msgs []provider.Message
	for _, tc := range calls {
		if ctx.Err() != nil {
			return msgs
		}
		tool, ok := toolMap[tc.Name]
		if !ok {
			if onToolCall != nil {
				onToolCall(ToolCallInfo{ToolCallID: tc.ID, ToolName: tc.Name, Step: step, Input: tc.Input, Error: ErrUnknownTool})
			}
			msgs = append(msgs, ToolMessage(tc.ID, tc.Name, "error: unknown tool"))
			continue
		}
		start := time.Now()
		output, err := tool.Execute(ctx, tc.Input)
		if onToolCall != nil {
			info := ToolCallInfo{ToolCallID: tc.ID, ToolName: tc.Name, Step: step, Input: tc.Input, Output: output, StartTime: start, Duration: time.Since(start), Error: err}
			var parsed any
			if err == nil && json.Unmarshal([]byte(output), &parsed) == nil {
				info.OutputObject = parsed
			}
			onToolCall(info)
		}
		if err != nil {
			msgs = append(msgs, ToolMessage(tc.ID, tc.Name, "error: "+err.Error()))
			continue
		}
		msgs = append(msgs, ToolMessage(tc.ID, tc.Name, output))
	}
	return msgs
}

// appendToolRoundTrip appends an assistant message (with tool_use parts) and tool result messages.
func appendToolRoundTrip(msgs []provider.Message, result *provider.GenerateResult, toolMsgs []provider.Message) []provider.Message {
	// Build assistant message with text + tool_use parts.
	var parts []provider.Part
	if result.Text != "" {
		parts = append(parts, provider.Part{Type: provider.PartText, Text: result.Text})
	}
	for _, tc := range result.ToolCalls {
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
