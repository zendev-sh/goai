// Package otel provides OpenTelemetry tracing and metrics hooks for goai.
//
// Basic usage with global providers:
//
//	result, err := goai.GenerateText(ctx, model,
//	    otel.WithTracing(),
//	    goai.WithPrompt("hello"),
//	)
//
// With custom providers:
//
//	result, err := goai.GenerateText(ctx, model,
//	    otel.WithTracing(
//	        otel.WithTracerProvider(tp),
//	        otel.WithMeterProvider(mp),
//	        otel.WithSpanName("my-agent"),
//	        otel.WithAttributes(attribute.String("user.id", "123")),
//	    ),
//	    goai.WithPrompt("hello"),
//	)
//
// Each call creates a fresh trace with isolated state. Concurrent calls are safe.
//
// Span hierarchy:
//
//	chat (root)
//	├── chat {model} (step-1) - LLM API call
//	├── execute_tool {tool-name} - tool execution
//	└── chat {model} (step-2) - final LLM call
package otel

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

const tracerName = "github.com/zendev-sh/goai/observability/otel"

// config holds the resolved configuration for a single tracing call.
type config struct {
	tracerProvider trace.TracerProvider
	meterProvider  metric.MeterProvider
	spanName       string
	attrs          []attribute.KeyValue
	recordInput    bool
	recordOutput   bool
	recordToolIO   bool // A7: record tool output as span events
}

// TracingOption configures WithTracing.
type TracingOption func(*config)

// WithTracerProvider sets a custom TracerProvider (default: global).
func WithTracerProvider(tp trace.TracerProvider) TracingOption {
	return func(c *config) { c.tracerProvider = tp }
}

// WithMeterProvider sets a custom MeterProvider (default: global).
func WithMeterProvider(mp metric.MeterProvider) TracingOption {
	return func(c *config) { c.meterProvider = mp }
}

// WithSpanName sets the root span name (default: "chat").
func WithSpanName(name string) TracingOption {
	return func(c *config) { c.spanName = name }
}

// WithAttributes adds custom attributes to the root span.
func WithAttributes(attrs ...attribute.KeyValue) TracingOption {
	return func(c *config) { c.attrs = append(c.attrs, attrs...) }
}

// RecordInputMessages controls whether input messages are recorded as span
// events (default: false, for privacy).
func RecordInputMessages(b bool) TracingOption {
	return func(c *config) { c.recordInput = b }
}

// RecordOutputMessages controls whether output messages are recorded as span
// events (default: false, for privacy).
func RecordOutputMessages(b bool) TracingOption {
	return func(c *config) { c.recordOutput = b }
}

// RecordToolIO controls whether tool input/output are recorded as span events
// on tool spans (default: false, for privacy). Similar to RecordInputMessages/
// RecordOutputMessages but for tool execution payloads.
func RecordToolIO(b bool) TracingOption {
	return func(c *config) { c.recordToolIO = b }
}

// instruments holds pre-created metric instruments to avoid repeated creation.
type instruments struct {
	tokenUsage        metric.Int64Histogram
	operationDuration metric.Float64Histogram
	toolDuration      metric.Float64Histogram
	earlyStopCounter  metric.Int64Counter      // D1: loop early stop counter
	messageCount      metric.Int64Gauge         // D4: conversation message count
}

func newInstruments(mp metric.MeterProvider) instruments {
	m := mp.Meter(tracerName)
	tokenUsage, _ := m.Int64Histogram("gen_ai.client.token.usage",
		metric.WithDescription("Token usage by type"),
		metric.WithUnit("{token}"),
	)
	operationDuration, _ := m.Float64Histogram("gen_ai.client.operation.duration",
		metric.WithDescription("Duration of LLM API calls"),
		metric.WithUnit("s"),
	)
	toolDuration, _ := m.Float64Histogram("goai.tool.duration",
		metric.WithDescription("Duration of tool executions"),
		metric.WithUnit("s"),
	)
	earlyStopCounter, _ := m.Int64Counter("goai.loop.early_stop",
		metric.WithDescription("Number of tool loops stopped early by OnBeforeStep hook"),
	)
	messageCount, _ := m.Int64Gauge("goai.conversation.message_count",
		metric.WithDescription("Number of messages in conversation per step"),
	)
	return instruments{
		tokenUsage:        tokenUsage,
		operationDuration: operationDuration,
		toolDuration:      toolDuration,
		earlyStopCounter:  earlyStopCounter,
		messageCount:      messageCount,
	}
}

// WithTracing returns a goai.Option that enables OTel tracing for a single call.
// Each invocation creates a fresh trace with isolated state -- safe for concurrent use.
//
// Ordering: WithTracing wraps the singleton hooks (OnBeforeToolExecute, OnAfterToolExecute,
// OnBeforeStep) to add observability without replacing user hooks. For this to work,
// WithTracing must be applied AFTER any user-registered singleton hooks. Place it last
// in the option list, or use WithOptions to group user hooks before WithTracing.
func WithTracing(opts ...TracingOption) goai.Option {
	cfg := config{
		spanName: "chat",
	}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.tracerProvider == nil {
		cfg.tracerProvider = otel.GetTracerProvider()
	}
	if cfg.meterProvider == nil {
		cfg.meterProvider = otel.GetMeterProvider()
	}

	return goai.WithOptions(run(cfg)...)
}

// run returns a fresh set of goai options scoped to a single call.
//
// Per-run state is isolated per run() call. OnRequest, OnResponse, and
// OnStepFinish are called sequentially by goai. However, OnToolCallStart
// and OnToolCall fire inside per-tool goroutines when multiple tools execute
// in parallel, so toolSpans requires mutex protection.
func run(cfg config) []goai.Option {
	tracer := cfg.tracerProvider.Tracer(tracerName)
	inst := newInstruments(cfg.meterProvider)

	// Per-run state. Isolated per run() call.
	// OnRequest/OnResponse/OnStepFinish are called sequentially.
	// OnToolCallStart/OnToolCall run in parallel goroutines (one per tool),
	// so toolSpans is protected by toolMu.
	var (
		rootSpan  trace.Span
		rootCtx   context.Context
		llmSpan   trace.Span
		step      int
		toolMu    sync.Mutex
		toolSpans = make(map[string]trace.Span) // keyed by toolCallID

		// B1-B3: per-tool state from OnBeforeToolExecute (keyed by toolCallID).
		toolSkipReasons    = make(map[string]string)
		toolInputOverrides = make(map[string]bool)
		toolCtxOverridden  = make(map[string]bool)   // B3: context was overridden
		toolCtxDeadlines   = make(map[string]string) // B3: deadline string (empty if no deadline)

		// C1-C2: per-tool state from OnAfterToolExecute (keyed by toolCallID).
		toolOutputModified = make(map[string]bool)
		toolErrorModified  = make(map[string]string)

		// D1-D2: termination tracking.
		hookStopped       bool
		hookStoppedAtStep int
		terminationReason string

		// D3: injected message count for next generation.
		pendingInjectedMessages int

		// Pending text/finishReason from OnStepFinish, for OnFinish to flush.
		pendingText         string
		pendingFinishReason provider.FinishReason
	)

	// end is declared before opts so WithOnStepFinish can reference it.
	var end func(text string, finishReason provider.FinishReason)

	opts := []goai.Option{
		goai.WithOnRequest(func(info goai.RequestInfo) {
			step++

			if rootSpan == nil {
				// First request: start root span.
				attrs := []attribute.KeyValue{
					attribute.String("gen_ai.system", "goai"),
					attribute.String("gen_ai.operation.name", "chat"),
					attribute.String("gen_ai.request.model", info.Model),
				}
				attrs = append(attrs, cfg.attrs...)

				parentCtx := info.Ctx
				if parentCtx == nil {
					parentCtx = context.Background()
				}
				// A4: use info.Timestamp for span start time (more accurate than
				// in-hook wall clock -- set by goai before OnRequest fires).
				rootCtx, rootSpan = tracer.Start(parentCtx, cfg.spanName,
					trace.WithAttributes(attrs...),
					trace.WithTimestamp(info.Timestamp),
				)

				if cfg.recordInput {
					rootSpan.AddEvent("gen_ai.content.prompt", trace.WithAttributes(
						attribute.String("gen_ai.prompt", marshalMessages(info.Messages)),
					))
				}
			}

			// A3: message count and tool count on LLM span.
			llmAttrs := []attribute.KeyValue{
				attribute.String("gen_ai.system", "goai"),
				attribute.String("gen_ai.operation.name", "chat"),
				attribute.String("gen_ai.request.model", info.Model),
				attribute.Int("goai.step", step),
				attribute.Int("goai.request.message_count", info.MessageCount),
				attribute.Int("goai.request.tool_count", info.ToolCount),
			}

			// D3: annotate injected messages from OnBeforeStep.
			if pendingInjectedMessages > 0 {
				llmAttrs = append(llmAttrs, attribute.Int("goai.step.injected_messages", pendingInjectedMessages))
				pendingInjectedMessages = 0
			}

			// Start LLM call span as child of root.
			// A4: use info.Timestamp for span start time.
			_, llmSpan = tracer.Start(rootCtx, fmt.Sprintf("chat %s", info.Model),
				trace.WithAttributes(llmAttrs...),
				trace.WithTimestamp(info.Timestamp),
			)
		}),

		goai.WithOnResponse(func(info goai.ResponseInfo) {
			if llmSpan == nil {
				return
			}

			duration := info.Latency.Seconds()

			// Set response attributes on the LLM span.
			attrs := []attribute.KeyValue{
				attribute.Int("gen_ai.usage.input_tokens", info.Usage.InputTokens),
				attribute.Int("gen_ai.usage.output_tokens", info.Usage.OutputTokens),
			}
			if info.Usage.TotalTokens > 0 {
				attrs = append(attrs, attribute.Int("gen_ai.usage.total_tokens", info.Usage.TotalTokens))
			}
			if info.FinishReason != "" {
				attrs = append(attrs, attribute.StringSlice("gen_ai.response.finish_reasons", []string{string(info.FinishReason)}))
			}
			if info.Usage.ReasoningTokens > 0 {
				attrs = append(attrs, attribute.Int("goai.usage.reasoning_tokens", info.Usage.ReasoningTokens))
			}
			if info.Usage.CacheReadTokens > 0 {
				attrs = append(attrs, attribute.Int("gen_ai.usage.cache_read.input_tokens", info.Usage.CacheReadTokens))
			}
			if info.Usage.CacheWriteTokens > 0 {
				attrs = append(attrs, attribute.Int("gen_ai.usage.cache_creation.input_tokens", info.Usage.CacheWriteTokens))
			}
			if info.StatusCode > 0 {
				attrs = append(attrs, attribute.Int("http.response.status_code", info.StatusCode))
			}

			llmSpan.SetAttributes(attrs...)

			// Record metrics. Only record token counts when at least one is non-zero
			// to avoid inflating histograms with zero-value data points on error paths.
			if info.Usage.InputTokens > 0 || info.Usage.OutputTokens > 0 {
				inst.tokenUsage.Record(context.Background(), int64(info.Usage.InputTokens),
					metric.WithAttributes(
						attribute.String("gen_ai.operation.name", "chat"),
						attribute.String("gen_ai.token.type", "input"),
					),
				)
				inst.tokenUsage.Record(context.Background(), int64(info.Usage.OutputTokens),
					metric.WithAttributes(
						attribute.String("gen_ai.operation.name", "chat"),
						attribute.String("gen_ai.token.type", "output"),
					),
				)
			}
			inst.operationDuration.Record(context.Background(), duration,
				metric.WithAttributes(attribute.String("gen_ai.operation.name", "chat")),
			)

			if info.Error != nil {
				llmSpan.SetStatus(codes.Error, info.Error.Error())
				llmSpan.End()
				llmSpan = nil
				if rootSpan != nil {
					rootSpan.SetStatus(codes.Error, info.Error.Error())
				}
				end("", "")
				return
			}

			llmSpan.End()
			llmSpan = nil
		}),

		goai.WithOnToolCallStart(func(info goai.ToolCallStartInfo) {
			if rootCtx == nil {
				return
			}

			_, toolSpan := tracer.Start(rootCtx, fmt.Sprintf("execute_tool %s", info.ToolName),
				trace.WithAttributes(
					attribute.String("gen_ai.tool.name", info.ToolName),
					attribute.String("gen_ai.tool.call.id", info.ToolCallID),
					attribute.Int("goai.step", info.Step),
				),
			)

			// A7: record tool input as span event (respecting privacy opt-in).
			if cfg.recordToolIO && len(info.Input) > 0 {
				toolSpan.AddEvent("goai.tool.input", trace.WithAttributes(
					attribute.String("goai.tool.input", string(info.Input)),
				))
			}

			toolMu.Lock()
			toolSpans[info.ToolCallID] = toolSpan
			toolMu.Unlock()
		}),

		goai.WithOnToolCall(func(info goai.ToolCallInfo) {
			toolMu.Lock()
			toolSpan, ok := toolSpans[info.ToolCallID]
			if ok {
				delete(toolSpans, info.ToolCallID)
			}

			// Read per-tool state from B/C hooks.
			skipReason := toolSkipReasons[info.ToolCallID]
			delete(toolSkipReasons, info.ToolCallID)
			inputOverridden := toolInputOverrides[info.ToolCallID]
			delete(toolInputOverrides, info.ToolCallID)
			ctxOverridden := toolCtxOverridden[info.ToolCallID]
			delete(toolCtxOverridden, info.ToolCallID)
			ctxDeadline := toolCtxDeadlines[info.ToolCallID]
			delete(toolCtxDeadlines, info.ToolCallID)
			outputModified := toolOutputModified[info.ToolCallID]
			delete(toolOutputModified, info.ToolCallID)
			errModified := toolErrorModified[info.ToolCallID]
			delete(toolErrorModified, info.ToolCallID)
			toolMu.Unlock()

			if !ok {
				return
			}

			// A1: annotate skipped tools.
			if info.Skipped {
				toolSpan.SetAttributes(attribute.Bool("goai.tool.skipped", true))
				// B1: skip reason attribution.
				if skipReason != "" {
					toolSpan.AddEvent("goai.tool.skipped", trace.WithAttributes(
						attribute.String("goai.tool.skip_reason", skipReason),
					))
				}
			}

			// A2: record metadata from OnAfterToolExecute.
			for k, v := range info.Metadata {
				switch val := v.(type) {
				case string:
					toolSpan.SetAttributes(attribute.String("goai.tool.metadata."+k, val))
				case int:
					toolSpan.SetAttributes(attribute.Int("goai.tool.metadata."+k, val))
				case int64:
					toolSpan.SetAttributes(attribute.Int64("goai.tool.metadata."+k, val))
				case float64:
					toolSpan.SetAttributes(attribute.Float64("goai.tool.metadata."+k, val))
				case bool:
					toolSpan.SetAttributes(attribute.Bool("goai.tool.metadata."+k, val))
				}
			}

			// B2: input override detection.
			if inputOverridden {
				toolSpan.SetAttributes(attribute.Bool("goai.tool.input_overridden", true))
			}

			// B3: context override detection -- set unconditionally when ctx overridden.
			if ctxOverridden {
				toolSpan.SetAttributes(attribute.Bool("goai.tool.context_overridden", true))
				if ctxDeadline != "" {
					toolSpan.SetAttributes(attribute.String("goai.tool.deadline", ctxDeadline))
				}
			}

			// C1: output modification detection.
			if outputModified {
				toolSpan.SetAttributes(attribute.Bool("goai.tool.output_modified", true))
			}

			// C2: error injection/replacement detection.
			if errModified != "" {
				toolSpan.AddEvent("goai.tool.error_modified", trace.WithAttributes(
					attribute.String("goai.tool.error_modification", errModified),
				))
			}

			// A7: record tool output as span event (respecting privacy opt-in).
			if cfg.recordToolIO && info.Output != "" {
				toolSpan.AddEvent("goai.tool.output", trace.WithAttributes(
					attribute.String("goai.tool.output", info.Output),
				))
			}

			if info.Error != nil {
				toolSpan.SetStatus(codes.Error, info.Error.Error())
			}

			toolSpan.End()

			// A1: skip recording tool duration metric when Skipped=true (0s pollutes histogram).
			if !info.Skipped {
				inst.toolDuration.Record(context.Background(), info.Duration.Seconds(),
					metric.WithAttributes(attribute.String("gen_ai.tool.name", info.ToolName)),
				)
			}
		}),

		goai.WithOnStepFinish(func(sr goai.StepResult) {
			// A6: record Sources, Response.Model, Response.ID, ProviderMetadata.
			// The per-step LLM span has already ended in OnResponse, so these are set
			// on rootSpan. In multi-step runs, later steps overwrite earlier values --
			// only the last step's metadata survives. This is acceptable because the
			// root span represents the overall operation, and the last step's response
			// is the final answer. Per-step metadata is available on LLM spans via
			// OnResponse attributes (usage, finish reason, etc.).
			if rootSpan != nil {
				if sr.Response.Model != "" {
					rootSpan.SetAttributes(attribute.String("gen_ai.response.model", sr.Response.Model))
				}
				if sr.Response.ID != "" {
					rootSpan.SetAttributes(attribute.String("gen_ai.response.id", sr.Response.ID))
				}
				if len(sr.Sources) > 0 {
					sourceNames := make([]string, 0, len(sr.Sources))
					for _, s := range sr.Sources {
						if s.URL != "" {
							sourceNames = append(sourceNames, s.URL)
						} else if s.Title != "" {
							sourceNames = append(sourceNames, s.Title)
						}
					}
					if len(sourceNames) > 0 {
						rootSpan.AddEvent("gen_ai.content.sources", trace.WithAttributes(
							attribute.StringSlice("gen_ai.sources", sourceNames),
						))
					}
				}
				if len(sr.ProviderMetadata) > 0 {
					for ns, data := range sr.ProviderMetadata {
						for k, v := range data {
							switch val := v.(type) {
							case string:
								rootSpan.SetAttributes(attribute.String("goai.provider_metadata."+ns+"."+k, val))
							case float64:
								rootSpan.SetAttributes(attribute.Float64("goai.provider_metadata."+ns+"."+k, val))
							case bool:
								rootSpan.SetAttributes(attribute.Bool("goai.provider_metadata."+ns+"."+k, val))
							}
						}
					}
				}
			}

			// D2: track termination reason.
			// "natural" and "hook_stopped" are detected here in OnStepFinish.
			// "max_steps" is detected in OnFinish via StepsExhausted.
			if sr.FinishReason == provider.FinishToolCalls {
				return // intermediate step -- more tool calls follow
			}
			if !hookStopped {
				terminationReason = "natural"
			}

			// Store pending text/finishReason for OnFinish to flush.
			// Do NOT call end() here -- OnFinish is the single closer.
			pendingText = sr.Text
			pendingFinishReason = sr.FinishReason
		}),
	}

	// OnFinish: fires once after all steps complete. Handles termination reason
	// detection (including max_steps via StepsExhausted) and ensures end() is called
	// for all exit paths.
	opts = append(opts, goai.WithOnFinish(func(info goai.FinishInfo) {
		if info.StepsExhausted {
			terminationReason = "max_steps"
		}
		// For hook_stopped, terminationReason was already set in OnBeforeStep wrapper.
		// For natural, it was set in OnStepFinish (which stored text in pendingText).
		// OnFinish is the single closer -- end() is only called here (and OnResponse error).
		text := pendingText
		fr := pendingFinishReason
		if fr == "" {
			fr = info.FinishReason
		}
		end(text, fr)
	}))

	// Section 7.1: Wrapper pattern for singleton hooks (B/C/D categories).

	// B1-B4: OnBeforeToolExecute wrapper.
	opts = append(opts, goai.WrapOnBeforeToolExecute(func(userHook func(goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult) func(goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
		return func(info goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			var result goai.BeforeToolExecuteResult
			if userHook != nil {
				result = userHook(info)
			}

			toolMu.Lock()
			// B4: add pre-execution span event.
			if toolSpan, ok := toolSpans[info.ToolCallID]; ok {
				toolSpan.AddEvent("goai.tool.execute_start")
			}

			// B1: capture skip reason.
			if result.Skip {
				reason := "skipped by hook"
				if result.Error != nil {
					reason = result.Error.Error()
				}
				toolSkipReasons[info.ToolCallID] = reason
			}
			// B2: detect input override.
			if result.Input != nil {
				toolInputOverrides[info.ToolCallID] = true
			}
			// B3: detect context override with deadline extraction.
			if result.Ctx != nil {
				toolCtxOverridden[info.ToolCallID] = true
				if dl, ok := result.Ctx.Deadline(); ok {
					toolCtxDeadlines[info.ToolCallID] = dl.Format("2006-01-02T15:04:05.000Z07:00")
				}
			}
			toolMu.Unlock()

			return result
		}
	}))

	// C1-C3: OnAfterToolExecute wrapper.
	opts = append(opts, goai.WrapOnAfterToolExecute(func(userHook func(goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult) func(goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
		return func(info goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
			var result goai.AfterToolExecuteResult
			if userHook != nil {
				result = userHook(info)
			}

			toolMu.Lock()
			// C3: add timing boundary event.
			if toolSpan, ok := toolSpans[info.ToolCallID]; ok {
				toolSpan.AddEvent("goai.tool.after_execute")
			}

			// C1: detect output modification.
			if result.Output != "" && result.Output != info.Output {
				toolOutputModified[info.ToolCallID] = true
			}
			// C2: detect error injection/replacement.
			if result.Error != nil {
				if info.Error == nil {
					toolErrorModified[info.ToolCallID] = "injected"
				} else if result.Error.Error() != info.Error.Error() {
					toolErrorModified[info.ToolCallID] = "replaced"
				}
			}
			toolMu.Unlock()

			return result
		}
	}))

	// D1, D3-D4: OnBeforeStep wrapper.
	opts = append(opts, goai.WrapOnBeforeStep(func(userHook func(goai.BeforeStepInfo) goai.BeforeStepResult) func(goai.BeforeStepInfo) goai.BeforeStepResult {
		return func(info goai.BeforeStepInfo) goai.BeforeStepResult {
			var result goai.BeforeStepResult
			if userHook != nil {
				result = userHook(info)
			}

			// D1: detect loop termination signal.
			if result.Stop {
				hookStopped = true
				hookStoppedAtStep = info.Step
				terminationReason = "hook_stopped"
				if rootSpan != nil {
					rootSpan.AddEvent("goai.loop.stopped", trace.WithAttributes(
						attribute.Int("goai.step", info.Step),
					))
					inst.earlyStopCounter.Add(context.Background(), 1)
				}
				// end() is called by OnFinish (which fires after the loop exits).
			}

			// D3: track injected messages.
			if len(result.ExtraMessages) > 0 {
				pendingInjectedMessages = len(result.ExtraMessages)
			}

			// D4: conversation size monitoring.
			if rootSpan != nil {
				inst.messageCount.Record(context.Background(), int64(len(info.Messages)),
					metric.WithAttributes(attribute.Int("goai.step", info.Step)),
				)
			}

			return result
		}
	}))

	end = func(text string, finishReason provider.FinishReason) {
		if rootSpan == nil {
			return
		}

		if cfg.recordOutput && text != "" {
			rootSpan.AddEvent("gen_ai.content.completion", trace.WithAttributes(
				attribute.String("gen_ai.completion", text),
			))
		}

		if finishReason != "" {
			rootSpan.SetAttributes(
				attribute.StringSlice("gen_ai.response.finish_reasons", []string{string(finishReason)}),
			)
		}

		// D1: annotate root span with hook_stopped.
		if hookStopped {
			rootSpan.SetAttributes(
				attribute.Bool("goai.stopped_by_hook", true),
				attribute.Int("goai.stopped_at_step", hookStoppedAtStep),
			)
		}

		// D2: set termination reason on root span.
		if terminationReason != "" {
			rootSpan.SetAttributes(attribute.String("goai.termination_reason", terminationReason))
		}

		// End any orphaned LLM span (e.g., drainStep error or empty step where
		// OnResponse never fired to end it). Safe to call even if llmSpan is nil.
		if llmSpan != nil {
			llmSpan.SetStatus(codes.Error, "stream terminated before response")
			llmSpan.End()
			llmSpan = nil
		}

		rootSpan.End()
		rootSpan = nil
		rootCtx = nil
	}

	return opts
}

// marshalMessages serializes messages to JSON for span events.
// Captures all text parts (concatenated), reasoning content, and tool call names.
func marshalMessages(msgs []provider.Message) string {
	type msgEntry struct {
		Role      string   `json:"role"`
		Content   string   `json:"content,omitempty"`
		Reasoning string   `json:"reasoning,omitempty"`
		ToolCalls []string `json:"tool_calls,omitempty"`
	}
	entries := make([]msgEntry, 0, len(msgs))
	for _, m := range msgs {
		var textParts []string
		var reasoningParts []string
		var toolCalls []string
		for _, p := range m.Content {
			switch p.Type {
			case provider.PartText:
				if p.Text != "" {
					textParts = append(textParts, p.Text)
				}
			case provider.PartReasoning:
				if p.Text != "" {
					reasoningParts = append(reasoningParts, p.Text)
				}
			case provider.PartToolCall:
				toolCalls = append(toolCalls, p.ToolName)
			}
		}
		entries = append(entries, msgEntry{
			Role:      string(m.Role),
			Content:   strings.Join(textParts, ""),
			Reasoning: strings.Join(reasoningParts, ""),
			ToolCalls: toolCalls,
		})
	}
	b, _ := json.Marshal(entries)
	return string(b)
}
