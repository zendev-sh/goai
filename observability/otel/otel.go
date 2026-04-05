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

// instruments holds pre-created metric instruments to avoid repeated creation.
type instruments struct {
	tokenUsage        metric.Int64Histogram
	operationDuration metric.Float64Histogram
	toolDuration      metric.Float64Histogram
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
	return instruments{
		tokenUsage:        tokenUsage,
		operationDuration: operationDuration,
		toolDuration:      toolDuration,
	}
}

// WithTracing returns a goai.Option that enables OTel tracing for a single call.
// Each invocation creates a fresh trace with isolated state -- safe for concurrent use.
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
				rootCtx, rootSpan = tracer.Start(parentCtx, cfg.spanName,
					trace.WithAttributes(attrs...),
				)

				if cfg.recordInput {
					rootSpan.AddEvent("gen_ai.content.prompt", trace.WithAttributes(
						attribute.String("gen_ai.prompt", marshalMessages(info.Messages)),
					))
				}
			}

			// Start LLM call span as child of root.
			_, llmSpan = tracer.Start(rootCtx, fmt.Sprintf("chat %s", info.Model),
				trace.WithAttributes(
					attribute.String("gen_ai.system", "goai"),
					attribute.String("gen_ai.operation.name", "chat"),
					attribute.String("gen_ai.request.model", info.Model),
					attribute.Int("goai.step", step),
				),
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
				// On error, mark root span with error status and end immediately.
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
			toolMu.Unlock()

			if !ok {
				return
			}

			if info.Error != nil {
				toolSpan.SetStatus(codes.Error, info.Error.Error())
			}

			toolSpan.End()

			inst.toolDuration.Record(context.Background(), info.Duration.Seconds(),
				metric.WithAttributes(attribute.String("gen_ai.tool.name", info.ToolName)),
			)
		}),

		goai.WithOnStepFinish(func(sr goai.StepResult) {
			if sr.FinishReason == provider.FinishToolCalls {
				return // intermediate step -- more tool calls follow
			}
			end(sr.Text, sr.FinishReason)
		}),
	}

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
