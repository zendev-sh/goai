package otel

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// mockModel implements provider.LanguageModel for testing.
type mockModel struct {
	id         string
	generateFn func(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error)
	streamFn   func(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error)
}

func (m *mockModel) ModelID() string { return m.id }
func (m *mockModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	return m.generateFn(ctx, params)
}
func (m *mockModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if m.streamFn != nil {
		return m.streamFn(ctx, params)
	}
	return nil, nil
}
func (m *mockModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{}
}

// streamFromChunks creates a StreamResult from pre-built chunks.
func streamFromChunks(chunks ...provider.StreamChunk) *provider.StreamResult {
	ch := make(chan provider.StreamChunk, len(chunks))
	for _, c := range chunks {
		ch <- c
	}
	close(ch)
	return &provider.StreamResult{Stream: ch}
}

// newTestProviders creates a TracerProvider with SpanRecorder and a MeterProvider
// with ManualReader for test assertions.
func newTestProviders(t *testing.T) (*sdktrace.TracerProvider, *tracetest.SpanRecorder, *sdkmetric.MeterProvider, *sdkmetric.ManualReader) {
	t.Helper()
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	mr := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(mr))
	t.Cleanup(func() {
		_ = tp.Shutdown(context.Background())
		_ = mp.Shutdown(context.Background())
	})
	return tp, sr, mp, mr
}

// spanByName finds the first ended span with the given name.
func spanByName(sr *tracetest.SpanRecorder, name string) tracetest.SpanStub {
	for _, s := range tracetest.SpanStubsFromReadOnlySpans(sr.Ended()) {
		if s.Name == name {
			return s
		}
	}
	return tracetest.SpanStub{}
}

// spansByName returns all ended spans with the given name.
func spansByName(sr *tracetest.SpanRecorder, name string) []tracetest.SpanStub {
	var out []tracetest.SpanStub
	for _, s := range tracetest.SpanStubsFromReadOnlySpans(sr.Ended()) {
		if s.Name == name {
			out = append(out, s)
		}
	}
	return out
}

// spansByPrefix returns all ended spans whose name starts with the given prefix.
func spansByPrefix(sr *tracetest.SpanRecorder, prefix string) []tracetest.SpanStub {
	var out []tracetest.SpanStub
	for _, s := range tracetest.SpanStubsFromReadOnlySpans(sr.Ended()) {
		if strings.HasPrefix(s.Name, prefix) {
			out = append(out, s)
		}
	}
	return out
}

// attrValue returns the attribute value for the given key, or empty.
func attrValue(attrs []attribute.KeyValue, key string) attribute.Value {
	for _, a := range attrs {
		if string(a.Key) == key {
			return a.Value
		}
	}
	return attribute.Value{}
}

// collectMetrics reads all metrics from the ManualReader.
func collectMetrics(t *testing.T, mr *sdkmetric.ManualReader) metricdata.ResourceMetrics {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := mr.Collect(context.Background(), &rm); err != nil {
		t.Fatalf("collect metrics: %v", err)
	}
	return rm
}

// findMetric finds a metric by name in ResourceMetrics.
func findMetric(rm metricdata.ResourceMetrics, name string) *metricdata.Metrics {
	for _, sm := range rm.ScopeMetrics {
		for i := range sm.Metrics {
			if sm.Metrics[i].Name == name {
				return &sm.Metrics[i]
			}
		}
	}
	return nil
}

func TestWithTracing_SingleStep(t *testing.T) {
	tp, sr, mp, mr := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "Hello!",
				FinishReason: provider.FinishStop,
				Usage: provider.Usage{
					InputTokens:  10,
					OutputTokens: 5,
				},
			}, nil
		},
	}

	result, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if result.Text != "Hello!" {
		t.Fatalf("got text %q, want %q", result.Text, "Hello!")
	}

	// Verify spans.
	ended := sr.Ended()
	if len(ended) != 2 {
		t.Fatalf("got %d ended spans, want 2", len(ended))
	}

	// Root span.
	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}
	if v := attrValue(root.Attributes, "gen_ai.system"); v.AsString() != "goai" {
		t.Errorf("gen_ai.system = %q, want %q", v.AsString(), "goai")
	}
	if v := attrValue(root.Attributes, "gen_ai.request.model"); v.AsString() != "test-model" {
		t.Errorf("gen_ai.request.model = %q, want %q", v.AsString(), "test-model")
	}

	// LLM call span.
	llm := spanByName(sr, "chat test-model")
	if llm.Name == "" {
		t.Fatal("llm span not found")
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.input_tokens"); v.AsInt64() != 10 {
		t.Errorf("input_tokens = %d, want 10", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.output_tokens"); v.AsInt64() != 5 {
		t.Errorf("output_tokens = %d, want 5", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "goai.step"); v.AsInt64() != 1 {
		t.Errorf("step = %d, want 1", v.AsInt64())
	}

	// LLM span should be child of root (compare TraceID).
	if llm.Parent.TraceID() != root.SpanContext.TraceID() {
		t.Error("llm span is not a child of root span")
	}

	// Verify metrics.
	rm := collectMetrics(t, mr)
	tokenMetric := findMetric(rm, "gen_ai.client.token.usage")
	if tokenMetric == nil {
		t.Fatal("token.usage metric not found")
	}
	durationMetric := findMetric(rm, "gen_ai.client.operation.duration")
	if durationMetric == nil {
		t.Fatal("operation.duration metric not found")
	}
}

func TestWithTracing_MultiStep(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	call := 0
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			call++
			if call == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "tc1", Name: "get_weather", Input: []byte(`{"city":"NYC"}`)},
					},
					Usage: provider.Usage{InputTokens: 20, OutputTokens: 10},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "It's sunny in NYC!",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 30, OutputTokens: 15},
			}, nil
		},
	}

	weatherTool := goai.Tool{
		Name: "get_weather",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return `{"temp":72}`, nil
		},
	}

	result, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("What's the weather in NYC?"),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if result.Text != "It's sunny in NYC!" {
		t.Fatalf("got text %q", result.Text)
	}

	// Should have: root + 2 LLM calls + 1 tool = 4 spans.
	ended := sr.Ended()
	if len(ended) != 4 {
		var names []string
		for _, s := range ended {
			names = append(names, s.Name())
		}
		t.Fatalf("got %d ended spans (names: %v), want 4", len(ended), names)
	}

	// Verify tool span exists.
	toolSpan := spanByName(sr, "execute_tool get_weather")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}
	if v := attrValue(toolSpan.Attributes, "gen_ai.tool.name"); v.AsString() != "get_weather" {
		t.Errorf("tool.name = %q, want %q", v.AsString(), "get_weather")
	}
	if v := attrValue(toolSpan.Attributes, "gen_ai.tool.call.id"); v.AsString() != "tc1" {
		t.Errorf("tool.call_id = %q, want %q", v.AsString(), "tc1")
	}

	// Verify 2 LLM call spans.
	llmSpans := spansByName(sr, "chat test-model")
	if len(llmSpans) != 2 {
		t.Fatalf("got %d llm spans, want 2", len(llmSpans))
	}
}

func TestWithTracing_Error(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	apiErr := errors.New("rate limit exceeded")
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, apiErr
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
		goai.WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected error")
	}

	// Both root and LLM spans should be ended.
	ended := sr.Ended()
	if len(ended) != 2 {
		t.Fatalf("got %d ended spans, want 2", len(ended))
	}

	// LLM span should have error status.
	llm := spanByName(sr, "chat test-model")
	if llm.Status.Code != codes.Error {
		t.Errorf("llm span status = %v, want Error", llm.Status.Code)
	}

	// Root span should also have error status.
	root := spanByName(sr, "chat")
	if root.Status.Code != codes.Error {
		t.Errorf("root span status = %v, want Error", root.Status.Code)
	}
}

func TestWithTracing_ToolError(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	call := 0
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			call++
			if call == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "tc1", Name: "failing_tool", Input: []byte(`{}`)},
					},
					Usage: provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "Tool failed, sorry.",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	failTool := goai.Tool{
		Name: "failing_tool",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "", errors.New("tool broke")
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("use tool"),
		goai.WithTools(failTool),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	// Tool span should have error status.
	toolSpan := spanByName(sr, "execute_tool failing_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}
	if toolSpan.Status.Code != codes.Error {
		t.Errorf("tool span status = %v, want Error", toolSpan.Status.Code)
	}
	if toolSpan.Status.Description != "tool broke" {
		t.Errorf("tool span status message = %q, want %q", toolSpan.Status.Description, "tool broke")
	}
}

func TestWithTracing_CustomTracerProvider(t *testing.T) {
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp)),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	ended := sr.Ended()
	if len(ended) != 2 {
		t.Fatalf("got %d spans, want 2", len(ended))
	}

	// Verify the instrumentation scope uses our tracer name.
	stubs := tracetest.SpanStubsFromReadOnlySpans(ended)
	if stubs[0].InstrumentationScope.Name != tracerName {
		t.Errorf("scope name = %q, want %q", stubs[0].InstrumentationScope.Name, tracerName)
	}
}

func TestWithTracing_CustomMeterProvider(t *testing.T) {
	tp, _, _, _ := newTestProviders(t)

	// Create a separate meter provider.
	customMR := sdkmetric.NewManualReader()
	customMP := sdkmetric.NewMeterProvider(sdkmetric.WithReader(customMR))
	t.Cleanup(func() { _ = customMP.Shutdown(context.Background()) })

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(customMP),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	rm := collectMetrics(t, customMR)
	tokenMetric := findMetric(rm, "gen_ai.client.token.usage")
	if tokenMetric == nil {
		t.Fatal("token.usage metric not found in custom meter provider")
	}
}

func TestWithTracing_CustomSpanName(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
			WithSpanName("my-agent.chat"),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "my-agent.chat")
	if root.Name == "" {
		t.Fatal("custom-named root span not found")
	}
}

func TestWithTracing_CustomAttributes(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
			WithAttributes(
				attribute.String("user.id", "u-123"),
				attribute.String("env", "test"),
			),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if v := attrValue(root.Attributes, "user.id"); v.AsString() != "u-123" {
		t.Errorf("user.id = %q, want %q", v.AsString(), "u-123")
	}
	if v := attrValue(root.Attributes, "env"); v.AsString() != "test" {
		t.Errorf("env = %q, want %q", v.AsString(), "test")
	}
}

func TestWithTracing_DefaultProviders(t *testing.T) {
	// WithTracing() without any options should not panic.
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText with default providers: %v", err)
	}
}

func TestWithTracing_TokenUsage(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage: provider.Usage{
					InputTokens:     100,
					OutputTokens:    50,
					TotalTokens:     160,
					ReasoningTokens:  10,
					CacheReadTokens:  20,
					CacheWriteTokens: 5,
				},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	llm := spanByName(sr, "chat test-model")
	if v := attrValue(llm.Attributes, "gen_ai.usage.input_tokens"); v.AsInt64() != 100 {
		t.Errorf("input_tokens = %d, want 100", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.output_tokens"); v.AsInt64() != 50 {
		t.Errorf("output_tokens = %d, want 50", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.total_tokens"); v.AsInt64() != 160 {
		t.Errorf("total_tokens = %d, want 160", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "goai.usage.reasoning_tokens"); v.AsInt64() != 10 {
		t.Errorf("reasoning_tokens = %d, want 10", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.cache_read.input_tokens"); v.AsInt64() != 20 {
		t.Errorf("cache_read_tokens = %d, want 20", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.cache_creation.input_tokens"); v.AsInt64() != 5 {
		t.Errorf("cache_write_tokens = %d, want 5", v.AsInt64())
	}
}

func TestWithTracing_Metrics(t *testing.T) {
	tp, _, mp, mr := newTestProviders(t)

	call := 0
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			call++
			if call == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "tc1", Name: "my_tool", Input: []byte(`{}`)},
					},
					Usage: provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	myTool := goai.Tool{
		Name: "my_tool",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			time.Sleep(5 * time.Millisecond)
			return "result", nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("use tool"),
		goai.WithTools(myTool),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	rm := collectMetrics(t, mr)

	// Token usage histogram should exist.
	tokenMetric := findMetric(rm, "gen_ai.client.token.usage")
	if tokenMetric == nil {
		t.Fatal("gen_ai.client.token.usage metric not found")
	}

	// Operation duration histogram should exist.
	opMetric := findMetric(rm, "gen_ai.client.operation.duration")
	if opMetric == nil {
		t.Fatal("gen_ai.client.operation.duration metric not found")
	}

	// Tool duration histogram should exist.
	toolMetric := findMetric(rm, "goai.tool.duration")
	if toolMetric == nil {
		t.Fatal("goai.tool.duration metric not found")
	}
}

func TestWithTracing_RecordMessages(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "Hello there!",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
			RecordInputMessages(true),
			RecordOutputMessages(true),
		),
		goai.WithPrompt("hello"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if len(root.Events) == 0 {
		t.Fatal("expected events on root span when recording messages")
	}

	// Check for input event.
	foundInput := false
	foundOutput := false
	for _, ev := range root.Events {
		if ev.Name == "gen_ai.content.prompt" {
			foundInput = true
		}
		if ev.Name == "gen_ai.content.completion" {
			foundOutput = true
		}
	}
	if !foundInput {
		t.Error("gen_ai.input event not found")
	}
	if !foundOutput {
		t.Error("gen_ai.output event not found")
	}
}

func TestWithTracing_RecordMessages_Disabled(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "Hello!",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
			// recording disabled by default
		),
		goai.WithPrompt("hello"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if len(root.Events) != 0 {
		t.Errorf("expected no events on root span when recording disabled, got %d", len(root.Events))
	}
}

func TestWithTracing_Streaming(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkText, Text: " world"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 8, OutputTokens: 4},
				},
			), nil
		},
	}

	stream, err := goai.StreamText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}
	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if result.Text != "Hello world" {
		t.Fatalf("got text %q, want %q", result.Text, "Hello world")
	}

	// Verify spans were created: root + LLM call = 2.
	ended := sr.Ended()
	if len(ended) != 2 {
		var names []string
		for _, s := range ended {
			names = append(names, s.Name())
		}
		t.Fatalf("got %d ended spans (names: %v), want 2", len(ended), names)
	}

	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}
}

func TestWithTracing_ConcurrentCalls(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	const n = 10
	var wg sync.WaitGroup
	wg.Add(n)
	for range n {
		go func() {
			defer wg.Done()
			_, err := goai.GenerateText(context.Background(), model,
				WithTracing(
					WithTracerProvider(tp),
					WithMeterProvider(mp),
				),
				goai.WithPrompt("hi"),
			)
			if err != nil {
				t.Errorf("GenerateText: %v", err)
			}
		}()
	}
	wg.Wait()

	// Each call produces 2 spans (root + llm), so we expect 2*n.
	ended := sr.Ended()
	if len(ended) != 2*n {
		t.Errorf("got %d ended spans, want %d", len(ended), 2*n)
	}
}

func TestWithTracing_FinishReasonOnRoot(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	v := attrValue(root.Attributes, "gen_ai.response.finish_reasons")
	reasons := v.AsStringSlice()
	if len(reasons) != 1 || reasons[0] != "stop" {
		t.Errorf("finish_reasons = %v, want [stop]", reasons)
	}
}

func TestWithTracing_StatusCode(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	// Note: StatusCode is set in ResponseInfo by the framework; with mock model
	// it will be 0, so we verify the attribute is NOT set when status is 0.
	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	llm := spanByName(sr, "chat test-model")
	// StatusCode 0 should not produce the attribute.
	v := attrValue(llm.Attributes, "http.response.status_code")
	if v.Type() != attribute.INVALID {
		t.Errorf("expected no http.status_code attribute for status 0, got %v", v)
	}
}

func TestWithTracing_ParallelToolCalls(t *testing.T) {
	tp, sr, mp, mr := newTestProviders(t)

	call := 0
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			call++
			if call == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "tc1", Name: "tool_a", Input: []byte(`{"x":1}`)},
						{ID: "tc2", Name: "tool_b", Input: []byte(`{"x":2}`)},
						{ID: "tc3", Name: "tool_c", Input: []byte(`{"x":3}`)},
					},
					Usage: provider.Usage{InputTokens: 20, OutputTokens: 10},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "all done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 50, OutputTokens: 20},
			}, nil
		},
	}

	toolA := goai.Tool{
		Name: "tool_a",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "a_result", nil
		},
	}
	toolB := goai.Tool{
		Name: "tool_b",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "b_result", nil
		},
	}
	toolC := goai.Tool{
		Name: "tool_c",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "c_result", nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("use all tools"),
		goai.WithTools(toolA, toolB, toolC),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	// Should have: root + 2 LLM calls + 3 tool spans = 6 spans.
	ended := sr.Ended()
	if len(ended) != 6 {
		var names []string
		for _, s := range ended {
			names = append(names, s.Name())
		}
		t.Fatalf("got %d ended spans (names: %v), want 6", len(ended), names)
	}

	// Verify 3 tool spans with distinct tool names.
	toolSpans := spansByPrefix(sr, "execute_tool")
	if len(toolSpans) != 3 {
		t.Fatalf("got %d tool spans, want 3", len(toolSpans))
	}
	toolNames := map[string]bool{}
	for _, ts := range toolSpans {
		name := attrValue(ts.Attributes, "gen_ai.tool.name").AsString()
		toolNames[name] = true
	}
	for _, expected := range []string{"tool_a", "tool_b", "tool_c"} {
		if !toolNames[expected] {
			t.Errorf("tool span for %q not found", expected)
		}
	}

	// Verify tool duration metrics recorded for all 3.
	rm := collectMetrics(t, mr)
	toolMetric := findMetric(rm, "goai.tool.duration")
	if toolMetric == nil {
		t.Fatal("goai.tool.duration metric not found")
	}
}

func TestWithTracing_ErrorRecordsMetrics(t *testing.T) {
	tp, _, mp, mr := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, errors.New("api error")
		},
	}

	_, _ = goai.GenerateText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
		goai.WithMaxRetries(0),
	)

	// Duration metric should be recorded even on error.
	rm := collectMetrics(t, mr)
	durationMetric := findMetric(rm, "gen_ai.client.operation.duration")
	if durationMetric == nil {
		t.Fatal("gen_ai.client.operation.duration metric not found on error path")
	}
	// Token metrics should NOT be recorded when both counts are zero (error path
	// with no usage data) to avoid inflating histograms with zero-value data points.
	tokenMetric := findMetric(rm, "gen_ai.client.token.usage")
	if tokenMetric != nil {
		t.Fatal("gen_ai.client.token.usage should not be recorded with zero-value tokens")
	}
}

func TestWithTracing_StreamingError(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, errors.New("stream init failed")
		},
	}

	_, err := goai.StreamText(context.Background(), model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
		goai.WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected error from StreamText")
	}

	// LLM and root spans should both be ended.
	ended := sr.Ended()
	if len(ended) != 2 {
		var names []string
		for _, s := range ended {
			names = append(names, s.Name())
		}
		t.Fatalf("got %d ended spans (names: %v), want 2", len(ended), names)
	}

	// LLM span should have error status.
	llm := spanByName(sr, "chat test-model")
	if llm.Name == "" {
		t.Fatal("llm span not found")
	}
	if llm.Status.Code != codes.Error {
		t.Errorf("llm span status = %v, want Error", llm.Status.Code)
	}

	// Root span should also be ended with error status.
	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found after streaming error")
	}
	if root.Status.Code != codes.Error {
		t.Errorf("root span status = %v, want Error", root.Status.Code)
	}
}

func TestMarshalMessages_MultiPart(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "Let me check "},
				{Type: provider.PartText, Text: "the weather."},
				{Type: provider.PartToolCall, ToolName: "get_weather", ToolCallID: "tc1"},
			},
		},
	}
	result := marshalMessages(msgs)
	if !strings.Contains(result, "Let me check the weather.") {
		t.Errorf("expected concatenated text, got %q", result)
	}
	if !strings.Contains(result, "get_weather") {
		t.Errorf("expected tool call name, got %q", result)
	}
}

func TestMarshalMessages_Reasoning(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{Type: provider.PartReasoning, Text: "thinking about this..."},
				{Type: provider.PartText, Text: "the answer is 42"},
			},
		},
	}
	result := marshalMessages(msgs)
	if !strings.Contains(result, "thinking about this...") {
		t.Errorf("expected reasoning content, got %q", result)
	}
	if !strings.Contains(result, "the answer is 42") {
		t.Errorf("expected text content, got %q", result)
	}
}

func TestMarshalMessages(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleUser,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "hello"},
			},
		},
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "hi there"},
			},
		},
	}

	result := marshalMessages(msgs)
	expected := `[{"role":"user","content":"hello"},{"role":"assistant","content":"hi there"}]`
	if result != expected {
		t.Errorf("marshalMessages = %q, want %q", result, expected)
	}
}

func TestMarshalMessages_Empty(t *testing.T) {
	if got := marshalMessages(nil); got != "[]" {
		t.Errorf("marshalMessages(nil) = %q, want %q", got, "[]")
	}
	if got := marshalMessages([]provider.Message{}); got != "[]" {
		t.Errorf("marshalMessages(empty) = %q, want %q", got, "[]")
	}
}

func TestWithTracing_ParentContext(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	// Create a parent span to act as the caller's existing trace context.
	tracer := tp.Tracer("test-parent")
	parentCtx, parentSpan := tracer.Start(context.Background(), "http.request")

	_, err := goai.GenerateText(parentCtx, model,
		WithTracing(
			WithTracerProvider(tp),
			WithMeterProvider(mp),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	// Capture the parent span context before ending.
	parentSC := parentSpan.SpanContext()

	// End the parent span so it appears in the recorder.
	parentSpan.End()

	// The goai.generate root span should be a child of the parent span.
	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}

	// Same trace ID means the OTel span is part of the parent's trace.
	if root.SpanContext.TraceID() != parentSC.TraceID() {
		t.Errorf("root span trace ID = %v, want %v (parent trace)", root.SpanContext.TraceID(), parentSC.TraceID())
	}

	// The root span's parent should be the parent span.
	if root.Parent.SpanID() != parentSC.SpanID() {
		t.Errorf("root span parent ID = %v, want %v (parent span)", root.Parent.SpanID(), parentSC.SpanID())
	}
}

// ---------------------------------------------------------------------------
// Helpers for enhancement tests
// ---------------------------------------------------------------------------

// hasEvent returns true if the span has an event with the given name.
func hasEvent(span tracetest.SpanStub, name string) bool {
	for _, e := range span.Events {
		if e.Name == name {
			return true
		}
	}
	return false
}

// eventAttrValue returns the attribute value for the given event name and key.
func eventAttrValue(span tracetest.SpanStub, eventName, key string) attribute.Value {
	for _, e := range span.Events {
		if e.Name == eventName {
			return attrValue(e.Attributes, key)
		}
	}
	return attribute.Value{}
}

// toolLoopModel returns a mock model that issues a single tool call on the first
// request and returns finalText on the second. The tool call uses the given name
// and ID. Response.Model and Response.ID are set on both results.
func toolLoopModel(toolName, toolCallID, finalText string) *mockModel {
	call := 0
	return &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			call++
			if call == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: toolCallID, Name: toolName, Input: []byte(`{"q":"test"}`)},
					},
					Usage:    provider.Usage{InputTokens: 10, OutputTokens: 5},
					Response: provider.ResponseMetadata{Model: "test-model-actual", ID: "resp-001"},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         finalText,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
				Response:     provider.ResponseMetadata{Model: "test-model-actual", ID: "resp-002"},
			}, nil
		},
	}
}

// simpleTool returns a goai.Tool with the given name that returns output.
func simpleTool(name, output string) goai.Tool {
	return goai.Tool{
		Name: name,
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return output, nil
		},
	}
}

// ---------------------------------------------------------------------------
// A1: Skipped tools
// ---------------------------------------------------------------------------

func TestA1_SkippedTools(t *testing.T) {
	tp, sr, mp, mr := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeToolExecute(func(_ goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			return goai.BeforeToolExecuteResult{Skip: true, Result: "skipped"}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	// Assert goai.tool.skipped=true attribute.
	if v := attrValue(toolSpan.Attributes, "goai.tool.skipped"); v.AsBool() != true {
		t.Errorf("goai.tool.skipped = %v, want true", v.AsBool())
	}

	// Assert tool duration metric is NOT recorded for skipped tools.
	rm := collectMetrics(t, mr)
	toolMetric := findMetric(rm, "goai.tool.duration")
	if toolMetric != nil {
		t.Error("goai.tool.duration should not be recorded for skipped tools")
	}
}

// ---------------------------------------------------------------------------
// A2: Metadata
// ---------------------------------------------------------------------------

func TestA2_Metadata(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnAfterToolExecute(func(_ goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
			return goai.AfterToolExecuteResult{
				Metadata: map[string]any{
					"region":  "us-east-1",
					"latency": 42,
					"cached":  true,
					"score":   0.95,
				},
			}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if v := attrValue(toolSpan.Attributes, "goai.tool.metadata.region"); v.AsString() != "us-east-1" {
		t.Errorf("metadata.region = %q, want %q", v.AsString(), "us-east-1")
	}
	if v := attrValue(toolSpan.Attributes, "goai.tool.metadata.latency"); v.AsInt64() != 42 {
		t.Errorf("metadata.latency = %d, want 42", v.AsInt64())
	}
	if v := attrValue(toolSpan.Attributes, "goai.tool.metadata.cached"); v.AsBool() != true {
		t.Errorf("metadata.cached = %v, want true", v.AsBool())
	}
	if v := attrValue(toolSpan.Attributes, "goai.tool.metadata.score"); v.AsFloat64() != 0.95 {
		t.Errorf("metadata.score = %f, want 0.95", v.AsFloat64())
	}
}

// ---------------------------------------------------------------------------
// A3: MessageCount / ToolCount
// ---------------------------------------------------------------------------

func TestA3_MessageCountAndToolCount(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	llmSpans := spansByName(sr, "chat test-model")
	if len(llmSpans) == 0 {
		t.Fatal("no LLM spans found")
	}

	// First LLM call should have message_count and tool_count attributes.
	first := llmSpans[0]
	if v := attrValue(first.Attributes, "goai.request.message_count"); v.AsInt64() == 0 {
		t.Error("goai.request.message_count should be > 0 on first LLM span")
	}
	if v := attrValue(first.Attributes, "goai.request.tool_count"); v.AsInt64() != 1 {
		t.Errorf("goai.request.tool_count = %d, want 1", v.AsInt64())
	}
}

// ---------------------------------------------------------------------------
// A6: StepResult fields (Response.Model, Response.ID)
// ---------------------------------------------------------------------------

func TestA6_StepResultFields(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
				Response:     provider.ResponseMetadata{Model: "gpt-4o-2024-05-13", ID: "chatcmpl-abc123"},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}

	if v := attrValue(root.Attributes, "gen_ai.response.model"); v.AsString() != "gpt-4o-2024-05-13" {
		t.Errorf("gen_ai.response.model = %q, want %q", v.AsString(), "gpt-4o-2024-05-13")
	}
	if v := attrValue(root.Attributes, "gen_ai.response.id"); v.AsString() != "chatcmpl-abc123" {
		t.Errorf("gen_ai.response.id = %q, want %q", v.AsString(), "chatcmpl-abc123")
	}
}

// ---------------------------------------------------------------------------
// A7: Tool output recording
// ---------------------------------------------------------------------------

func TestA7_ToolOutputRecording(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", `{"temp":72}`)

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp), RecordToolIO(true)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	// Assert goai.tool.output event exists.
	if !hasEvent(toolSpan, "goai.tool.output") {
		t.Error("expected goai.tool.output event on tool span")
	}
	if v := eventAttrValue(toolSpan, "goai.tool.output", "goai.tool.output"); v.AsString() != `{"temp":72}` {
		t.Errorf("goai.tool.output = %q, want %q", v.AsString(), `{"temp":72}`)
	}

	// Also verify tool input event.
	if !hasEvent(toolSpan, "goai.tool.input") {
		t.Error("expected goai.tool.input event on tool span")
	}
}

// ---------------------------------------------------------------------------
// B1: Skip reason
// ---------------------------------------------------------------------------

func TestB1_SkipReason(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	skipErr := errors.New("permission denied: admin only")
	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeToolExecute(func(_ goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			return goai.BeforeToolExecuteResult{Skip: true, Error: skipErr}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	// Assert goai.tool.skipped attribute.
	if v := attrValue(toolSpan.Attributes, "goai.tool.skipped"); v.AsBool() != true {
		t.Error("expected goai.tool.skipped=true")
	}

	// Assert goai.tool.skipped event with skip_reason.
	if !hasEvent(toolSpan, "goai.tool.skipped") {
		t.Fatal("expected goai.tool.skipped event")
	}
	if v := eventAttrValue(toolSpan, "goai.tool.skipped", "goai.tool.skip_reason"); v.AsString() != "permission denied: admin only" {
		t.Errorf("skip_reason = %q, want %q", v.AsString(), "permission denied: admin only")
	}
}

// ---------------------------------------------------------------------------
// B2: Input override
// ---------------------------------------------------------------------------

func TestB2_InputOverride(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeToolExecute(func(_ goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			return goai.BeforeToolExecuteResult{Input: json.RawMessage(`{"overridden":true}`)}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if v := attrValue(toolSpan.Attributes, "goai.tool.input_overridden"); v.AsBool() != true {
		t.Errorf("goai.tool.input_overridden = %v, want true", v.AsBool())
	}
}

// ---------------------------------------------------------------------------
// B3: Context override
// ---------------------------------------------------------------------------

func TestB3_ContextOverride(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	deadline := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	ctxWithDeadline, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeToolExecute(func(_ goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			return goai.BeforeToolExecuteResult{Ctx: ctxWithDeadline}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if v := attrValue(toolSpan.Attributes, "goai.tool.context_overridden"); v.AsBool() != true {
		t.Errorf("goai.tool.context_overridden = %v, want true", v.AsBool())
	}
	if v := attrValue(toolSpan.Attributes, "goai.tool.deadline"); v.AsString() == "" {
		t.Error("goai.tool.deadline should be set when context has deadline")
	}
}

// ---------------------------------------------------------------------------
// B4: Pre-execution span event
// ---------------------------------------------------------------------------

func TestB4_PreExecutionSpanEvent(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if !hasEvent(toolSpan, "goai.tool.execute_start") {
		t.Error("expected goai.tool.execute_start event on tool span")
	}
}

// ---------------------------------------------------------------------------
// C1: Output modified
// ---------------------------------------------------------------------------

func TestC1_OutputModified(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "original_output")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnAfterToolExecute(func(_ goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
			return goai.AfterToolExecuteResult{Output: "modified_output"}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if v := attrValue(toolSpan.Attributes, "goai.tool.output_modified"); v.AsBool() != true {
		t.Errorf("goai.tool.output_modified = %v, want true", v.AsBool())
	}
}

// ---------------------------------------------------------------------------
// C2: Error injected
// ---------------------------------------------------------------------------

func TestC2_ErrorInjected(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnAfterToolExecute(func(_ goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
			return goai.AfterToolExecuteResult{Error: errors.New("injected error")}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if !hasEvent(toolSpan, "goai.tool.error_modified") {
		t.Fatal("expected goai.tool.error_modified event")
	}
	if v := eventAttrValue(toolSpan, "goai.tool.error_modified", "goai.tool.error_modification"); v.AsString() != "injected" {
		t.Errorf("error_modification = %q, want %q", v.AsString(), "injected")
	}
}

// ---------------------------------------------------------------------------
// C3: Timing boundary
// ---------------------------------------------------------------------------

func TestC3_TimingBoundary(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnAfterToolExecute(func(_ goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
			return goai.AfterToolExecuteResult{}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	if !hasEvent(toolSpan, "goai.tool.after_execute") {
		t.Error("expected goai.tool.after_execute event on tool span")
	}
}

// ---------------------------------------------------------------------------
// D1: Hook stopped
// ---------------------------------------------------------------------------

func TestD1_HookStopped(t *testing.T) {
	tp, sr, mp, mr := newTestProviders(t)

	// Use a model that does two tool-call rounds so OnBeforeStep fires.
	call := 0
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			call++
			if call <= 2 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "tc" + strings.Repeat("x", call), Name: "my_tool", Input: []byte(`{}`)},
					},
					Usage:    provider.Usage{InputTokens: 10, OutputTokens: 5},
					Response: provider.ResponseMetadata{Model: "test-model-actual", ID: "resp-001"},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
				Response:     provider.ResponseMetadata{Model: "test-model-actual", ID: "resp-002"},
			}, nil
		},
	}
	tool := simpleTool("my_tool", "result")

	stopped := false
	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeStep(func(info goai.BeforeStepInfo) goai.BeforeStepResult {
			// Stop on step 2 (first OnBeforeStep invocation).
			if info.Step == 2 {
				stopped = true
				return goai.BeforeStepResult{Stop: true}
			}
			return goai.BeforeStepResult{}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(5),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if !stopped {
		t.Fatal("OnBeforeStep hook was not called")
	}

	// The WrapOnBeforeStep wrapper calls end() when Stop=true, so the root span
	// is ended with hook_stopped attributes.
	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}
	if v := attrValue(root.Attributes, "goai.stopped_by_hook"); v.AsBool() != true {
		t.Errorf("goai.stopped_by_hook = %v, want true", v.AsBool())
	}
	if v := attrValue(root.Attributes, "goai.stopped_at_step"); v.AsInt64() != 2 {
		t.Errorf("goai.stopped_at_step = %d, want 2", v.AsInt64())
	}
	if v := attrValue(root.Attributes, "goai.termination_reason"); v.AsString() != "hook_stopped" {
		t.Errorf("goai.termination_reason = %q, want %q", v.AsString(), "hook_stopped")
	}
	// Also verify the span event and metric.
	if !hasEvent(root, "goai.loop.stopped") {
		t.Error("root span missing goai.loop.stopped event")
	}
	rm := collectMetrics(t, mr)
	earlyStopMetric := findMetric(rm, "goai.loop.early_stop")
	if earlyStopMetric == nil {
		t.Fatal("goai.loop.early_stop metric not found")
	}
}

// ---------------------------------------------------------------------------
// D2: Termination reason
// ---------------------------------------------------------------------------

func TestD2_TerminationReason_Natural(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}
	if v := attrValue(root.Attributes, "goai.termination_reason"); v.AsString() != "natural" {
		t.Errorf("goai.termination_reason = %q, want %q", v.AsString(), "natural")
	}
}

func TestD2_TerminationReason_HookStopped(t *testing.T) {
	tp, sr, mp, mr := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	stopped := false
	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeStep(func(_ goai.BeforeStepInfo) goai.BeforeStepResult {
			stopped = true
			return goai.BeforeStepResult{Stop: true}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	if !stopped {
		t.Fatal("OnBeforeStep hook was not called")
	}

	// The WrapOnBeforeStep wrapper calls end() when Stop=true, ending the root span
	// with termination_reason="hook_stopped".
	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}
	if v := attrValue(root.Attributes, "goai.termination_reason"); v.AsString() != "hook_stopped" {
		t.Errorf("goai.termination_reason = %q, want %q", v.AsString(), "hook_stopped")
	}
	// Also verify the metric.
	rm := collectMetrics(t, mr)
	earlyStopMetric := findMetric(rm, "goai.loop.early_stop")
	if earlyStopMetric == nil {
		t.Fatal("goai.loop.early_stop metric not found -- hook_stopped should increment counter")
	}
}

// ---------------------------------------------------------------------------
// D3: ExtraMessages
// ---------------------------------------------------------------------------

func TestD3_ExtraMessages(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		goai.WithOnBeforeStep(func(_ goai.BeforeStepInfo) goai.BeforeStepResult {
			return goai.BeforeStepResult{
				ExtraMessages: []provider.Message{
					{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "extra context"}}},
				},
			}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	// The second LLM span should have goai.step.injected_messages attribute.
	llmSpans := spansByName(sr, "chat test-model")
	if len(llmSpans) < 2 {
		t.Fatalf("expected at least 2 LLM spans, got %d", len(llmSpans))
	}

	second := llmSpans[1]
	if v := attrValue(second.Attributes, "goai.step.injected_messages"); v.AsInt64() != 1 {
		t.Errorf("goai.step.injected_messages = %d, want 1", v.AsInt64())
	}
}

// ---------------------------------------------------------------------------
// D4: Message count gauge
// ---------------------------------------------------------------------------

func TestD4_MessageCountGauge(t *testing.T) {
	tp, _, mp, mr := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	rm := collectMetrics(t, mr)
	msgMetric := findMetric(rm, "goai.conversation.message_count")
	if msgMetric == nil {
		t.Fatal("goai.conversation.message_count metric not found")
	}
	// Verify the gauge has data points with non-zero values.
	// OnBeforeStep fires at step 2 with the accumulated messages from step 1.
	gauge, ok := msgMetric.Data.(metricdata.Gauge[int64])
	if !ok {
		t.Fatalf("expected Gauge[int64], got %T", msgMetric.Data)
	}
	if len(gauge.DataPoints) == 0 {
		t.Fatal("goai.conversation.message_count has no data points")
	}
	// The message count should be > 0 (at least the original prompt + tool round-trip).
	for _, dp := range gauge.DataPoints {
		if dp.Value <= 0 {
			t.Errorf("goai.conversation.message_count data point value = %d, want > 0", dp.Value)
		}
	}
}

// ---------------------------------------------------------------------------
// Wrapper pattern: User OnBeforeToolExecute + WithTracing
// ---------------------------------------------------------------------------

func TestWrapperPattern_UserSkipHonored(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := toolLoopModel("my_tool", "tc1", "done")
	tool := simpleTool("my_tool", "result")

	userHookCalled := false
	_, err := goai.GenerateText(context.Background(), model,
		// User hook registered BEFORE WithTracing.
		goai.WithOnBeforeToolExecute(func(_ goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			userHookCalled = true
			return goai.BeforeToolExecuteResult{Skip: true, Result: "user-skip", Error: errors.New("blocked")}
		}),
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if !userHookCalled {
		t.Error("user's OnBeforeToolExecute hook was not called")
	}

	toolSpan := spanByName(sr, "execute_tool my_tool")
	if toolSpan.Name == "" {
		t.Fatal("tool span not found")
	}

	// User's Skip=true should be honored.
	if v := attrValue(toolSpan.Attributes, "goai.tool.skipped"); v.AsBool() != true {
		t.Errorf("goai.tool.skipped = %v, want true (user skip should be honored)", v.AsBool())
	}

	// Observability annotations should still be present.
	if !hasEvent(toolSpan, "goai.tool.execute_start") {
		t.Error("expected goai.tool.execute_start event (observability wrapper)")
	}

	// B1: skip reason from user's error should be captured.
	if !hasEvent(toolSpan, "goai.tool.skipped") {
		t.Fatal("expected goai.tool.skipped event")
	}
	if v := eventAttrValue(toolSpan, "goai.tool.skipped", "goai.tool.skip_reason"); v.AsString() != "blocked" {
		t.Errorf("skip_reason = %q, want %q", v.AsString(), "blocked")
	}
}

func TestD2_TerminationReason_MaxSteps(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	// Model always returns tool calls, so MaxSteps will be exhausted.
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}
	tool := simpleTool("my_tool", "result")

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(2),
		goai.WithTools(tool),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}
	if v := attrValue(root.Attributes, "goai.termination_reason"); v.AsString() != "max_steps" {
		t.Errorf("goai.termination_reason = %q, want %q", v.AsString(), "max_steps")
	}
}

func TestA6_SourcesAndProviderMetadata(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "hello",
				FinishReason: provider.FinishStop,
				Usage: provider.Usage{
					InputTokens:  10,
					OutputTokens: 5,
				},
				Sources: []provider.Source{
					{Title: "Wikipedia", URL: "https://en.wikipedia.org"},
				},
				ProviderMetadata: map[string]map[string]any{
					"openai": {"system_fingerprint": "fp_abc123"},
				},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	root := spanByName(sr, "chat")
	if root.Name == "" {
		t.Fatal("root span not found")
	}

	// A6: Sources should produce a gen_ai.content.sources event with URLs.
	if !hasEvent(root, "gen_ai.content.sources") {
		t.Fatal("expected gen_ai.content.sources event on root span")
	}
	sourcesVal := eventAttrValue(root, "gen_ai.content.sources", "gen_ai.sources")
	sources := sourcesVal.AsStringSlice()
	if len(sources) != 1 || sources[0] != "https://en.wikipedia.org" {
		t.Errorf("gen_ai.sources = %v, want [https://en.wikipedia.org]", sources)
	}

	// A6: ProviderMetadata should produce goai.provider_metadata.* attributes.
	if v := attrValue(root.Attributes, "goai.provider_metadata.openai.system_fingerprint"); v.AsString() != "fp_abc123" {
		t.Errorf("provider_metadata attr = %q, want %q", v.AsString(), "fp_abc123")
	}
}

func TestWithTracing_CacheTokens(t *testing.T) {
	tp, sr, mp, _ := newTestProviders(t)

	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "hello",
				FinishReason: provider.FinishStop,
				Usage: provider.Usage{
					InputTokens:     10,
					OutputTokens:    5,
					CacheReadTokens:  100,
					CacheWriteTokens: 50,
				},
			}, nil
		},
	}

	_, err := goai.GenerateText(context.Background(), model,
		WithTracing(WithTracerProvider(tp), WithMeterProvider(mp)),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	llm := spanByName(sr, "chat test-model")
	if llm.Name == "" {
		t.Fatal("llm span not found")
	}

	if v := attrValue(llm.Attributes, "gen_ai.usage.cache_read.input_tokens"); v.AsInt64() != 100 {
		t.Errorf("cache_read.input_tokens = %d, want 100", v.AsInt64())
	}
	if v := attrValue(llm.Attributes, "gen_ai.usage.cache_creation.input_tokens"); v.AsInt64() != 50 {
		t.Errorf("cache_creation.input_tokens = %d, want 50", v.AsInt64())
	}
}
