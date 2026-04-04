package langfuse

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
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

// newCaptureServer starts an httptest.Server that captures all ingestion events.
// The returned function can be called to retrieve all captured events (decoded).
func newCaptureServer(t *testing.T) (*httptest.Server, func() []map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var allEvents []map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Logf("capture server: read body: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		var payload struct {
			Batch []map[string]any `json:"batch"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Logf("capture server: decode: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		mu.Lock()
		allEvents = append(allEvents, payload.Batch...)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))

	getEvents := func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		out := make([]map[string]any, len(allEvents))
		copy(out, allEvents)
		return out
	}
	return srv, getEvents
}

// newHooks creates a Hooks instance pointing at the given test server.
func newHooks(t *testing.T, srv *httptest.Server, cfg Config) *Hooks {
	t.Helper()
	cfg.Host = srv.URL
	cfg.PublicKey = "pub"
	cfg.SecretKey = "sec"
	return New(cfg)
}

// eventTypes returns the "type" field of each event.
func eventTypes(events []map[string]any) []string {
	out := make([]string, 0, len(events))
	for _, e := range events {
		if t, ok := e["type"].(string); ok {
			out = append(out, t)
		}
	}
	return out
}

// bodyOf returns the body map for the first event matching the given type.
func bodyOf(t *testing.T, events []map[string]any, eventType string) map[string]any {
	t.Helper()
	for _, e := range events {
		if e["type"] == eventType {
			if b, ok := e["body"].(map[string]any); ok {
				return b
			}
		}
	}
	t.Fatalf("no event of type %q found in %d events", eventType, len(events))
	return nil
}

// --- Tests ------------------------------------------------------------------

func TestNew_DefaultTraceName(t *testing.T) {
	h := New(Config{})
	if h.cfg.TraceName != "agent" {
		t.Errorf("TraceName = %q, want %q", h.cfg.TraceName, "agent")
	}
}

func TestRun_SingleStep(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "test-trace"})

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "hello",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	if len(events) != 3 {
		t.Fatalf("event count = %d, want 3; types: %v", len(events), eventTypes(events))
	}

	types := eventTypes(events)
	hasTrace, hasSpan, hasGen := false, false, false
	for _, tp := range types {
		switch tp {
		case eventTrace:
			hasTrace = true
		case eventSpan:
			hasSpan = true
		case eventGeneration:
			hasGen = true
		}
	}
	if !hasTrace {
		t.Error("missing trace-create event")
	}
	if !hasSpan {
		t.Error("missing span-create event")
	}
	if !hasGen {
		t.Error("missing generation-create event")
	}

	traceBody := bodyOf(t, events, eventTrace)
	if traceBody["name"] != "test-trace" {
		t.Errorf("trace name = %q, want %q", traceBody["name"], "test-trace")
	}

	genBody := bodyOf(t, events, eventGeneration)
	if genBody["model"] != "gpt-4o" {
		t.Errorf("generation model = %q, want %q", genBody["model"], "gpt-4o")
	}
	usage, ok := genBody["usage"].(map[string]any)
	if !ok {
		t.Fatalf("generation usage missing or wrong type: %T", genBody["usage"])
	}
	if usage["input"] != float64(10) {
		t.Errorf("usage.input = %v, want 10", usage["input"])
	}
	if usage["output"] != float64(5) {
		t.Errorf("usage.output = %v, want 5", usage["output"])
	}
}

func TestRun_MultiStep_WithToolCalls(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "multi-step"})

	callCount := 0
	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
			}, nil
		},
	}

	opts := append(h.Run(),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(goai.Tool{
			Name:    "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "tool result", nil },
		}),
	)
	_, err := goai.GenerateText(t.Context(), model, opts...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	if len(events) != 5 {
		t.Fatalf("event count = %d, want 5; types: %v", len(events), eventTypes(events))
	}

	// Count event types
	counts := map[string]int{}
	for _, tp := range eventTypes(events) {
		counts[tp]++
	}
	if counts[eventTrace] != 1 {
		t.Errorf("trace events = %d, want 1", counts[eventTrace])
	}
	if counts[eventSpan] != 2 {
		t.Errorf("span events = %d, want 2 (agent span + tool span)", counts[eventSpan])
	}
	if counts[eventGeneration] != 2 {
		t.Errorf("generation events = %d, want 2", counts[eventGeneration])
	}

	// Find tool span (not the agent span)
	var toolSpan map[string]any
	for _, e := range events {
		if e["type"] == eventSpan {
			b, ok := e["body"].(map[string]any)
			if !ok {
				continue
			}
			if b["name"] == "my_tool" {
				toolSpan = b
				break
			}
		}
	}
	if toolSpan == nil {
		t.Fatal("tool span not found")
	}
	if toolSpan["name"] != "my_tool" {
		t.Errorf("tool span name = %q, want %q", toolSpan["name"], "my_tool")
	}
}

func TestRun_ConcurrentRuns(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "concurrent"})

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	var wg sync.WaitGroup
	const numGoroutines = 5
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			opts := append(h.Run(), goai.WithPrompt("hi"))
			_, _ = goai.GenerateText(context.Background(), model, opts...)
		}()
	}
	wg.Wait()

	events := getEvents()
	const wantEvents = numGoroutines * 3
	if len(events) != wantEvents {
		t.Fatalf("event count = %d, want %d (%d runs × 3 events each); types: %v",
			len(events), wantEvents, numGoroutines, eventTypes(events))
	}

	// Collect all trace IDs
	traceIDs := map[string]bool{}
	for _, e := range events {
		if e["type"] == eventTrace {
			if b, ok := e["body"].(map[string]any); ok {
				if id, ok := b["id"].(string); ok {
					traceIDs[id] = true
				}
			}
		}
	}
	if len(traceIDs) != numGoroutines {
		t.Errorf("distinct trace IDs = %d, want %d", len(traceIDs), numGoroutines)
	}
}

func TestRun_TokenUsage(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "result",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 100, OutputTokens: 42},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hello"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	genBody := bodyOf(t, events, eventGeneration)
	usage, ok := genBody["usage"].(map[string]any)
	if !ok {
		t.Fatalf("usage missing or wrong type: %T", genBody["usage"])
	}
	if usage["input"] != float64(100) {
		t.Errorf("usage.input = %v, want 100", usage["input"])
	}
	if usage["output"] != float64(42) {
		t.Errorf("usage.output = %v, want 42", usage["output"])
	}
	if usage["total"] != float64(142) {
		t.Errorf("usage.total = %v, want 142", usage["total"])
	}
}

func TestRun_ToolError(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})

	callCount := 0
	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "broken_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	opts := append(h.Run(),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(goai.Tool{
			Name: "broken_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "", fmt.Errorf("tool exploded")
			},
		}),
	)
	_, err := goai.GenerateText(t.Context(), model, opts...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()

	// Find tool span
	var toolSpan map[string]any
	for _, e := range events {
		if e["type"] == eventSpan {
			b, ok := e["body"].(map[string]any)
			if !ok {
				continue
			}
			if b["name"] == "broken_tool" {
				toolSpan = b
				break
			}
		}
	}
	if toolSpan == nil {
		t.Fatal("tool span not found")
	}
	if toolSpan["level"] != levelError {
		t.Errorf("tool span level = %q, want %q", toolSpan["level"], levelError)
	}
	if toolSpan["statusMessage"] != "tool exploded" {
		t.Errorf("tool span statusMessage = %q, want %q", toolSpan["statusMessage"], "tool exploded")
	}
}

func TestWith_CallsRunPerInvocation(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})
	factory := h.With()

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	extractTraceIDs := func(events []map[string]any) []string {
		var ids []string
		for _, e := range events {
			if e["type"] == eventTrace {
				if b, ok := e["body"].(map[string]any); ok {
					if id, ok := b["id"].(string); ok {
						ids = append(ids, id)
					}
				}
			}
		}
		return ids
	}

	// Run twice via the same factory
	opts1 := append(factory(), goai.WithPrompt("hi"))
	if _, err := goai.GenerateText(t.Context(), model, opts1...); err != nil {
		t.Fatalf("run 1: %v", err)
	}
	opts2 := append(factory(), goai.WithPrompt("hi"))
	if _, err := goai.GenerateText(t.Context(), model, opts2...); err != nil {
		t.Fatalf("run 2: %v", err)
	}

	traceIDs := extractTraceIDs(getEvents())
	if len(traceIDs) != 2 {
		t.Fatalf("expected 2 traces, got %d", len(traceIDs))
	}
	if traceIDs[0] == traceIDs[1] {
		t.Errorf("With() factory should produce separate trace IDs per invocation; both = %q", traceIDs[0])
	}
}

func TestRun_WithSystemMessage(t *testing.T) {
	// Exercises the requestMessages(system != "") code path - system message
	// is prepended to info.Messages so it appears first in the generation input.
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "hi", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model,
		append(h.Run(), goai.WithSystem("you are a helper"), goai.WithPrompt("hello"))...,
	)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	genBody := bodyOf(t, events, eventGeneration)
	input, ok := genBody["input"].([]any)
	if !ok || len(input) == 0 {
		t.Fatalf("generation input not a non-empty list: %T %v", genBody["input"], genBody["input"])
	}
	// System message is prepended first
	first, ok := input[0].(map[string]any)
	if !ok || first["role"] != "system" {
		t.Errorf("first input message role = %v, want system", first["role"])
	}
}

func TestRun_Environment(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{Environment: "staging"})

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	traceBody := bodyOf(t, events, eventTrace)
	meta, ok := traceBody["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("trace metadata not a map: %T", traceBody["metadata"])
	}
	if meta["environment"] != "staging" {
		t.Errorf("trace metadata.environment = %v, want staging", meta["environment"])
	}
}

func TestRun_FinishLengthWarning(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "truncated...",
				FinishReason: provider.FinishLength,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 10},
			}, nil
		},
	}

	_, _ = goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)

	events := getEvents()
	genBody := bodyOf(t, events, eventGeneration)
	if genBody["level"] != levelWarning {
		t.Errorf("generation level = %v, want %q", genBody["level"], levelWarning)
	}
}

func TestRun_CacheTokensInMetadata(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage: provider.Usage{
					InputTokens:      10,
					OutputTokens:     5,
					CacheReadTokens:  8,
					CacheWriteTokens: 3,
				},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	genBody := bodyOf(t, events, eventGeneration)
	meta, ok := genBody["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("generation metadata not a map: %T", genBody["metadata"])
	}
	if meta["cache_read_tokens"] != float64(8) {
		t.Errorf("cache_read_tokens = %v, want 8", meta["cache_read_tokens"])
	}
	if meta["cache_write_tokens"] != float64(3) {
		t.Errorf("cache_write_tokens = %v, want 3", meta["cache_write_tokens"])
	}
}

func TestRun_EnvVarCredentials(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	t.Setenv("LANGFUSE_PUBLIC_KEY", "env-pub")
	t.Setenv("LANGFUSE_SECRET_KEY", "env-sec")
	t.Setenv("LANGFUSE_HOST", srv.URL)

	h := New(Config{}) // no credentials set - must fall back to env vars

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	// If env vars were not read, no events would arrive at the server.
	if len(getEvents()) == 0 {
		t.Error("no events received - env var credentials were not used")
	}
}

func TestMergeMeta_NonMapBase(t *testing.T) {
	result := mergeMeta("not-a-map", map[string]any{"k": "v"})
	if result["k"] != "v" {
		t.Errorf("mergeMeta[k] = %v, want v", result["k"])
	}
	if len(result) != 1 {
		t.Errorf("mergeMeta len = %d, want 1", len(result))
	}
}

func TestMergeMeta_WithBaseMap(t *testing.T) {
	base := map[string]any{"a": 1, "b": 2}
	result := mergeMeta(base, map[string]any{"b": 99, "c": 3})
	if result["a"] != 1 {
		t.Errorf("result[a] = %v, want 1", result["a"])
	}
	if result["b"] != 99 {
		t.Errorf("result[b] = %v, want 99 (extra overrides base)", result["b"])
	}
	if result["c"] != 3 {
		t.Errorf("result[c] = %v, want 3", result["c"])
	}
}

func TestLastAssistantOutput_TextOnly(t *testing.T) {
	msg := provider.Message{
		Role: provider.RoleAssistant,
		Content: []provider.Part{
			{Type: provider.PartText, Text: "hello world"},
		},
	}
	out := lastAssistantOutput(msg)
	if out != "hello world" {
		t.Errorf("lastAssistantOutput = %v, want %q", out, "hello world")
	}
}

func TestLastAssistantOutput_MultipleTextParts(t *testing.T) {
	msg := provider.Message{
		Role: provider.RoleAssistant,
		Content: []provider.Part{
			{Type: provider.PartText, Text: "part one"},
			{Type: provider.PartText, Text: "part two"},
		},
	}
	out := lastAssistantOutput(msg)
	parts, ok := out.([]string)
	if !ok {
		t.Fatalf("type = %T, want []string", out)
	}
	if len(parts) != 2 || parts[0] != "part one" || parts[1] != "part two" {
		t.Errorf("parts = %v, want [part one part two]", parts)
	}
}

func TestToolCallMap_InvalidJSONInput(t *testing.T) {
	p := provider.Part{
		Type:       provider.PartToolCall,
		ToolCallID: "tc1",
		ToolName:   "my_tool",
		ToolInput:  json.RawMessage(`not-valid-json`),
	}
	tc := toolCallMap(p)
	if tc["input"] != "not-valid-json" {
		t.Errorf("input = %v, want raw string fallback", tc["input"])
	}
}

func TestRun_TraceHierarchy(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "hierarchy-test"})

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "result",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("test"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()

	// Get agent span ID
	agentSpanBody := bodyOf(t, events, eventSpan)
	agentSpanID, ok := agentSpanBody["id"].(string)
	if !ok || agentSpanID == "" {
		t.Fatalf("agent span ID missing; body: %v", agentSpanBody)
	}

	// Get generation parentObservationId
	genBody := bodyOf(t, events, eventGeneration)
	parentID, ok := genBody["parentObservationId"].(string)
	if !ok || parentID == "" {
		t.Fatalf("generation parentObservationId missing; body: %v", genBody)
	}

	if parentID != agentSpanID {
		t.Errorf("generation parentObservationId = %q, want agent span ID %q", parentID, agentSpanID)
	}
}

func TestRun_FlushError_OnFlushErrorCalled(t *testing.T) {
	// Server always returns 401 so flush fails.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	var flushErr error
	h := New(Config{
		Host:         srv.URL,
		PublicKey:    "bad",
		SecretKey:    "creds",
		OnFlushError: func(err error) { flushErr = err },
	})

	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	if flushErr == nil {
		t.Error("OnFlushError should have been called with a non-nil error")
	}
}

func TestRun_RunError_PartialTraceFlushes(t *testing.T) {
	// Verify that when the model call fails, end() is still called so partial
	// trace events are flushed rather than silently dropped.
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "error-test"})

	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, fmt.Errorf("model unavailable")
		},
	}

	_, _ = goai.GenerateText(t.Context(), model,
		append(h.Run(), goai.WithPrompt("hi"), goai.WithMaxRetries(0))...,
	)

	events := getEvents()
	types := eventTypes(events)

	// Even on failure we expect a trace, agent span, and one generation.
	if len(events) == 0 {
		t.Fatal("expected partial trace events to be flushed on error, got none")
	}

	hasTrace := false
	for _, ty := range types {
		if ty == eventTrace {
			hasTrace = true
		}
	}
	if !hasTrace {
		t.Errorf("expected trace-create event; got types: %v", types)
	}

	// The generation should carry an ERROR level.
	genBody := bodyOf(t, events, eventGeneration)
	if genBody["level"] != levelError {
		t.Errorf("generation level = %v, want ERROR", genBody["level"])
	}
}

func TestRun_EventOrdering(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "ordering-test"})

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	if len(events) < 3 {
		t.Fatalf("expected at least 3 events, got %d; types: %v", len(events), eventTypes(events))
	}

	// The batch is built in deterministic order: trace-create, span-create, generation-create.
	if events[0]["type"] != eventTrace {
		t.Errorf("events[0].type = %q, want %q", events[0]["type"], eventTrace)
	}
	if events[1]["type"] != eventSpan {
		t.Errorf("events[1].type = %q, want %q", events[1]["type"], eventSpan)
	}
	if events[2]["type"] != eventGeneration {
		t.Errorf("events[2].type = %q, want %q", events[2]["type"], eventGeneration)
	}
}

func TestRun_ToolSpanParentID(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "tool-parent-test"})

	callCount := 0
	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
			}, nil
		},
	}

	opts := append(h.Run(),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(goai.Tool{
			Name:    "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "result", nil },
		}),
	)
	_, err := goai.GenerateText(t.Context(), model, opts...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()

	// Find the agent span (span-create whose name matches the trace name, i.e. no parentObservationId).
	var agentSpanID string
	for _, e := range events {
		if e["type"] != eventSpan {
			continue
		}
		b, ok := e["body"].(map[string]any)
		if !ok {
			continue
		}
		// The agent span has no parentObservationId.
		if _, hasParent := b["parentObservationId"]; !hasParent {
			agentSpanID, _ = b["id"].(string)
			break
		}
	}
	if agentSpanID == "" {
		t.Fatal("agent span not found")
	}

	// Find the tool span.
	var toolParentID string
	for _, e := range events {
		if e["type"] != eventSpan {
			continue
		}
		b, ok := e["body"].(map[string]any)
		if !ok {
			continue
		}
		if b["name"] == "my_tool" {
			toolParentID, _ = b["parentObservationId"].(string)
			break
		}
	}
	if toolParentID == "" {
		t.Fatal("tool span not found or missing parentObservationId")
	}

	if toolParentID != agentSpanID {
		t.Errorf("tool span parentObservationId = %q, want agent span ID %q", toolParentID, agentSpanID)
	}
}

func TestRun_ErrorOnStep2(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "error-step2"})

	callCount := 0
	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				// Step 1 succeeds with a tool call.
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			// Step 2 model call fails.
			return nil, fmt.Errorf("step 2 failed")
		},
	}

	opts := append(h.Run(),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithMaxRetries(0),
		goai.WithTools(goai.Tool{
			Name:    "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "tool result", nil },
		}),
	)
	_, _ = goai.GenerateText(t.Context(), model, opts...)

	events := getEvents()
	if len(events) == 0 {
		t.Fatal("expected partial trace to be flushed, got no events")
	}

	// Find the generation for step 2 - it should carry ERROR level.
	var errorGenFound bool
	for _, e := range events {
		if e["type"] != eventGeneration {
			continue
		}
		b, ok := e["body"].(map[string]any)
		if !ok {
			continue
		}
		if b["level"] == levelError {
			errorGenFound = true
			break
		}
	}
	if !errorGenFound {
		t.Errorf("expected a generation-create event with level ERROR; types: %v", eventTypes(events))
	}
}

func TestRun_ReasoningTokens(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{})

	model := &mockModel{
		id: "o1",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "reasoned answer",
				FinishReason: provider.FinishStop,
				Usage: provider.Usage{
					InputTokens:     10,
					OutputTokens:    20,
					ReasoningTokens: 50,
				},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("think"))...)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	genBody := bodyOf(t, events, eventGeneration)
	meta, ok := genBody["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("generation metadata not a map: %T", genBody["metadata"])
	}
	if meta["reasoning_tokens"] != float64(50) {
		t.Errorf("metadata.reasoning_tokens = %v, want 50", meta["reasoning_tokens"])
	}
}

func TestRun_BaseURLEnvVar(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	// Set LANGFUSE_BASE_URL but NOT LANGFUSE_HOST - exercises the fallback path.
	t.Setenv("LANGFUSE_BASE_URL", srv.URL)
	t.Setenv("LANGFUSE_PUBLIC_KEY", "env-pub")
	t.Setenv("LANGFUSE_SECRET_KEY", "env-sec")

	h := New(Config{}) // no Host set - must fall back to LANGFUSE_BASE_URL

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}

	if len(getEvents()) == 0 {
		t.Error("no events received - LANGFUSE_BASE_URL fallback did not work")
	}
}

func TestStreamText_SingleStep(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "stream-single"})

	model := &mockModel{
		id: "gpt-4o",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{Type: provider.ChunkText, Text: " world"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := goai.StreamText(t.Context(), model, append(h.Run(), goai.WithPrompt("hi"))...)
	if err != nil {
		t.Fatal(err)
	}
	// Drain stream.
	for range stream.TextStream() {
	}
	if stream.Err() != nil {
		t.Fatal(stream.Err())
	}

	events := getEvents()
	if len(events) != 3 {
		t.Fatalf("event count = %d, want 3; types: %v", len(events), eventTypes(events))
	}

	counts := map[string]int{}
	for _, tp := range eventTypes(events) {
		counts[tp]++
	}
	if counts[eventTrace] != 1 {
		t.Errorf("trace events = %d, want 1", counts[eventTrace])
	}
	if counts[eventSpan] != 1 {
		t.Errorf("span events = %d, want 1", counts[eventSpan])
	}
	if counts[eventGeneration] != 1 {
		t.Errorf("generation events = %d, want 1", counts[eventGeneration])
	}

	genBody := bodyOf(t, events, eventGeneration)
	if genBody["model"] != "gpt-4o" {
		t.Errorf("generation model = %q, want %q", genBody["model"], "gpt-4o")
	}
	usage, ok := genBody["usage"].(map[string]any)
	if !ok {
		t.Fatalf("generation usage missing or wrong type: %T", genBody["usage"])
	}
	if usage["input"] != float64(10) {
		t.Errorf("usage.input = %v, want 10", usage["input"])
	}
	if usage["output"] != float64(5) {
		t.Errorf("usage.output = %v, want 5", usage["output"])
	}
}

func TestStreamText_MultiStep_WithToolCalls(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	h := newHooks(t, srv, Config{TraceName: "stream-multi-step"})

	callCount := 0
	model := &mockModel{
		id: "gpt-4o",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			callCount++
			if callCount == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "calling tool"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "my_tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 8}},
			), nil
		},
	}

	opts := append(h.Run(),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(goai.Tool{
			Name:    "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "tool result", nil },
		}),
	)
	stream, err := goai.StreamText(t.Context(), model, opts...)
	if err != nil {
		t.Fatal(err)
	}
	// Drain stream.
	for range stream.TextStream() {
	}
	if stream.Err() != nil {
		t.Fatal(stream.Err())
	}

	events := getEvents()
	// Expect: trace-create, span-create (agent), generation-create (step-1),
	// span-create (tool), generation-create (step-2) = 5 events.
	if len(events) != 5 {
		t.Fatalf("event count = %d, want 5; types: %v", len(events), eventTypes(events))
	}

	counts := map[string]int{}
	for _, tp := range eventTypes(events) {
		counts[tp]++
	}
	if counts[eventTrace] != 1 {
		t.Errorf("trace events = %d, want 1", counts[eventTrace])
	}
	if counts[eventSpan] != 2 {
		t.Errorf("span events = %d, want 2 (agent span + tool span)", counts[eventSpan])
	}
	if counts[eventGeneration] != 2 {
		t.Errorf("generation events = %d, want 2", counts[eventGeneration])
	}

	// Find tool span.
	var toolSpan map[string]any
	for _, e := range events {
		if e["type"] == eventSpan {
			b, ok := e["body"].(map[string]any)
			if !ok {
				continue
			}
			if b["name"] == "my_tool" {
				toolSpan = b
				break
			}
		}
	}
	if toolSpan == nil {
		t.Fatal("tool span not found")
	}
	if toolSpan["name"] != "my_tool" {
		t.Errorf("tool span name = %q, want %q", toolSpan["name"], "my_tool")
	}

	// Verify trace output.
	traceBody := bodyOf(t, events, eventTrace)
	if traceBody["name"] != "stream-multi-step" {
		t.Errorf("trace name = %q, want %q", traceBody["name"], "stream-multi-step")
	}
}

// --- WithTracing tests ------------------------------------------------------

func TestWithTracing_SingleStep(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "hello",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model,
		WithTracing(
			TraceName("single-step-test"),
			PublicKey("pub"),
			SecretKey("sec"),
			Host(srv.URL),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	if len(events) != 3 {
		t.Fatalf("event count = %d, want 3; types: %v", len(events), eventTypes(events))
	}

	traceBody := bodyOf(t, events, eventTrace)
	if traceBody["name"] != "single-step-test" {
		t.Errorf("trace name = %q, want %q", traceBody["name"], "single-step-test")
	}

	genBody := bodyOf(t, events, eventGeneration)
	if genBody["model"] != "gpt-4o" {
		t.Errorf("generation model = %q, want %q", genBody["model"], "gpt-4o")
	}
	usage, ok := genBody["usage"].(map[string]any)
	if !ok {
		t.Fatalf("generation usage missing: %T", genBody["usage"])
	}
	if usage["input"] != float64(10) {
		t.Errorf("usage.input = %v, want 10", usage["input"])
	}
}

func TestWithTracing_AllOptions(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model,
		WithTracing(
			TraceName("opts-test"),
			PublicKey("pub"),
			SecretKey("sec"),
			Host(srv.URL),
			UserID("user-42"),
			SessionID("sess-abc"),
			Tags("tag1", "tag2"),
			Metadata(map[string]any{"key": "val"}),
			Release("v1.0"),
			Version("1.0.0"),
			Environment("staging"),
			PromptName("my-prompt"),
			PromptVersion(3),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	traceBody := bodyOf(t, events, eventTrace)
	if traceBody["name"] != "opts-test" {
		t.Errorf("trace name = %q, want %q", traceBody["name"], "opts-test")
	}
	if traceBody["userId"] != "user-42" {
		t.Errorf("userId = %q, want %q", traceBody["userId"], "user-42")
	}
	if traceBody["sessionId"] != "sess-abc" {
		t.Errorf("sessionId = %q, want %q", traceBody["sessionId"], "sess-abc")
	}
	if traceBody["release"] != "v1.0" {
		t.Errorf("release = %q, want %q", traceBody["release"], "v1.0")
	}
	if traceBody["version"] != "1.0.0" {
		t.Errorf("version = %q, want %q", traceBody["version"], "1.0.0")
	}

	tags, ok := traceBody["tags"].([]any)
	if !ok || len(tags) != 2 {
		t.Errorf("tags = %v, want [tag1 tag2]", traceBody["tags"])
	}

	// Verify Environment is in trace metadata.
	traceMeta, ok := traceBody["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("trace metadata not a map: %T", traceBody["metadata"])
	}
	if traceMeta["environment"] != "staging" {
		t.Errorf("trace metadata.environment = %v, want staging", traceMeta["environment"])
	}

	// Verify Metadata values are present in trace metadata.
	if traceMeta["key"] != "val" {
		t.Errorf("trace metadata.key = %v, want val", traceMeta["key"])
	}

	genBody := bodyOf(t, events, eventGeneration)
	if genBody["promptName"] != "my-prompt" {
		t.Errorf("promptName = %q, want %q", genBody["promptName"], "my-prompt")
	}
	if genBody["promptVersion"] != float64(3) {
		t.Errorf("promptVersion = %v, want 3", genBody["promptVersion"])
	}
}

func TestWithTracing_EnvVars(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	t.Setenv("LANGFUSE_PUBLIC_KEY", "env-pub")
	t.Setenv("LANGFUSE_SECRET_KEY", "env-sec")
	t.Setenv("LANGFUSE_HOST", srv.URL)

	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model,
		WithTracing(),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(getEvents()) == 0 {
		t.Error("no events received - env var credentials were not used")
	}
}

func TestWithTracing_MultiStep_WithToolCalls(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	callCount := 0
	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
			}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model,
		WithTracing(
			TraceName("tool-test"),
			PublicKey("pub"),
			SecretKey("sec"),
			Host(srv.URL),
		),
		goai.WithPrompt("go"),
		goai.WithMaxSteps(3),
		goai.WithTools(goai.Tool{
			Name:    "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "tool result", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	events := getEvents()
	if len(events) != 5 {
		t.Fatalf("event count = %d, want 5; types: %v", len(events), eventTypes(events))
	}

	counts := map[string]int{}
	for _, tp := range eventTypes(events) {
		counts[tp]++
	}
	if counts[eventTrace] != 1 {
		t.Errorf("trace events = %d, want 1", counts[eventTrace])
	}
	if counts[eventSpan] != 2 {
		t.Errorf("span events = %d, want 2", counts[eventSpan])
	}
	if counts[eventGeneration] != 2 {
		t.Errorf("generation events = %d, want 2", counts[eventGeneration])
	}
}

func TestWithTracing_ConcurrentRuns(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	model := &mockModel{
		id: "gpt-4o",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	var wg sync.WaitGroup
	const numGoroutines = 5
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = goai.GenerateText(context.Background(), model,
				WithTracing(
					PublicKey("pub"),
					SecretKey("sec"),
					Host(srv.URL),
				),
				goai.WithPrompt("hi"),
			)
		}()
	}
	wg.Wait()

	events := getEvents()
	const wantEvents = numGoroutines * 3
	if len(events) != wantEvents {
		t.Fatalf("event count = %d, want %d (%d runs x 3 events each); types: %v",
			len(events), wantEvents, numGoroutines, eventTypes(events))
	}

	// Collect all trace IDs
	traceIDs := map[string]bool{}
	for _, e := range events {
		if e["type"] == eventTrace {
			if b, ok := e["body"].(map[string]any); ok {
				if id, ok := b["id"].(string); ok {
					traceIDs[id] = true
				}
			}
		}
	}
	if len(traceIDs) != numGoroutines {
		t.Errorf("distinct trace IDs = %d, want %d", len(traceIDs), numGoroutines)
	}
}

func TestWithTracing_MissingCredentials(t *testing.T) {
	// Clear any env vars that might provide credentials.
	t.Setenv("LANGFUSE_PUBLIC_KEY", "")
	t.Setenv("LANGFUSE_SECRET_KEY", "")
	t.Setenv("LANGFUSE_HOST", "")

	// Capture stderr output.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStderr := os.Stderr
	os.Stderr = w

	opt := WithTracing()

	_ = w.Close()
	var buf bytes.Buffer
	_, _ = buf.ReadFrom(r)
	os.Stderr = oldStderr

	stderrOutput := buf.String()
	if !strings.Contains(stderrOutput, "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set") {
		t.Errorf("expected warning on stderr, got: %q", stderrOutput)
	}

	// Verify the returned option is a no-op (doesn't panic when applied).
	model := &mockModel{
		id: "m",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err = goai.GenerateText(t.Context(), model,
		opt,
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("no-op tracing should not cause errors: %v", err)
	}
}

func TestWithTracing_StreamText_SingleStep(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	model := &mockModel{
		id: "gpt-4o",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{Type: provider.ChunkText, Text: " world"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := goai.StreamText(t.Context(), model,
		WithTracing(
			TraceName("stream-with-tracing"),
			PublicKey("pub"),
			SecretKey("sec"),
			Host(srv.URL),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatal(err)
	}
	for range stream.TextStream() {
	}
	if stream.Err() != nil {
		t.Fatal(stream.Err())
	}

	events := getEvents()
	if len(events) != 3 {
		t.Fatalf("event count = %d, want 3; types: %v", len(events), eventTypes(events))
	}

	traceBody := bodyOf(t, events, eventTrace)
	if traceBody["name"] != "stream-with-tracing" {
		t.Errorf("trace name = %q, want %q", traceBody["name"], "stream-with-tracing")
	}

	genBody := bodyOf(t, events, eventGeneration)
	usage, ok := genBody["usage"].(map[string]any)
	if !ok {
		t.Fatalf("generation usage missing: %T", genBody["usage"])
	}
	if usage["input"] != float64(10) {
		t.Errorf("usage.input = %v, want 10", usage["input"])
	}
}

func TestWithTracing_OnFlushError(t *testing.T) {
	// Server always returns 401 so flush fails.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	var flushErr error
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := goai.GenerateText(t.Context(), model,
		WithTracing(
			PublicKey("bad"),
			SecretKey("creds"),
			Host(srv.URL),
			OnFlushError(func(err error) { flushErr = err }),
		),
		goai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if flushErr == nil {
		t.Error("OnFlushError should have been called with a non-nil error via WithTracing")
	}
}
