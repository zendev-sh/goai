package langfuse

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestAsyncParallelEagerSpansWaitBeforeFinalBatch(t *testing.T) {
	t.Run("release_oldest_first", func(t *testing.T) { checkParallelEagerRelease(t, true) })
	t.Run("release_newest_first", func(t *testing.T) { checkParallelEagerRelease(t, false) })
}

func checkParallelEagerRelease(t *testing.T, oldestFirst bool) {
	t.Helper()
	const firstCall, secondCall, toolName = "call-1", "call-2", "test_tool"
	outputs := map[string]string{firstCall: "first result", secondCall: "second result"}
	firstResponse, secondResponse := make(chan struct{}), make(chan struct{})
	gates := map[string]<-chan struct{}{firstCall: firstResponse, secondCall: secondResponse}
	releaseFirst := sync.OnceFunc(func() { close(firstResponse) })
	releaseSecond := sync.OnceFunc(func() { close(secondResponse) })
	requests := make(chan []map[string]any, 3)
	srv, _ := newCaptureServerWithGate(t, func(ctx context.Context, batch []map[string]any) {
		select {
		case requests <- batch:
		default:
			t.Error("received more than three ingestion batches")
			return
		}
		if len(batch) == 1 && batch[0]["type"] == eventSpan {
			body, _ := batch[0]["body"].(map[string]any)
			meta, _ := body["metadata"].(map[string]any)
			callID, _ := meta["tool_call_id"].(string)
			if gate := gates[callID]; gate != nil {
				select {
				case <-gate:
				case <-ctx.Done():
				}
			}
		}
	})
	defer srv.Close()
	defer releaseFirst()
	defer releaseSecond()

	runCtx, cancelRun := context.WithTimeout(t.Context(), tracingTestTimeout)
	defer cancelRun()
	firstSubmitted := make(chan struct{})
	tool := goai.Tool{
		Name: toolName,
		Execute: func(ctx context.Context, _ json.RawMessage) (string, error) {
			callID := goai.ToolCallIDFromContext(ctx)
			if callID == firstCall {
				close(firstSubmitted)
			}
			return outputs[callID], nil
		},
	}
	calls := []provider.ToolCall{
		{ID: firstCall, Name: toolName, Input: json.RawMessage(`{}`)},
		{ID: secondCall, Name: toolName, Input: json.RawMessage(`{}`)},
	}
	model := &mockModel{
		id: "test",
		generateFn: func(context.Context, provider.GenerateParams) (*provider.GenerateResult, error) {
			if calls != nil {
				step := &provider.GenerateResult{ToolCalls: calls, FinishReason: provider.FinishToolCalls}
				calls = nil
				return step, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}
	finished := make(chan struct{})
	var result *goai.TextResult
	var runErr error
	go func() {
		defer close(finished)
		result, runErr = goai.GenerateText(runCtx, model,
			goai.WithPrompt("run both tools"),
			goai.WithTools(tool),
			goai.WithMaxSteps(2),
			goai.WithOnToolCallStart(func(info goai.ToolCallStartInfo) {
				if info.ToolCallID == secondCall {
					// The first Execute starts after its eager submission. Enforce
					// submission order without depending on HTTP arrival order.
					select {
					case <-firstSubmitted:
					case <-runCtx.Done():
					}
				}
			}),
			tracingToServer(srv.URL, true, EagerToolSpans(true)),
		)
	}()

	eagerIDs := map[string]string{}
	for range len(outputs) {
		batch := awaitTracing(t, requests, "both eager requests before releasing either HTTP response")
		if len(batch) != 1 {
			t.Fatalf("eager batch event types = %v, want one span create", eventTypes(batch))
		}
		body := bodyOf(t, batch, eventSpan)
		meta, _ := body["metadata"].(map[string]any)
		callID, _ := meta["tool_call_id"].(string)
		id, _ := body["id"].(string)
		if _, valid := outputs[callID]; !valid || id == "" || eagerIDs[callID] != "" || meta["status"] != "in_progress" {
			t.Fatalf("unexpected eager span: %v; previous eager IDs: %v", body, eagerIDs)
		}
		eagerIDs[callID] = id
	}
	if eagerIDs[firstCall] == eagerIDs[secondCall] {
		t.Fatalf("eager spans reused ID %s", eagerIDs[firstCall])
	}
	awaitTracing(t, finished, "GenerateText completion with both eager responses blocked")
	if runErr != nil || result == nil || result.Text != "done" {
		t.Fatalf("GenerateText result = %+v, error = %v; want successful final text", result, runErr)
	}

	releaseOne, releaseOther := releaseFirst, releaseSecond
	if !oldestFirst {
		releaseOne, releaseOther = releaseOther, releaseOne
	}
	releaseOne()
	assertTracingBlocked(t, requests, "final batch arrived while one eager request was blocked")
	releaseOther()
	flushTracing(t)
	batch := awaitTracing(t, requests, "final ingestion batch")
	bodyOf(t, batch, eventTrace)
	for _, event := range batch {
		if event["type"] != eventSpanUpdate {
			continue
		}
		body, _ := event["body"].(map[string]any)
		meta, _ := body["metadata"].(map[string]any)
		callID, _ := meta["tool_call_id"].(string)
		if wantID := eagerIDs[callID]; wantID == "" || body["id"] != wantID || body["output"] != outputs[callID] {
			t.Fatalf("unexpected span update for %q: %v; wanted eager ID %q and output %q", callID, body, wantID, outputs[callID])
		}
		delete(eagerIDs, callID)
	}
	if len(eagerIDs) != 0 {
		t.Errorf("missing span updates for eager IDs %v", eagerIDs)
	}
}
