package langfuse

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestAsyncShutdownAfterCanceledToolStream(t *testing.T) {
	srv, getEvents := newCaptureServer(t)
	defer srv.Close()

	runCtx, cancelRun := context.WithCancel(t.Context())
	defer cancelRun()
	toolStarted := make(chan struct{})
	cleaningUp := make(chan struct{})
	cleanupGate := make(chan struct{})
	releaseCleanup := sync.OnceFunc(func() { close(cleanupGate) })
	defer releaseCleanup()
	tool := goai.Tool{
		Name: "gated_tool",
		Execute: func(ctx context.Context, _ json.RawMessage) (string, error) {
			close(toolStarted)
			<-ctx.Done()
			close(cleaningUp)
			<-cleanupGate
			return "", ctx.Err()
		},
	}

	source := make(chan provider.StreamChunk)
	model := &mockModel{
		id: "test",
		streamFn: func(context.Context, provider.GenerateParams) (*provider.StreamResult, error) {
			return &provider.StreamResult{Stream: source}, nil
		},
	}
	traceFinished := make(chan struct{})
	stream, err := goai.StreamText(runCtx, model,
		goai.WithPrompt("run the tool"),
		goai.WithTools(tool),
		goai.WithMaxSteps(2),
		// Stop after tool execution so Langfuse submits in OnFinish, not through a step-2 error.
		goai.WithStopWhen(func([]goai.StepResult) bool { return true }),
		WithTracing(PublicKey("pub"), SecretKey("sec"), Host(srv.URL), Async(true)),
		goai.WithOnFinish(func(goai.FinishInfo) { close(traceFinished) }),
	)
	if err != nil {
		t.Fatal(err)
	}

	chunks := stream.Stream()
	func() {
		defer close(source)
		feedCtx, cancel := context.WithTimeout(t.Context(), tracingTestTimeout)
		defer cancel()
		send := func(chunk provider.StreamChunk) {
			select {
			case source <- chunk:
			case <-feedCtx.Done():
				t.Fatalf("sending provider chunk: %v", feedCtx.Err())
			}
		}
		// Fill the unread consumer buffer; forwarding the tool call then blocks
		// the consumer, but the producer can proceed to Execute.
		for range cap(chunks) {
			send(provider.StreamChunk{Type: provider.ChunkText, Text: "x"})
		}
		send(provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: tool.Name, ToolInput: `{}`})
		send(provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls})
	}()
	awaitTracing(t, toolStarted, "tool execution")
	cancelRun()
	awaitTracing(t, cleaningUp, "tool cancellation cleanup")

	// Err observes consumer completion without reading Result fields that the
	// producer has yet to write. It can return while the tool is still cleaning up.
	consumerDone := make(chan error, 1)
	go func() { consumerDone <- stream.Err() }()
	if err := awaitTracing(t, consumerDone, "canceled stream consumer"); !errors.Is(err, context.Canceled) {
		t.Fatalf("stream.Err() = %v, want context.Canceled", err)
	}
	select {
	case <-traceFinished:
		t.Fatal("OnFinish ran while the tool was still cleaning up")
	default:
	}
	flushTracing(t)
	if events := getEvents(); len(events) != 0 {
		t.Fatalf("events before releasing tool cleanup = %v, want none", eventTypes(events))
	}

	shutdown := func(ctx context.Context) error {
		select {
		case <-traceFinished:
			return Flush(ctx)
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	expiredCtx, cancelExpired := context.WithDeadline(t.Context(), time.Now())
	defer cancelExpired()
	if err := shutdown(expiredCtx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("shutdown while tool cleanup is blocked = %v, want context.DeadlineExceeded", err)
	}

	releaseCleanup()
	shutdownCtx, cancelShutdown := context.WithTimeout(t.Context(), tracingTestTimeout)
	defer cancelShutdown()
	if err := shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown after releasing tool cleanup: %v", err)
	}
	events := getEvents()
	bodyOf(t, events, eventTrace)
	spans := toolSpanBodies(t, events)
	if len(spans) != 1 || spans[0]["name"] != tool.Name || spans[0]["statusMessage"] != context.Canceled.Error() {
		t.Errorf("tool spans = %v, want one %q span reporting cancellation", spans, tool.Name)
	}
}
