package openai

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

const completedResponsesEvent = "event: response.completed\ndata: {\"response\":{\"id\":\"resp_1\",\"model\":\"o3\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"

func isResponsesIdleTimeout(err error) bool {
	var target *StreamIdleTimeoutError
	return errors.As(err, &target)
}

func isResponsesProtocolError(err error) bool {
	var target *StreamProtocolError
	return errors.As(err, &target)
}

type blockingReadCloser struct {
	readStarted chan struct{}
	closed      chan struct{}
	readOnce    sync.Once
	closeOnce   sync.Once
	closeCount  atomic.Int32
}

type blockingCloseReadCloser struct {
	readStarted  chan struct{}
	closeStarted chan struct{}
	releaseClose chan struct{}
	releaseRead  chan struct{}
	readDone     chan struct{}
	readOnce     sync.Once
	closeOnce    sync.Once
}

func newBlockingCloseReadCloser() *blockingCloseReadCloser {
	return &blockingCloseReadCloser{
		readStarted:  make(chan struct{}),
		closeStarted: make(chan struct{}),
		releaseClose: make(chan struct{}),
		releaseRead:  make(chan struct{}),
		readDone:     make(chan struct{}),
	}
}

func (r *blockingCloseReadCloser) Read(_ []byte) (int, error) {
	r.readOnce.Do(func() { close(r.readStarted) })
	<-r.releaseRead
	close(r.readDone)
	return 0, io.ErrClosedPipe
}

func (r *blockingCloseReadCloser) Close() error {
	r.closeOnce.Do(func() { close(r.closeStarted) })
	<-r.releaseClose
	close(r.releaseRead)
	return nil
}

func newBlockingReadCloser() *blockingReadCloser {
	return &blockingReadCloser{
		readStarted: make(chan struct{}),
		closed:      make(chan struct{}),
	}
}

func (r *blockingReadCloser) Read(_ []byte) (int, error) {
	r.readOnce.Do(func() { close(r.readStarted) })
	<-r.closed
	return 0, io.ErrClosedPipe
}

func (r *blockingReadCloser) Close() error {
	r.closeCount.Add(1)
	r.closeOnce.Do(func() { close(r.closed) })
	return nil
}

func TestResponsesStreamIdleTimeoutErrorContract(t *testing.T) {
	body := newBlockingReadCloser()
	out := make(chan provider.StreamChunk, 4)
	idle := 25 * time.Millisecond
	go streamResponsesWithConfig(t.Context(), body, out, responsesStreamConfig{idleTimeout: idle})

	select {
	case <-body.readStarted:
	case <-time.After(time.Second):
		t.Fatal("stream reader did not start")
	}

	chunk := receiveChunk(t, out)
	if chunk.Type != provider.ChunkError {
		t.Fatalf("chunk type = %q, want %q", chunk.Type, provider.ChunkError)
	}
	var idleErr *StreamIdleTimeoutError
	if !errors.As(chunk.Error, &idleErr) {
		t.Fatalf("error = %T %v, want *StreamIdleTimeoutError", chunk.Error, chunk.Error)
	}
	if idleErr.Idle != idle || !isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("idle error = %#v", idleErr)
	}
	if idleErr.Error() != "openai responses stream idle for 25ms" || !idleErr.Temporary() {
		t.Fatalf("idle error contract = %q, temporary = %v", idleErr.Error(), idleErr.Temporary())
	}
	var netErr net.Error
	if !errors.As(chunk.Error, &netErr) || !netErr.Timeout() {
		t.Fatalf("error does not satisfy transient net.Error timeout contract: %v", chunk.Error)
	}

	drainChunks(t, out)
	if got := body.closeCount.Load(); got != 1 {
		t.Fatalf("body Close calls = %d, want 1", got)
	}
}

func TestResponsesStreamEventActivityResetsIdleTimeout(t *testing.T) {
	tests := []struct {
		name      string
		eventType string
		data      string
	}{
		{name: "in progress", eventType: "response.in_progress", data: `{"response":{"id":"resp_1"}}`},
		{name: "unknown valid event", eventType: "response.rate_limits.updated", data: `{"remaining":10}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader, writer := io.Pipe()
			out := make(chan provider.StreamChunk, 8)
			go streamResponsesWithConfig(t.Context(), reader, out, responsesStreamConfig{
				idleTimeout: 200 * time.Millisecond,
			})

			writerDone := make(chan struct{})
			go func() {
				defer close(writerDone)
				defer func() { _ = writer.Close() }()
				for range 5 {
					time.Sleep(50 * time.Millisecond)
					if _, err := io.WriteString(writer, sseLine(tt.eventType, tt.data)); err != nil {
						return
					}
				}
				time.Sleep(50 * time.Millisecond)
				_, _ = io.WriteString(writer, completedResponsesEvent)
			}()

			var gotFinish, gotError bool
			for chunk := range out {
				gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
				gotError = gotError || chunk.Type == provider.ChunkError
			}
			<-writerDone
			if !gotFinish || gotError {
				t.Fatalf("finish = %v, error = %v", gotFinish, gotError)
			}
		})
	}
}

func TestResponsesStreamOutputBackpressureDoesNotCauseIdleTimeout(t *testing.T) {
	idle := 10 * time.Millisecond
	input := sseLine("response.output_text.delta", `{"delta":"hello"}`) + completedResponsesEvent

	for range 20 {
		out := make(chan provider.StreamChunk)
		go streamResponsesWithConfig(t.Context(), io.NopCloser(strings.NewReader(input)), out, responsesStreamConfig{
			idleTimeout: idle,
		})

		// Keep projection blocked beyond the idle window while the reader queues
		// the already-complete terminal event.
		time.Sleep(3 * idle)

		var gotFinish bool
		for chunk := range out {
			if isResponsesIdleTimeout(chunk.Error) {
				t.Fatal("downstream backpressure was reported as provider inactivity")
			}
			if chunk.Type == provider.ChunkError {
				t.Fatalf("unexpected stream error: %v", chunk.Error)
			}
			gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
		}
		if !gotFinish {
			t.Fatal("stream did not finish after downstream resumed")
		}
	}
}

func TestResponsesStreamCommentsDoNotResetIdleTimeout(t *testing.T) {
	reader, writer := io.Pipe()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(t.Context(), reader, out, responsesStreamConfig{idleTimeout: 60 * time.Millisecond})

	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		defer func() { _ = writer.Close() }()
		for {
			time.Sleep(15 * time.Millisecond)
			if _, err := io.WriteString(writer, ": ping\n\n"); err != nil {
				return
			}
		}
	}()

	chunk := receiveChunk(t, out)
	if !isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("error = %T %v, want idle timeout", chunk.Error, chunk.Error)
	}
	drainChunks(t, out)
	<-writerDone
}

func TestResponsesStreamPartialEventDoesNotResetIdleTimeout(t *testing.T) {
	reader, writer := io.Pipe()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(t.Context(), reader, out, responsesStreamConfig{idleTimeout: 40 * time.Millisecond})

	if _, err := io.WriteString(writer, "event: response.in_progress\n"); err != nil {
		t.Fatalf("write partial event: %v", err)
	}
	chunk := receiveChunk(t, out)
	if !isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("error = %T %v, want idle timeout", chunk.Error, chunk.Error)
	}
	drainChunks(t, out)
	_ = writer.Close()
}

func TestResponsesStreamReasoningThenIdleTimeout(t *testing.T) {
	reader, writer := io.Pipe()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(t.Context(), reader, out, responsesStreamConfig{idleTimeout: 40 * time.Millisecond})

	input := sseLine("response.reasoning_summary_text.delta", `{"item_id":"rs_1","summary_index":0,"delta":"thinking"}`)
	if _, err := io.WriteString(writer, input); err != nil {
		t.Fatalf("write reasoning event: %v", err)
	}
	if chunk := receiveChunk(t, out); chunk.Type != provider.ChunkReasoning || chunk.Text != "thinking" {
		t.Fatalf("reasoning chunk = %#v", chunk)
	}
	if chunk := receiveChunk(t, out); !isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("error = %T %v, want idle timeout", chunk.Error, chunk.Error)
	}
	drainChunks(t, out)
	_ = writer.Close()
}

func TestResponsesStreamZeroIdleTimeoutDisablesWatchdog(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	body := newBlockingReadCloser()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(ctx, body, out, responsesStreamConfig{})

	select {
	case chunk := <-out:
		t.Fatalf("received chunk with disabled watchdog: %#v", chunk)
	case <-time.After(50 * time.Millisecond):
	}
	cancel()

	chunk := receiveChunk(t, out)
	if !errors.Is(chunk.Error, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", chunk.Error)
	}
	drainChunks(t, out)
}

func TestResponsesStreamCallerCancellationIdentity(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	body := newBlockingReadCloser()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(ctx, body, out, responsesStreamConfig{idleTimeout: time.Second})

	select {
	case <-body.readStarted:
	case <-time.After(time.Second):
		t.Fatal("stream reader did not start")
	}
	cancel()

	chunk := receiveChunk(t, out)
	if !errors.Is(chunk.Error, context.Canceled) || isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("error = %T %v, want context.Canceled", chunk.Error, chunk.Error)
	}
	drainChunks(t, out)
}

func TestResponsesStreamCallerDeadlineIdentity(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 25*time.Millisecond)
	defer cancel()
	body := newBlockingReadCloser()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(ctx, body, out, responsesStreamConfig{idleTimeout: time.Second})

	chunk := receiveChunk(t, out)
	if !errors.Is(chunk.Error, context.DeadlineExceeded) || isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("error = %T %v, want context.DeadlineExceeded", chunk.Error, chunk.Error)
	}
	drainChunks(t, out)
}

func TestResponsesStreamTerminalClosesBlockedBody(t *testing.T) {
	reader, writer := io.Pipe()
	body := &countingReadCloser{ReadCloser: reader}
	out := make(chan provider.StreamChunk, 8)
	go streamResponsesWithConfig(t.Context(), body, out, responsesStreamConfig{idleTimeout: time.Second})

	writeDone := make(chan error, 1)
	go func() {
		_, err := io.WriteString(writer, completedResponsesEvent)
		writeDone <- err
	}()
	if err := <-writeDone; err != nil {
		t.Fatalf("write terminal event: %v", err)
	}

	var gotFinish bool
	for chunk := range out {
		gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
	}
	if !gotFinish {
		t.Fatal("terminal event did not emit finish chunk")
	}
	if got := body.closeCount.Load(); got != 1 {
		t.Fatalf("body Close calls = %d, want 1", got)
	}
	if _, err := io.WriteString(writer, "still open"); err == nil {
		t.Fatal("body remained open after terminal event")
	}
	_ = writer.Close()
}

func TestResponsesStreamBlockingBodyCloseDoesNotBlockOutput(t *testing.T) {
	body := newBlockingCloseReadCloser()
	out := make(chan provider.StreamChunk, 4)
	go streamResponsesWithConfig(t.Context(), body, out, responsesStreamConfig{idleTimeout: 20 * time.Millisecond})

	select {
	case <-body.readStarted:
	case <-time.After(time.Second):
		t.Fatal("stream reader did not start")
	}

	chunk := receiveChunk(t, out)
	if !isResponsesIdleTimeout(chunk.Error) {
		t.Fatalf("error = %T %v, want idle timeout", chunk.Error, chunk.Error)
	}
	select {
	case <-body.closeStarted:
	case <-time.After(time.Second):
		t.Fatal("body Close did not start")
	}

	// Close remains blocked, but the coordinator must still finish after its
	// bounded reader wait and close the output channel.
	drainChunks(t, out)
	close(body.releaseClose)
	select {
	case <-body.readDone:
	case <-time.After(time.Second):
		t.Fatal("stream reader did not exit after releasing body Close")
	}
}

func TestResponsesStreamProtocolErrorContract(t *testing.T) {
	out := make(chan provider.StreamChunk, 4)
	streamResponses(t.Context(), io.NopCloser(strings.NewReader("event: response.created\ndata: {}\n\n")), out)

	chunk := receiveChunk(t, out)
	var protocolErr *StreamProtocolError
	if !errors.As(chunk.Error, &protocolErr) || !isResponsesProtocolError(chunk.Error) {
		t.Fatalf("error = %T %v, want *StreamProtocolError", chunk.Error, chunk.Error)
	}
	if protocolErr.Provider != responsesStreamProvider || protocolErr.API != responsesStreamAPI {
		t.Fatalf("protocol error = %#v", protocolErr)
	}
	if strings.Contains(protocolErr.Error(), "response.created") {
		t.Fatalf("premature EOF error unexpectedly includes event payload/type: %q", protocolErr.Error())
	}
	drainChunks(t, out)
}

func TestResponsesStreamAcceptsDataOnlyEvents(t *testing.T) {
	input := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"
	out := make(chan provider.StreamChunk, 4)
	streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)

	var text string
	var gotFinish bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			t.Fatalf("unexpected stream error: %v", chunk.Error)
		}
		if chunk.Type == provider.ChunkText {
			text += chunk.Text
		}
		gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
	}
	if text != "hello" || !gotFinish {
		t.Fatalf("text = %q, finish = %v; want hello and finish", text, gotFinish)
	}
}

func TestResponsesStreamRejectsMismatchedPayloadType(t *testing.T) {
	tests := []struct {
		name      string
		eventType string
		data      string
	}{
		{
			name:      "nonterminal",
			eventType: "response.output_text.delta",
			data:      `{"type":"response.refusal.delta","delta":"no"}`,
		},
		{
			name:      "terminal",
			eventType: "response.completed",
			data:      `{"type":"response.failed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make(chan provider.StreamChunk, 4)
			streamResponses(t.Context(), io.NopCloser(strings.NewReader(sseLine(tt.eventType, tt.data))), out)

			chunk := receiveChunk(t, out)
			var protocolErr *StreamProtocolError
			if !errors.As(chunk.Error, &protocolErr) || protocolErr.EventType != tt.eventType {
				t.Fatalf("error = %T %v, want payload mismatch for %q", chunk.Error, chunk.Error, tt.eventType)
			}
			if protocolErr.Reason != "event type does not match payload type" {
				t.Fatalf("reason = %q", protocolErr.Reason)
			}
			if !strings.Contains(protocolErr.Error(), tt.eventType) {
				t.Fatalf("error %q does not identify event type %q", protocolErr.Error(), tt.eventType)
			}
			drainChunks(t, out)
		})
	}
}

func TestResponsesIdleTimerReset(t *testing.T) {
	disabled := newResponsesIdleTimer(0)
	disabled.reset(time.Second)
	disabled.stop()

	timer := newResponsesIdleTimer(time.Hour)
	t.Cleanup(timer.stop)
	timer.reset(20 * time.Millisecond)
	select {
	case <-timer.c:
	case <-time.After(time.Second):
		t.Fatal("reset timer did not fire")
	}
}

func TestResponsesStreamRejectsMalformedNonterminalEvent(t *testing.T) {
	out := make(chan provider.StreamChunk, 4)
	streamResponses(t.Context(), io.NopCloser(strings.NewReader(sseLine("response.output_text.delta", `{not-json}`))), out)

	chunk := receiveChunk(t, out)
	var protocolErr *StreamProtocolError
	if chunk.Type != provider.ChunkError || !errors.As(chunk.Error, &protocolErr) {
		t.Fatalf("chunk = %#v, want protocol error", chunk)
	}
	if protocolErr.EventType != "response.output_text.delta" || protocolErr.Reason != "event data is not valid JSON" {
		t.Fatalf("protocol error = %#v", protocolErr)
	}
	drainChunks(t, out)
}

func TestResponsesStreamRejectsUntypedDataOnlyEvent(t *testing.T) {
	for _, data := range []string{`{}`, `null`} {
		t.Run(data, func(t *testing.T) {
			reader, writer := io.Pipe()
			t.Cleanup(func() {
				_ = reader.Close()
				_ = writer.Close()
			})

			out := make(chan provider.StreamChunk, 4)
			go streamResponsesWithConfig(t.Context(), reader, out, responsesStreamConfig{
				idleTimeout: 100 * time.Millisecond,
			})
			if _, err := io.WriteString(writer, "data: "+data+"\n\n"); err != nil {
				t.Fatalf("write event: %v", err)
			}

			chunk := receiveChunk(t, out)
			var protocolErr *StreamProtocolError
			if chunk.Type != provider.ChunkError || !errors.As(chunk.Error, &protocolErr) {
				t.Fatalf("chunk = %#v, want protocol error", chunk)
			}
			if protocolErr.EventType != "" || protocolErr.Reason != "event is missing type" {
				t.Fatalf("protocol error = %#v", protocolErr)
			}
			if isResponsesIdleTimeout(chunk.Error) {
				t.Fatalf("untyped event incorrectly counted as stream activity: %v", chunk.Error)
			}
			drainChunks(t, out)
		})
	}
}

func TestResponsesStreamRejectsInvalidRecognizedEventSchema(t *testing.T) {
	tests := []struct {
		name       string
		eventType  string
		data       string
		dataOnly   bool
		wantReason string
	}{
		{
			name:      "text delta",
			eventType: "response.output_text.delta",
			data:      `{"type":"response.output_text.delta","delta":123}`,
			dataOnly:  true,
		},
		{
			name:      "refusal delta",
			eventType: "response.refusal.delta",
			data:      `{"delta":123}`,
		},
		{
			name:      "reasoning summary delta",
			eventType: "response.reasoning_summary_text.delta",
			data:      `{"summary_index":"first"}`,
		},
		{
			name:      "reasoning text delta",
			eventType: "response.reasoning_text.delta",
			data:      `{"content_index":"first"}`,
		},
		{
			name:      "reasoning summary part added",
			eventType: "response.reasoning_summary_part.added",
			data:      `{"item_id":123,"output_index":0,"summary_index":0}`,
		},
		{
			name:      "output item added",
			eventType: "response.output_item.added",
			data:      `{"output_index":"first","item":{"type":"function_call"}}`,
		},
		{
			name:      "tool call delta",
			eventType: "response.function_call_arguments.delta",
			data:      `{"type":"response.function_call_arguments.delta","output_index":0,"delta":123}`,
		},
		{
			name:      "tool call done",
			eventType: "response.function_call_arguments.done",
			data:      `{"output_index":"first"}`,
		},
		{
			name:      "output item done envelope",
			eventType: "response.output_item.done",
			data:      `{"output_index":"first","item":{"type":"message"}}`,
		},
		{
			name:      "output item done item",
			eventType: "response.output_item.done",
			data:      `{"output_index":0,"item":123}`,
		},
		{
			name:      "output item done item type",
			eventType: "response.output_item.done",
			data:      `{"output_index":0,"item":{"type":123}}`,
		},
		{
			name:      "reasoning item",
			eventType: "response.output_item.done",
			data:      `{"output_index":0,"item":{"type":"reasoning","encrypted_content":123}}`,
		},
		{
			name:      "server-executed item",
			eventType: "response.output_item.done",
			data:      `{"output_index":0,"item":{"type":"web_search_call","score":1e1000}}`,
		},
		{
			name:      "server-executed item id",
			eventType: "response.output_item.done",
			data:      `{"output_index":0,"item":{"type":"web_search_call","id":123}}`,
		},
		{
			name:      "server-executed item name",
			eventType: "response.output_item.done",
			data:      `{"output_index":0,"item":{"type":"web_search_call","name":true}}`,
		},
		{
			name:       "reasoning part item id null",
			eventType:  "response.reasoning_summary_part.added",
			data:       `{"item_id":null,"output_index":0,"summary_index":0,"sequence_number":0,"part":{}}`,
			wantReason: "event payload has null item_id",
		},
		{
			name:       "reasoning part output index null",
			eventType:  "response.reasoning_summary_part.added",
			data:       `{"item_id":"rs_1","output_index":null,"summary_index":0,"sequence_number":0,"part":{}}`,
			wantReason: "event payload has null output_index",
		},
		{
			name:       "reasoning part summary index null",
			eventType:  "response.reasoning_summary_part.added",
			data:       `{"item_id":"rs_1","output_index":0,"summary_index":null,"sequence_number":0,"part":{}}`,
			wantReason: "event payload has null summary_index",
		},
		{
			name:       "reasoning part sequence number null",
			eventType:  "response.reasoning_summary_part.added",
			data:       `{"item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":null,"part":{}}`,
			wantReason: "event payload has null sequence_number",
		},
		{
			name:       "reasoning part null",
			eventType:  "response.reasoning_summary_part.added",
			data:       `{"item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":0,"part":null}`,
			wantReason: "event payload has null part",
		},
		{
			name:       "server-executed item id null",
			eventType:  "response.output_item.done",
			data:       `{"output_index":0,"item":{"type":"web_search_call","id":null}}`,
			wantReason: "event payload has null item.id",
		},
		{
			name:       "server-executed item name null",
			eventType:  "response.output_item.done",
			data:       `{"output_index":0,"item":{"type":"web_search_call","name":null}}`,
			wantReason: "event payload has null item.name",
		},
		{
			name:       "text delta missing",
			eventType:  "response.output_text.delta",
			data:       `{}`,
			wantReason: "event payload is missing required delta",
		},
		{
			name:       "text event null payload",
			eventType:  "response.output_text.delta",
			data:       `null`,
			wantReason: "event payload is missing required delta",
		},
		{
			name:       "refusal delta null",
			eventType:  "response.refusal.delta",
			data:       `{"delta":null}`,
			wantReason: "event payload is missing required delta",
		},
		{
			name:       "reasoning item id null",
			eventType:  "response.reasoning_summary_text.delta",
			data:       `{"item_id":null,"summary_index":0,"delta":"thinking"}`,
			wantReason: "event payload has null item_id",
		},
		{
			name:       "reasoning summary index null",
			eventType:  "response.reasoning_summary_text.delta",
			data:       `{"item_id":"rs_1","summary_index":null,"delta":"thinking"}`,
			wantReason: "event payload has null summary_index",
		},
		{
			name:       "reasoning delta null",
			eventType:  "response.reasoning_summary_text.delta",
			data:       `{"item_id":"rs_1","summary_index":0,"delta":null}`,
			wantReason: "event payload is missing required delta",
		},
		{
			name:       "reasoning text item id null",
			eventType:  "response.reasoning_text.delta",
			data:       `{"item_id":null,"content_index":0,"delta":"thinking"}`,
			wantReason: "event payload has null item_id",
		},
		{
			name:       "reasoning text content index null",
			eventType:  "response.reasoning_text.delta",
			data:       `{"item_id":"rs_1","content_index":null,"delta":"thinking"}`,
			wantReason: "event payload has null content_index",
		},
		{
			name:       "reasoning text delta null",
			eventType:  "response.reasoning_text.delta",
			data:       `{"item_id":"rs_1","content_index":0,"delta":null}`,
			wantReason: "event payload is missing required delta",
		},
		{
			name:       "output item added index null",
			eventType:  "response.output_item.added",
			data:       `{"output_index":null,"item":{"type":"function_call"}}`,
			wantReason: "event payload has null output_index",
		},
		{
			name:       "output item added item null",
			eventType:  "response.output_item.added",
			data:       `{"output_index":0,"item":null}`,
			wantReason: "event payload is missing required item",
		},
		{
			name:       "output item added type null",
			eventType:  "response.output_item.added",
			data:       `{"output_index":0,"item":{"type":null}}`,
			wantReason: "event payload is missing required item.type",
		},
		{
			name:       "tool call delta index null",
			eventType:  "response.function_call_arguments.delta",
			data:       `{"output_index":null,"delta":"{"}`,
			wantReason: "event payload has null output_index",
		},
		{
			name:       "tool call delta null",
			eventType:  "response.function_call_arguments.delta",
			data:       `{"output_index":0,"delta":null}`,
			wantReason: "event payload is missing required delta",
		},
		{
			name:       "tool call done index null",
			eventType:  "response.function_call_arguments.done",
			data:       `{"output_index":null}`,
			wantReason: "event payload has null output_index",
		},
		{
			name:       "output item done index null",
			eventType:  "response.output_item.done",
			data:       `{"output_index":null}`,
			wantReason: "event payload has null output_index",
		},
		{
			name:       "output item done item null",
			eventType:  "response.output_item.done",
			data:       `{"output_index":0,"item":null}`,
			wantReason: "event payload has null item",
		},
		{
			name:       "output item done type null",
			eventType:  "response.output_item.done",
			data:       `{"output_index":0,"item":{"type":null}}`,
			wantReason: "event payload is missing required item.type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := sseLine(tt.eventType, tt.data)
			if tt.dataOnly {
				input = "data: " + tt.data + "\n\n"
			}
			input += completedResponsesEvent

			out := make(chan provider.StreamChunk, 4)
			streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)

			chunk := receiveChunk(t, out)
			var protocolErr *StreamProtocolError
			if chunk.Type != provider.ChunkError || !errors.As(chunk.Error, &protocolErr) {
				t.Fatalf("chunk = %#v, want protocol error", chunk)
			}
			wantReason := tt.wantReason
			if wantReason == "" {
				wantReason = "event payload does not match event schema"
			}
			if protocolErr.EventType != tt.eventType || protocolErr.Reason != wantReason {
				t.Fatalf("protocol error = %#v", protocolErr)
			}
			if tt.wantReason == "" && errors.Unwrap(protocolErr) == nil {
				t.Fatalf("protocol error does not preserve decoding cause: %#v", protocolErr)
			}
			drainChunks(t, out)
		})
	}
}

func TestResponsesStreamPreservesOptionalFieldCompatibility(t *testing.T) {
	input := sseLine("response.output_text.delta", `{"delta":""}`) +
		sseLine("response.reasoning_summary_text.delta", `{"delta":""}`) +
		sseLine("response.reasoning_summary_part.added", `{}`) +
		sseLine("response.function_call_arguments.delta", `{"delta":""}`) +
		sseLine("response.output_item.done", `{"item":{"type":"web_search_call"}}`) +
		completedResponsesEvent
	out := make(chan provider.StreamChunk, 8)
	streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)

	var gotFinish bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			t.Fatalf("unexpected stream error: %v", chunk.Error)
		}
		gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
	}
	if !gotFinish {
		t.Fatal("stream did not finish after present empty delta fields")
	}
}

func TestResponsesStreamPreservesReasoningItemFallbacks(t *testing.T) {
	input := sseLine("response.output_item.added", `{"output_index":0,"item":{"type":"reasoning"}}`) +
		sseLine("response.output_item.done", `{"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"encrypted"}}`) +
		completedResponsesEvent
	out := make(chan provider.StreamChunk, 8)
	streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)

	var reasoning *provider.StreamChunk
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			t.Fatalf("unexpected stream error: %v", chunk.Error)
		}
		if chunk.Type == provider.ChunkReasoning {
			copy := chunk
			reasoning = &copy
		}
	}
	if reasoning == nil {
		t.Fatal("missing encrypted reasoning chunk")
	}
	if got := reasoning.Metadata["reasoningId"]; got != "rs_1:0" {
		t.Fatalf("reasoningId = %#v, want rs_1:0", got)
	}
	openai, ok := reasoning.Metadata["openai"].(map[string]any)
	if !ok || openai["itemId"] != "rs_1" || openai["encryptedContent"] != "encrypted" {
		t.Fatalf("openai reasoning metadata = %#v", reasoning.Metadata["openai"])
	}
}

func TestResponsesStreamRejectsUnterminatedTerminalEvent(t *testing.T) {
	input := "event: response.completed\ndata: {\"response\":{\"id\":\"resp_1\"}}"
	out := make(chan provider.StreamChunk, 4)
	streamResponses(t.Context(), io.NopCloser(strings.NewReader(input)), out)

	chunk := receiveChunk(t, out)
	var protocolErr *StreamProtocolError
	if chunk.Type != provider.ChunkError || !errors.As(chunk.Error, &protocolErr) {
		t.Fatalf("chunk = %#v, want protocol error", chunk)
	}
	if protocolErr.Reason != "stream ended before a terminal event" {
		t.Fatalf("reason = %q", protocolErr.Reason)
	}
	drainChunks(t, out)
}

func TestResponsesStreamTerminalUsageCompatibility(t *testing.T) {
	tests := []struct {
		name         string
		usage        string
		wantFinish   bool
		wantProtocol bool
		wantInput    int
		wantOutput   int
	}{
		{name: "usage omitted", wantFinish: true},
		{name: "complete usage", usage: `,"usage":{"input_tokens":3,"output_tokens":2}`, wantFinish: true, wantInput: 3, wantOutput: 2},
		{name: "empty usage", usage: `,"usage":{}`, wantProtocol: true},
		{name: "missing input tokens", usage: `,"usage":{"output_tokens":2}`, wantProtocol: true},
		{name: "missing output tokens", usage: `,"usage":{"input_tokens":3}`, wantProtocol: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload := `{"response":{"id":"resp_1"` + tt.usage + `}}`
			out := make(chan provider.StreamChunk, 4)
			streamResponses(t.Context(), io.NopCloser(strings.NewReader(sseLine("response.completed", payload))), out)

			var gotFinish bool
			var gotProtocol bool
			for chunk := range out {
				gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
				var protocolErr *StreamProtocolError
				gotProtocol = gotProtocol || errors.As(chunk.Error, &protocolErr)
				if chunk.Type == provider.ChunkFinish && (chunk.Usage.InputTokens != tt.wantInput || chunk.Usage.OutputTokens != tt.wantOutput) {
					t.Fatalf("finish usage = %#v, want %d input and %d output tokens", chunk.Usage, tt.wantInput, tt.wantOutput)
				}
			}
			if gotFinish != tt.wantFinish || gotProtocol != tt.wantProtocol {
				t.Fatalf("finish = %v, protocol error = %v; want finish = %v, protocol error = %v", gotFinish, gotProtocol, tt.wantFinish, tt.wantProtocol)
			}
		})
	}
}

func TestResponsesStreamOptions(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		model := Chat("o3", WithAPIKey("key")).(*chatModel)
		if got := model.opts.responsesStreamIdleTimeout; got != DefaultResponsesStreamIdleTimeout {
			t.Fatalf("default idle timeout = %s, want %s", got, DefaultResponsesStreamIdleTimeout)
		}
	})

	t.Run("override", func(t *testing.T) {
		model := Chat(
			"o3",
			WithAPIKey("key"),
			WithResponsesStreamIdleTimeout(3*time.Second),
		).(*chatModel)
		if got := model.opts.responsesStreamIdleTimeout; got != 3*time.Second {
			t.Fatalf("idle timeout = %s, want 3s", got)
		}
	})

	t.Run("done compatibility", func(t *testing.T) {
		model := Chat(
			"gpt-5",
			WithAPIKey("key"),
			WithResponsesStreamDoneCompatibility(true),
		).(*chatModel)
		if !model.opts.responsesStreamAllowDone {
			t.Fatal("[DONE] compatibility option was not applied")
		}
	})

	t.Run("negative rejected before request", func(t *testing.T) {
		model := Chat(
			"o3",
			WithAPIKey("key"),
			WithBaseURL("http://invalid.invalid"),
			WithResponsesStreamIdleTimeout(-time.Second),
		)
		result, err := model.DoStream(t.Context(), provider.GenerateParams{})
		if err == nil || result != nil {
			t.Fatalf("DoStream() = (%v, %v), want nil result and configuration error", result, err)
		}
		if !strings.Contains(err.Error(), "must be non-negative") {
			t.Fatalf("error = %q", err)
		}
	})

	t.Run("chat completions unaffected", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		}))
		defer server.Close()

		model := Chat(
			"gpt-4o",
			WithAPIKey("key"),
			WithBaseURL(server.URL),
			WithResponsesStreamIdleTimeout(-time.Second),
		)
		result, err := model.DoStream(t.Context(), provider.GenerateParams{ProviderOptions: chatCompletionsOpts})
		if err != nil {
			t.Fatalf("DoStream() error = %v", err)
		}
		var gotFinish bool
		for chunk := range result.Stream {
			gotFinish = gotFinish || chunk.Type == provider.ChunkFinish
		}
		if !gotFinish {
			t.Fatal("Chat Completions stream did not preserve [DONE] behavior")
		}
	})
}

func TestResponsesStreamErrorPropagatesThroughTextStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseLine("response.created", `{"response":{"id":"resp_1"}}`))
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	stream, err := goai.StreamText(t.Context(), model, goai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText() error = %v", err)
	}
	stream.Result()
	var protocolErr *StreamProtocolError
	if err := stream.Err(); !errors.As(err, &protocolErr) {
		t.Fatalf("stream.Err() = %T %v, want *StreamProtocolError", err, err)
	}
}

func TestResponsesStreamTerminalRaces(t *testing.T) {
	t.Run("completion and cancellation", func(t *testing.T) {
		for range 100 {
			ctx, cancel := context.WithCancel(t.Context())
			reader, writer := io.Pipe()
			body := &countingReadCloser{ReadCloser: reader}
			out := make(chan provider.StreamChunk, 8)
			go streamResponsesWithConfig(ctx, body, out, responsesStreamConfig{idleTimeout: time.Second})

			start := make(chan struct{})
			var wg sync.WaitGroup
			wg.Go(func() {
				<-start
				_, _ = io.WriteString(writer, completedResponsesEvent)
			})
			wg.Go(func() {
				<-start
				cancel()
			})
			close(start)

			terminalOutcomes := countTerminalOutcomes(out)
			wg.Wait()
			_ = writer.Close()
			if terminalOutcomes > 1 {
				t.Fatalf("terminal outcomes = %d, want at most 1", terminalOutcomes)
			}
			if got := body.closeCount.Load(); got != 1 {
				t.Fatalf("body Close calls = %d, want 1", got)
			}
		}
	})

	t.Run("completion and timeout", func(t *testing.T) {
		for range 50 {
			reader, writer := io.Pipe()
			body := &countingReadCloser{ReadCloser: reader}
			out := make(chan provider.StreamChunk, 8)
			go streamResponsesWithConfig(t.Context(), body, out, responsesStreamConfig{idleTimeout: 2 * time.Millisecond})

			writerDone := make(chan struct{})
			go func() {
				defer close(writerDone)
				time.Sleep(2 * time.Millisecond)
				_, _ = io.WriteString(writer, completedResponsesEvent)
			}()
			terminalOutcomes := countTerminalOutcomes(out)
			<-writerDone
			_ = writer.Close()
			if terminalOutcomes != 1 {
				t.Fatalf("terminal outcomes = %d, want 1", terminalOutcomes)
			}
			if got := body.closeCount.Load(); got != 1 {
				t.Fatalf("body Close calls = %d, want 1", got)
			}
		}
	})
}

type countingReadCloser struct {
	io.ReadCloser
	closeCount atomic.Int32
}

func (r *countingReadCloser) Close() error {
	r.closeCount.Add(1)
	return r.ReadCloser.Close()
}

func receiveChunk(t *testing.T, out <-chan provider.StreamChunk) provider.StreamChunk {
	t.Helper()
	select {
	case chunk, ok := <-out:
		if !ok {
			t.Fatal("stream closed without a chunk")
		}
		return chunk
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for stream chunk")
		return provider.StreamChunk{}
	}
}

func drainChunks(t *testing.T, out <-chan provider.StreamChunk) {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for {
		select {
		case _, ok := <-out:
			if !ok {
				return
			}
		case <-timer.C:
			t.Fatal("timed out waiting for stream close")
		}
	}
}

func countTerminalOutcomes(out <-chan provider.StreamChunk) int {
	var count int
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish || chunk.Type == provider.ChunkError {
			count++
		}
	}
	return count
}

func TestStreamResponses_ReasoningTextCancelledWhileSending(t *testing.T) {
	// A reasoning_text delta waiting to be received when the caller cancels
	// is dropped, and the stream ends without delivering it.
	ctx, cancel := context.WithCancel(t.Context())
	input := "event: response.reasoning_text.delta\n" +
		`data: {"type":"response.reasoning_text.delta","content_index":0,"delta":"thinking","item_id":"rs_1","output_index":0}` + "\n\n"

	out := make(chan provider.StreamChunk) // unbuffered, and not received from yet
	done := make(chan struct{})
	go func() {
		streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
		close(done)
	}()
	time.Sleep(50 * time.Millisecond) // let the delta reach the send
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("stream did not stop after cancellation")
	}
	for chunk := range out {
		if chunk.Type == provider.ChunkReasoning {
			t.Fatalf("reasoning delivered after cancellation: %#v", chunk)
		}
	}
}
