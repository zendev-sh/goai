package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

const (
	// DefaultResponsesStreamIdleTimeout is the maximum time a Responses API
	// stream waits for a complete provider event by default.
	DefaultResponsesStreamIdleTimeout = 5 * time.Minute

	responsesStreamProvider = "openai"
	responsesStreamAPI      = "responses"
	responsesReaderWait     = 250 * time.Millisecond
)

// WithResponsesStreamIdleTimeout configures the event-level idle timeout for
// OpenAI Responses streams. A positive duration enables the watchdog, zero
// disables it, and a negative duration makes Responses streaming fail during
// request setup. The default is [DefaultResponsesStreamIdleTimeout].
func WithResponsesStreamIdleTimeout(timeout time.Duration) Option {
	return func(o *options) {
		o.responsesStreamIdleTimeout = timeout
	}
}

// WithResponsesStreamDoneCompatibility allows a non-standard Responses
// endpoint to terminate a stream with a bare [DONE] sentinel. It is disabled by
// default because the Responses API uses typed terminal events and accepting a
// bare sentinel can hide a truncated response.
func WithResponsesStreamDoneCompatibility(enabled bool) Option {
	return func(o *options) {
		o.responsesStreamAllowDone = enabled
	}
}

// StreamIdleTimeoutError reports that an OpenAI Responses stream received no
// complete provider event within its configured idle window.
type StreamIdleTimeoutError struct {
	Provider string
	API      string
	Idle     time.Duration
}

func (e *StreamIdleTimeoutError) Error() string {
	return fmt.Sprintf("%s %s stream idle for %s", e.Provider, e.API, e.Idle)
}

// Timeout reports that this error is a timeout, satisfying net.Error.
func (e *StreamIdleTimeoutError) Timeout() bool { return true }

// Temporary reports that an idle stream interruption may be retried.
func (e *StreamIdleTimeoutError) Temporary() bool { return true }

// StreamProtocolError reports an invalid or interrupted OpenAI Responses
// stream. Event payloads are intentionally excluded from this type.
type StreamProtocolError struct {
	Provider  string
	API       string
	EventType string
	Reason    string
	cause     error
}

func (e *StreamProtocolError) Error() string {
	if e.EventType != "" {
		return fmt.Sprintf("%s %s stream protocol error for %s: %s", e.Provider, e.API, e.EventType, e.Reason)
	}
	return fmt.Sprintf("%s %s stream protocol error: %s", e.Provider, e.API, e.Reason)
}

// Unwrap returns the underlying body-read or decode error, when present.
func (e *StreamProtocolError) Unwrap() error { return e.cause }

type responsesStreamConfig struct {
	idleTimeout time.Duration
	allowDone   bool
	// retryHeaders of the 200 response, attached to stream errors so callers
	// can honor the delay (Azure sends its rate limit as an error event).
	retryHeaders map[string]string
}

func defaultResponsesStreamConfig() responsesStreamConfig {
	return responsesStreamConfig{idleTimeout: DefaultResponsesStreamIdleTimeout}
}

func newStreamProtocolError(eventType, reason string, cause error) *StreamProtocolError {
	return &StreamProtocolError{
		Provider:  responsesStreamProvider,
		API:       responsesStreamAPI,
		EventType: eventType,
		Reason:    reason,
		cause:     cause,
	}
}

type responsesReadResult struct {
	event sse.Event
	err   error
	eof   bool
}

type responsesEventReader struct {
	results   <-chan responsesReadResult
	cancel    context.CancelFunc
	closeBody func()
	closeDone <-chan struct{}
	done      <-chan struct{}
}

func newResponsesEventReader(ctx context.Context, body io.ReadCloser) *responsesEventReader {
	readerCtx, cancel := context.WithCancel(ctx)
	closeDone := make(chan struct{})
	closeBody := sync.OnceFunc(func() {
		// A custom RoundTripper may return a body whose Close blocks. Keep that
		// implementation bug from preventing the coordinator and output channel
		// from reaching their bounded terminal state.
		go func() {
			defer close(closeDone)
			_ = body.Close()
		}()
	})
	results := make(chan responsesReadResult, 1)
	done := make(chan struct{})

	go func() {
		defer close(done)
		scanner := sse.NewScanner(body)
		for {
			event, ok := scanner.NextEvent()
			if !ok {
				result := responsesReadResult{err: scanner.Err(), eof: scanner.Err() == nil}
				select {
				case results <- result:
				case <-readerCtx.Done():
				}
				return
			}
			select {
			case results <- responsesReadResult{event: event}:
			case <-readerCtx.Done():
				return
			}
		}
	}()

	return &responsesEventReader{
		results:   results,
		cancel:    cancel,
		closeBody: closeBody,
		closeDone: closeDone,
		done:      done,
	}
}

func (r *responsesEventReader) close() {
	r.cancel()
	r.closeBody()
	timer := time.NewTimer(responsesReaderWait)
	defer timer.Stop()
	readerDone := r.done
	bodyCloseDone := r.closeDone
	for readerDone != nil || bodyCloseDone != nil {
		select {
		case <-readerDone:
			readerDone = nil
		case <-bodyCloseDone:
			bodyCloseDone = nil
		case <-timer.C:
			return
		}
	}
}

type responsesIdleTimer struct {
	timer *time.Timer
	c     <-chan time.Time
}

func newResponsesIdleTimer(timeout time.Duration) *responsesIdleTimer {
	if timeout == 0 {
		return &responsesIdleTimer{}
	}
	timer := time.NewTimer(timeout)
	return &responsesIdleTimer{timer: timer, c: timer.C}
}

func (t *responsesIdleTimer) reset(timeout time.Duration) {
	if t.timer == nil {
		return
	}
	if !t.timer.Stop() {
		select {
		case <-t.timer.C:
		default:
		}
	}
	t.timer.Reset(timeout)
}

func (t *responsesIdleTimer) stop() {
	if t.timer != nil {
		t.timer.Stop()
	}
}

func responsesEventType(event sse.Event) (string, error) {
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(event.Data, &envelope); err != nil {
		return "", newStreamProtocolError(event.Type, "event data is not valid JSON", err)
	}
	if event.Type != "" && envelope.Type != "" && event.Type != envelope.Type {
		return "", newStreamProtocolError(event.Type, "event type does not match payload type", nil)
	}
	if event.Type != "" {
		return event.Type, nil
	}
	if envelope.Type == "" {
		return "", newStreamProtocolError("", "event is missing type", nil)
	}
	return envelope.Type, nil
}

func decodeResponsesEvent(eventType string, data []byte, target any) error {
	if err := json.Unmarshal(data, target); err != nil {
		return newStreamProtocolError(eventType, "event payload does not match event schema", err)
	}
	return nil
}

func missingResponsesEventField(eventType, field string) error {
	return newStreamProtocolError(eventType, fmt.Sprintf("event payload is missing required %s", field), nil)
}

func nullResponsesEventField(eventType, field string) error {
	return newStreamProtocolError(eventType, fmt.Sprintf("event payload has null %s", field), nil)
}

func trySendResponsesError(ctx context.Context, out chan<- provider.StreamChunk, err error) bool {
	chunk := provider.StreamChunk{Type: provider.ChunkError, Error: err}
	if ctx.Err() == nil {
		return provider.TrySend(ctx, out, chunk)
	}
	select {
	case out <- chunk:
		return true
	default:
		return false
	}
}
