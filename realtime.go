package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zendev-sh/goai/provider"
)

const (
	defaultRealtimeEventBuffer         = 64
	defaultRealtimeMaxParallelTools    = 4
	defaultRealtimeToolTimeout         = 30 * time.Second
	defaultRealtimeBackpressureTimeout = time.Second
)

// RealtimeOption configures OpenRealtimeSession.
type RealtimeOption func(*realtimeOptions)

type realtimeOptions struct {
	params           provider.RealtimeParams
	tools            []Tool
	eventBuffer      int
	backpressureWait time.Duration
	maxParallelTools int
	toolTimeout      time.Duration
	sequentialTools  bool
	onEvent          []func(provider.RealtimeEvent)
	onToolCallStart  []func(ToolCallStartInfo)
	onToolCall       []func(ToolCallInfo)
	onBeforeExecute  func(BeforeToolExecuteInfo) BeforeToolExecuteResult
	onAfterExecute   func(AfterToolExecuteInfo) AfterToolExecuteResult
	onPanic          []func(PanicInfo)
}

func defaultRealtimeOptions() realtimeOptions {
	return realtimeOptions{
		eventBuffer:      defaultRealtimeEventBuffer,
		backpressureWait: defaultRealtimeBackpressureTimeout,
		maxParallelTools: defaultRealtimeMaxParallelTools,
		toolTimeout:      defaultRealtimeToolTimeout,
	}
}

// WithRealtimeSystem sets the session system instructions.
func WithRealtimeSystem(system string) RealtimeOption {
	return func(o *realtimeOptions) { o.params.System = system }
}

// WithRealtimeTools registers tools available to the model. Tools with Execute
// handlers are dispatched automatically when the provider emits a tool call.
func WithRealtimeTools(tools ...Tool) RealtimeOption {
	return func(o *realtimeOptions) { o.tools = append([]Tool(nil), tools...) }
}

// WithRealtimeToolChoice controls provider tool selection.
func WithRealtimeToolChoice(choice string) RealtimeOption {
	return func(o *realtimeOptions) { o.params.ToolChoice = choice }
}

// WithRealtimeAudio configures input and output audio formats.
func WithRealtimeAudio(input, output provider.AudioFormat) RealtimeOption {
	return func(o *realtimeOptions) {
		o.params.InputAudio = input
		o.params.OutputAudio = output
	}
}

// WithRealtimeVoice requests a provider voice. Supported values are model-specific.
func WithRealtimeVoice(voice string) RealtimeOption {
	return func(o *realtimeOptions) { o.params.Voice = voice }
}

// WithRealtimeTurnDetection configures turn detection and interruption behavior.
func WithRealtimeTurnDetection(config provider.TurnDetection) RealtimeOption {
	return func(o *realtimeOptions) { o.params.TurnDetection = config }
}

// WithRealtimeInputTranscription configures input transcription.
func WithRealtimeInputTranscription(config provider.TranscriptionConfig) RealtimeOption {
	return func(o *realtimeOptions) { o.params.InputTranscription = config }
}

// WithRealtimeOutputTranscription configures output transcription.
func WithRealtimeOutputTranscription(config provider.TranscriptionConfig) RealtimeOption {
	return func(o *realtimeOptions) { o.params.OutputTranscription = config }
}

// WithRealtimeMaxOutputTokens limits response output tokens. Zero uses the
// provider default.
func WithRealtimeMaxOutputTokens(n int) RealtimeOption {
	return func(o *realtimeOptions) { o.params.MaxOutputTokens = n }
}

// WithRealtimeProviderOptions sets experimental provider-specific session fields.
func WithRealtimeProviderOptions(options map[string]any) RealtimeOption {
	return func(o *realtimeOptions) { o.params.ProviderOptions = cloneAnyMap(options) }
}

// WithRealtimeEventBuffer configures the bounded event queue exposed to the
// consumer. Values below one are rejected by OpenRealtimeSession. When the
// queue is full the session waits up to the backpressure timeout (see
// WithRealtimeBackpressureTimeout) before failing with ErrRealtimeBackpressure.
func WithRealtimeEventBuffer(size int) RealtimeOption {
	return func(o *realtimeOptions) { o.eventBuffer = size }
}

// WithRealtimeBackpressureTimeout sets how long the session waits for the
// consumer to drain a full event queue before terminating with
// ErrRealtimeBackpressure. Failing fast avoids accumulating stale audio. The
// default is one second; values below zero are rejected by OpenRealtimeSession.
func WithRealtimeBackpressureTimeout(timeout time.Duration) RealtimeOption {
	return func(o *realtimeOptions) { o.backpressureWait = timeout }
}

// WithRealtimeToolTimeout sets the timeout for each automatically dispatched tool.
// A non-positive duration disables the per-tool timeout.
func WithRealtimeToolTimeout(timeout time.Duration) RealtimeOption {
	return func(o *realtimeOptions) { o.toolTimeout = timeout }
}

// WithRealtimeMaxParallelTools limits concurrently executing realtime tools.
func WithRealtimeMaxParallelTools(max int) RealtimeOption {
	return func(o *realtimeOptions) { o.maxParallelTools = max }
}

// WithRealtimeSequentialTools forces realtime tools to execute one at a time.
func WithRealtimeSequentialTools(enabled bool) RealtimeOption {
	return func(o *realtimeOptions) { o.sequentialTools = enabled }
}

// WithRealtimeOnEvent adds a callback invoked before an event is published to
// the session event channel. Panics are recovered and reported to OnPanic hooks.
func WithRealtimeOnEvent(fn func(provider.RealtimeEvent)) RealtimeOption {
	return func(o *realtimeOptions) {
		if fn != nil {
			o.onEvent = append(o.onEvent, fn)
		}
	}
}

// WithRealtimeOnToolCallStart adds a callback invoked before a realtime tool executes.
func WithRealtimeOnToolCallStart(fn func(ToolCallStartInfo)) RealtimeOption {
	return func(o *realtimeOptions) {
		if fn != nil {
			o.onToolCallStart = append(o.onToolCallStart, fn)
		}
	}
}

// WithRealtimeOnToolCall adds a callback invoked after a realtime tool completes.
func WithRealtimeOnToolCall(fn func(ToolCallInfo)) RealtimeOption {
	return func(o *realtimeOptions) {
		if fn != nil {
			o.onToolCall = append(o.onToolCall, fn)
		}
	}
}

// WithRealtimeOnBeforeToolExecute configures the realtime pre-execution tool hook.
func WithRealtimeOnBeforeToolExecute(fn func(BeforeToolExecuteInfo) BeforeToolExecuteResult) RealtimeOption {
	return func(o *realtimeOptions) { o.onBeforeExecute = fn }
}

// WithRealtimeOnAfterToolExecute configures the realtime post-execution tool hook.
func WithRealtimeOnAfterToolExecute(fn func(AfterToolExecuteInfo) AfterToolExecuteResult) RealtimeOption {
	return func(o *realtimeOptions) { o.onAfterExecute = fn }
}

// WithRealtimeOnPanic adds a callback for recovered realtime callback/tool panics.
func WithRealtimeOnPanic(fn func(PanicInfo)) RealtimeOption {
	return func(o *realtimeOptions) {
		if fn != nil {
			o.onPanic = append(o.onPanic, fn)
		}
	}
}

// RealtimeSession is the high-level lifecycle wrapper around a provider
// RealtimeConnection. It optionally dispatches registered GoAI tools.
//
// All methods are safe for concurrent use. Events are delivered in the order
// the provider emitted them, each as a copy the consumer owns. Input buffers are
// copied before they are sent, so callers may reuse them once a method returns.
type RealtimeSession struct {
	ctx    context.Context
	cancel context.CancelFunc
	conn   provider.RealtimeConnection

	events           chan provider.RealtimeEvent
	done             chan struct{}
	backpressureWait time.Duration

	closeOnce sync.Once
	wg        sync.WaitGroup

	errMu sync.RWMutex
	err   error

	tools         map[string]Tool
	toolHooks     toolHooks
	toolTimeout   time.Duration
	toolSlots     chan struct{}
	toolCancelsMu sync.Mutex
	toolCancels   map[string]context.CancelFunc
	toolSequence  atomic.Int64
	onEvent       []func(provider.RealtimeEvent)
	onPanic       []func(PanicInfo)
}

// OpenRealtimeSession opens a realtime provider connection and starts its event
// lifecycle. The returned session owns the connection until Close.
//
// Experimental: see provider.RealtimeModel.
func OpenRealtimeSession(ctx context.Context, model provider.RealtimeModel, opts ...RealtimeOption) (*RealtimeSession, error) {
	if ctx == nil {
		return nil, errors.New("goai: realtime context is nil")
	}
	if model == nil {
		return nil, errors.New("goai: realtime model is nil")
	}

	o := defaultRealtimeOptions()
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	if o.eventBuffer < 1 {
		return nil, fmt.Errorf("%w: event buffer must be positive", provider.ErrInvalidRealtimeConfig)
	}
	if o.backpressureWait < 0 {
		return nil, fmt.Errorf("%w: backpressure timeout cannot be negative", provider.ErrInvalidRealtimeConfig)
	}
	if o.maxParallelTools < 1 {
		return nil, fmt.Errorf("%w: max parallel tools must be positive", provider.ErrInvalidRealtimeConfig)
	}

	o.params.Tools = ToolDefinitions(o.tools)
	o.params = cloneRealtimeParams(o.params)
	if err := o.params.Validate(); err != nil {
		return nil, err
	}

	conn, err := model.OpenRealtime(ctx, o.params)
	if err != nil {
		return nil, err
	}
	if conn == nil {
		return nil, errors.New("goai: realtime model returned a nil connection")
	}

	sessionCtx, cancel := context.WithCancel(ctx)
	toolMap := make(map[string]Tool, len(o.tools))
	for _, tool := range o.tools {
		toolMap[tool.Name] = tool
	}
	parallel := o.maxParallelTools
	if o.sequentialTools {
		parallel = 1
	}

	s := &RealtimeSession{
		ctx:              sessionCtx,
		cancel:           cancel,
		conn:             conn,
		events:           make(chan provider.RealtimeEvent, o.eventBuffer),
		done:             make(chan struct{}),
		backpressureWait: o.backpressureWait,
		tools:            toolMap,
		toolHooks: toolHooks{
			sequential:      true,
			onToolCallStart: o.onToolCallStart,
			onToolCall:      o.onToolCall,
			onBeforeExecute: o.onBeforeExecute,
			onAfterExecute:  o.onAfterExecute,
			onPanic:         o.onPanic,
		},
		toolTimeout: o.toolTimeout,
		toolSlots:   make(chan struct{}, parallel),
		toolCancels: make(map[string]context.CancelFunc),
		onEvent:     append([]func(provider.RealtimeEvent){}, o.onEvent...),
		onPanic:     append([]func(PanicInfo){}, o.onPanic...),
	}

	s.wg.Add(1)
	go s.consume()
	go func() {
		s.wg.Wait()
		close(s.events)
		close(s.done)
	}()
	return s, nil
}

// Events returns the normalized provider event stream.
func (s *RealtimeSession) Events() <-chan provider.RealtimeEvent { return s.events }

// InputAudio appends an audio frame using the format configured for the session.
func (s *RealtimeSession) InputAudio(ctx context.Context, audio []byte) error {
	return s.send(ctx, provider.RealtimeClientEvent{
		Type:  provider.RealtimeInputAudioAppend,
		Audio: append([]byte(nil), audio...),
	})
}

// CommitInput commits buffered input audio.
func (s *RealtimeSession) CommitInput(ctx context.Context) error {
	return s.send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioCommit})
}

// StartInputActivity signals the beginning of client-managed speech activity.
func (s *RealtimeSession) StartInputActivity(ctx context.Context) error {
	return s.send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputActivityStart})
}

// EndInputActivity signals the end of client-managed speech activity.
func (s *RealtimeSession) EndInputActivity(ctx context.Context) error {
	return s.send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeInputActivityEnd})
}

// SendText sends textual input into the realtime conversation.
func (s *RealtimeSession) SendText(ctx context.Context, text string) error {
	return s.send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: text})
}

// RequestResponse explicitly asks the provider to create a response.
func (s *RealtimeSession) RequestResponse(ctx context.Context) error {
	return s.send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeResponseCreate})
}

// Interrupt cancels the active response. cursor should identify audio actually
// played when the provider supports output truncation.
func (s *RealtimeSession) Interrupt(ctx context.Context, cursor *provider.PlaybackCursor) error {
	var cloned *provider.PlaybackCursor
	if cursor != nil {
		copy := *cursor
		cloned = &copy
	}
	return s.send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeResponseCancel, Cursor: cloned})
}

// Close terminates the session and waits for its internal goroutines. It is idempotent.
func (s *RealtimeSession) Close() error {
	s.closeOnce.Do(func() {
		s.cancel()
		if err := s.conn.Close(); err != nil {
			s.setErr(err)
		}
		s.cancelAllTools()
	})
	<-s.done
	return s.Err()
}

// Err returns the terminal session error, if any.
func (s *RealtimeSession) Err() error {
	s.errMu.RLock()
	defer s.errMu.RUnlock()
	return s.err
}

func (s *RealtimeSession) send(ctx context.Context, event provider.RealtimeClientEvent) error {
	if ctx == nil {
		return errors.New("goai: realtime send context is nil")
	}
	select {
	case <-s.done:
		return provider.ErrRealtimeSessionClosed
	default:
	}
	select {
	case <-s.ctx.Done():
		return provider.ErrRealtimeSessionClosed
	default:
	}
	return s.conn.Send(ctx, cloneRealtimeClientEvent(event))
}

func (s *RealtimeSession) consume() {
	defer s.wg.Done()
	defer s.cancelAllTools()

	for {
		select {
		case <-s.ctx.Done():
			return
		case event, ok := <-s.conn.Events():
			if !ok {
				if err := s.conn.Err(); err != nil {
					s.setErr(err)
				}
				return
			}

			event = cloneRealtimeEvent(event)
			s.handleInternalEvent(event)
			s.fireEventHooks(event)

			if !s.publish(event) {
				return
			}
		}
	}
}

// publish delivers event to the consumer. A full queue gets a bounded grace
// period so a briefly slow consumer does not kill a healthy session.
func (s *RealtimeSession) publish(event provider.RealtimeEvent) bool {
	select {
	case s.events <- event:
		return true
	case <-s.ctx.Done():
		return false
	default:
	}

	timer := time.NewTimer(s.backpressureWait)
	defer timer.Stop()
	select {
	case s.events <- event:
		return true
	case <-s.ctx.Done():
		return false
	case <-timer.C:
		s.setErr(provider.ErrRealtimeBackpressure)
		s.cancel()
		_ = s.conn.Close()
		return false
	}
}

func (s *RealtimeSession) handleInternalEvent(event provider.RealtimeEvent) {
	switch event.Type {
	case provider.RealtimeToolCallEvent:
		if event.ToolCall != nil {
			s.startTool(*event.ToolCall)
		}
	case provider.RealtimeToolCallCancelled:
		if event.ToolCallID != "" {
			s.cancelTool(event.ToolCallID)
		}
	}
}

func (s *RealtimeSession) startTool(call provider.RealtimeToolCall) {
	input := append(json.RawMessage(nil), call.Input...)
	call.Input = input

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		select {
		case s.toolSlots <- struct{}{}:
			defer func() { <-s.toolSlots }()
		case <-s.ctx.Done():
			return
		}

		toolCtx, cancel := context.WithCancel(s.ctx)
		if s.toolTimeout > 0 {
			toolCtx, cancel = context.WithTimeout(s.ctx, s.toolTimeout)
		}
		s.toolCancelsMu.Lock()
		s.toolCancels[call.ID] = cancel
		s.toolCancelsMu.Unlock()
		defer func() {
			cancel()
			s.toolCancelsMu.Lock()
			delete(s.toolCancels, call.ID)
			s.toolCancelsMu.Unlock()
		}()

		step := int(s.toolSequence.Add(1))
		_, results := executeToolsParallel(toolCtx, []provider.ToolCall{{
			ID: call.ID, Name: call.Name, Input: call.Input,
		}}, s.tools, step, s.toolHooks)
		if len(results) == 0 || toolCtx.Err() != nil && errors.Is(toolCtx.Err(), context.Canceled) {
			return
		}

		result := results[0]
		output := encodeRealtimeToolOutput(result.Output, result.Error)
		err := s.conn.Send(s.ctx, provider.RealtimeClientEvent{
			Type: provider.RealtimeToolResultEvent,
			ToolResult: &provider.RealtimeToolResult{
				ToolCallID: call.ID,
				Output:     output,
				IsError:    result.IsError,
			},
		})
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, provider.ErrRealtimeSessionClosed) {
			s.setErr(err)
			s.cancel()
			_ = s.conn.Close()
		}
	}()
}

func encodeRealtimeToolOutput(output string, err error) json.RawMessage {
	if err != nil {
		message := err.Error()
		runes := []rune(message)
		if len(runes) > 500 {
			message = string(runes[:500]) + "..."
		}
		encoded, _ := json.Marshal(map[string]string{"error": message})
		return encoded
	}
	if json.Valid([]byte(output)) {
		return append(json.RawMessage(nil), output...)
	}
	encoded, _ := json.Marshal(map[string]string{"text": output})
	return encoded
}

func (s *RealtimeSession) cancelTool(id string) {
	s.toolCancelsMu.Lock()
	cancel := s.toolCancels[id]
	s.toolCancelsMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (s *RealtimeSession) cancelAllTools() {
	s.toolCancelsMu.Lock()
	cancels := make([]context.CancelFunc, 0, len(s.toolCancels))
	for _, cancel := range s.toolCancels {
		cancels = append(cancels, cancel)
	}
	s.toolCancelsMu.Unlock()
	for _, cancel := range cancels {
		cancel()
	}
}

func (s *RealtimeSession) fireEventHooks(event provider.RealtimeEvent) {
	for _, fn := range s.onEvent {
		func() {
			defer func() {
				if r := recover(); r != nil {
					firePanic(s.onPanic, "RealtimeOnEvent", r)
				}
			}()
			fn(cloneRealtimeEvent(event))
		}()
	}
}

func (s *RealtimeSession) setErr(err error) {
	if err == nil || errors.Is(err, context.Canceled) {
		return
	}
	s.errMu.Lock()
	if s.err == nil {
		s.err = err
	}
	s.errMu.Unlock()
}

func cloneRealtimeParams(params provider.RealtimeParams) provider.RealtimeParams {
	params.Tools = append([]provider.ToolDefinition(nil), params.Tools...)
	for i := range params.Tools {
		params.Tools[i].InputSchema = append(json.RawMessage(nil), params.Tools[i].InputSchema...)
		params.Tools[i].ProviderDefinedOptions = cloneAnyMap(params.Tools[i].ProviderDefinedOptions)
	}
	params.TurnDetection.ProviderOptions = cloneAnyMap(params.TurnDetection.ProviderOptions)
	params.InputTranscription.ProviderOptions = cloneAnyMap(params.InputTranscription.ProviderOptions)
	params.OutputTranscription.ProviderOptions = cloneAnyMap(params.OutputTranscription.ProviderOptions)
	params.ProviderOptions = cloneAnyMap(params.ProviderOptions)
	return params
}

func cloneRealtimeClientEvent(event provider.RealtimeClientEvent) provider.RealtimeClientEvent {
	event.Audio = append([]byte(nil), event.Audio...)
	event.Metadata = cloneAnyMap(event.Metadata)
	if event.Cursor != nil {
		cursor := *event.Cursor
		event.Cursor = &cursor
	}
	if event.ToolResult != nil {
		result := *event.ToolResult
		result.Output = append(json.RawMessage(nil), result.Output...)
		event.ToolResult = &result
	}
	return event
}

func cloneRealtimeEvent(event provider.RealtimeEvent) provider.RealtimeEvent {
	event.Audio = append([]byte(nil), event.Audio...)
	event.RateLimits = append([]provider.RealtimeRateLimit(nil), event.RateLimits...)
	event.ProviderMetadata = cloneAnyMap(event.ProviderMetadata)
	if event.ToolCall != nil {
		call := *event.ToolCall
		call.Input = append(json.RawMessage(nil), call.Input...)
		event.ToolCall = &call
	}
	return event
}

func cloneAnyMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = cloneAnyValue(value)
	}
	return out
}

func cloneAnyValue(value any) any {
	switch v := value.(type) {
	case map[string]any:
		return cloneAnyMap(v)
	case []any:
		out := make([]any, len(v))
		for i := range v {
			out[i] = cloneAnyValue(v[i])
		}
		return out
	case []string:
		return append([]string(nil), v...)
	case []byte:
		return append([]byte(nil), v...)
	case json.RawMessage:
		return append(json.RawMessage(nil), v...)
	default:
		return value
	}
}
