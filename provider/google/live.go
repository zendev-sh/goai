package google

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/jsonmerge"
	"github.com/zendev-sh/goai/internal/realtimews"
	"github.com/zendev-sh/goai/provider"
)

var (
	_ provider.RealtimeModel        = (*liveModel)(nil)
	_ provider.RealtimeCapableModel = (*liveModel)(nil)
	_ provider.RealtimeConnection   = (*liveConn)(nil)
)

const (
	livePath             = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	liveVertexPath       = "/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
	liveInputSampleRate  = 16_000
	liveOutputSampleRate = 24_000
	liveReadLimit        = 16 << 20
	liveEventBuffer      = 64
	liveMaxErrorBody     = 1024
)

// LiveToolBehavior selects whether the model waits for a function result
// before continuing the conversation.
type LiveToolBehavior string

// LiveResponseScheduling selects how a NON_BLOCKING function result enters the
// conversation.
type LiveResponseScheduling string

const (
	// LiveBlocking pauses the conversation until the function result arrives.
	LiveBlocking LiveToolBehavior = "BLOCKING"
	// LiveNonBlocking lets the model keep talking while the function runs.
	LiveNonBlocking LiveToolBehavior = "NON_BLOCKING"

	// LiveSilent adds the result to the context without announcing it.
	LiveSilent LiveResponseScheduling = "SILENT"
	// LiveWhenIdle reports the result once the model finishes its current activity.
	LiveWhenIdle LiveResponseScheduling = "WHEN_IDLE"
	// LiveInterrupt interrupts the current activity to report the result.
	LiveInterrupt LiveResponseScheduling = "INTERRUPT"
)

// LiveToolOptions returns the ProviderDefinedOptions of a function tool used in
// a Gemini Live session. Empty values keep the model default (Gemini 3.8 Live
// runs functions NON_BLOCKING and schedules results WHEN_IDLE).
func LiveToolOptions(behavior LiveToolBehavior, scheduling LiveResponseScheduling) map[string]any {
	opts := map[string]any{}
	if behavior != "" {
		opts["behavior"] = string(behavior)
	}
	if scheduling != "" {
		opts["scheduling"] = string(scheduling)
	}
	return opts
}

// Live creates a Gemini Live API realtime model for the given model ID, such as
// "gemini-3.8-live". It speaks the Live WebSocket protocol from the backend and
// reuses the provider options for the API key, base URL, headers, and HTTP
// client.
//
// With WithVertex the session runs on Vertex AI: it connects to the regional
// endpoint, authenticates with the OAuth token from WithTokenSource, and names
// the model as a Vertex resource. WithVertexBaseURL replaces the endpoint host.
//
// Session settings without a common field go under ProviderOptions["google"]
// using Live setup field names, for example "sessionResumption",
// "contextWindowCompression", or "historyConfig".
//
// Experimental: see provider.RealtimeModel.
func Live(modelID string, opts ...Option) provider.RealtimeModel {
	return &liveModel{id: modelID, opts: resolveOptions(opts...)}
}

type liveModel struct {
	id   string
	opts options
}

func (m *liveModel) ModelID() string { return m.id }

// RealtimeCapabilities describes the Gemini 3.8 Live protocol: provider or
// client activity detection, barge-in, blocking and non-blocking functions with
// cancellation, session resumption, and context compression.
func (m *liveModel) RealtimeCapabilities() provider.RealtimeCapabilities {
	return provider.RealtimeCapabilities{
		InputAudio:           true,
		OutputAudio:          true,
		InputTranscription:   true,
		OutputTranscription:  true,
		TextInput:            true,
		ProviderVAD:          true,
		ClientVAD:            true,
		BargeIn:              true,
		FunctionCalling:      true,
		BlockingToolCalls:    true,
		NonBlockingToolCalls: true,
		ParallelToolCalls:    true,
		ToolCallCancellation: true,
		SessionResumption:    true,
		ContextCompression:   true,
		UsageUpdates:         true,
		SupportedInputAudio:  []provider.AudioFormat{liveInputFormat},
		SupportedOutputAudio: []provider.AudioFormat{liveOutputFormat},
	}
}

var (
	liveInputFormat  = provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: liveInputSampleRate, Channels: 1}
	liveOutputFormat = provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: liveOutputSampleRate, Channels: 1}
)

// OpenRealtime dials the Live endpoint, sends the setup message, and returns
// once the server confirms it. ctx bounds the whole session.
func (m *liveModel) OpenRealtime(ctx context.Context, params provider.RealtimeParams) (provider.RealtimeConnection, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}
	endpoint, modelName, err := m.target()
	if err != nil {
		return nil, err
	}
	setup, scheduling, err := buildLiveSetup(modelName, params)
	if err != nil {
		return nil, err
	}
	token, err := resolveAuthToken(ctx, m.opts)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	// Caller headers first, credential last, so a caller header cannot override it.
	header := http.Header{}
	for k, v := range m.opts.headers {
		header.Set(k, v)
	}
	if m.opts.isVertex {
		header.Set("Authorization", "Bearer "+token)
	} else {
		header.Set("x-goog-api-key", token)
	}

	ws, resp, err := realtimews.Dial(ctx, endpoint, realtimews.Options{
		HTTPClient: m.opts.httpClient,
		Headers:    header,
		ReadLimit:  liveReadLimit,
	})
	if err != nil {
		return nil, liveDialError(err, resp)
	}

	connCtx, cancel := context.WithCancel(ctx)
	c := &liveConn{
		ws:             ws,
		ctx:            connCtx,
		cancel:         cancel,
		events:         make(chan provider.RealtimeEvent, liveEventBuffer),
		done:           make(chan struct{}),
		closing:        make(chan struct{}),
		calls:          make(map[string]string),
		scheduling:     scheduling,
		clientActivity: params.TurnDetection.Mode == provider.TurnModeClientVAD,
	}
	if err := c.write(ctx, liveClientMessage{Setup: setup}); err != nil {
		c.abort()
		return nil, fmt.Errorf("google: sending live setup: %w", err)
	}
	if err := c.awaitSetup(ctx); err != nil {
		c.abort()
		return nil, err
	}
	c.events <- provider.RealtimeEvent{Type: provider.RealtimeSessionReady}
	go c.readLoop()
	return c, nil
}

// target resolves the WebSocket endpoint and the model resource name for the
// configured backend: the Gemini API or Vertex AI.
func (m *liveModel) target() (endpoint, modelName string, err error) {
	if !m.opts.isVertex {
		endpoint, err = liveURL(m.opts.baseURL, livePath, true)
		return endpoint, liveModelName(m.id), err
	}
	if !validGCPIdentifier(m.opts.project) {
		return "", "", fmt.Errorf("google: invalid vertex project %q", m.opts.project)
	}
	if !validGCPIdentifier(m.opts.location) {
		return "", "", fmt.Errorf("google: invalid vertex location %q", m.opts.location)
	}
	base := cmp.Or(m.opts.vertexBaseURL, "https://"+vertexHost(m.opts.location))
	endpoint, err = liveURL(base, liveVertexPath, false)
	return endpoint, liveVertexModelName(m.id, m.opts.project, m.opts.location), err
}

// liveURL turns an HTTP(S) base into the WebSocket endpoint for path. The base
// path is kept for the Gemini API base URL; a Vertex override only sets the host.
func liveURL(base, path string, keepBasePath bool) (string, error) {
	u, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil {
		return "", fmt.Errorf("google: invalid base URL: %w", err)
	}
	if !keepBasePath {
		u.Path = ""
	}
	switch u.Scheme {
	case "https", "wss":
		u.Scheme = "wss"
	case "http", "ws":
		u.Scheme = "ws"
	default:
		return "", fmt.Errorf("google: unsupported base URL scheme %q for Live", u.Scheme)
	}
	u.Path += path
	return u.String(), nil
}

// liveDialError maps a failed WebSocket upgrade to a goai API error when the
// server answered with an HTTP status.
func liveDialError(err error, resp *http.Response) error {
	if resp == nil || resp.StatusCode == http.StatusSwitchingProtocols {
		return fmt.Errorf("google: connecting to live api: %w", err)
	}
	var body []byte
	if resp.Body != nil {
		body, _ = io.ReadAll(io.LimitReader(resp.Body, liveMaxErrorBody))
		_ = resp.Body.Close()
	}
	return goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, body, resp.Header)
}

// buildLiveSetup maps RealtimeParams to the Live setup message. It also returns
// the response scheduling configured per function name.
func buildLiveSetup(modelName string, params provider.RealtimeParams) (map[string]any, map[string]string, error) {
	if !params.InputAudio.IsZero() && params.InputAudio != liveInputFormat {
		return nil, nil, fmt.Errorf("%w: gemini live input must be 16 kHz mono PCM16", provider.ErrUnsupportedAudioFormat)
	}
	if !params.OutputAudio.IsZero() && params.OutputAudio != liveOutputFormat {
		return nil, nil, fmt.Errorf("%w: gemini live output is 24 kHz mono PCM16", provider.ErrUnsupportedAudioFormat)
	}

	setup := map[string]any{"model": modelName}
	if si := buildSystemInstruction(params.System); si != nil {
		setup["systemInstruction"] = si
	}

	toolChoice := params.ToolChoice
	if toolChoice == "auto" {
		toolChoice = ""
	}
	if toolChoice != "" {
		return nil, nil, fmt.Errorf("%w: gemini live has no tool choice", provider.ErrRealtimeUnsupported)
	}
	tools, scheduling, err := buildLiveTools(params.Tools)
	if err != nil {
		return nil, nil, err
	}
	if tools != nil {
		setup["tools"] = tools
	}

	gen := map[string]any{"responseModalities": []string{"AUDIO"}}
	if params.MaxOutputTokens > 0 {
		gen["maxOutputTokens"] = params.MaxOutputTokens
	}
	if params.Voice != "" {
		gen["speechConfig"] = map[string]any{
			"voiceConfig": map[string]any{
				"prebuiltVoiceConfig": map[string]any{"voiceName": params.Voice},
			},
		}
	}
	setup["generationConfig"] = gen

	inputConfig, err := buildLiveInputConfig(params.TurnDetection)
	if err != nil {
		return nil, nil, err
	}
	if inputConfig != nil {
		setup["realtimeInputConfig"] = inputConfig
	}

	for key, cfg := range map[string]provider.TranscriptionConfig{
		"inputAudioTranscription":  params.InputTranscription,
		"outputAudioTranscription": params.OutputTranscription,
	} {
		tc, err := buildLiveTranscription(cfg)
		if err != nil {
			return nil, nil, err
		}
		if tc != nil {
			setup[key] = tc
		}
	}

	setup, err = mergeLiveProviderOptions(setup, params.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}
	return setup, scheduling, nil
}

func liveModelName(id string) string {
	if strings.HasPrefix(id, "models/") {
		return id
	}
	return "models/" + id
}

// liveVertexModelName returns the Vertex publisher model resource; a full
// "projects/..." resource name is kept as given.
func liveVertexModelName(id, project, location string) string {
	if strings.HasPrefix(id, "projects/") {
		return id
	}
	id = strings.TrimPrefix(id, "models/")
	id = strings.TrimPrefix(id, "publishers/google/models/")
	return fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", project, location, id)
}

// buildLiveTools reuses the chat tool serialization and adds the Live-only
// behavior of each function declaration.
func buildLiveTools(defs []provider.ToolDefinition) (any, map[string]string, error) {
	tools, toolConfig, err := buildToolsAndConfig(defs, "", nil)
	if err != nil {
		return nil, nil, err
	}
	if toolConfig != nil {
		return nil, nil, fmt.Errorf("%w: gemini live has no tool config", provider.ErrRealtimeUnsupported)
	}

	behaviors := map[string]string{}
	scheduling := map[string]string{}
	for _, def := range defs {
		if def.ProviderDefinedType != "" {
			continue
		}
		if b, ok := def.ProviderDefinedOptions["behavior"]; ok {
			value, _ := b.(string)
			if value != string(LiveBlocking) && value != string(LiveNonBlocking) {
				return nil, nil, fmt.Errorf("%w: invalid live behavior %v for tool %s", provider.ErrInvalidRealtimeConfig, b, def.Name)
			}
			behaviors[def.Name] = value
		}
		if s, ok := def.ProviderDefinedOptions["scheduling"]; ok {
			value, _ := s.(string)
			switch LiveResponseScheduling(value) {
			case LiveSilent, LiveWhenIdle, LiveInterrupt:
				scheduling[def.Name] = value
			default:
				return nil, nil, fmt.Errorf("%w: invalid live scheduling %v for tool %s", provider.ErrInvalidRealtimeConfig, s, def.Name)
			}
		}
	}

	if list, ok := tools.([]map[string]any); ok && len(behaviors) > 0 {
		for _, tool := range list {
			decls, _ := tool["functionDeclarations"].([]map[string]any)
			for _, decl := range decls {
				if b, ok := behaviors[decl["name"].(string)]; ok {
					decl["behavior"] = b
				}
			}
		}
	}
	return tools, scheduling, nil
}

func buildLiveInputConfig(td provider.TurnDetection) (map[string]any, error) {
	detection := map[string]any{}
	switch td.Mode {
	case "", provider.TurnModeProviderVAD:
	case provider.TurnModeClientVAD:
		detection["disabled"] = true
	default:
		return nil, fmt.Errorf("%w: gemini live has no %s turn mode; use client_vad", provider.ErrRealtimeUnsupported, td.Mode)
	}
	if td.DisableAutoResponse {
		return nil, fmt.Errorf("%w: gemini live always responds at the end of a turn", provider.ErrRealtimeUnsupported)
	}
	for k, v := range td.ProviderOptions {
		detection[k] = v
	}

	config := map[string]any{}
	if len(detection) > 0 {
		config["automaticActivityDetection"] = detection
	}
	if td.DisableInterruption {
		config["activityHandling"] = "NO_INTERRUPTION"
	}
	if len(config) == 0 {
		return nil, nil
	}
	return config, nil
}

func buildLiveTranscription(cfg provider.TranscriptionConfig) (map[string]any, error) {
	if !cfg.Enabled {
		return nil, nil
	}
	if cfg.Prompt != "" {
		return nil, fmt.Errorf("%w: gemini live transcription has no prompt", provider.ErrRealtimeUnsupported)
	}
	tc := map[string]any{}
	if cfg.Language != "" {
		tc["languageCodes"] = []string{cfg.Language}
	}
	for k, v := range cfg.ProviderOptions {
		tc[k] = v
	}
	return tc, nil
}

// mergeLiveProviderOptions overlays ProviderOptions["google"] on the setup;
// nested objects merge at every level. Fields the SDK derives from
// RealtimeParams cannot be overridden.
func mergeLiveProviderOptions(setup map[string]any, providerOptions map[string]any) (map[string]any, error) {
	gopts, _ := providerOptions["google"].(map[string]any)
	for key := range gopts {
		switch key {
		case "model", "systemInstruction", "tools":
			return nil, fmt.Errorf("%w: %s is derived from the realtime params", provider.ErrInvalidRealtimeConfig, key)
		}
	}
	return jsonmerge.Deep(setup, gopts), nil
}

// liveConn is one Live WebSocket session. A single reader goroutine translates
// server messages into events; writes are serialized by the transport.
type liveConn struct {
	ws      *realtimews.Conn
	ctx     context.Context
	cancel  context.CancelFunc
	events  chan provider.RealtimeEvent
	done    chan struct{}
	closing chan struct{}

	closeOnce sync.Once

	errMu sync.Mutex
	err   error

	// calls maps pending function call IDs to names; the toolResponse needs both.
	callsMu        sync.Mutex
	calls          map[string]string
	scheduling     map[string]string
	clientActivity bool

	// Reader-goroutine state for the current model turn.
	seq           int64
	responding    bool
	interrupted   bool
	turnHadOutput bool
	turnHadTools  bool
	inputText     strings.Builder
	outputText    strings.Builder
}

func (c *liveConn) Events() <-chan provider.RealtimeEvent { return c.events }

func (c *liveConn) Err() error {
	c.errMu.Lock()
	defer c.errMu.Unlock()
	return c.err
}

// Close performs the WebSocket close handshake and waits for the reader to
// finish. It is idempotent. The reader stops delivering events first so it can
// keep reading until the server acknowledges the close.
func (c *liveConn) Close() error {
	c.closeOnce.Do(func() {
		close(c.closing)
		_ = c.ws.Close()
		c.cancel()
	})
	<-c.done
	return nil
}

// abort releases a connection whose reader never started.
func (c *liveConn) abort() {
	c.closeOnce.Do(func() {
		_ = c.ws.CloseNow()
		c.cancel()
		close(c.events)
		close(c.done)
	})
}

func (c *liveConn) Send(ctx context.Context, event provider.RealtimeClientEvent) error {
	if c.ctx.Err() != nil {
		return provider.ErrRealtimeSessionClosed
	}
	msg, err := c.encode(event)
	if err != nil {
		return err
	}
	if err := c.write(ctx, msg); err != nil {
		if c.ctx.Err() != nil {
			return provider.ErrRealtimeSessionClosed
		}
		return fmt.Errorf("google: sending live message: %w", err)
	}
	return nil
}

func (c *liveConn) encode(event provider.RealtimeClientEvent) (liveClientMessage, error) {
	switch event.Type {
	case provider.RealtimeInputAudioAppend:
		if len(event.Audio) == 0 {
			return liveClientMessage{}, fmt.Errorf("%w: empty audio frame", provider.ErrInvalidRealtimeConfig)
		}
		return liveClientMessage{RealtimeInput: &liveRealtimeInput{Audio: &liveBlob{
			MimeType: fmt.Sprintf("audio/pcm;rate=%d", liveInputSampleRate),
			Data:     event.Audio,
		}}}, nil
	case provider.RealtimeInputActivityStart, provider.RealtimeInputActivityEnd:
		if !c.clientActivity {
			return liveClientMessage{}, fmt.Errorf("%w: activity signals require the client_vad turn mode", provider.ErrRealtimeUnsupported)
		}
		input := &liveRealtimeInput{}
		if event.Type == provider.RealtimeInputActivityStart {
			input.ActivityStart = &struct{}{}
		} else {
			input.ActivityEnd = &struct{}{}
		}
		return liveClientMessage{RealtimeInput: input}, nil
	case provider.RealtimeTextInput:
		return liveClientMessage{RealtimeInput: &liveRealtimeInput{Text: event.Text}}, nil
	case provider.RealtimeToolResultEvent:
		return c.encodeToolResult(event.ToolResult)
	default:
		return liveClientMessage{}, fmt.Errorf("%w: gemini live does not support %s", provider.ErrRealtimeUnsupported, event.Type)
	}
}

func (c *liveConn) encodeToolResult(result *provider.RealtimeToolResult) (liveClientMessage, error) {
	if result == nil || result.ToolCallID == "" {
		return liveClientMessage{}, fmt.Errorf("%w: tool result needs a tool call ID", provider.ErrInvalidRealtimeConfig)
	}
	c.callsMu.Lock()
	name, ok := c.calls[result.ToolCallID]
	delete(c.calls, result.ToolCallID)
	c.callsMu.Unlock()
	if !ok {
		return liveClientMessage{}, fmt.Errorf("%w: unknown tool call %q", provider.ErrInvalidRealtimeConfig, result.ToolCallID)
	}
	return liveClientMessage{ToolResponse: &liveToolResponse{FunctionResponses: []liveFunctionResponse{{
		ID:         result.ToolCallID,
		Name:       name,
		Response:   liveResponseObject(result.Output),
		Scheduling: c.scheduling[name],
	}}}}, nil
}

// liveResponseObject wraps non-object results: functionResponse.response must
// be a JSON object.
func liveResponseObject(output json.RawMessage) json.RawMessage {
	trimmed := strings.TrimSpace(string(output))
	if strings.HasPrefix(trimmed, "{") && json.Valid(output) {
		return append(json.RawMessage(nil), output...)
	}
	if trimmed == "" || !json.Valid(output) {
		encoded, _ := json.Marshal(map[string]string{"output": trimmed})
		return encoded
	}
	encoded, _ := json.Marshal(map[string]json.RawMessage{"output": output})
	return encoded
}

func (c *liveConn) write(ctx context.Context, msg liveClientMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.ws.Write(ctx, realtimews.MessageText, data)
}

func (c *liveConn) awaitSetup(ctx context.Context) error {
	for {
		_, data, err := c.ws.Read(ctx)
		if err != nil {
			if closeErr := liveCloseError(err); closeErr != nil {
				return closeErr
			}
			return fmt.Errorf("google: waiting for live setup: %w", err)
		}
		var msg liveServerMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return fmt.Errorf("google: decoding live message: %w", err)
		}
		if msg.Error != nil {
			return msg.Error.err()
		}
		if msg.SetupComplete != nil {
			return nil
		}
	}
}

func (c *liveConn) readLoop() {
	defer close(c.done)
	defer close(c.events)
	for {
		_, data, err := c.ws.Read(c.ctx)
		if err != nil {
			if c.ctx.Err() == nil {
				c.setErr(liveCloseError(err))
			}
			return
		}
		var msg liveServerMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			c.setErr(fmt.Errorf("google: decoding live message: %w", err))
			_ = c.ws.CloseNow()
			return
		}
		for _, event := range c.translate(msg) {
			c.seq++
			event.Sequence = c.seq
			select {
			case c.events <- event:
			case <-c.closing:
			case <-c.ctx.Done():
				return
			}
		}
	}
}

// liveCloseError turns the server's close frame into the terminal error; Live
// reports setup and quota failures this way. A normal closure is not an error.
func liveCloseError(err error) error {
	code, reason, ok := realtimews.CloseStatus(err)
	if !ok {
		return fmt.Errorf("google: live connection lost: %w", err)
	}
	if code == realtimews.StatusNormalClosure {
		return nil
	}
	return fmt.Errorf("google: live session closed with status %d: %s", code, reason)
}

func (e *liveError) err() error {
	return fmt.Errorf("google: live api error %d %s: %s", e.Code, e.Status, e.Message)
}

func (c *liveConn) setErr(err error) {
	if err == nil {
		return
	}
	c.errMu.Lock()
	if c.err == nil {
		c.err = err
	}
	c.errMu.Unlock()
}

// translate maps one server message to common events. It runs only on the
// reader goroutine, which owns the turn state.
func (c *liveConn) translate(msg liveServerMessage) []provider.RealtimeEvent {
	var events []provider.RealtimeEvent
	startResponse := func() {
		if !c.responding {
			c.responding = true
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeResponseStarted})
		}
	}

	if msg.Error != nil {
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeError, Error: msg.Error.err()})
	}

	if va := msg.VoiceActivity; va != nil {
		switch va.Type {
		case "ACTIVITY_START":
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStarted})
		case "ACTIVITY_END":
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputSpeechStopped})
		}
	}

	if sc := msg.ServerContent; sc != nil {
		if tr := sc.InputTranscription; tr != nil && tr.Text != "" {
			c.inputText.WriteString(tr.Text)
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputTranscript, Text: tr.Text})
		}
		if turn := sc.ModelTurn; turn != nil {
			for _, part := range turn.Parts {
				switch {
				case part.Thought:
					// Interleaved reasoning is internal to the model; it is not output.
				case part.InlineData != nil && strings.HasPrefix(part.InlineData.MimeType, "audio/"):
					startResponse()
					c.turnHadOutput = true
					events = append(events, provider.RealtimeEvent{
						Type:        provider.RealtimeOutputAudio,
						Audio:       part.InlineData.Data,
						AudioFormat: liveOutputFormat,
					})
				case part.Text != "":
					startResponse()
					c.turnHadOutput = true
					events = append(events, provider.RealtimeEvent{Type: provider.RealtimeTextDelta, Text: part.Text})
				}
			}
		}
		if tr := sc.OutputTranscription; tr != nil && tr.Text != "" {
			startResponse()
			c.outputText.WriteString(tr.Text)
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeOutputTranscript, Text: tr.Text})
		}
		if sc.Interrupted {
			c.interrupted = true
		}
		if sc.TurnComplete {
			events = append(events, c.finishTurn()...)
		}
	}

	if tc := msg.ToolCall; tc != nil {
		for _, call := range tc.FunctionCalls {
			startResponse()
			c.turnHadTools = true
			c.callsMu.Lock()
			c.calls[call.ID] = call.Name
			c.callsMu.Unlock()
			input := call.Args
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			events = append(events, provider.RealtimeEvent{
				Type:       provider.RealtimeToolCallEvent,
				ToolCallID: call.ID,
				ToolCall:   &provider.RealtimeToolCall{ID: call.ID, Name: call.Name, Input: input},
			})
		}
	}

	if cancel := msg.ToolCallCancellation; cancel != nil {
		for _, id := range cancel.IDs {
			c.callsMu.Lock()
			delete(c.calls, id)
			c.callsMu.Unlock()
			events = append(events, provider.RealtimeEvent{Type: provider.RealtimeToolCallCancelled, ToolCallID: id})
		}
	}

	if ga := msg.GoAway; ga != nil {
		left, _ := time.ParseDuration(ga.TimeLeft)
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeSessionGoAway, TimeToClose: left})
	}

	if ru := msg.SessionResumptionUpdate; ru != nil && ru.Resumable && ru.NewHandle != "" {
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeSessionResumeState, ResumeToken: ru.NewHandle})
	}

	if um := msg.UsageMetadata; um != nil {
		usage := provider.Usage{
			CacheReadTokens: um.CachedContentTokenCount,
			InputTokens:     max(um.PromptTokenCount-um.CachedContentTokenCount, 0),
			OutputTokens:    cmp.Or(um.ResponseTokenCount, um.CandidatesTokenCount),
			ReasoningTokens: um.ThoughtsTokenCount,
			TotalTokens:     um.TotalTokenCount,
		}
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeUsageUpdated, Usage: usage})
	}
	return events
}

// finishTurn closes the current model turn: final transcripts, then the
// response finish with the reason the turn ended.
func (c *liveConn) finishTurn() []provider.RealtimeEvent {
	var events []provider.RealtimeEvent
	if c.inputText.Len() > 0 {
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeInputTranscript, Text: c.inputText.String(), IsFinal: true})
	}
	if c.outputText.Len() > 0 {
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeOutputTranscript, Text: c.outputText.String(), IsFinal: true})
	}
	if c.responding || c.interrupted {
		reason := provider.FinishStop
		switch {
		case c.interrupted:
			reason = provider.FinishInterrupted
		case c.turnHadTools && !c.turnHadOutput:
			reason = provider.FinishToolCalls
		}
		events = append(events, provider.RealtimeEvent{Type: provider.RealtimeResponseFinished, FinishReason: reason})
	}
	c.responding, c.interrupted, c.turnHadOutput, c.turnHadTools = false, false, false, false
	c.inputText.Reset()
	c.outputText.Reset()
	return events
}
