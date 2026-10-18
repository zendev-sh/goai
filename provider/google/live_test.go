package google

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

// liveTestServer is a minimal Gemini Live endpoint. handle runs once per
// connection after the upgrade; it owns the server side of the socket.
func liveTestServer(t *testing.T, handle func(ctx context.Context, conn *websocket.Conn, r *http.Request)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != livePath && r.URL.Path != liveVertexPath {
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		handle(r.Context(), conn, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func readClientMessage(t *testing.T, ctx context.Context, conn *websocket.Conn) map[string]json.RawMessage {
	t.Helper()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Errorf("server read: %v", err)
		return nil
	}
	var msg map[string]json.RawMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Errorf("client sent invalid JSON %s: %v", data, err)
	}
	return msg
}

func sendServerMessage(t *testing.T, ctx context.Context, conn *websocket.Conn, msg string) {
	t.Helper()
	if err := conn.Write(ctx, websocket.MessageText, []byte(msg)); err != nil {
		t.Errorf("server write: %v", err)
	}
}

// acceptSetup reads the setup message, confirms it, and returns it.
func acceptSetup(t *testing.T, ctx context.Context, conn *websocket.Conn) map[string]any {
	t.Helper()
	msg := readClientMessage(t, ctx, conn)
	var setup map[string]any
	if err := json.Unmarshal(msg["setup"], &setup); err != nil {
		t.Errorf("first message is not a setup: %v", msg)
	}
	sendServerMessage(t, ctx, conn, `{"setupComplete":{}}`)
	return setup
}

// drainUntilClose keeps reading so the client's close handshake completes.
func drainUntilClose(ctx context.Context, conn *websocket.Conn) {
	for {
		if _, _, err := conn.Read(ctx); err != nil {
			return
		}
	}
}

func nextEvent(t *testing.T, conn provider.RealtimeConnection) provider.RealtimeEvent {
	t.Helper()
	select {
	case event, ok := <-conn.Events():
		if !ok {
			t.Fatalf("events closed early; Err() = %v", conn.Err())
		}
		return event
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for a live event")
		return provider.RealtimeEvent{}
	}
}

func TestLive_SetupMessage(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	setupCh := make(chan map[string]any, 1)
	headerCh := make(chan http.Header, 1)
	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, r *http.Request) {
		headerCh <- r.Header.Clone()
		setupCh <- acceptSetup(t, ctx, conn)
		drainUntilClose(ctx, conn)
	})

	model := Live("gemini-3.8-live",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"X-Custom": "yes", "x-goog-api-key": "spoofed"}),
	)
	conn, err := model.OpenRealtime(t.Context(), provider.RealtimeParams{
		System:      "Be brief.",
		ToolChoice:  "auto",
		Voice:       "Kore",
		InputAudio:  liveInputFormat,
		OutputAudio: liveOutputFormat,
		Tools: []provider.ToolDefinition{
			{
				Name:                   "lookup",
				Description:            "Looks things up",
				InputSchema:            json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
				ProviderDefinedOptions: LiveToolOptions(LiveNonBlocking, LiveWhenIdle),
			},
			{Name: "plain", Description: "No options"},
		},
		TurnDetection: provider.TurnDetection{
			Mode:                provider.TurnModeClientVAD,
			DisableInterruption: true,
			ProviderOptions:     map[string]any{"silenceDurationMs": 500},
		},
		InputTranscription:  provider.TranscriptionConfig{Enabled: true, Language: "es-MX"},
		OutputTranscription: provider.TranscriptionConfig{Enabled: true},
		MaxOutputTokens:     256,
		ProviderOptions: map[string]any{"google": map[string]any{
			"sessionResumption":        map[string]any{"handle": "prev"},
			"contextWindowCompression": map[string]any{"slidingWindow": map[string]any{}},
			"generationConfig":         map[string]any{"temperature": 0.5, "speechConfig": map[string]any{"languageCode": "es-US"}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	if ev := nextEvent(t, conn); ev.Type != provider.RealtimeSessionReady {
		t.Fatalf("first event = %s, want session ready", ev.Type)
	}

	header := <-headerCh
	if header.Get("x-goog-api-key") != "test-key" || header.Get("X-Custom") != "yes" {
		t.Fatalf("unexpected handshake headers: %v", header)
	}

	setup := <-setupCh
	got, _ := json.Marshal(setup)
	for _, want := range []string{
		`"model":"models/gemini-3.8-live"`,
		`"systemInstruction":{"parts":[{"text":"Be brief."}]}`,
		`"behavior":"NON_BLOCKING"`,
		`"responseModalities":["AUDIO"]`,
		`"maxOutputTokens":256`,
		`"temperature":0.5`,
		`"voiceName":"Kore"`,
		`"languageCode":"es-US"`,
		`"automaticActivityDetection":{"disabled":true,"silenceDurationMs":500}`,
		`"activityHandling":"NO_INTERRUPTION"`,
		`"inputAudioTranscription":{"languageCodes":["es-MX"]}`,
		`"outputAudioTranscription":{}`,
		`"sessionResumption":{"handle":"prev"}`,
		`"contextWindowCompression":{"slidingWindow":{}}`,
	} {
		if !strings.Contains(string(got), want) {
			t.Errorf("setup missing %s\nsetup: %s", want, got)
		}
	}
	if strings.Contains(string(got), "toolConfig") {
		t.Errorf("setup must not carry toolConfig: %s", got)
	}
	if strings.Count(string(got), `"behavior"`) != 1 {
		t.Errorf("only the configured tool carries a behavior: %s", got)
	}
}

func TestLive_VertexBackend(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	type handshake struct {
		path, auth, apiKey, model string
	}
	got := make(chan handshake, 1)
	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, r *http.Request) {
		setup := acceptSetup(t, ctx, conn)
		model, _ := setup["model"].(string)
		got <- handshake{r.URL.Path, r.Header.Get("Authorization"), r.Header.Get("x-goog-api-key"), model}
		drainUntilClose(ctx, conn)
	})

	conn, err := Live("gemini-3.8-live",
		WithVertex("my-project", "us-central1"),
		WithVertexBaseURL(server.URL+"/ignored/path"),
		WithTokenSource(provider.StaticToken("oauth-token")),
	).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	h := <-got
	if h.path != liveVertexPath {
		t.Errorf("path = %s, want %s", h.path, liveVertexPath)
	}
	if h.auth != "Bearer oauth-token" || h.apiKey != "" {
		t.Errorf("auth headers = %q / %q", h.auth, h.apiKey)
	}
	if want := "projects/my-project/locations/us-central1/publishers/google/models/gemini-3.8-live"; h.model != want {
		t.Errorf("model = %s, want %s", h.model, want)
	}

	for in, want := range map[string]string{
		"gemini-3.8-live":                                   "projects/p/locations/l/publishers/google/models/gemini-3.8-live",
		"models/gemini-3.8-live":                            "projects/p/locations/l/publishers/google/models/gemini-3.8-live",
		"publishers/google/models/gemini-3.8-live":          "projects/p/locations/l/publishers/google/models/gemini-3.8-live",
		"projects/x/locations/y/publishers/google/models/z": "projects/x/locations/y/publishers/google/models/z",
	} {
		if got := liveVertexModelName(in, "p", "l"); got != want {
			t.Errorf("liveVertexModelName(%q) = %s, want %s", in, got, want)
		}
	}
	m := &liveModel{id: "m", opts: options{isVertex: true, project: "p", location: "global"}}
	if endpoint, _, _ := m.target(); endpoint != "wss://aiplatform.googleapis.com"+liveVertexPath {
		t.Errorf("global endpoint = %s", endpoint)
	}
}

func TestLive_SetupValidation(t *testing.T) {
	model := Live("gemini-3.8-live", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	pcm := func(rate int) provider.AudioFormat {
		return provider.AudioFormat{Encoding: provider.AudioPCM16, SampleRate: rate, Channels: 1}
	}
	tests := []struct {
		name   string
		params provider.RealtimeParams
		want   error
	}{
		{"input rate", provider.RealtimeParams{InputAudio: pcm(24_000)}, provider.ErrUnsupportedAudioFormat},
		{"output rate", provider.RealtimeParams{OutputAudio: pcm(16_000)}, provider.ErrUnsupportedAudioFormat},
		{"manual turns", provider.RealtimeParams{TurnDetection: provider.TurnDetection{Mode: provider.TurnModeManual}}, provider.ErrRealtimeUnsupported},
		{"no auto response", provider.RealtimeParams{TurnDetection: provider.TurnDetection{DisableAutoResponse: true}}, provider.ErrRealtimeUnsupported},
		{"tool choice", provider.RealtimeParams{ToolChoice: "required"}, provider.ErrRealtimeUnsupported},
		{"transcription prompt", provider.RealtimeParams{InputTranscription: provider.TranscriptionConfig{Enabled: true, Prompt: "x"}}, provider.ErrRealtimeUnsupported},
		{"bad behavior", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "t", ProviderDefinedOptions: map[string]any{"behavior": "ASYNC"}}}}, provider.ErrInvalidRealtimeConfig},
		{"bad scheduling", provider.RealtimeParams{Tools: []provider.ToolDefinition{{Name: "t", ProviderDefinedOptions: map[string]any{"scheduling": "LATER"}}}}, provider.ErrInvalidRealtimeConfig},
		{"override model", provider.RealtimeParams{ProviderOptions: map[string]any{"google": map[string]any{"model": "x"}}}, provider.ErrInvalidRealtimeConfig},
		{"common validation", provider.RealtimeParams{MaxOutputTokens: -1}, provider.ErrInvalidRealtimeConfig},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := model.OpenRealtime(t.Context(), tt.params)
			if !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}

	if _, err := Live("m", WithAPIKey("k"), WithVertex("p", "us-central1")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected Vertex to reject an API key")
	}
	if _, err := Live("m", WithTokenSource(provider.StaticToken("t")), WithVertex("bad/project", "us-central1")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected an invalid Vertex project to be rejected")
	}
	if _, err := Live("m", WithTokenSource(provider.StaticToken("t")), WithVertex("p", "us central")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected an invalid Vertex location to be rejected")
	}
	if _, err := Live("m", WithAPIKey("k"), WithBaseURL("ftp://example.com")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil {
		t.Fatal("expected unsupported scheme error")
	}
	failing := provider.CachedTokenSource(func(context.Context) (*provider.Token, error) { return nil, errors.New("no token") })
	if _, err := Live("m", WithTokenSource(failing), WithBaseURL("http://127.0.0.1:1")).OpenRealtime(t.Context(), provider.RealtimeParams{}); err == nil || !strings.Contains(err.Error(), "no token") {
		t.Fatalf("token error = %v", err)
	}
}

func TestLive_ServerEvents(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSetup(t, ctx, conn)
		for _, msg := range []string{
			`{"voiceActivity":{"type":"ACTIVITY_START"}}`,
			`{"serverContent":{"inputTranscription":{"text":"hola "}}}`,
			`{"serverContent":{"inputTranscription":{"text":"mundo"}}}`,
			`{"voiceActivity":{"type":"ACTIVITY_END"}}`,
			`{"serverContent":{"modelTurn":{"parts":[{"text":"plan","thought":true},{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQID"}}]}}}`,
			`{"serverContent":{"outputTranscription":{"text":"Hi"}}}`,
			`{"serverContent":{"generationComplete":true}}`,
			`{"serverContent":{"turnComplete":true}}`,
			`{"goAway":{"timeLeft":"5.5s"}}`,
			`{"sessionResumptionUpdate":{"newHandle":"h1","resumable":true}}`,
			`{"sessionResumptionUpdate":{"resumable":false}}`,
			`{"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":40,"responseTokenCount":20,"thoughtsTokenCount":5,"totalTokenCount":125}}`,
			`{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":7,"totalTokenCount":17}}`,
			`{"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}`,
		} {
			sendServerMessage(t, ctx, conn, msg)
		}
		drainUntilClose(ctx, conn)
	})

	conn, err := Live("gemini-3.8-live", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	want := []provider.RealtimeEventType{
		provider.RealtimeSessionReady,
		provider.RealtimeInputSpeechStarted,
		provider.RealtimeInputTranscript,
		provider.RealtimeInputTranscript,
		provider.RealtimeInputSpeechStopped,
		provider.RealtimeResponseStarted,
		provider.RealtimeOutputAudio,
		provider.RealtimeOutputTranscript,
		provider.RealtimeInputTranscript,  // final
		provider.RealtimeOutputTranscript, // final
		provider.RealtimeResponseFinished,
		provider.RealtimeSessionGoAway,
		provider.RealtimeSessionResumeState,
		provider.RealtimeUsageUpdated,
		provider.RealtimeUsageUpdated, // Vertex AI naming
		provider.RealtimeError,
	}
	events := make([]provider.RealtimeEvent, 0, len(want))
	for range want {
		events = append(events, nextEvent(t, conn))
	}
	for i, ev := range events {
		if ev.Type != want[i] {
			t.Fatalf("event %d = %s, want %s (all: %+v)", i, ev.Type, want[i], events)
		}
		if i > 0 && ev.Sequence <= events[i-1].Sequence {
			t.Fatalf("sequence not increasing at %d: %d after %d", i, ev.Sequence, events[i-1].Sequence)
		}
	}
	if ev := events[6]; string(ev.Audio) != "\x01\x02\x03" || ev.AudioFormat != liveOutputFormat {
		t.Fatalf("audio event = %+v", ev)
	}
	if ev := events[8]; !ev.IsFinal || ev.Text != "hola mundo" {
		t.Fatalf("final input transcript = %+v", ev)
	}
	if ev := events[9]; !ev.IsFinal || ev.Text != "Hi" {
		t.Fatalf("final output transcript = %+v", ev)
	}
	if ev := events[10]; ev.FinishReason != provider.FinishStop {
		t.Fatalf("finish reason = %s", ev.FinishReason)
	}
	if ev := events[11]; ev.TimeToClose != 5500*time.Millisecond {
		t.Fatalf("time to close = %s", ev.TimeToClose)
	}
	if ev := events[12]; ev.ResumeToken != "h1" {
		t.Fatalf("resume token = %q", ev.ResumeToken)
	}
	wantUsage := provider.Usage{InputTokens: 60, CacheReadTokens: 40, OutputTokens: 20, ReasoningTokens: 5, TotalTokens: 125}
	if ev := events[13]; ev.Usage != wantUsage {
		t.Fatalf("usage = %+v, want %+v", ev.Usage, wantUsage)
	}
	if ev := events[14]; ev.Usage.OutputTokens != 7 || ev.Usage.InputTokens != 10 {
		t.Fatalf("vertex usage = %+v", ev.Usage)
	}
	if ev := events[15]; ev.Error == nil || !strings.Contains(ev.Error.Error(), "RESOURCE_EXHAUSTED") {
		t.Fatalf("error event = %+v", ev)
	}
}

func TestLive_InterruptedAndToolOnlyTurns(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSetup(t, ctx, conn)
		for _, msg := range []string{
			`{"serverContent":{"modelTurn":{"parts":[{"text":"partial"}]}}}`,
			`{"serverContent":{"interrupted":true}}`,
			`{"serverContent":{"turnComplete":true}}`,
			`{"toolCall":{"functionCalls":[{"id":"c1","name":"lookup"}]}}`,
			`{"serverContent":{"turnComplete":true}}`,
			`{"serverContent":{"turnComplete":true}}`,
		} {
			sendServerMessage(t, ctx, conn, msg)
		}
		drainUntilClose(ctx, conn)
	})

	conn, err := Live("gemini-3.8-live", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	nextEvent(t, conn) // ready
	if ev := nextEvent(t, conn); ev.Type != provider.RealtimeResponseStarted {
		t.Fatalf("got %s", ev.Type)
	}
	if ev := nextEvent(t, conn); ev.Type != provider.RealtimeTextDelta || ev.Text != "partial" {
		t.Fatalf("got %+v", ev)
	}
	if ev := nextEvent(t, conn); ev.FinishReason != provider.FinishInterrupted {
		t.Fatalf("interrupted turn finished with %+v", ev)
	}
	nextEvent(t, conn) // response started for the tool turn
	call := nextEvent(t, conn)
	if call.Type != provider.RealtimeToolCallEvent || call.ToolCall == nil || string(call.ToolCall.Input) != `{}` {
		t.Fatalf("tool call = %+v", call)
	}
	if ev := nextEvent(t, conn); ev.FinishReason != provider.FinishToolCalls {
		t.Fatalf("tool-only turn finished with %+v", ev)
	}

	// A turnComplete with no response activity emits nothing; the next event
	// proves it by being the channel close after the server is told to stop.
	_ = conn.Close()
	for ev := range conn.Events() {
		t.Fatalf("unexpected event after an empty turn: %+v", ev)
	}
}

func TestLive_ClientMessages(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	received := make(chan map[string]json.RawMessage, 8)
	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSetup(t, ctx, conn)
		sendServerMessage(t, ctx, conn, `{"toolCall":{"functionCalls":[{"id":"c1","name":"lookup","args":{"q":"go"}},{"id":"c2","name":"plain"}]}}`)
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]json.RawMessage
			_ = json.Unmarshal(data, &msg)
			received <- msg
		}
	})

	conn, err := Live("gemini-3.8-live", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{
		TurnDetection: provider.TurnDetection{Mode: provider.TurnModeClientVAD},
		Tools: []provider.ToolDefinition{
			{Name: "lookup", ProviderDefinedOptions: LiveToolOptions(LiveNonBlocking, LiveSilent)},
			{Name: "plain"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	for range 4 { // ready, response started, two tool calls
		nextEvent(t, conn)
	}

	ctx := t.Context()
	send := func(ev provider.RealtimeClientEvent) string {
		t.Helper()
		if err := conn.Send(ctx, ev); err != nil {
			t.Fatalf("Send(%s): %v", ev.Type, err)
		}
		select {
		case msg := <-received:
			out, _ := json.Marshal(msg)
			return string(out)
		case <-time.After(2 * time.Second):
			t.Fatalf("server did not receive %s", ev.Type)
			return ""
		}
	}

	if got := send(provider.RealtimeClientEvent{Type: provider.RealtimeInputAudioAppend, Audio: []byte{1, 2, 3}}); got != `{"realtimeInput":{"audio":{"mimeType":"audio/pcm;rate=16000","data":"AQID"}}}` {
		t.Errorf("audio message = %s", got)
	}
	if got := send(provider.RealtimeClientEvent{Type: provider.RealtimeInputActivityStart}); got != `{"realtimeInput":{"activityStart":{}}}` {
		t.Errorf("activity start = %s", got)
	}
	if got := send(provider.RealtimeClientEvent{Type: provider.RealtimeInputActivityEnd}); got != `{"realtimeInput":{"activityEnd":{}}}` {
		t.Errorf("activity end = %s", got)
	}
	if got := send(provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "hola"}); got != `{"realtimeInput":{"text":"hola"}}` {
		t.Errorf("text = %s", got)
	}
	got := send(provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{
		ToolCallID: "c1", Output: json.RawMessage(`{"answer":42}`),
	}})
	if want := `{"toolResponse":{"functionResponses":[{"id":"c1","name":"lookup","response":{"answer":42},"scheduling":"SILENT"}]}}`; got != want {
		t.Errorf("tool response = %s, want %s", got, want)
	}
	got = send(provider.RealtimeClientEvent{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{
		ToolCallID: "c2", Output: json.RawMessage(`[1,2]`),
	}})
	if want := `{"toolResponse":{"functionResponses":[{"id":"c2","name":"plain","response":{"output":[1,2]}}]}}`; got != want {
		t.Errorf("non-object tool response = %s, want %s", got, want)
	}

	for _, ev := range []provider.RealtimeClientEvent{
		{Type: provider.RealtimeInputAudioCommit},
		{Type: provider.RealtimeResponseCreate},
		{Type: provider.RealtimeResponseCancel},
	} {
		if err := conn.Send(ctx, ev); !errors.Is(err, provider.ErrRealtimeUnsupported) {
			t.Errorf("Send(%s) = %v, want ErrRealtimeUnsupported", ev.Type, err)
		}
	}
	for _, ev := range []provider.RealtimeClientEvent{
		{Type: provider.RealtimeInputAudioAppend},
		{Type: provider.RealtimeToolResultEvent, ToolResult: &provider.RealtimeToolResult{ToolCallID: "c1"}}, // already answered
		{Type: provider.RealtimeToolResultEvent},
	} {
		if err := conn.Send(ctx, ev); !errors.Is(err, provider.ErrInvalidRealtimeConfig) {
			t.Errorf("Send(%s) = %v, want ErrInvalidRealtimeConfig", ev.Type, err)
		}
	}

	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if err := conn.Close(); err != nil {
		t.Fatal("second Close:", err)
	}
	if err := conn.Send(ctx, provider.RealtimeClientEvent{Type: provider.RealtimeTextInput, Text: "x"}); !errors.Is(err, provider.ErrRealtimeSessionClosed) {
		t.Fatalf("Send after Close = %v", err)
	}
	if conn.Err() != nil {
		t.Fatalf("Err after normal close = %v", conn.Err())
	}
}

func TestLive_ActivitySignalsNeedClientVAD(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSetup(t, ctx, conn)
		drainUntilClose(ctx, conn)
	})
	conn, err := Live("gemini-3.8-live", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	if err := conn.Send(t.Context(), provider.RealtimeClientEvent{Type: provider.RealtimeInputActivityStart}); !errors.Is(err, provider.ErrRealtimeUnsupported) {
		t.Fatalf("activity start without client VAD = %v", err)
	}
}

func TestLive_ConnectionErrors(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	t.Run("http error on upgrade", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"error":{"code":403,"message":"API key not valid"}}`))
		}))
		defer server.Close()
		_, err := Live("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		var apiErr *goai.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusForbidden {
			t.Fatalf("err = %v, want APIError 403", err)
		}
	})

	t.Run("setup rejected", func(t *testing.T) {
		server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			readClientMessage(t, ctx, conn)
			_ = conn.Close(websocket.StatusInvalidFramePayloadData, "models/x is not found")
		})
		_, err := Live("x", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err == nil || !strings.Contains(err.Error(), "models/x is not found") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("server closes mid session", func(t *testing.T) {
		server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			acceptSetup(t, ctx, conn)
			_ = conn.Close(websocket.StatusInternalError, "internal")
		})
		conn, err := Live("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = conn.Close() }()
		for range conn.Events() {
		}
		if err := conn.Err(); err == nil || !strings.Contains(err.Error(), "1011") {
			t.Fatalf("Err() = %v", err)
		}
	})

	t.Run("malformed server message", func(t *testing.T) {
		server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			acceptSetup(t, ctx, conn)
			sendServerMessage(t, ctx, conn, `{not json`)
			drainUntilClose(ctx, conn)
		})
		conn, err := Live("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = conn.Close() }()
		for range conn.Events() {
		}
		if conn.Err() == nil {
			t.Fatal("expected a decode error")
		}
	})

	t.Run("error message during setup", func(t *testing.T) {
		server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			readClientMessage(t, ctx, conn)
			sendServerMessage(t, ctx, conn, `{"error":{"code":400,"message":"Invalid JSON payload","status":"INVALID_ARGUMENT"}}`)
			drainUntilClose(ctx, conn)
		})
		_, err := Live("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(t.Context(), provider.RealtimeParams{})
		if err == nil || !strings.Contains(err.Error(), "Invalid JSON payload") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("context cancelled during setup", func(t *testing.T) {
		server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
			readClientMessage(t, ctx, conn)
			drainUntilClose(ctx, conn)
		})
		ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
		defer cancel()
		if _, err := Live("m", WithAPIKey("k"), WithBaseURL(server.URL)).OpenRealtime(ctx, provider.RealtimeParams{}); err == nil {
			t.Fatal("expected setup to fail when ctx expires")
		}
	})
}

// TestLive_RealtimeSessionToolRoundTrip drives the adapter through the goai
// session runtime: the model's tool call is executed and answered automatically.
func TestLive_RealtimeSessionToolRoundTrip(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	response := make(chan string, 1)
	server := liveTestServer(t, func(ctx context.Context, conn *websocket.Conn, _ *http.Request) {
		acceptSetup(t, ctx, conn)
		sendServerMessage(t, ctx, conn, `{"toolCall":{"functionCalls":[{"id":"call-1","name":"weather","args":{"city":"Lima"}}]}}`)
		msg := readClientMessage(t, ctx, conn)
		response <- string(msg["toolResponse"])
		drainUntilClose(ctx, conn)
	})

	weather := goai.Tool{
		Name:        "weather",
		Description: "Current weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct{ City string }
			_ = json.Unmarshal(input, &args)
			return "sunny in " + args.City, nil
		},
	}
	session, err := goai.OpenRealtimeSession(t.Context(),
		Live("gemini-3.8-live", WithAPIKey("k"), WithBaseURL(server.URL)),
		goai.WithRealtimeTools(weather),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	select {
	case got := <-response:
		if want := `{"functionResponses":[{"id":"call-1","name":"weather","response":{"text":"sunny in Lima"}}]}`; got != want {
			t.Fatalf("toolResponse = %s, want %s", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("tool response never reached the server")
	}
}

func TestLiveResponseObject(t *testing.T) {
	tests := map[string]string{
		`{"a":1}`:  `{"a":1}`,
		`"text"`:   `{"output":"text"}`,
		`[1]`:      `{"output":[1]}`,
		`not json`: `{"output":"not json"}`,
		``:         `{"output":""}`,
	}
	for in, want := range tests {
		if got := string(liveResponseObject(json.RawMessage(in))); got != want {
			t.Errorf("liveResponseObject(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestLiveRealtimeCapabilities(t *testing.T) {
	model := Live("gemini-3.8-live")
	if model.ModelID() != "gemini-3.8-live" {
		t.Fatalf("ModelID = %q", model.ModelID())
	}
	caps := provider.RealtimeCapabilitiesOf(model)
	if !caps.FunctionCalling || !caps.NonBlockingToolCalls || caps.ExplicitResponseRequest || caps.OutputAudioTruncation {
		t.Fatalf("unexpected capabilities: %+v", caps)
	}
	if len(caps.SupportedInputAudio) != 1 || caps.SupportedInputAudio[0] != liveInputFormat {
		t.Fatalf("input formats = %+v", caps.SupportedInputAudio)
	}
	if got, _ := liveURL("https://example.com/base/", livePath, true); got != "wss://example.com/base"+livePath {
		t.Fatalf("liveURL = %s", got)
	}
	if got, _ := liveURL("https://example.com/v1beta1/projects/p", liveVertexPath, false); got != "wss://example.com"+liveVertexPath {
		t.Fatalf("vertex liveURL = %s", got)
	}
	if liveModelName("models/x") != "models/x" {
		t.Fatal("liveModelName must keep an explicit models/ prefix")
	}
}

func FuzzLiveTranslate(f *testing.F) {
	for _, seed := range []string{
		`{"setupComplete":{}}`,
		`{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm","data":"AQID"}}]},"turnComplete":true}}`,
		`{"toolCall":{"functionCalls":[{"id":"1","name":"f","args":{}}]}}`,
		`{"toolCallCancellation":{"ids":["1"]}}`,
		`{"goAway":{"timeLeft":"bogus"}}`,
		`{"usageMetadata":{"promptTokenCount":-5}}`,
		`{"voiceActivity":{"type":"ACTIVITY_START"}}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(_ *testing.T, data []byte) {
		var msg liveServerMessage
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		c := &liveConn{calls: map[string]string{}}
		_ = c.translate(msg)
		_ = c.finishTurn()
	})
}
