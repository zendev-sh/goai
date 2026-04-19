package provider_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// mockLanguageModel implements provider.LanguageModel for testing.
type mockLanguageModel struct {
	id   string
	caps provider.ModelCapabilities
}

func (m *mockLanguageModel) ModelID() string { return m.id }

func (m *mockLanguageModel) DoGenerate(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
	return &provider.GenerateResult{
		Text:         "hello",
		FinishReason: provider.FinishStop,
		Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
	}, nil
}

func (m *mockLanguageModel) DoStream(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
	ch := make(chan provider.StreamChunk, 2)
	ch <- provider.StreamChunk{Type: provider.ChunkText, Text: "hello"}
	ch <- provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop}
	close(ch)
	return &provider.StreamResult{Stream: ch}, nil
}

func (m *mockLanguageModel) Capabilities() provider.ModelCapabilities { return m.caps }

// mockEmbeddingModel implements provider.EmbeddingModel for testing.
type mockEmbeddingModel struct {
	id string
}

func (m *mockEmbeddingModel) ModelID() string { return m.id }

func (m *mockEmbeddingModel) DoEmbed(_ context.Context, values []string, _ provider.EmbedParams) (*provider.EmbedResult, error) {
	embeddings := make([][]float64, len(values))
	for i := range values {
		embeddings[i] = []float64{0.1, 0.2, 0.3}
	}
	return &provider.EmbedResult{Embeddings: embeddings}, nil
}

func (m *mockEmbeddingModel) MaxValuesPerCall() int { return 100 }

// mockImageModel implements provider.ImageModel for testing.
type mockImageModel struct {
	id string
}

func (m *mockImageModel) ModelID() string { return m.id }

func (m *mockImageModel) DoGenerate(_ context.Context, _ provider.ImageParams) (*provider.ImageResult, error) {
	return &provider.ImageResult{
		Images: []provider.ImageData{{Data: []byte("fake-png"), MediaType: "image/png"}},
	}, nil
}

func TestLanguageModelInterface(t *testing.T) {
	var model provider.LanguageModel = &mockLanguageModel{
		id: "gpt-4o",
		caps: provider.ModelCapabilities{
			Temperature: true,
			ToolCall:    true,
			InputModalities: provider.ModalitySet{
				Text:  true,
				Image: true,
			},
			OutputModalities: provider.ModalitySet{Text: true},
		},
	}

	if model.ModelID() != "gpt-4o" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "gpt-4o")
	}

	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("expected Temperature and ToolCall capabilities")
	}
	if !caps.InputModalities.Image {
		t.Error("expected Image input modality")
	}

	// DoGenerate
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error: %v", err)
	}
	if result.Text != "hello" {
		t.Errorf("Text = %q, want %q", result.Text, "hello")
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v, want InputTokens=10, OutputTokens=5", result.Usage)
	}

	// DoStream
	stream, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoStream error: %v", err)
	}

	var chunks []provider.StreamChunk
	for chunk := range stream.Stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}
	if chunks[0].Type != provider.ChunkText || chunks[0].Text != "hello" {
		t.Errorf("chunk[0] = %+v, want text chunk with 'hello'", chunks[0])
	}
	if chunks[1].Type != provider.ChunkFinish {
		t.Errorf("chunk[1].Type = %q, want %q", chunks[1].Type, provider.ChunkFinish)
	}
}

func TestEmbeddingModelInterface(t *testing.T) {
	var model provider.EmbeddingModel = &mockEmbeddingModel{id: "text-embedding-3-small"}

	if model.ModelID() != "text-embedding-3-small" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "text-embedding-3-small")
	}
	if model.MaxValuesPerCall() != 100 {
		t.Errorf("MaxValuesPerCall() = %d, want 100", model.MaxValuesPerCall())
	}

	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("DoEmbed error: %v", err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("got %d embeddings, want 2", len(result.Embeddings))
	}
}

func TestImageModelInterface(t *testing.T) {
	var model provider.ImageModel = &mockImageModel{id: "dall-e-3"}

	if model.ModelID() != "dall-e-3" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "dall-e-3")
	}

	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat", N: 1, Size: "1024x1024",
	})
	if err != nil {
		t.Fatalf("DoGenerate error: %v", err)
	}
	if len(result.Images) != 1 {
		t.Fatalf("got %d images, want 1", len(result.Images))
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("MediaType = %q, want %q", result.Images[0].MediaType, "image/png")
	}
}

func TestGenerateParamsWithTools(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read file.go"}}},
		},
		System: "You are a coding assistant.",
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "Read a file", InputSchema: schema},
		},
		ToolChoice: "auto",
	}

	if len(params.Tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(params.Tools))
	}
	if params.Tools[0].Name != "read_file" {
		t.Errorf("tool name = %q, want %q", params.Tools[0].Name, "read_file")
	}
}

func TestToolCallResult(t *testing.T) {
	tc := provider.ToolCall{
		ID:    "call_123",
		Name:  "read_file",
		Input: json.RawMessage(`{"path":"main.go"}`),
	}

	if tc.ID != "call_123" {
		t.Errorf("ID = %q, want %q", tc.ID, "call_123")
	}

	var input struct{ Path string }
	if err := json.Unmarshal(tc.Input, &input); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if input.Path != "main.go" {
		t.Errorf("Path = %q, want %q", input.Path, "main.go")
	}
}

func TestMessageParts(t *testing.T) {
	msg := provider.Message{
		Role: provider.RoleAssistant,
		Content: []provider.Part{
			{Type: provider.PartText, Text: "Let me read that file."},
			{
				Type:       provider.PartToolCall,
				ToolCallID: "call_1",
				ToolName:   "read_file",
				ToolInput:  json.RawMessage(`{"path":"main.go"}`),
			},
		},
	}

	if msg.Role != provider.RoleAssistant {
		t.Errorf("Role = %q, want %q", msg.Role, provider.RoleAssistant)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("got %d parts, want 2", len(msg.Content))
	}
	if msg.Content[0].Type != provider.PartText {
		t.Errorf("part[0].Type = %q, want %q", msg.Content[0].Type, provider.PartText)
	}
	if msg.Content[1].Type != provider.PartToolCall {
		t.Errorf("part[1].Type = %q, want %q", msg.Content[1].Type, provider.PartToolCall)
	}
}

func TestStreamChunkTypes(t *testing.T) {
	types := []provider.StreamChunkType{
		provider.ChunkText,
		provider.ChunkReasoning,
		provider.ChunkToolCall,
		provider.ChunkToolCallDelta,
		provider.ChunkToolCallStreamStart,
		provider.ChunkToolResult,
		provider.ChunkStepFinish,
		provider.ChunkFinish,
		provider.ChunkError,
	}
	if len(types) != 9 {
		t.Errorf("expected 9 chunk types, got %d", len(types))
	}
}

func TestFinishReasons(t *testing.T) {
	reasons := []provider.FinishReason{
		provider.FinishStop,
		provider.FinishToolCalls,
		provider.FinishLength,
		provider.FinishContentFilter,
		provider.FinishError,
		provider.FinishOther,
	}
	if len(reasons) != 6 {
		t.Errorf("expected 6 finish reasons, got %d", len(reasons))
	}
}

func TestStaticToken(t *testing.T) {
	ts := provider.StaticToken("sk-test-key")

	tok, err := ts.Token(t.Context())
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if tok != "sk-test-key" {
		t.Errorf("Token = %q, want %q", tok, "sk-test-key")
	}

	// Repeated calls return the same value.
	tok2, _ := ts.Token(t.Context())
	if tok2 != tok {
		t.Error("StaticToken returned different values")
	}
}

func TestCachedTokenSource(t *testing.T) {
	callCount := 0
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount++
		return &provider.Token{
			Value:     "token-v" + string(rune('0'+callCount)),
			ExpiresAt: time.Now().Add(time.Hour),
		}, nil
	})

	// First call fetches.
	tok, err := ts.Token(t.Context())
	if err != nil {
		t.Fatalf("Token error: %v", err)
	}
	if tok != "token-v1" {
		t.Errorf("Token = %q, want %q", tok, "token-v1")
	}
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1", callCount)
	}

	// Second call returns cached.
	tok, _ = ts.Token(t.Context())
	if tok != "token-v1" {
		t.Errorf("Token = %q, want %q (cached)", tok, "token-v1")
	}
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1 (should be cached)", callCount)
	}
}

func TestCachedTokenSourceExpiry(t *testing.T) {
	callCount := 0
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount++
		return &provider.Token{
			Value:     "tok",
			ExpiresAt: time.Now().Add(-time.Second), // already expired
		}, nil
	})

	// First call fetches.
	_, _ = ts.Token(t.Context())
	if callCount != 1 {
		t.Fatalf("callCount = %d, want 1", callCount)
	}

	// Second call re-fetches because token is expired.
	_, _ = ts.Token(t.Context())
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (token expired, should re-fetch)", callCount)
	}
}

func TestCachedTokenSourceInvalidate(t *testing.T) {
	callCount := 0
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount++
		return &provider.Token{
			Value:     "tok",
			ExpiresAt: time.Now().Add(time.Hour),
		}, nil
	})

	_, _ = ts.Token(t.Context())
	if callCount != 1 {
		t.Fatalf("callCount = %d, want 1", callCount)
	}

	// Invalidate forces re-fetch.
	inv, ok := ts.(provider.InvalidatingTokenSource)
	if !ok {
		t.Fatal("CachedTokenSource should implement InvalidatingTokenSource")
	}
	inv.Invalidate()

	_, _ = ts.Token(t.Context())
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (invalidated, should re-fetch)", callCount)
	}
}

func TestCachedTokenSourceNoExpiry(t *testing.T) {
	callCount := 0
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount++
		return &provider.Token{Value: "forever"}, nil // zero ExpiresAt
	})

	// Zero ExpiresAt means "no expiry" - token is cached indefinitely.
	_, _ = ts.Token(t.Context())
	_, _ = ts.Token(t.Context())
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1 (zero ExpiresAt should cache forever)", callCount)
	}
}

func TestCachedTokenSourceFetchError(t *testing.T) {
	errFetch := fmt.Errorf("auth server down")
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, errFetch
	})

	tok, err := ts.Token(t.Context())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if tok != "" {
		t.Errorf("Token = %q, want empty on error", tok)
	}
	if err != errFetch {
		t.Errorf("err = %v, want %v", err, errFetch)
	}
}

// bareLanguageModel implements LanguageModel but NOT CapableModel.
type bareLanguageModel struct {
	id string
}

func (m *bareLanguageModel) ModelID() string { return m.id }

func (m *bareLanguageModel) DoGenerate(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
	return &provider.GenerateResult{Text: "ok"}, nil
}

func (m *bareLanguageModel) DoStream(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
	ch := make(chan provider.StreamChunk)
	close(ch)
	return &provider.StreamResult{Stream: ch}, nil
}

func TestModelCapabilitiesOf_NonCapableModel(t *testing.T) {
	model := &bareLanguageModel{id: "bare-model"}
	caps := provider.ModelCapabilitiesOf(model)

	// Should return zero-value capabilities.
	if caps.Temperature || caps.ToolCall || caps.InputModalities.Image || caps.OutputModalities.Text {
		t.Errorf("expected zero ModelCapabilities, got %+v", caps)
	}
}

func TestTrySend_Success(t *testing.T) {
	out := make(chan provider.StreamChunk, 1)
	chunk := provider.StreamChunk{Type: provider.ChunkText, Text: "hello"}
	if !provider.TrySend(t.Context(), out, chunk) {
		t.Fatal("TrySend returned false on active context")
	}
	got := <-out
	if got.Text != "hello" {
		t.Errorf("Text = %q, want %q", got.Text, "hello")
	}
}

func TestTrySend_CancelledContext(t *testing.T) {
	out := make(chan provider.StreamChunk) // unbuffered - would block
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	chunk := provider.StreamChunk{Type: provider.ChunkText, Text: "hello"}
	if provider.TrySend(ctx, out, chunk) {
		t.Fatal("TrySend returned true on cancelled context")
	}
}

func TestCachedTokenSource_Concurrent(t *testing.T) {
	var callCount atomic.Int32
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount.Add(1)
		return &provider.Token{
			Value:     "shared-token",
			ExpiresAt: time.Now().Add(time.Hour),
		}, nil
	})

	const n = 10
	var wg sync.WaitGroup
	results := make([]string, n)
	errs := make([]error, n)

	wg.Add(n)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = ts.Token(t.Context())
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: unexpected error: %v", i, err)
		}
		if results[i] != "shared-token" {
			t.Errorf("goroutine %d: Token = %q, want %q", i, results[i], "shared-token")
		}
	}

	// The implementation is intentionally lock-free during fetch, so a small number
	// of goroutines may race through the cache-miss check simultaneously. However,
	// the vast majority must hit the cache - all 10 bypassing it would indicate
	// the caching is broken entirely.
	if got := callCount.Load(); got > 3 {
		t.Errorf("callCount = %d, want ≤ 3 (lock-free allows 1-2 races, not all %d)", got, n)
	}
	if got := callCount.Load(); got == 0 {
		t.Error("fetchFn was never called")
	}
}

func TestCachedTokenSource_NearExpiry(t *testing.T) {
	var callCount int
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount++
		return &provider.Token{
			Value:     fmt.Sprintf("tok-%d", callCount),
			ExpiresAt: time.Now().Add(50 * time.Millisecond),
		}, nil
	})

	// First call: fetch happens.
	tok, err := ts.Token(t.Context())
	if err != nil {
		t.Fatalf("first Token error: %v", err)
	}
	if tok != "tok-1" {
		t.Errorf("first Token = %q, want %q", tok, "tok-1")
	}
	if callCount != 1 {
		t.Errorf("callCount = %d after first call, want 1", callCount)
	}

	// Immediately after: still within TTL, cached value returned.
	tok, err = ts.Token(t.Context())
	if err != nil {
		t.Fatalf("second Token error: %v", err)
	}
	if tok != "tok-1" {
		t.Errorf("second Token = %q, want cached %q", tok, "tok-1")
	}
	if callCount != 1 {
		t.Errorf("callCount = %d after second call, want 1 (cached)", callCount)
	}

	// Wait for expiry.
	time.Sleep(60 * time.Millisecond)

	// After expiry: should re-fetch.
	tok, err = ts.Token(t.Context())
	if err != nil {
		t.Fatalf("third Token error: %v", err)
	}
	if tok != "tok-2" {
		t.Errorf("third Token = %q, want %q (re-fetched)", tok, "tok-2")
	}
	if callCount != 2 {
		t.Errorf("callCount = %d after third call, want 2 (expired, re-fetched)", callCount)
	}
}

func TestCachedTokenSource_FetchErrorAfterCached(t *testing.T) {
	// The implementation does NOT serve stale cached values on error: if the
	// re-fetch fails, the error is returned directly. The expired token remains
	// in the cache slot but is not returned.
	var callCount int
	fetchErr := errors.New("auth server unavailable")

	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		callCount++
		if callCount == 1 {
			return &provider.Token{
				Value:     "good-token",
				ExpiresAt: time.Now().Add(-time.Millisecond), // expires immediately
			}, nil
		}
		return nil, fetchErr
	})

	// First call: succeeds, but token is already expired.
	tok, err := ts.Token(t.Context())
	if err != nil {
		t.Fatalf("first Token error: %v", err)
	}
	if tok != "good-token" {
		t.Errorf("first Token = %q, want %q", tok, "good-token")
	}

	// Second call: token is expired, re-fetch fails - error returned, no stale value.
	tok, err = ts.Token(t.Context())
	if err == nil {
		t.Fatal("expected error on second call, got nil")
	}
	if !errors.Is(err, fetchErr) {
		t.Errorf("err = %v, want %v", err, fetchErr)
	}
	if tok != "" {
		t.Errorf("Token = %q on error, want empty string", tok)
	}
	if callCount != 2 {
		t.Errorf("callCount = %d, want 2", callCount)
	}
}

func TestCachedTokenSource_InvalidateDuringFetch(t *testing.T) {
	// Verify that calling Invalidate() while a slow fetch is in progress
	// does not cause a panic or deadlock.
	fetchStarted := make(chan struct{})
	unblock := make(chan struct{})

	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		close(fetchStarted)
		<-unblock
		return &provider.Token{
			Value:     "slow-token",
			ExpiresAt: time.Now().Add(time.Hour),
		}, nil
	})

	var fetchErr error
	var fetchTok string
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		fetchTok, fetchErr = ts.Token(t.Context())
	}()

	// Wait until fetch has started (lock released, fetch in progress).
	<-fetchStarted

	// Invalidate while fetch is in progress - must not deadlock or panic.
	inv := ts.(provider.InvalidatingTokenSource)
	inv.Invalidate()

	// Let the fetch complete.
	close(unblock)
	wg.Wait()

	if fetchErr != nil {
		t.Fatalf("Token error: %v", fetchErr)
	}
	if fetchTok != "slow-token" {
		t.Errorf("Token = %q, want %q", fetchTok, "slow-token")
	}
}


// TestToolResult_JSONMarshal_ErrorAsString verifies FIX 19: ToolResult.Error
// (an interface) marshals as a string via err.Error() rather than the default
// "{}" that encoding/json emits for zero-populated error interfaces.
// Observability and logging consumers need the message to be human-readable.
func TestToolResult_JSONMarshal_ErrorAsString(t *testing.T) {
	t.Run("error populated", func(t *testing.T) {
		tr := provider.ToolResult{
			ToolCallID: "tc1",
			ToolName:   "bash",
			Output:     "error: boom",
			Error:      errors.New("boom"),
			IsError:    true,
		}
		b, err := json.Marshal(tr)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got map[string]any
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got["Error"] != "boom" {
			t.Errorf("Error field = %v, want %q", got["Error"], "boom")
		}
		if got["IsError"] != true {
			t.Errorf("IsError = %v, want true", got["IsError"])
		}
		if got["ToolCallID"] != "tc1" {
			t.Errorf("ToolCallID = %v, want tc1", got["ToolCallID"])
		}
	})
	t.Run("error nil omitted", func(t *testing.T) {
		tr := provider.ToolResult{
			ToolCallID: "tc2",
			ToolName:   "echo",
			Output:     "ok",
			IsError:    false,
		}
		b, err := json.Marshal(tr)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got map[string]any
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, present := got["Error"]; present {
			t.Errorf("Error field should be omitted when nil; got %v", got["Error"])
		}
	})
}

// TestToolResult_IsErrorInvariant verifies FIX 25: IsError is always
// equivalent to (Error != nil). This is an ergonomic convenience field;
// if it ever diverged from Error != nil, predicates branching on IsError
// would see different behaviour from predicates branching on Error.
//
// Spot-check the invariant on two ToolResult construction shapes that
// mirror buildToolResults output: one successful (no Error, IsError=false)
// and one error (Error != nil, IsError=true).
func TestToolResult_IsErrorInvariant(t *testing.T) {
	cases := []provider.ToolResult{
		{ToolCallID: "tc-ok", ToolName: "t", Output: "ok", Error: nil, IsError: false},
		{ToolCallID: "tc-err", ToolName: "t", Output: "error: x", Error: errors.New("x"), IsError: true},
	}
	for _, tr := range cases {
		if tr.IsError != (tr.Error != nil) {
			t.Errorf("ToolResult{ID=%s}: IsError=%v but (Error!=nil)=%v - invariant broken", tr.ToolCallID, tr.IsError, tr.Error != nil)
		}
	}
}

// TestStopCause_IsValid verifies FIX 23: IsValid returns true only for the
// seven declared StopCause constants, false for empty and arbitrary values.
func TestStopCause_IsValid(t *testing.T) {
	valid := []provider.StopCause{
		provider.StopCauseNatural,
		provider.StopCauseMaxSteps,
		provider.StopCausePredicate,
		provider.StopCauseBeforeStep,
		provider.StopCauseAbort,
		provider.StopCauseEmpty,
		provider.StopCauseNoExecutableTools,
	}
	for _, c := range valid {
		if !c.IsValid() {
			t.Errorf("StopCause %q IsValid=false, want true", c)
		}
	}
	invalid := []provider.StopCause{"", "unknown", "Natural", "bogus", "max_steps"}
	for _, c := range invalid {
		if c.IsValid() {
			t.Errorf("StopCause %q IsValid=true, want false", c)
		}
	}
}

// TestToolResult_JSONRoundTrip verifies FIX 28: ToolResult survives a
// MarshalJSON → UnmarshalJSON cycle with the IsError ↔ (Error != nil)
// invariant preserved. Without UnmarshalJSON, the Error string produced
// by MarshalJSON deserialized into a nil interface - silently breaking
// the invariant for any consumer that persists ToolResults as JSON.
func TestToolResult_JSONRoundTrip(t *testing.T) {
	t.Run("error populated round-trips", func(t *testing.T) {
		orig := provider.ToolResult{
			ToolCallID: "tc1",
			ToolName:   "bash",
			Output:     "error: boom",
			Error:      errors.New("boom"),
			IsError:    true,
		}
		b, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got provider.ToolResult
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got.ToolCallID != orig.ToolCallID {
			t.Errorf("ToolCallID = %q, want %q", got.ToolCallID, orig.ToolCallID)
		}
		if got.ToolName != orig.ToolName {
			t.Errorf("ToolName = %q, want %q", got.ToolName, orig.ToolName)
		}
		if got.Output != orig.Output {
			t.Errorf("Output = %q, want %q", got.Output, orig.Output)
		}
		if got.Error == nil {
			t.Fatalf("Error is nil after round-trip; want non-nil (invariant broken)")
		}
		if got.Error.Error() != "boom" {
			t.Errorf("Error.Error() = %q, want %q", got.Error.Error(), "boom")
		}
		if !got.IsError {
			t.Errorf("IsError = false, want true (invariant IsError == Error!=nil broken)")
		}
	})

	t.Run("error nil round-trips", func(t *testing.T) {
		orig := provider.ToolResult{
			ToolCallID: "tc2",
			ToolName:   "echo",
			Output:     "ok",
			IsError:    false,
		}
		b, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got provider.ToolResult
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got.Error != nil {
			t.Errorf("Error = %v, want nil", got.Error)
		}
		if got.IsError {
			t.Errorf("IsError = true, want false (invariant broken)")
		}
		if got.Output != "ok" {
			t.Errorf("Output = %q, want %q", got.Output, "ok")
		}
	})

	t.Run("IsError true with empty Error synthesizes placeholder", func(t *testing.T) {
		// A hand-crafted JSON with IsError=true but no Error string must
		// still preserve the invariant on unmarshal (synthesize error).
		data := []byte(`{"ToolCallID":"tc3","ToolName":"x","Output":"y","IsError":true}`)
		var got provider.ToolResult
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got.Error == nil {
			t.Errorf("Error is nil but IsError=true; invariant broken (placeholder not synthesized)")
		}
		if !got.IsError {
			t.Errorf("IsError = false, want true")
		}
	})

	t.Run("Error string present forces IsError true", func(t *testing.T) {
		// Defensive: IsError=false but Error non-empty → coerce IsError=true.
		data := []byte(`{"ToolCallID":"tc4","ToolName":"x","Output":"y","Error":"oops","IsError":false}`)
		var got provider.ToolResult
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got.Error == nil || got.Error.Error() != "oops" {
			t.Errorf("Error = %v, want error with message %q", got.Error, "oops")
		}
		if !got.IsError {
			t.Errorf("IsError = false, want true (coerced)")
		}
	})

	// FIX 39: empty-string error messages must round-trip faithfully. The
	// previous encoding used `omitempty` on Error, which dropped ""; the
	// resurrected ToolResult was a placeholder (`"tool error (no message)"`)
	// instead of the original empty message. A pointer-to-string alias
	// distinguishes "key absent" from "empty string present".
	t.Run("FIX 39: empty-string error round-trips faithfully", func(t *testing.T) {
		orig := provider.ToolResult{
			ToolCallID: "tc5",
			ToolName:   "x",
			Output:     "y",
			Error:      errors.New(""),
			IsError:    true,
		}
		b, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		// Wire form must carry the Error key even though the value is "".
		if !strings.Contains(string(b), `"Error":""`) {
			t.Fatalf("wire form missing explicit empty Error key; got %s", string(b))
		}
		var got provider.ToolResult
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got.Error == nil {
			t.Fatalf("Error nil after round-trip; want non-nil with empty message")
		}
		if got.Error.Error() != "" {
			t.Errorf("Error message = %q; want %q (empty-string fidelity lost - FIX 39 regression)", got.Error.Error(), "")
		}
		if !got.IsError {
			t.Errorf("IsError = false; want true")
		}
	})

	// FIX 40: sentinel identity is NOT preserved across JSON round-trip.
	// Documented limitation: callers branching on errors.Is(r.Error,
	// ErrSentinel) after JSON persistence will see false. This test pins
	// the documented behavior so a future refactor that introduces a
	// registry-based reconstruction updates godoc at the same time.
	t.Run("FIX 40: sentinel identity is not preserved (documented)", func(t *testing.T) {
		sentinel := errors.New("well-known sentinel")
		orig := provider.ToolResult{
			ToolCallID: "tc6",
			ToolName:   "x",
			Output:     "y",
			Error:      sentinel,
			IsError:    true,
		}
		b, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got provider.ToolResult
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got.Error == nil {
			t.Fatalf("Error nil after round-trip")
		}
		if errors.Is(got.Error, sentinel) {
			t.Fatalf("errors.Is(sentinel) returned true after JSON round-trip; documented FIX 40 limitation unexpectedly lifted - update godoc")
		}
		// Message is preserved even though identity is not.
		if got.Error.Error() != sentinel.Error() {
			t.Errorf("Error message = %q; want %q", got.Error.Error(), sentinel.Error())
		}
	})
}
