package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// orderedTurn is an assistant turn in the shape Claude Opus 5.5 / Fable 5.1
// produce with tools: reasoning, an empty (omitted) reasoning block, narration,
// a progress-update thinking block before each tool_use, and redacted thinking.
// Replaying it on the next step must reproduce every block, in place.
var orderedTurn = []map[string]any{
	{"type": "redacted_thinking", "data": "enc-0"},
	{"type": "thinking", "thinking": "Reasoning about the lookup.", "signature": "sig-a"},
	{"type": "thinking", "thinking": "", "signature": "sig-b"},
	{"type": "text", "text": "Looking that up."},
	{"type": "thinking", "thinking": "Checking the first source.", "signature": "sig-c"},
	{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": map[string]any{"q": "one"}},
	{"type": "thinking", "thinking": "", "signature": "sig-d"},
	{"type": "tool_use", "id": "toolu_2", "name": "lookup", "input": map[string]any{"q": "two"}},
}

// leadingTurn is an assistant turn whose thinking blocks all precede the
// tool_use, as every model before Claude 5.x produces: consecutive blocks,
// each with its own signature, one of them empty (omitted display).
var leadingTurn = []map[string]any{
	{"type": "redacted_thinking", "data": "enc-0"},
	{"type": "thinking", "thinking": "Reasoning about the lookup.", "signature": "sig-a"},
	{"type": "thinking", "thinking": "", "signature": "sig-b"},
	{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": map[string]any{"q": "one"}},
}

// writeSSE streams blocks as Anthropic SSE events, splitting text-bearing
// deltas in two so accumulation is exercised.
func writeSSE(w http.ResponseWriter, blocks []map[string]any, stopReason string) {
	w.Header().Set("Content-Type", "text/event-stream")
	event := func(v map[string]any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", v["type"], b)
	}
	halves := func(s string) []string { return []string{s[:len(s)/2], s[len(s)/2:]} }
	delta := func(i int, d map[string]any) {
		event(map[string]any{"type": "content_block_delta", "index": i, "delta": d})
	}
	event(map[string]any{"type": "message_start", "message": map[string]any{"id": "msg", "model": "claude-opus-5-5", "content": []any{}, "usage": map[string]any{"input_tokens": 1}}})
	for i, b := range blocks {
		switch b["type"] {
		case "thinking":
			event(map[string]any{"type": "content_block_start", "index": i, "content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""}})
			// Omitted display still sends one empty thinking_delta.
			for _, h := range halves(b["thinking"].(string)) {
				delta(i, map[string]any{"type": "thinking_delta", "thinking": h})
			}
			delta(i, map[string]any{"type": "signature_delta", "signature": b["signature"]})
		case "text":
			event(map[string]any{"type": "content_block_start", "index": i, "content_block": map[string]any{"type": "text", "text": ""}})
			for _, h := range halves(b["text"].(string)) {
				delta(i, map[string]any{"type": "text_delta", "text": h})
			}
		case "tool_use":
			event(map[string]any{"type": "content_block_start", "index": i, "content_block": map[string]any{"type": "tool_use", "id": b["id"], "name": b["name"], "input": map[string]any{}}})
			in, _ := json.Marshal(b["input"])
			for _, h := range halves(string(in)) {
				delta(i, map[string]any{"type": "input_json_delta", "partial_json": h})
			}
		default:
			event(map[string]any{"type": "content_block_start", "index": i, "content_block": b})
		}
		event(map[string]any{"type": "content_block_stop", "index": i})
	}
	event(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": stopReason}, "usage": map[string]any{"output_tokens": 1}})
	event(map[string]any{"type": "message_stop"})
}

// replayCase selects the transport a replay test runs over.
type replayCase struct {
	name          string
	stream        bool // StreamText instead of GenerateText
	autoStreaming bool // GenerateText over the SSE transport
}

// runReplay runs a tool loop whose first step returns turn, and returns the
// assistant content the second request replays along with the loop's steps.
func runReplay(t *testing.T, tc replayCase, turn []map[string]any) (replayed []any, steps []goai.StepResult) {
	t.Helper()
	var mu sync.Mutex
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		mu.Lock()
		calls++
		call := calls
		if call == 2 {
			msgs, _ := body["messages"].([]any)
			if len(msgs) > 1 {
				replayed, _ = msgs[1].(map[string]any)["content"].([]any)
			}
		}
		mu.Unlock()
		blocks, stop := turn, "tool_use"
		if call > 1 {
			blocks, stop = []map[string]any{{"type": "text", "text": "done"}}, "end_turn"
		}
		if body["stream"] == true {
			writeSSE(w, blocks, stop)
		} else {
			t.Error("non-streaming request")
		}
	}))
	defer srv.Close()

	model := Chat("claude-opus-5-5", WithAPIKey("test"), WithBaseURL(srv.URL), WithAutoStreaming(tc.autoStreaming))
	type lookupIn struct {
		Q string `json:"q"`
	}
	tool := goai.NewTool("lookup", "Look something up.", func(_ context.Context, in lookupIn) (string, error) {
		return "result " + in.Q, nil
	})
	opts := []goai.Option{goai.WithPrompt("go"), goai.WithTools(tool), goai.WithMaxSteps(3)}
	if tc.stream {
		ts, err := goai.StreamText(t.Context(), model, opts...)
		if err != nil {
			t.Fatal(err)
		}
		for range ts.Stream() {
		}
		if err := ts.Err(); err != nil {
			t.Fatal(err)
		}
		steps = ts.Result().Steps
	} else {
		res, err := goai.GenerateText(t.Context(), model, opts...)
		if err != nil {
			t.Fatal(err)
		}
		steps = res.Steps
	}
	mu.Lock()
	defer mu.Unlock()
	return replayed, steps
}

// assertReplayed checks that replayed is exactly turn, byte for byte.
func assertReplayed(t *testing.T, replayed []any, turn []map[string]any) {
	t.Helper()
	want := make([]any, len(turn))
	for i, b := range turn {
		want[i] = b
	}
	wantJSON, _ := json.Marshal(want)
	gotJSON, _ := json.Marshal(replayed)
	if string(gotJSON) != string(wantJSON) {
		t.Errorf("replayed assistant turn:\n got %s\nwant %s", gotJSON, wantJSON)
	}
}

// TestReplayLeadingThinking checks that the tool loop sends back every
// thinking block of a turn -- each with its own signature, empty ones and
// redacted ones included -- on every transport.
func TestReplayLeadingThinking(t *testing.T) {
	for _, tc := range []replayCase{
		{name: "stream", stream: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			replayed, _ := runReplay(t, tc, leadingTurn)
			assertReplayed(t, replayed, leadingTurn)
		})
	}
}

// TestStreamReasoningBlockIDs checks that every reasoning chunk names its
// content block, so consecutive thinking blocks -- even text-less ones -- stay
// distinct for consumers of the raw chunk stream.
func TestStreamReasoningBlockIDs(t *testing.T) {
	rec := httptest.NewRecorder()
	writeSSE(rec, orderedTurn, "tool_use")
	out := make(chan provider.StreamChunk, 64)
	go parseSSE(t.Context(), strings.NewReader(rec.Body.String()), out, false)
	var got []string
	for c := range out {
		if c.Type == provider.ChunkReasoning {
			got = append(got, fmt.Sprintf("%v:%q", c.Metadata["blockId"], c.Text))
		}
	}
	want := []string{
		`0:""`,
		`1:"Reasoning abo"`, `1:"ut the lookup."`, `1:""`,
		`2:""`,
		`4:"Checking the "`, `4:"first source."`, `4:""`,
		`6:""`,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("reasoning chunks = %v, want %v", got, want)
	}
}
