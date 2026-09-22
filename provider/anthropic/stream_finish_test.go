package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// TestStreamFinishReason checks that a streamed stop_reason survives to the
// result: ChunkFinish used to carry an empty reason that overwrote the one
// from message_delta, so a refusal looked like a normal (empty) completion.
func TestStreamFinishReason(t *testing.T) {
	for stop, want := range map[string]provider.FinishReason{
		"refusal":    provider.FinishContentFilter,
		"max_tokens": provider.FinishLength,
		"end_turn":   provider.FinishStop,
	} {
		for _, tools := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/tools=%v", stop, tools), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg\",\"model\":\"claude-opus-5-5\",\"content\":[],\"usage\":{\"input_tokens\":1}}}\n\n")
					fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
					fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n")
					fmt.Fprint(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
					fmt.Fprintf(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":%q},\"usage\":{\"output_tokens\":1}}\n\n", stop)
					fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
				}))
				defer srv.Close()
				opts := []goai.Option{goai.WithPrompt("go")}
				if tools {
					tool := goai.NewTool("lookup", "Look something up.", func(context.Context, struct{}) (string, error) { return "", nil })
					opts = append(opts, goai.WithTools(tool), goai.WithMaxSteps(3))
				}
				ts, err := goai.StreamText(t.Context(), Chat("claude-opus-5-5", WithAPIKey("test"), WithBaseURL(srv.URL)), opts...)
				if err != nil {
					t.Fatal(err)
				}
				for range ts.Stream() {
				}
				res := ts.Result()
				if res.FinishReason != want {
					t.Errorf("FinishReason = %q, want %q", res.FinishReason, want)
				}
				if len(res.Steps) > 0 && res.Steps[len(res.Steps)-1].FinishReason != want {
					t.Errorf("last step FinishReason = %q, want %q", res.Steps[len(res.Steps)-1].FinishReason, want)
				}
			})
		}
	}
}
