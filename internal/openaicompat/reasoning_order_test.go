package openaicompat

import (
	"strings"
	"testing"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// A single SSE delta can carry both reasoning_content and content at the
// reasoning→answer transition (seen with GLM via NVIDIA NIM). Within one
// delta, reasoning always precedes the answer, so the reasoning chunk must
// be emitted before the text chunk.
func TestParseStream_ReasoningBeforeTextInCombinedDelta(t *testing.T) {
	input := `data: {"choices":[{"delta":{"reasoning_content":"existing aesthetic"},"index":0}]}
data: {"choices":[{"delta":{"reasoning_content":".","content":"V"},"index":0}]}
data: {"choices":[{"delta":{"content":"oy a explorar"},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	go ParseStream(t.Context(), scanner, out)

	var order []string
	for chunk := range out {
		switch chunk.Type {
		case provider.ChunkText:
			order = append(order, "text:"+chunk.Text)
		case provider.ChunkReasoning:
			order = append(order, "reasoning:"+chunk.Text)
		}
	}

	want := []string{"reasoning:existing aesthetic", "reasoning:.", "text:V", "text:oy a explorar"}
	if len(order) != len(want) {
		t.Fatalf("got %d text/reasoning chunks %v, want %d", len(order), order, len(want))
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("chunk[%d] = %q, want %q (full order: %v)", i, order[i], want[i], order)
		}
	}
}
