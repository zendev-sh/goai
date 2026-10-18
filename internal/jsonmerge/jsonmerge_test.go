package jsonmerge

import (
	"encoding/json"
	"testing"
)

func TestDeep(t *testing.T) {
	base := map[string]any{
		"audio": map[string]any{"output": map[string]any{"format": "pcm", "voice": "marin"}},
		"keep":  1,
	}
	overlay := map[string]any{
		"audio": map[string]any{"output": map[string]any{"speed": 1.1}},
		"keep":  map[string]any{"replaced": true},
		"new":   "x",
	}
	got, _ := json.Marshal(Deep(base, overlay))
	want := `{"audio":{"output":{"format":"pcm","speed":1.1,"voice":"marin"}},"keep":{"replaced":true},"new":"x"}`
	if string(got) != want {
		t.Fatalf("Deep = %s\nwant   %s", got, want)
	}
	before, _ := json.Marshal(base)
	if string(before) != `{"audio":{"output":{"format":"pcm","voice":"marin"}},"keep":1}` {
		t.Fatalf("base was mutated: %s", before)
	}
	if out := Deep(nil, nil); len(out) != 0 {
		t.Fatalf("Deep(nil, nil) = %v", out)
	}
}
