package sse

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestScanner_BasicEvents(t *testing.T) {
	input := "data: hello\ndata: world\ndata: [DONE]\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "hello" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "hello")
	}

	data, ok = s.Next()
	if !ok || data != "world" {
		t.Errorf("second: got %q, %v; want %q, true", data, ok, "world")
	}

	data, ok = s.Next()
	if ok {
		t.Errorf("after DONE: got %q, %v; want false", data, ok)
	}
	if !s.IsDone() {
		t.Error("IsDone should be true after [DONE]")
	}
}

func TestScanner_SkipsNonDataLines(t *testing.T) {
	input := "event: message\nid: 1\ndata: payload\n\nretry: 5000\ndata: [DONE]\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "payload" {
		t.Errorf("got %q, %v; want %q, true", data, ok, "payload")
	}

	data, ok = s.Next()
	if ok {
		t.Errorf("after DONE: got %q, %v; want false", data, ok)
	}
}

func TestScanner_EmptyStream(t *testing.T) {
	s := NewScanner(strings.NewReader(""))

	data, ok := s.Next()
	if ok {
		t.Errorf("empty stream: got %q, %v; want false", data, ok)
	}
	if s.IsDone() {
		t.Error("IsDone should be false for empty stream (no [DONE] seen)")
	}
}

func TestScanner_NoDataPrefix(t *testing.T) {
	input := "event: ping\n\nevent: pong\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if ok {
		t.Errorf("no data lines: got %q, %v; want false", data, ok)
	}
}

func TestScanner_JSONPayloads(t *testing.T) {
	input := `data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}
data: {"id":"2","choices":[{"delta":{"content":" there"}}]}
data: [DONE]
`
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok {
		t.Fatal("expected first event")
	}
	if !strings.Contains(data, `"content":"hi"`) {
		t.Errorf("first event missing content: %s", data)
	}

	data, ok = s.Next()
	if !ok {
		t.Fatal("expected second event")
	}
	if !strings.Contains(data, `"content":" there"`) {
		t.Errorf("second event missing content: %s", data)
	}

	_, ok = s.Next()
	if ok {
		t.Error("expected false after DONE")
	}
}

func TestScanner_DoneIdempotent(t *testing.T) {
	input := "data: first\ndata: [DONE]\ndata: after-done\n"
	s := NewScanner(strings.NewReader(input))

	s.Next() // "first"
	s.Next() // DONE

	// Calling Next after DONE should keep returning false.
	for i := 0; i < 3; i++ {
		_, ok := s.Next()
		if ok {
			t.Errorf("call %d after DONE returned ok=true", i)
		}
	}
}

func TestScanner_Err(t *testing.T) {
	input := "data: ok\n"
	s := NewScanner(strings.NewReader(input))
	s.Next()
	s.Next() // EOF

	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil", err)
	}
}

func TestScanner_EmptyDataPayload(t *testing.T) {
	// "data:" with no space or value should yield an empty string token, not be skipped.
	input := "data:\n\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok {
		t.Fatal("expected ok=true for empty data payload, got false")
	}
	if data != "" {
		t.Errorf("expected empty string, got %q", data)
	}
}

// errReader is an io.Reader that returns a fixed error on every Read call.
type errReader struct{ err error }

func (e *errReader) Read(_ []byte) (int, error) { return 0, e.err }

func TestScanner_ReadError(t *testing.T) {
	injected := errors.New("stream broken")
	// First reader yields one valid event; second reader immediately returns the error.
	r := io.MultiReader(
		strings.NewReader("data: first\n\n"),
		&errReader{err: injected},
	)
	s := NewScanner(r)

	data, ok := s.Next()
	if !ok || data != "first" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "first")
	}

	_, ok = s.Next()
	if ok {
		t.Error("expected ok=false after read error")
	}

	if err := s.Err(); !errors.Is(err, injected) {
		t.Errorf("Err() = %v; want injected error %v", err, injected)
	}
}

func TestScanner_VeryLongLine(t *testing.T) {
	// Regression test for "bufio.Scanner: token too long" (issue #70).
	// The scanner must accept SSE data lines larger than bufio.Scanner's
	// MaxScanTokenSize (64KiB) and the previous 1MiB cap.
	payload := strings.Repeat("x", 4*1024*1024) // 4 MiB
	input := "data: " + payload + "\ndata: [DONE]\n"

	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok {
		t.Fatalf("expected ok=true for long line; Err=%v", s.Err())
	}
	if len(data) != len(payload) {
		t.Errorf("got len=%d, want len=%d", len(data), len(payload))
	}
	if data != payload {
		t.Errorf("payload mismatch")
	}

	if _, ok := s.Next(); ok {
		t.Error("expected false after DONE")
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil", err)
	}
}

func TestScanner_LineExceedsMaxSize(t *testing.T) {
	// A single line larger than MaxLineSize must be rejected with an error,
	// not silently consumed (DoS protection).
	oversized := strings.Repeat("x", MaxLineSize+1)
	input := "data: " + oversized + "\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if ok {
		t.Errorf("expected ok=false for oversized line, got data len=%d", len(data))
	}
	if err := s.Err(); err == nil {
		t.Fatal("expected Err() to report oversized line, got nil")
	} else if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected size-limit error, got: %v", err)
	}
}

func TestScanner_LineWithoutTrailingNewline(t *testing.T) {
	// A final "data:" line lacking a trailing newline must still be emitted.
	input := "data: first\ndata: last"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "first" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "first")
	}

	data, ok = s.Next()
	if !ok || data != "last" {
		t.Errorf("last: got %q, %v; want %q, true", data, ok, "last")
	}

	if _, ok := s.Next(); ok {
		t.Error("expected false at EOF")
	}
}

func TestScanner_CRLFLineEndings(t *testing.T) {
	// SSE spec allows CRLF line endings; the scanner must strip \r as well as \n.
	input := "data: hello\r\ndata: world\r\ndata: [DONE]\r\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "hello" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "hello")
	}

	data, ok = s.Next()
	if !ok || data != "world" {
		t.Errorf("second: got %q, %v; want %q, true", data, ok, "world")
	}

	if _, ok := s.Next(); ok {
		t.Error("expected false after DONE")
	}
}

func TestScanner_MultiLineData(t *testing.T) {
	// Two consecutive data: lines in one event block.
	// The implementation does NOT concatenate; each data: line is returned independently.
	input := "data: line1\ndata: line2\n\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "line1" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "line1")
	}

	data, ok = s.Next()
	if !ok || data != "line2" {
		t.Errorf("second: got %q, %v; want %q, true", data, ok, "line2")
	}

	_, ok = s.Next()
	if ok {
		t.Error("expected false after all lines consumed")
	}
}

func TestScanner_NextLine(t *testing.T) {
	// NextLine must return every line (event:, data:, blank, comment) with
	// trailing CR/LF stripped, so callers parsing event-typed SSE can do
	// their own dispatch.
	input := "event: ping\r\ndata: {\"x\":1}\r\n\r\n: comment\nevent: done\ndata: [DONE]\n"
	s := NewScanner(strings.NewReader(input))

	want := []string{
		"event: ping",
		"data: {\"x\":1}",
		"",
		": comment",
		"event: done",
		"data: [DONE]",
	}
	for i, w := range want {
		got, ok := s.NextLine()
		if !ok {
			t.Fatalf("line %d: ok=false; Err=%v", i, s.Err())
		}
		if got != w {
			t.Errorf("line %d: got %q, want %q", i, got, w)
		}
	}
	if _, ok := s.NextLine(); ok {
		t.Error("expected ok=false at EOF")
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil", err)
	}
}

func TestScanner_NextLine_ReadError(t *testing.T) {
	injected := errors.New("stream broken")
	r := io.MultiReader(
		strings.NewReader("event: ping\n"),
		&errReader{err: injected},
	)
	s := NewScanner(r)

	line, ok := s.NextLine()
	if !ok || line != "event: ping" {
		t.Fatalf("first: got %q, %v; want %q, true", line, ok, "event: ping")
	}

	_, ok = s.NextLine()
	if ok {
		t.Error("expected ok=false after read error")
	}
	if err := s.Err(); !errors.Is(err, injected) {
		t.Errorf("Err() = %v; want injected %v", err, injected)
	}

	// Subsequent calls must short-circuit on the cached error.
	if _, ok := s.NextLine(); ok {
		t.Error("expected ok=false on repeat call after error")
	}
}

func TestScanner_NextLine_LargeLine(t *testing.T) {
	// Regression test for issue #70: NextLine must accept lines past the
	// historical 1 MiB bufio.Scanner cap, so event-typed SSE consumers
	// (OpenAI Responses API) don't choke on long deltas.
	payload := strings.Repeat("y", 4*1024*1024)
	input := "event: response.output_text.delta\ndata: " + payload + "\n"

	s := NewScanner(strings.NewReader(input))

	got, ok := s.NextLine()
	if !ok || got != "event: response.output_text.delta" {
		t.Fatalf("first line: got %q, %v", got, ok)
	}

	got, ok = s.NextLine()
	if !ok {
		t.Fatalf("expected long data line; Err=%v", s.Err())
	}
	if len(got) != len("data: ")+len(payload) {
		t.Errorf("got len=%d, want %d", len(got), len("data: ")+len(payload))
	}
}
