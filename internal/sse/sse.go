// Package sse provides a scanner for Server-Sent Events (SSE) streams.
//
// It handles the "data: " prefix, blank line skipping, and [DONE] sentinel.
// JSON deserialization is left to the caller.
package sse

import (
	"bufio"
	"io"
	"strings"
)

// Scanner reads SSE data payloads from an io.Reader.
type Scanner struct {
	scanner *bufio.Scanner
	done    bool
}

// NewScanner creates an SSE scanner with a 1MB buffer.
func NewScanner(r io.Reader) *Scanner {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return &Scanner{scanner: s}
}

// Next returns the next SSE data payload (with "data: " prefix stripped).
// Returns ("", false) at EOF or after [DONE].
// Skips non-"data: " lines and blank lines.
func (s *Scanner) Next() (data string, ok bool) {
	if s.done {
		return "", false
	}

	for s.scanner.Scan() {
		line := s.scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data = strings.TrimPrefix(line, "data:")
		data = strings.TrimPrefix(data, " ")

		if data == "[DONE]" {
			s.done = true
			return "", false
		}

		return data, true
	}

	return "", false
}

// Err returns the first non-EOF error encountered by the scanner.
func (s *Scanner) Err() error {
	return s.scanner.Err()
}

// IsDone reports whether the scanner has encountered the [DONE] sentinel.
func (s *Scanner) IsDone() bool {
	return s.done
}
