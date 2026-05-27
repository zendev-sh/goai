// Package sse provides a scanner for Server-Sent Events (SSE) streams.
//
// It handles the "data: " prefix, blank line skipping, and [DONE] sentinel.
// JSON deserialization is left to the caller.
package sse

import (
	"bufio"
	"errors"
	"io"
	"strings"
)

// Scanner reads SSE data payloads from an io.Reader.
type Scanner struct {
	reader *bufio.Reader
	err    error
	done   bool
}

// NewScanner creates an SSE scanner backed by a bufio.Reader.
//
// Unlike bufio.Scanner, the underlying reader grows as needed and has no
// fixed maximum line length, so it accepts arbitrarily large SSE payloads
// (e.g. long tool-call argument deltas or reasoning blocks).
func NewScanner(r io.Reader) *Scanner {
	return &Scanner{reader: bufio.NewReader(r)}
}

// Next returns the next SSE data payload (with "data: " prefix stripped).
// Returns ("", false) at EOF or after [DONE].
// Skips non-"data: " lines and blank lines.
func (s *Scanner) Next() (data string, ok bool) {
	if s.done || s.err != nil {
		return "", false
	}

	for {
		line, err := s.reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			if strings.HasPrefix(line, "data:") {
				payload := strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " ")
				if payload == "[DONE]" {
					s.done = true
					return "", false
				}
				return payload, true
			}
			// Non-"data:" line: skip and continue reading.
		}

		if err != nil {
			if !errors.Is(err, io.EOF) {
				s.err = err
			}
			return "", false
		}
	}
}

// Err returns the first non-EOF error encountered while reading.
func (s *Scanner) Err() error {
	return s.err
}

// IsDone reports whether the scanner has encountered the [DONE] sentinel.
func (s *Scanner) IsDone() bool {
	return s.done
}
