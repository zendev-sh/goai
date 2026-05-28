// Package sse provides a scanner for Server-Sent Events (SSE) streams.
//
// It handles the "data: " prefix, blank line skipping, and [DONE] sentinel.
// JSON deserialization is left to the caller.
package sse

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

// MaxLineSize is the upper bound on a single SSE line, in bytes.
//
// Large enough to comfortably hold long tool-call argument deltas and
// reasoning blocks observed from production providers, while preventing a
// malicious or buggy upstream from forcing unbounded client-side allocation
// (a DoS vector). Lines exceeding this size cause Next to stop and Err to
// report the violation.
const MaxLineSize = 16 << 20 // 16 MiB

// Scanner reads SSE data payloads from an io.Reader.
type Scanner struct {
	reader *bufio.Reader
	err    error
	done   bool
}

// NewScanner creates an SSE scanner backed by a bufio.Reader.
//
// Lines up to [MaxLineSize] bytes are accepted; longer lines cause the
// scanner to stop with an error reported via Err. This lifts the 1 MiB
// limit imposed by bufio.Scanner while still bounding memory use.
func NewScanner(r io.Reader) *Scanner {
	return &Scanner{reader: bufio.NewReader(r)}
}

// NextLine returns the next line from the SSE stream with trailing CR/LF
// stripped. Unlike [Scanner.Next], it returns every line including blank
// lines and non-"data:" lines, so callers parsing event-typed SSE
// (interleaved "event:" + "data:" pairs, e.g. the OpenAI Responses API)
// can implement their own line dispatch while still benefiting from
// [MaxLineSize]-bounded reads. Returns ("", false) at EOF or after a read
// error (which is reported via [Scanner.Err]).
//
// Mix Next and NextLine on the same scanner at your own risk; pick one
// mode per stream.
func (s *Scanner) NextLine() (line string, ok bool) {
	if s.err != nil {
		return "", false
	}
	raw, err := s.readLine()
	if err != nil {
		if !errors.Is(err, io.EOF) {
			s.err = err
		}
		if len(raw) == 0 {
			return "", false
		}
	}
	return strings.TrimRight(raw, "\r\n"), true
}

// Next returns the next SSE data payload (with "data: " prefix stripped).
// Returns ("", false) at EOF or after [DONE].
// Skips non-"data: " lines and blank lines.
func (s *Scanner) Next() (data string, ok bool) {
	if s.done || s.err != nil {
		return "", false
	}

	for {
		line, err := s.readLine()
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

// readLine reads one '\n'-terminated line, accumulating across the
// underlying bufio.Reader's internal buffer so we are not limited by its
// size. It enforces [MaxLineSize] to prevent unbounded memory growth from
// a hostile or malformed stream.
//
// Returns the line (which may include the trailing '\n') together with
// any terminal error. A final partial line at EOF is returned with
// io.EOF.
func (s *Scanner) readLine() (string, error) {
	var buf []byte
	for {
		slice, err := s.reader.ReadSlice('\n')
		if len(buf)+len(slice) > MaxLineSize {
			return "", fmt.Errorf("sse: line exceeds %d bytes", MaxLineSize)
		}
		// ReadSlice returns a reference into the reader's internal buffer
		// that may be overwritten on the next read; append copies it out.
		buf = append(buf, slice...)
		if err == nil {
			return string(buf), nil
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return string(buf), err
	}
}

// Err returns the first non-EOF error encountered while reading,
// including line-size violations.
func (s *Scanner) Err() error {
	return s.err
}

// IsDone reports whether the scanner has encountered the [DONE] sentinel.
func (s *Scanner) IsDone() bool {
	return s.done
}
