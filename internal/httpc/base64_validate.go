package httpc

import (
	"encoding/base64"
	"strings"
)

const base64ValidationChunk = 4096

// validStdBase64 reports whether base64.StdEncoding.DecodeString accepts data
// without allocating the decoded bytes. Request builders validate every inline
// image on every request, so long conversations would otherwise decode each
// image again per turn only to discard it.
//
// DecodeString ignores CR and LF anywhere and rejects anything but line breaks
// after a padded quantum. Dropping line breaks and decoding fixed chunks of a
// multiple of four symbols therefore preserves its result: a quantum never
// spans chunks, and a padded chunk must be the last one with symbols.
func validStdBase64(data string) bool {
	var chunk [base64ValidationChunk]byte
	n, padded := 0, false
	for {
		run, rest, lineBreak := cutLineBreak(data)
		for run != "" {
			if padded {
				return false
			}
			copied := copy(chunk[n:], run)
			n += copied
			run = run[copied:]
			if n == len(chunk) {
				if !decodeStdBase64Chunk(chunk[:], &padded) {
					return false
				}
				n = 0
			}
		}
		if !lineBreak {
			return n == 0 || decodeStdBase64Chunk(chunk[:n], &padded)
		}
		data = rest
	}
}

// cutLineBreak splits s around its first CR or LF without scanning past it.
func cutLineBreak(s string) (before, after string, found bool) {
	end := strings.IndexAny(s, "\r\n")
	if end < 0 {
		return s, "", false
	}
	return s[:end], s[end+1:], true
}

func decodeStdBase64Chunk(src []byte, padded *bool) bool {
	var dst [base64ValidationChunk / 4 * 3]byte
	written, err := base64.StdEncoding.Decode(dst[:], src)
	*padded = written < len(src)/4*3
	return err == nil
}
