package bedrock

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
)

// eventStreamFrame represents a decoded AWS EventStream frame.
type eventStreamFrame struct {
	MessageType string // ":message-type" header (e.g., "event", "exception")
	EventType   string // ":event-type" header (e.g., "contentBlockDelta")
	ContentType string // ":content-type" header
	Payload     []byte
}

// eventStreamDecoder reads AWS EventStream binary frames from an io.Reader.
//
// AWS EventStream frame layout:
//
//	[4B totalLength BE][4B headersLength BE][4B preludeCRC32][headers...][payload...][4B messageCRC32]
type eventStreamDecoder struct {
	r io.Reader
}

// newEventStreamDecoder creates a decoder wrapping the given reader.
func newEventStreamDecoder(r io.Reader) *eventStreamDecoder {
	return &eventStreamDecoder{r: r}
}

// Next reads and returns the next frame. Returns io.EOF when the stream ends.
func (d *eventStreamDecoder) Next() (*eventStreamFrame, error) {
	// Read 12-byte prelude: totalLength (4) + headersLength (4) + preludeCRC (4).
	var prelude [12]byte
	if _, err := io.ReadFull(d.r, prelude[:]); err != nil {
		return nil, err
	}

	totalLen := binary.BigEndian.Uint32(prelude[0:4])
	headersLen := binary.BigEndian.Uint32(prelude[4:8])
	preludeCRC := binary.BigEndian.Uint32(prelude[8:12])

	// Validate prelude CRC.
	if got := crc32.ChecksumIEEE(prelude[0:8]); got != preludeCRC {
		return nil, fmt.Errorf("eventstream: prelude CRC mismatch: got %08x, want %08x", got, preludeCRC)
	}

	// Remaining bytes after prelude = totalLen - 12 (prelude) - 4 (message CRC).
	const maxFrameSize = 10 * 1024 * 1024 // 10MB
	remaining := int(totalLen) - 12 - 4
	if remaining < 0 || remaining > maxFrameSize {
		return nil, fmt.Errorf("eventstream: invalid total length %d", totalLen)
	}

	if int(headersLen) > remaining {
		return nil, fmt.Errorf("eventstream: headers length %d exceeds frame payload %d", headersLen, remaining)
	}

	buf := make([]byte, remaining+4) // +4 for message CRC
	if _, err := io.ReadFull(d.r, buf); err != nil {
		return nil, fmt.Errorf("eventstream: reading frame body: %w", err)
	}

	// Validate message CRC over entire frame (prelude + headers + payload).
	messageCRC := binary.BigEndian.Uint32(buf[remaining:])
	crcCalc := crc32.NewIEEE()
	crcCalc.Write(prelude[:])
	crcCalc.Write(buf[:remaining])
	if crcCalc.Sum32() != messageCRC {
		return nil, fmt.Errorf("eventstream: message CRC mismatch")
	}

	// Parse headers.
	headers := buf[:headersLen]
	payloadBytes := buf[headersLen:remaining]

	frame := &eventStreamFrame{
		Payload: payloadBytes,
	}

	// Parse headers: [1B nameLen][name...][1B typeTag][value per type]
	for off := 0; off < len(headers); {
		nameLen := int(headers[off])
		off++
		if off+nameLen > len(headers) {
			return nil, fmt.Errorf("eventstream: header name overflow")
		}
		name := string(headers[off : off+nameLen])
		off += nameLen

		if off >= len(headers) {
			return nil, fmt.Errorf("eventstream: missing header type tag")
		}
		typeTag := headers[off]
		off++

		switch typeTag {
		case 7: // String
			if off+2 > len(headers) {
				return nil, fmt.Errorf("eventstream: string header value length overflow")
			}
			valLen := int(binary.BigEndian.Uint16(headers[off : off+2]))
			off += 2
			if off+valLen > len(headers) {
				return nil, fmt.Errorf("eventstream: string header value overflow")
			}
			val := string(headers[off : off+valLen])
			off += valLen

			switch name {
			case ":message-type":
				frame.MessageType = val
			case ":event-type":
				frame.EventType = val
			case ":content-type":
				frame.ContentType = val
			}
		case 0, 1: // bool true/false: no value bytes
			// skip
		case 2: // byte
			off += 1
		case 3: // short
			off += 2
		case 4: // int
			off += 4
		case 5, 8: // long, timestamp
			off += 8
		case 9: // uuid
			off += 16
		case 6: // bytes: 2B length prefix + data
			if off+2 > len(headers) {
				return nil, fmt.Errorf("eventstream: bytes header length overflow")
			}
			bLen := int(binary.BigEndian.Uint16(headers[off : off+2]))
			if off+2+bLen > len(headers) {
				return nil, fmt.Errorf("eventstream: bytes header value overflows header block")
			}
			off += 2 + bLen
		default:
			return nil, fmt.Errorf("eventstream: unknown header type tag %d", typeTag)
		}
	}

	return frame, nil
}
