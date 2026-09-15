package langfuse

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// client accumulates Langfuse ingestion events and sends them in a single batch on flush.
type client struct {
	host       string
	auth       string // base64(publicKey:secretKey)
	httpClient *http.Client

	mu     sync.Mutex
	events []ingestionEvent
}

type ingestionEvent struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Timestamp string `json:"timestamp"`
	Body      any    `json:"body"`
}

func newClient(host, publicKey, secretKey string) *client {
	return &client{
		host:       strings.TrimRight(host, "/"),
		auth:       base64.StdEncoding.EncodeToString([]byte(publicKey + ":" + secretKey)),
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// appendEvents adds pre-built events to the batch in a single lock operation.
func (c *client) appendEvents(events []ingestionEvent) {
	c.mu.Lock()
	c.events = append(c.events, events...)
	c.mu.Unlock()
}

// flush sends all buffered events to Langfuse in a single POST and clears the queue.
// Events are cleared before the POST to avoid double-sending on retry.
// If the POST fails, those events are permanently lost - this is intentional:
// observability is best-effort and must never block or retry indefinitely.
//
// A successful status does not mean every event was accepted: Langfuse answers
// 207 Multi-Status and lists per-event outcomes in the body. Events it
// rejected are reported as an *IngestionRejectedError.
func (c *client) flush(ctx context.Context) error {
	c.mu.Lock()
	if len(c.events) == 0 {
		c.mu.Unlock()
		return nil
	}
	events := c.events
	c.events = nil
	c.mu.Unlock()

	payload, err := json.Marshal(map[string]any{"batch": events})
	if err != nil {
		return fmt.Errorf("langfuse: marshal batch: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.host+"/api/public/ingestion", bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("langfuse: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Basic "+c.auth)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("langfuse: send batch: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	_, _ = io.Copy(io.Discard, resp.Body)

	if resp.StatusCode >= 400 {
		if s := snippet(string(body), 300); s != "" {
			return fmt.Errorf("langfuse: ingestion failed with status %d: %s", resp.StatusCode, s)
		}
		return fmt.Errorf("langfuse: ingestion failed with status %d", resp.StatusCode)
	}
	return rejectedEvents(body)
}

// maxResponseBytes caps how much of an ingestion response is read. A 207 body
// lists every event with its outcome, so it grows with the batch; the errors
// worth reporting fit well inside this.
const maxResponseBytes = 1 << 20

// IngestionRejectedError reports events Langfuse refused inside a response
// whose HTTP status said the batch was accepted.
//
// The ingestion endpoint answers 207 Multi-Status and puts each event's verdict
// in the body: {"successes":[...],"errors":[...]}. An event can be rejected for
// its content (an invalid timestamp, a wrong field type) while the status stays
// below 400, so without this the event would be dropped with no error and no
// OnFlushError call.
//
// Rejections are not retried: an event rejected for its content would be
// rejected again.
type IngestionRejectedError struct {
	Accepted int
	Rejected []RejectedEvent
}

// RejectedEvent is one entry of a 207 response's "errors" list.
type RejectedEvent struct {
	ID      string `json:"id"`
	Status  int    `json:"status"`
	Message string `json:"message"`
	Error   any    `json:"error"`
}

// Error returns a one-line summary naming up to three rejected events.
func (e *IngestionRejectedError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "langfuse: ingestion rejected %d of %d events", len(e.Rejected), len(e.Rejected)+e.Accepted)
	for i, r := range e.Rejected {
		if i == 3 {
			fmt.Fprintf(&b, "; and %d more", len(e.Rejected)-i)
			break
		}
		fmt.Fprintf(&b, "; %s: status %d %s", r.ID, r.Status, r.Message)
		if detail := errorDetail(r.Error); detail != "" {
			fmt.Fprintf(&b, " (%s)", snippet(detail, 200))
		}
	}
	return b.String()
}

// rejectedEvents returns an *IngestionRejectedError when an accepted response
// still lists rejected events, and nil otherwise - including for a body that
// is empty or not the documented shape, which says nothing about rejection.
func rejectedEvents(body []byte) error {
	if len(bytes.TrimSpace(body)) == 0 {
		return nil
	}
	var parsed struct {
		Successes []json.RawMessage `json:"successes"`
		Errors    []RejectedEvent   `json:"errors"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil || len(parsed.Errors) == 0 {
		return nil
	}
	return &IngestionRejectedError{Accepted: len(parsed.Successes), Rejected: parsed.Errors}
}

// errorDetail renders a rejected event's "error" field, which Langfuse sends
// as a string or as structured JSON.
func errorDetail(v any) string {
	switch d := v.(type) {
	case nil:
		return ""
	case string:
		return strings.Join(strings.Fields(d), " ")
	default:
		b, err := json.Marshal(d)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

// snippet collapses whitespace and caps s at max bytes on a rune boundary.
func snippet(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && s[cut]&0xC0 == 0x80 {
		cut--
	}
	return s[:cut] + "…"
}

// newID returns a random UUID v4 string using only the standard library.
func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant bits
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

// formatTime formats a time.Time as ISO 8601 with millisecond precision for Langfuse.
// Returns an empty string for zero values (used with omitempty).
func formatTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format("2006-01-02T15:04:05.000Z07:00")
}
