package langfuse

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
// Events are cleared before the POST; if the POST fails, those events are permanently
// lost. This is intentional: observability is best-effort and we avoid double-sending.
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
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("langfuse: ingestion failed with status %d", resp.StatusCode)
	}
	return nil
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
