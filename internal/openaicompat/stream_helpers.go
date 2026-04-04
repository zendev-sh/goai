package openaicompat

import (
	"context"
	"io"
	"sync"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// NewSSEStream sets up a standard SSE streaming pipeline for OpenAI-compatible
// providers. It creates a channel, spawns a goroutine that reads SSE events from
// body via ParseStream, and handles context cancellation to prevent goroutine leaks.
//
// The returned StreamResult owns the body; callers must not close it.
func NewSSEStream(ctx context.Context, body io.ReadCloser) *provider.StreamResult {
	out := make(chan provider.StreamChunk, 64)
	scanner := sse.NewScanner(body)
	go func() {
		var closeOnce sync.Once
		closeBody := func() { closeOnce.Do(func() { _ = body.Close() }) }
		defer closeBody()
		// Close body on context cancellation to unblock scanner.Scan().
		// Without this, the goroutine leaks if the server stalls mid-stream.
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				closeBody()
			case <-done:
			}
		}()
		ParseStream(ctx, scanner, out)
	}()
	return &provider.StreamResult{Stream: out}
}
