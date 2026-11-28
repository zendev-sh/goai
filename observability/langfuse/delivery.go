package langfuse

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
)

// Bound both concurrent HTTP requests and batches waiting for an earlier send.
const maxPendingBatches = 64

var deliveries = struct {
	sync.Mutex
	// Each channel tracks one accepted batch for Flush. Only eager batches
	// have an owner: their run's final batch depends on them.
	pending map[chan struct{}]*delivery
}{pending: make(map[chan struct{}]*delivery)}

// Flush waits for the process's already accepted asynchronous batches to finish their
// delivery attempts. Flush does not wait for active calls or tool cleanup to
// submit new batches; wait for their tracing hooks first when shutting down.
// Later submissions do not extend the wait. Cancellation stops only the wait.
// Delivery is best effort: individual errors go to OnFlushError, not Flush,
// and error callbacks may still be running when Flush returns.
func Flush(ctx context.Context) error {
	deliveries.Lock()
	pending := slices.Collect(maps.Keys(deliveries.pending))
	deliveries.Unlock()
	for _, done := range pending {
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// Each run owns its eager batches, even when legacy Hooks share an HTTP client.
type delivery struct {
	client  *client
	async   bool
	onError func(error)
}

type batchKind uint8

const (
	eagerBatch batchKind = iota
	finalBatch
)

// submit reports whether a send was attempted or queued, not whether Langfuse
// accepted the events. A locally rejected create cannot be updated later.
func (d *delivery) submit(kind batchKind, events []ingestionEvent) bool {
	// Bodies can reference caller-owned values. Freeze them before the hook returns.
	payload, err := json.Marshal(map[string]any{"batch": events})
	if err != nil {
		d.report(fmt.Errorf("langfuse: marshal batch: %w", err))
		return false
	}
	if !d.async {
		if err := d.client.send(context.Background(), payload); err != nil {
			d.report(err)
		}
		return true
	}
	deliveries.Lock()
	if len(deliveries.pending) == maxPendingBatches {
		deliveries.Unlock()
		d.report(errors.New("langfuse: trace queue is full"))
		return false
	}
	done := make(chan struct{})
	var creates []chan struct{}
	if kind == eagerBatch {
		deliveries.pending[done] = d
	} else {
		// Tool-start hooks have returned before the final batch is built.
		// Snapshot only this run's unfinished creates, not other traces.
		for pending, owner := range deliveries.pending {
			if owner == d {
				creates = append(creates, pending)
			}
		}
		deliveries.pending[done] = nil
	}
	deliveries.Unlock()
	go func() {
		for _, create := range creates {
			<-create
		}
		err := d.client.send(context.Background(), payload)
		deliveries.Lock()
		delete(deliveries.pending, done)
		close(done)
		deliveries.Unlock()
		if err != nil {
			d.report(err)
		}
	}()
	return true
}

// A blocking or panicking user callback must not stall delivery or generation.
var callbackSlots = make(chan struct{}, maxPendingBatches)

func (d *delivery) report(err error) {
	if d.onError == nil {
		return
	}
	if !d.async {
		d.onError(err)
		return
	}
	select {
	case callbackSlots <- struct{}{}:
		go func() {
			defer func() {
				_ = recover()
				<-callbackSlots
			}()
			d.onError(err)
		}()
	default: // Error reporting is also best effort when callbacks are saturated.
	}
}
