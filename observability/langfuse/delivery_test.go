package langfuse

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
)

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// No scheduling delays: each Wait establishes that the goroutines have reached
// a blocking operation, including Flush's wait on its original snapshot.
func TestDeliverySnapshotAndOrder(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		requests := make(chan string, 2)
		release := make(chan struct{})
		defer close(release)
		c := newClient("https://example.invalid", "pub", "sec")
		c.httpClient.Transport = transportFunc(func(r *http.Request) (*http.Response, error) {
			var data struct{ Batch []ingestionEvent }
			if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
				return nil, err
			}
			requests <- data.Batch[0].Body.(string)
			<-release
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(""))}, nil
		})
		d := delivery{async: true, client: c}
		text := "eager"
		d.submit(eagerBatch, []ingestionEvent{{Body: &text}})
		flushed := make(chan error, 1)
		go func() { flushed <- Flush(t.Context()) }()
		synctest.Wait()
		if first := <-requests; first != "eager" {
			t.Fatal(first)
		}
		text = "final"
		d.submit(finalBatch, []ingestionEvent{{Body: &text}})
		text = "mutated after return"
		synctest.Wait()
		select {
		case <-requests:
			t.Fatal("final batch overtook eager batch")
		default:
		}
		select {
		case err := <-flushed:
			t.Fatalf("Flush returned before first delivery: %v", err)
		default:
		}
		release <- struct{}{}
		if second := <-requests; second != "final" {
			t.Fatal(second)
		}
		synctest.Wait()
		select {
		case err := <-flushed:
			if err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatal("Flush waited for a batch submitted after its snapshot")
		}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		canceledFlush := make(chan error, 1)
		go func() { canceledFlush <- Flush(ctx) }()
		synctest.Wait()
		select {
		case err := <-canceledFlush:
			t.Fatalf("Flush returned before cancellation or delivery: %v", err)
		default:
		}
		cancel()
		if err := <-canceledFlush; !errors.Is(err, context.Canceled) {
			t.Fatalf("Flush = %v", err)
		}

		nextFlush := make(chan error, 1)
		go func() { nextFlush <- Flush(t.Context()) }()
		synctest.Wait()
		select {
		case err := <-nextFlush:
			t.Fatalf("Flush after cancellation returned before delivery: %v", err)
		default:
		}
		release <- struct{}{}
		if err := <-nextFlush; err != nil {
			t.Fatal(err)
		}
	})
}

func TestDeliveryAsyncErrorCallbackPanic(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		requests := make(chan string, 2)
		reported := make(chan error, 1)
		sendErr := errors.New("first delivery failed")
		c := newClient("https://example.invalid", "pub", "sec")
		c.httpClient.Transport = transportFunc(func(r *http.Request) (*http.Response, error) {
			var data struct{ Batch []ingestionEvent }
			if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
				return nil, err
			}
			if len(data.Batch) != 1 {
				return nil, errors.New("expected one event")
			}
			body, ok := data.Batch[0].Body.(string)
			if !ok {
				return nil, errors.New("expected string body")
			}
			requests <- body
			if body == "first" {
				return nil, sendErr
			}
			return &http.Response{StatusCode: 200, Body: http.NoBody}, nil
		})
		d := delivery{async: true, client: c, onError: func(err error) {
			reported <- err
			panic("error callback panic")
		}}
		d.submit(finalBatch, []ingestionEvent{{Body: "first"}})
		synctest.Wait()
		select {
		case err := <-reported:
			if !errors.Is(err, sendErr) {
				t.Fatalf("callback error = %v; want %v", err, sendErr)
			}
		default:
			t.Fatal("delivery error was not reported")
		}
		if got := <-requests; got != "first" {
			t.Fatalf("first delivery = %q", got)
		}

		d.submit(finalBatch, []ingestionEvent{{Body: "second"}})
		synctest.Wait()
		select {
		case got := <-requests:
			if got != "second" {
				t.Fatalf("delivery after callback panic = %q", got)
			}
		default:
			t.Fatal("delivery after callback panic was not attempted")
		}
		if err := Flush(t.Context()); err != nil {
			t.Fatal(err)
		}
	})
}

func TestDeliveryOverload(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		c := newClient("https://example.invalid", "pub", "sec")
		release := make(chan struct{})
		var sent atomic.Int32
		c.httpClient.Transport = transportFunc(func(*http.Request) (*http.Response, error) {
			<-release
			sent.Add(1)
			return &http.Response{StatusCode: 200, Body: http.NoBody}, nil
		})
		reported := make(chan error, 1)
		d := delivery{async: true, client: c, onError: func(err error) { reported <- err }}
		for range maxPendingBatches + 1 {
			d.submit(eagerBatch, []ingestionEvent{{ID: "trace"}})
		}
		if err := <-reported; !strings.Contains(err.Error(), "queue is full") {
			t.Fatal(err)
		}
		close(release)
		if err := Flush(t.Context()); err != nil {
			t.Fatal(err)
		}
		if got := sent.Load(); got != maxPendingBatches {
			t.Fatalf("sent %d batches; want %d", got, maxPendingBatches)
		}
	})
}
