package httpc

import (
	"bytes"
	"io"
	"net/http"
	"testing"
)

// A logging transport may inspect GetBody lazily, while consuming the response.
type replayOnRead struct {
	getBody func() (io.ReadCloser, error)
	io.ReadCloser
}

func (r *replayOnRead) Read(p []byte) (int, error) {
	if r.getBody != nil {
		body, err := r.getBody()
		if err != nil {
			return 0, err
		}
		r.ReadCloser, r.getBody = body, nil
	}
	return r.ReadCloser.Read(p)
}

func TestCustomTransportReplaysWhileReadingResponse(t *testing.T) {
	payload := bytes.Repeat([]byte("upload"), 16<<10)
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		_ = r.Body.Close()
		return &http.Response{StatusCode: http.StatusOK, Body: &replayOnRead{getBody: r.GetBody, ReadCloser: http.NoBody}}, nil
	})
	for _, path := range []string{"client", "round trip"} {
		t.Run(path, func(t *testing.T) {
			req := MustNewRequest(t.Context(), http.MethodPost, "https://example.test", payload)
			var resp *http.Response
			var err error
			if path == "client" {
				resp, err = Do(&http.Client{Transport: transport}, req)
			} else {
				resp, err = DoRoundTrip(transport, req)
			}
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			got, err := io.ReadAll(resp.Body)
			if err != nil || !bytes.Equal(got, payload) {
				t.Fatalf("replayed %d bytes, err = %v", len(got), err)
			}
		})
	}
}
