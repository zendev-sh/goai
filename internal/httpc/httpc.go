// Package httpc provides HTTP helper functions for provider implementations.
package httpc

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// MustMarshalJSON marshals v to JSON. It panics on failure, which should only
// occur with unmarshalable types (chan, func, cyclic structs) - a programming
// error, not a runtime condition. All call sites pass provider.GenerateParams
// or similar well-typed structs that are always marshalable.
func MustMarshalJSON(v any) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		panic("httpc: json.Marshal failed: " + err.Error())
	}
	return data
}

// MustNewRequest creates an HTTP request. It panics on failure, which should
// only occur with an invalid HTTP method or a nil context - both programming
// errors detectable at development time, not runtime conditions. All call
// sites pass a valid context and a constant HTTP method string.
func MustNewRequest(ctx context.Context, method, url string, body []byte) *http.Request {
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		panic("httpc: http.NewRequestWithContext failed: " + err.Error())
	}
	return req
}

// ParseDataURL extracts media type and base64 data from a data URL.
// Format: data:<mediaType>;base64,<data>
func ParseDataURL(url string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", "", false
	}
	rest := url[5:]
	semicolon := strings.Index(rest, ";base64,")
	if semicolon < 0 {
		return "", "", false
	}
	return rest[:semicolon], rest[semicolon+8:], true
}
