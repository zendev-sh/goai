// Package httpc provides HTTP helper functions for provider implementations.
package httpc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// RequestConfig holds common HTTP request parameters for DoJSONRequest.
type RequestConfig struct {
	URL        string            // Full URL to POST to.
	Token      string            // Bearer token (empty = no Authorization header).
	Body       map[string]any    // JSON body; "_headers" key is extracted and applied.
	Headers    map[string]string // Provider-level extra headers (e.g. WithHeaders).
	HTTPClient *http.Client      // Custom client (nil = http.DefaultClient).
	ProviderID string            // Provider name for error messages.
}

// ErrorParser is a function that converts an HTTP error response into a Go error.
// It receives the provider ID, status code, response body, and response headers.
type ErrorParser func(providerID string, statusCode int, body []byte, headers http.Header) error

// DoJSONRequest builds, sends, and validates a JSON POST request.
// It extracts "_headers" from the body map, marshals JSON, sets Content-Type
// and Authorization Bearer (if token is non-empty), applies provider and
// per-request headers, checks the HTTP status code, and returns the response
// or a parsed error.
func DoJSONRequest(ctx context.Context, cfg RequestConfig, parseError ErrorParser) (*http.Response, error) {
	// Extract per-request headers before marshaling (they must not appear in the JSON body).
	reqHeaders, _ := cfg.Body["_headers"].(map[string]string)
	delete(cfg.Body, "_headers")

	jsonBody := MustMarshalJSON(cfg.Body)
	req := MustNewRequest(ctx, "POST", cfg.URL, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	if cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.Token)
	}

	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range reqHeaders {
		req.Header.Set(k, v)
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, parseError(cfg.ProviderID, resp.StatusCode, respBody, resp.Header)
	}

	return resp, nil
}

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
