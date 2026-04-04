package httpc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMustMarshalJSON(t *testing.T) {
	data := MustMarshalJSON(map[string]any{"key": "value", "num": 42})
	if len(data) == 0 {
		t.Fatal("expected non-empty JSON")
	}
	want := `{"key":"value","num":42}`
	if string(data) != want {
		t.Errorf("got %s, want %s", data, want)
	}
}

func TestMustMarshalJSON_Panic(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for unmarshalable value")
		}
	}()

	// Channels cannot be marshaled to JSON.
	MustMarshalJSON(make(chan int))
}

func TestMustNewRequest(t *testing.T) {
	body := []byte(`{"prompt":"hello"}`)
	req := MustNewRequest(t.Context(), "POST", "https://api.example.com/v1/chat", body)

	if req.Method != "POST" {
		t.Errorf("Method = %q, want POST", req.Method)
	}
	if req.URL.String() != "https://api.example.com/v1/chat" {
		t.Errorf("URL = %q", req.URL.String())
	}

	reqBody, _ := io.ReadAll(req.Body)
	if string(reqBody) != string(body) {
		t.Errorf("Body = %q, want %q", reqBody, body)
	}
}

func TestMustNewRequest_NilBody(t *testing.T) {
	req := MustNewRequest(t.Context(), "GET", "https://api.example.com/v1/models", nil)

	if req.Method != "GET" {
		t.Errorf("Method = %q, want GET", req.Method)
	}
	if req.Body != nil {
		t.Error("expected nil body for GET request")
	}
}

func TestMustNewRequest_Panic(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil context")
		}
	}()

	// nil context causes http.NewRequestWithContext to fail.
	MustNewRequest(nil, "GET", "https://example.com", nil) //nolint:staticcheck
}

func TestParseDataURL(t *testing.T) {
	tests := []struct {
		name      string
		url       string
		wantMedia string
		wantData  string
		wantOK    bool
	}{
		{
			name:      "valid data URL",
			url:       "data:image/png;base64,abc123",
			wantMedia: "image/png",
			wantData:  "abc123",
			wantOK:    true,
		},
		{
			name:   "no base64 marker",
			url:    "data:image/png,abc",
			wantOK: false,
		},
		{
			name:   "not a data URL",
			url:    "https://example.com",
			wantOK: false,
		},
		{
			name:   "empty string",
			url:    "",
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			media, data, ok := ParseDataURL(tt.url)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if media != tt.wantMedia {
				t.Errorf("mediaType = %q, want %q", media, tt.wantMedia)
			}
			if data != tt.wantData {
				t.Errorf("data = %q, want %q", data, tt.wantData)
			}
		})
	}
}

// --- DoJSONRequest tests ---

// stubErrorParser returns a simple error with status code and body.
func stubErrorParser(providerID string, statusCode int, body []byte, _ http.Header) error {
	return fmt.Errorf("%s: HTTP %d: %s", providerID, statusCode, string(body))
}

func TestDoJSONRequest_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Method = %q, want POST", r.Method)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-token" {
			t.Errorf("Authorization = %q, want Bearer test-token", auth)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if body["model"] != "gpt-4o" {
			t.Errorf("body[model] = %v, want gpt-4o", body["model"])
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"resp-1"}`))
	}))
	defer srv.Close()

	resp, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:        srv.URL,
		Token:      "test-token",
		Body:       map[string]any{"model": "gpt-4o"},
		ProviderID: "openai",
	}, stubErrorParser)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	if string(data) != `{"id":"resp-1"}` {
		t.Errorf("response body = %q", data)
	}
}

func TestDoJSONRequest_ErrorResponse400(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	defer srv.Close()

	_, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:        srv.URL,
		Token:      "tok",
		Body:       map[string]any{"prompt": "hi"},
		ProviderID: "test",
	}, stubErrorParser)
	if err == nil {
		t.Fatal("expected error for 400 response")
	}
	if want := "test: HTTP 400"; !contains(err.Error(), want) {
		t.Errorf("error = %q, want containing %q", err, want)
	}
}

func TestDoJSONRequest_ErrorResponse500(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`server error`))
	}))
	defer srv.Close()

	_, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:        srv.URL,
		Token:      "tok",
		Body:       map[string]any{},
		ProviderID: "myprovider",
	}, stubErrorParser)
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
	if want := "myprovider: HTTP 500"; !contains(err.Error(), want) {
		t.Errorf("error = %q, want containing %q", err, want)
	}
}

func TestDoJSONRequest_CustomHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Custom"); got != "hello" {
			t.Errorf("X-Custom = %q, want hello", got)
		}
		if got := r.Header.Get("X-Api-Version"); got != "2024-01" {
			t.Errorf("X-Api-Version = %q, want 2024-01", got)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	resp, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:   srv.URL,
		Token: "tok",
		Body:  map[string]any{},
		Headers: map[string]string{
			"X-Custom":      "hello",
			"X-Api-Version": "2024-01",
		},
		ProviderID: "test",
	}, stubErrorParser)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp.Body.Close()
}

func TestDoJSONRequest_HeadersFromBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// _headers should be extracted from body and applied to request.
		if got := r.Header.Get("X-From-Body"); got != "yes" {
			t.Errorf("X-From-Body = %q, want yes", got)
		}
		// _headers must NOT appear in the JSON body.
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if _, exists := body["_headers"]; exists {
			t.Error("_headers should be removed from JSON body")
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	resp, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:   srv.URL,
		Token: "tok",
		Body: map[string]any{
			"model":    "test",
			"_headers": map[string]string{"X-From-Body": "yes"},
		},
		ProviderID: "test",
	}, stubErrorParser)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp.Body.Close()
}

func TestDoJSONRequest_NoToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("Authorization = %q, want empty (no token)", auth)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	resp, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:        srv.URL,
		Token:      "", // empty = no Authorization header
		Body:       map[string]any{"prompt": "hi"},
		ProviderID: "test",
	}, stubErrorParser)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp.Body.Close()
}

func TestDoJSONRequest_CustomHTTPClient(t *testing.T) {
	var called bool
	customTransport := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		called = true
		// Forward to a real server would be complex; just return a canned response.
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(io.LimitReader(nil, 0)),
			Header:     make(http.Header),
		}, nil
	})

	resp, err := DoJSONRequest(t.Context(), RequestConfig{
		URL:        "https://api.example.com/v1/chat",
		Token:      "tok",
		Body:       map[string]any{},
		HTTPClient: &http.Client{Transport: customTransport},
		ProviderID: "test",
	}, stubErrorParser)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp.Body.Close()
	if !called {
		t.Error("custom HTTP client transport was not used")
	}
}

func TestDoJSONRequest_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(t.Context())
	cancel() // cancel immediately

	_, err := DoJSONRequest(ctx, RequestConfig{
		URL:        srv.URL,
		Token:      "tok",
		Body:       map[string]any{},
		ProviderID: "test",
	}, stubErrorParser)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	if !contains(err.Error(), "sending request") {
		t.Errorf("error = %q, want containing 'sending request'", err)
	}
}

// --- helpers ---

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsImpl(s, substr))
}

func containsImpl(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
