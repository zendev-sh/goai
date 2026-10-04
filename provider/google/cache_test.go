package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// findToolEntry returns the tools[] entry that has the given key, or nil.
func findToolEntry(t *testing.T, body map[string]any, key string) map[string]any {
	t.Helper()
	tools, ok := body["tools"].([]any)
	if !ok {
		t.Fatalf("body.tools is not an array: %v", body["tools"])
	}
	for _, e := range tools {
		m, ok := e.(map[string]any)
		if !ok {
			continue
		}
		if _, has := m[key]; has {
			return m
		}
	}
	return nil
}

func TestCacheClient_Create_DirectShape(t *testing.T) {
	var gotMethod, gotPath, gotKey string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotPath, gotKey = r.Method, r.URL.Path, r.Header.Get("x-goog-api-key")
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"cachedContents/abc","expireTime":"2026-01-01T00:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("test-key"), WithBaseURL(server.URL))
	got, err := c.Create(context.Background(), CachedContentInput{
		Model:  "gemini-3.5-flash",
		System: "you are helpful",
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "reads a file", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: "auto",
		TTL:        5 * time.Minute,
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	if got.Name != "cachedContents/abc" {
		t.Errorf("Name = %q", got.Name)
	}
	if got.ExpiresAt.IsZero() {
		t.Error("ExpiresAt not parsed")
	}
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q", gotMethod)
	}
	if gotPath != "/v1beta/cachedContents" {
		t.Errorf("path = %q", gotPath)
	}
	if gotKey != "test-key" {
		t.Errorf("x-goog-api-key = %q", gotKey)
	}
	if body["model"] != "models/gemini-3.5-flash" {
		t.Errorf("model = %v", body["model"])
	}
	if body["ttl"] != "300s" {
		t.Errorf("ttl = %v", body["ttl"])
	}
	if _, ok := body["systemInstruction"]; !ok {
		t.Error("systemInstruction missing from cache body")
	}
	if findToolEntry(t, body, "functionDeclarations") == nil {
		t.Error("functionDeclarations missing from cache body")
	}
}

// The parity test: a cached resource must hold the SAME server tools and
// toolConfig that a generate request would send — otherwise a cached turn loses
// web_search/url_context and the mixed-tools config.
func TestCacheClient_Create_IncludesClientAndServerTools(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"cachedContents/x","expireTime":"2026-01-01T00:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := c.Create(context.Background(), CachedContentInput{
		Model:  "gemini-3.5-flash",
		System: "sys",
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "reads", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{ProviderDefinedType: "google.google_search"},
			{ProviderDefinedType: "google.url_context"},
		},
		TTL: time.Hour,
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	if findToolEntry(t, body, "functionDeclarations") == nil {
		t.Error("client function declarations missing")
	}
	if findToolEntry(t, body, "googleSearch") == nil {
		t.Error("server tool googleSearch missing from cache body")
	}
	if findToolEntry(t, body, "urlContext") == nil {
		t.Error("server tool urlContext missing from cache body")
	}
}

func TestCacheClient_Renew_PatchesTTL(t *testing.T) {
	var gotMethod, gotPath, gotQuery string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotPath, gotQuery = r.Method, r.URL.Path, r.URL.RawQuery
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"cachedContents/abc","expireTime":"2026-01-01T01:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	got, err := c.Renew(context.Background(), "cachedContents/abc", time.Hour)
	if err != nil {
		t.Fatalf("Renew: %v", err)
	}
	if got.ExpiresAt.IsZero() {
		t.Error("ExpiresAt not parsed")
	}
	if gotMethod != http.MethodPatch {
		t.Errorf("method = %q", gotMethod)
	}
	if gotPath != "/v1beta/cachedContents/abc" {
		t.Errorf("path = %q", gotPath)
	}
	if gotQuery != "updateMask=ttl" {
		t.Errorf("query = %q", gotQuery)
	}
	if body["ttl"] != "3600s" {
		t.Errorf("ttl = %v", body["ttl"])
	}
}

func TestCacheClient_Create_VertexShape(t *testing.T) {
	var gotPath, gotAuth string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotAuth = r.URL.Path, r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"name":"projects/proj/locations/us-central1/cachedContents/z","expireTime":"2026-01-01T00:00:00Z"}`)
	}))
	defer server.Close()

	c := NewCacheClient(
		WithTokenSource(provider.StaticToken("tok")),
		WithVertex("proj", "us-central1"),
		WithBaseURL(server.URL),
	)
	got, err := c.Create(context.Background(), CachedContentInput{Model: "gemini-3.5-flash", System: "x", TTL: time.Hour})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if got.Name != "projects/proj/locations/us-central1/cachedContents/z" {
		t.Errorf("Name = %q", got.Name)
	}
	if gotPath != "/v1beta1/projects/proj/locations/us-central1/cachedContents" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer tok" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if body["model"] != "projects/proj/locations/us-central1/publishers/google/models/gemini-3.5-flash" {
		t.Errorf("model = %v", body["model"])
	}
}

func TestCacheClient_Create_SurfacesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"bad"}}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: time.Minute}); err == nil {
		t.Error("expected error on 400")
	}
}

func TestCacheClient_ValidationAndRoutingErrors(t *testing.T) {
	// Bad project in Vertex mode
	badProj := NewCacheClient(WithVertex("bad/project", "us-central1"))
	if _, err := badProj.Create(context.Background(), CachedContentInput{Model: "gemini-3.5-flash"}); err == nil {
		t.Error("expected error for bad project on Create")
	}
	if _, err := badProj.Renew(context.Background(), "name", time.Hour); err == nil {
		t.Error("expected error for bad project on Renew")
	}

	// Bad location in Vertex mode
	badLoc := NewCacheClient(WithVertex("project", "bad/location"))
	if _, err := badLoc.Create(context.Background(), CachedContentInput{Model: "gemini-3.5-flash"}); err == nil {
		t.Error("expected error for bad location on Create")
	}
	if _, err := badLoc.Renew(context.Background(), "name", time.Hour); err == nil {
		t.Error("expected error for bad location on Renew")
	}

	// Invalid tool schema
	c := NewCacheClient(WithAPIKey("key"))
	if _, err := c.Create(context.Background(), CachedContentInput{
		Tools: []provider.ToolDefinition{
			{Name: "bad_tool", InputSchema: json.RawMessage(`{invalid json}`)},
		},
	}); err == nil {
		t.Error("expected error for invalid tool schema on Create")
	}
}

func TestCacheClient_VertexDefaultURL(t *testing.T) {
	// Tests default Vertex URLs without WithBaseURL override
	c := NewCacheClient(WithVertex("my-proj", "us-central1"))
	url, err := c.createURL()
	if err != nil {
		t.Fatalf("createURL error: %v", err)
	}
	wantURL := "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/cachedContents"
	if url != wantURL {
		t.Errorf("createURL = %q, want %q", url, wantURL)
	}

	resURL, err := c.resourceURL("projects/my-proj/locations/us-central1/cachedContents/123", "?updateMask=ttl")
	if err != nil {
		t.Fatalf("resourceURL error: %v", err)
	}
	wantResURL := "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/cachedContents/123?updateMask=ttl"
	if resURL != wantResURL {
		t.Errorf("resourceURL = %q, want %q", resURL, wantResURL)
	}

	// Also test global region vertexHost
	cGlobal := NewCacheClient(WithVertex("my-proj", "global"))
	globalURL, err := cGlobal.createURL()
	if err != nil {
		t.Fatalf("createURL error: %v", err)
	}
	wantGlobalURL := "https://aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/global/cachedContents"
	if globalURL != wantGlobalURL {
		t.Errorf("createURL = %q, want %q", globalURL, wantGlobalURL)
	}
}

func TestCacheClient_ResponseParsingErrors(t *testing.T) {
	tests := []struct {
		name     string
		response string
		wantErr  string
	}{
		{
			name:     "invalid json",
			response: `{not json}`,
			wantErr:  "parsing cachedContents response",
		},
		{
			name:     "missing name",
			response: `{"expireTime":"2026-01-01T00:00:00Z"}`,
			wantErr:  "cachedContents response missing name",
		},
		{
			name:     "invalid expireTime format",
			response: `{"name":"cachedContents/123","expireTime":"invalid-date"}`,
			wantErr:  "parsing cachedContents expireTime",
		},
		{
			name:     "invalid createTime format",
			response: `{"name":"cachedContents/123","expireTime":"2026-01-01T00:00:00Z","createTime":"invalid-date"}`,
			wantErr:  "parsing cachedContents createTime",
		},
		{
			name:     "invalid updateTime format",
			response: `{"name":"cachedContents/123","expireTime":"2026-01-01T00:00:00Z","updateTime":"invalid-date"}`,
			wantErr:  "parsing cachedContents updateTime",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, tt.response)
			}))
			defer server.Close()

			c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
			_, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: time.Hour})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Create() err = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestCacheClient_Renew_SurfacesErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = fmt.Fprint(w, `{"error":{"code":404,"message":"not found"}}`)
	}))
	defer server.Close()

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := c.Renew(context.Background(), "cachedContents/missing", time.Hour); err == nil {
		t.Error("expected error on 404 for Renew")
	}
}

type errorReadCloser struct{}

func (errorReadCloser) Read(p []byte) (n int, err error) {
	return 0, errors.New("simulated body read error")
}

func (errorReadCloser) Close() error {
	return nil
}

type errorBodyTransport struct{}

func (errorBodyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       errorReadCloser{},
		Header:     make(http.Header),
	}, nil
}

func TestCacheClient_BodyReadError(t *testing.T) {
	httpClient := &http.Client{Transport: errorBodyTransport{}}
	c := NewCacheClient(WithAPIKey("k"), WithHTTPClient(httpClient))
	_, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: time.Hour})
	if err == nil || !strings.Contains(err.Error(), "reading cachedContents response") {
		t.Fatalf("expected body read error, got: %v", err)
	}
}

func TestCacheClient_RejectsNonPositiveTTL(t *testing.T) {
	c := NewCacheClient(WithAPIKey("k"))
	if _, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: 0}); err == nil || !strings.Contains(err.Error(), "TTL must be positive") {
		t.Fatalf("Create(TTL=0) err = %v, want positive-TTL error", err)
	}
	if _, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: -time.Minute}); err == nil || !strings.Contains(err.Error(), "TTL must be positive") {
		t.Fatalf("Create(negative TTL) err = %v, want positive-TTL error", err)
	}
	if _, err := c.Renew(context.Background(), "cachedContents/x", 0); err == nil || !strings.Contains(err.Error(), "TTL must be positive") {
		t.Fatalf("Renew(TTL=0) err = %v, want positive-TTL error", err)
	}
}

type oversizedBodyTransport struct{}

func (oversizedBodyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(strings.Repeat("x", maxCacheResponseBytes+1))),
		Header:     make(http.Header),
	}, nil
}

func TestCacheClient_RejectsOversizedResponse(t *testing.T) {
	httpClient := &http.Client{Transport: oversizedBodyTransport{}}
	c := NewCacheClient(WithAPIKey("k"), WithHTTPClient(httpClient))
	_, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: time.Hour})
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("expected size-limit error, got: %v", err)
	}
}

// cacheResponseServer answers every cachedContents call with the given body.
func cacheResponseServer(t *testing.T, response string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, response)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestCacheClient_ParsesResourceMetadata(t *testing.T) {
	server := cacheResponseServer(t, `{
		"name": "cachedContents/abc",
		"displayName": "session prefix",
		"model": "models/gemini-3.5-flash",
		"createTime": "2026-01-01T00:00:00.123456Z",
		"updateTime": "2026-01-01T00:30:00Z",
		"expireTime": "2026-01-01T01:00:00Z",
		"usageMetadata": {"totalTokenCount": 12345}
	}`)

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	for name, call := range map[string]func() (CachedContent, error){
		"Create": func() (CachedContent, error) {
			return c.Create(context.Background(), CachedContentInput{Model: "gemini-3.5-flash", TTL: time.Hour})
		},
		"Renew": func() (CachedContent, error) {
			return c.Renew(context.Background(), "cachedContents/abc", time.Hour)
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := call()
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			want := CachedContent{
				Name:        "cachedContents/abc",
				DisplayName: "session prefix",
				Model:       "models/gemini-3.5-flash",
				CreatedAt:   time.Date(2026, 1, 1, 0, 0, 0, 123456000, time.UTC),
				UpdatedAt:   time.Date(2026, 1, 1, 0, 30, 0, 0, time.UTC),
				ExpiresAt:   time.Date(2026, 1, 1, 1, 0, 0, 0, time.UTC),
				Usage:       CachedContentUsage{TotalTokens: 12345},
			}
			if got != want {
				t.Errorf("got %+v\nwant %+v", got, want)
			}
		})
	}
}

// Vertex reports a per-modality breakdown alongside the total.
func TestCacheClient_ParsesVertexUsageMetadata(t *testing.T) {
	server := cacheResponseServer(t, `{
		"name": "projects/p/locations/us-central1/cachedContents/z",
		"expireTime": "2026-01-01T01:00:00Z",
		"usageMetadata": {
			"totalTokenCount": 9000,
			"textCount": 4000,
			"imageCount": 2,
			"videoDurationSeconds": 30,
			"audioDurationSeconds": 15
		}
	}`)

	c := NewCacheClient(WithTokenSource(provider.StaticToken("tok")), WithVertex("p", "us-central1"), WithBaseURL(server.URL))
	got, err := c.Create(context.Background(), CachedContentInput{Model: "gemini-3.5-flash", TTL: time.Hour})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	want := CachedContentUsage{TotalTokens: 9000, TextCount: 4000, ImageCount: 2, VideoDurationSeconds: 30, AudioDurationSeconds: 15}
	if got.Usage != want {
		t.Errorf("Usage = %+v, want %+v", got.Usage, want)
	}
}

// Optional fields absent from the response stay at their zero value.
func TestCacheClient_MissingOptionalFieldsAreZero(t *testing.T) {
	server := cacheResponseServer(t, `{"name":"cachedContents/abc","expireTime":"2026-01-01T00:00:00Z"}`)

	c := NewCacheClient(WithAPIKey("k"), WithBaseURL(server.URL))
	got, err := c.Create(context.Background(), CachedContentInput{Model: "m", TTL: time.Hour})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if !got.CreatedAt.IsZero() || !got.UpdatedAt.IsZero() || got.DisplayName != "" || got.Model != "" || got.Usage != (CachedContentUsage{}) {
		t.Errorf("optional fields not zero: %+v", got)
	}
}
