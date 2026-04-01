package vertex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestEmbedding_SingleValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/models/text-embedding-004:predict") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		if len(instances) != 1 {
			t.Fatalf("expected 1 instance, got %d", len(instances))
		}
		inst := instances[0].(map[string]any)
		if inst["content"] != "hello" {
			t.Errorf("content = %v", inst["content"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{
					"embeddings": map[string]any{
						"values":     []float64{0.1, 0.2, 0.3},
						"statistics": map[string]any{"token_count": 1},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(result.Embeddings[0]))
	}
	if result.Usage.InputTokens != 1 {
		t.Errorf("expected 1 input token, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_MultipleValues(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		if len(instances) != 3 {
			t.Errorf("expected 3 instances, got %d", len(instances))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1, 0.2}, "statistics": map[string]any{"token_count": 1}}},
				{"embeddings": map[string]any{"values": []float64{0.3, 0.4}, "statistics": map[string]any{"token_count": 2}}},
				{"embeddings": map[string]any{"values": []float64{0.5, 0.6}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"a", "b", "c"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Embeddings))
	}
	if result.Usage.InputTokens != 4 {
		t.Errorf("expected 4 total input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_ProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Check instances have task_type and title.
		instances := req["instances"].([]any)
		inst := instances[0].(map[string]any)
		if inst["task_type"] != "RETRIEVAL_QUERY" {
			t.Errorf("task_type = %v", inst["task_type"])
		}
		if inst["title"] != "My Doc" {
			t.Errorf("title = %v", inst["title"])
		}

		// Check parameters have outputDimensionality and autoTruncate.
		params := req["parameters"].(map[string]any)
		if params["outputDimensionality"] != float64(256) {
			t.Errorf("outputDimensionality = %v", params["outputDimensionality"])
		}
		if params["autoTruncate"] != true {
			t.Errorf("autoTruncate = %v", params["autoTruncate"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"taskType":             "RETRIEVAL_QUERY",
				"title":               "My Doc",
				"outputDimensionality": 256,
				"autoTruncate":         true,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedding_NoProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		inst := instances[0].(map[string]any)
		if _, ok := inst["task_type"]; ok {
			t.Error("task_type should not be set")
		}
		if _, ok := req["parameters"]; ok {
			t.Error("parameters should not be set when empty")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad request","code":400}}`))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedding_NoProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	model := Embedding("text-embedding-004", WithTokenSource(provider.StaticToken("tok")))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "PROJECT required") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("text-embedding-004", WithTokenSource(provider.StaticToken("tok")))
	if model.ModelID() != "text-embedding-004" {
		t.Errorf("ModelID = %q", model.ModelID())
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("text-embedding-004", WithTokenSource(provider.StaticToken("tok")))
	if got := model.MaxValuesPerCall(); got != 250 {
		t.Errorf("MaxValuesPerCall = %d, want 250", got)
	}
}

func TestEmbedding_ConnectionError(t *testing.T) {
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL("http://127.0.0.1:1"),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_UnmarshalError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not json`))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_TokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token failed")
	})
	model := Embedding("text-embedding-004", WithTokenSource(ts), WithBaseURL("http://fake"))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_WithHeaders(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Custom": "val"}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if gotHeader != "val" {
		t.Errorf("X-Custom = %q", gotHeader)
	}
}

func TestEmbedding_DefaultURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-east1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	expected := "https://us-east1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-east1/publishers/google/models/text-embedding-004:predict"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestEmbedding_NoTokenSource(t *testing.T) {
	// No token source + custom baseURL -- auth is skipped, request sent unauthenticated.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("unexpected auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("got %d embeddings, want 1", len(result.Embeddings))
	}
}

func TestEmbedding_ReadBodyError(t *testing.T) {
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

// embedAPIKeyURLTransport captures the URL for API-key embedding requests.
type embedAPIKeyURLTransport struct {
	captured string
}

func (tr *embedAPIKeyURLTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tr.captured = req.URL.String()
	body := `{"predictions":[{"embeddings":{"values":[0.1],"statistics":{"token_count":1}}}]}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}, nil
}

func TestEmbedding_APIKeySkipsBearerAuth(t *testing.T) {
	transport := &embedAPIKeyURLTransport{}
	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	// URL should contain ?key=
	if !strings.Contains(transport.captured, "?key=my-embed-key") {
		t.Errorf("URL should contain ?key=, got %q", transport.captured)
	}
	// Should use Gemini API base URL.
	if !strings.Contains(transport.captured, "generativelanguage.googleapis.com") {
		t.Errorf("URL should use generativelanguage.googleapis.com, got %q", transport.captured)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
		WithHTTPClient(&http.Client{}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_ResponseModelPopulated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{
					"embeddings": map[string]any{
						"values":     []float64{0.1, 0.2},
						"statistics": map[string]any{"token_count": 1},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Response.Model != "text-embedding-004" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "text-embedding-004")
	}
}
