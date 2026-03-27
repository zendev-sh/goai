package bedrock

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestEmbedding_Titan(t *testing.T) {
	wantEmbedding := []float64{0.1, 0.2, 0.3}
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/model/") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.Path, "/invoke") {
			t.Errorf("embedding should use /invoke endpoint, got %s", r.URL.Path)
		}
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256") {
			t.Errorf("expected SigV4 auth, got %s", auth)
		}

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)

		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           wantEmbedding,
			"inputTextTokenCount": 5,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AKIAIOSFODNN7EXAMPLE"),
		WithSecretKey("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
		WithBaseURL(server.URL),
	)

	if model.MaxValuesPerCall() != 1 {
		t.Errorf("MaxValuesPerCall = %d, want 1", model.MaxValuesPerCall())
	}

	result, err := model.DoEmbed(t.Context(), []string{"hello world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Fatalf("embedding length = %d, want 3", len(result.Embeddings[0]))
	}
	for i, v := range wantEmbedding {
		if result.Embeddings[0][i] != v {
			t.Errorf("embedding[%d] = %f, want %f", i, result.Embeddings[0][i], v)
		}
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("InputTokens = %d, want 5", result.Usage.InputTokens)
	}
	if gotReq["inputText"] != "hello world" {
		t.Errorf("inputText = %v, want 'hello world'", gotReq["inputText"])
	}
	if gotReq["normalize"] != true {
		t.Errorf("normalize = %v, want true", gotReq["normalize"])
	}
}

func TestEmbedding_TitanProviderOptions(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1},
			"inputTextTokenCount": 3,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"test"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"dimensions": 512,
			"normalize":  false,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if gotReq["dimensions"] != float64(512) {
		t.Errorf("dimensions = %v, want 512", gotReq["dimensions"])
	}
	if gotReq["normalize"] != false {
		t.Errorf("normalize = %v, want false", gotReq["normalize"])
	}
}

func TestEmbedding_Cohere(t *testing.T) {
	wantEmbeddings := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": wantEmbeddings,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("cohere.embed-english-v3",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	if model.MaxValuesPerCall() != 96 {
		t.Errorf("MaxValuesPerCall = %d, want 96", model.MaxValuesPerCall())
	}

	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"input_type": "search_query",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(result.Embeddings))
	}
	if gotReq["input_type"] != "search_query" {
		t.Errorf("input_type = %v, want 'search_query'", gotReq["input_type"])
	}
	texts, _ := gotReq["texts"].([]any)
	if len(texts) != 2 {
		t.Errorf("texts length = %d, want 2", len(texts))
	}
}

func TestEmbedding_TitanV1_NoExtraFields(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1, 0.2},
			"inputTextTokenCount": 2,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v1",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	// V1 must NOT send normalize or dimensions — they are unsupported.
	if _, ok := gotReq["normalize"]; ok {
		t.Error("titan v1 must not send 'normalize' field")
	}
	if _, ok := gotReq["dimensions"]; ok {
		t.Error("titan v1 must not send 'dimensions' field")
	}
	if gotReq["inputText"] != "hello" {
		t.Errorf("inputText = %v, want 'hello'", gotReq["inputText"])
	}
}

func TestEmbedding_TitanV2_EmbeddingTypes(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1},
			"inputTextTokenCount": 1,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"embeddingTypes": []string{"float", "binary"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	types, _ := gotReq["embeddingTypes"].([]any)
	if len(types) != 2 {
		t.Errorf("embeddingTypes = %v, want [float binary]", gotReq["embeddingTypes"])
	}
}

func TestEmbedding_CohereV4Format(t *testing.T) {
	// Cohere v4 returns {"embeddings": {"float": [...]}} instead of {"embeddings": [[...]]}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":{"float":[[0.1,0.2],[0.3,0.4]]}}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-v4:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 0.1 || result.Embeddings[1][0] != 0.3 {
		t.Errorf("unexpected embeddings: %v", result.Embeddings)
	}
}

func TestEmbedding_TitanMultimodal_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.1, 0.2},
			"inputTextTokenCount": 3,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-image-v1",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"outputEmbeddingLength": 384},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	cfg, _ := gotReq["embeddingConfig"].(map[string]any)
	if cfg == nil || cfg["outputEmbeddingLength"] != float64(384) {
		t.Errorf("embeddingConfig = %v, want outputEmbeddingLength=384", gotReq["embeddingConfig"])
	}
	if gotReq["inputText"] != "hello" {
		t.Errorf("inputText = %v, want 'hello'", gotReq["inputText"])
	}
	// Must not send normalize or dimensions.
	if _, ok := gotReq["normalize"]; ok {
		t.Error("titan-embed-image-v1 must not send 'normalize'")
	}
}

func TestEmbedding_Nova_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": []map[string]any{
				{"embeddingType": "TEXT", "embedding": []float64{0.5, 0.6, 0.7}},
			},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"test text"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"embeddingDimension": 1024,
			"embeddingPurpose":   "TEXT_RETRIEVAL",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 3 {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}

	params, _ := gotReq["singleEmbeddingParams"].(map[string]any)
	if params == nil {
		t.Fatal("singleEmbeddingParams missing")
	}
	if gotReq["schemaVersion"] != "nova-multimodal-embed-v1" {
		t.Errorf("schemaVersion = %v", gotReq["schemaVersion"])
	}
	if params["embeddingPurpose"] != "TEXT_RETRIEVAL" {
		t.Errorf("embeddingPurpose = %v, want TEXT_RETRIEVAL", params["embeddingPurpose"])
	}
	if params["embeddingDimension"] != float64(1024) {
		t.Errorf("embeddingDimension = %v, want 1024", params["embeddingDimension"])
	}
	textParam, _ := params["text"].(map[string]any)
	if textParam["value"] != "test text" {
		t.Errorf("text.value = %v, want 'test text'", textParam["value"])
	}
}

func TestEmbedding_Marengo27_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":       []float64{0.1, 0.2, 0.3},
			"embeddingOption": "visual-text",
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("twelvelabs.marengo-embed-2-7-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	if gotReq["inputType"] != "text" {
		t.Errorf("inputType = %v, want 'text'", gotReq["inputType"])
	}
	if gotReq["inputText"] != "hello" {
		t.Errorf("inputText = %v, want 'hello'", gotReq["inputText"])
	}
	if gotReq["textTruncate"] != "end" {
		t.Errorf("textTruncate = %v, want 'end'", gotReq["textTruncate"])
	}
}

func TestEmbedding_Marengo30_TextOnly(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"data": map[string]any{
				"embedding": []float64{0.4, 0.5},
			},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("twelvelabs.marengo-embed-3-0-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	result, err := model.DoEmbed(t.Context(), []string{"world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 2 {
		t.Fatalf("unexpected embeddings: %v", result.Embeddings)
	}
	if gotReq["inputType"] != "text" {
		t.Errorf("inputType = %v, want 'text'", gotReq["inputType"])
	}
	textParam, _ := gotReq["text"].(map[string]any)
	if textParam == nil || textParam["inputText"] != "world" {
		t.Errorf("text.inputText = %v, want 'world'", textParam)
	}
}

func TestEmbedding_Nova_Defaults(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": []map[string]any{
				{"embeddingType": "TEXT", "embedding": []float64{0.1}},
			},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	params, _ := gotReq["singleEmbeddingParams"].(map[string]any)
	if params["embeddingPurpose"] != "GENERIC_INDEX" {
		t.Errorf("embeddingPurpose = %v, want GENERIC_INDEX", params["embeddingPurpose"])
	}
	if params["embeddingDimension"] != float64(3072) {
		t.Errorf("embeddingDimension = %v, want 3072", params["embeddingDimension"])
	}
	textParam, _ := params["text"].(map[string]any)
	if textParam["truncationMode"] != "END" {
		t.Errorf("truncationMode = %v, want END", textParam["truncationMode"])
	}
}

func TestEmbedding_Nova_DimensionAsFloat64(t *testing.T) {
	// embeddingDimension passed as float64 (e.g. from JSON-decoded options) must be handled.
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embeddings": []map[string]any{{"embedding": []float64{0.1}}},
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"embeddingDimension": float64(1024)},
	})
	if err != nil {
		t.Fatal(err)
	}

	params, _ := gotReq["singleEmbeddingParams"].(map[string]any)
	if params["embeddingDimension"] != float64(1024) {
		t.Errorf("embeddingDimension = %v, want 1024", params["embeddingDimension"])
	}
}

func TestEmbedding_Nova_EmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[]}`))
	}))
	defer server.Close()

	model := Embedding("amazon.nova-2-multimodal-embeddings-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for empty nova response")
	}
	if !strings.Contains(err.Error(), "nova returned no embeddings") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message":"invalid model identifier"}`))
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for HTTP 400")
	}
	if !strings.Contains(err.Error(), "invalid model identifier") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_CohereInvalidFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":"unexpected_string"}`))
	}))
	defer server.Close()

	model := Embedding("cohere.embed-english-v3",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for invalid embeddings format")
	}
	if !strings.Contains(err.Error(), "unrecognised embeddings format") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_TitanMultimodal_DefaultLength(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embedding": []float64{0.1}, "inputTextTokenCount": 1})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-image-v1",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	cfg, _ := gotReq["embeddingConfig"].(map[string]any)
	if cfg == nil || cfg["outputEmbeddingLength"] != float64(1024) {
		t.Errorf("default outputEmbeddingLength = %v, want 1024", cfg)
	}
}

func TestEmbedding_Marengo27_TextTruncate(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embedding": []float64{0.1}})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("twelvelabs.marengo-embed-2-7-v1:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"textTruncate": "none"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if gotReq["textTruncate"] != "none" {
		t.Errorf("textTruncate = %v, want 'none'", gotReq["textTruncate"])
	}
}

func TestEmbedding_CohereDefaultInputType(t *testing.T) {
	var gotReq map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{"embeddings": [][]float64{{0.1}}})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("cohere.embed-english-v3",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if gotReq["input_type"] != "search_document" {
		t.Errorf("default input_type = %v, want 'search_document'", gotReq["input_type"])
	}
}

func TestEmbedding_InvalidRegion(t *testing.T) {
	model := &embeddingModel{
		id:   "amazon.titan-embed-text-v2:0",
		opts: options{region: "invalid region!", accessKey: "AK", secretKey: "SK"},
	}
	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for invalid region")
	}
	if !strings.Contains(err.Error(), "invalid AWS region") {
		t.Errorf("error = %v", err)
	}
}

func TestEmbedding_Empty(t *testing.T) {
	model := Embedding("amazon.titan-embed-text-v2:0",
		WithAccessKey("AK"),
		WithSecretKey("SK"),
	)
	result, err := model.DoEmbed(t.Context(), []string{}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 0 {
		t.Errorf("expected empty embeddings, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_MissingCredentials(t *testing.T) {
	model := &embeddingModel{
		id:   "amazon.titan-embed-text-v2:0",
		opts: options{region: "us-east-1"},
	}
	_, err := model.DoEmbed(t.Context(), []string{"test"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for missing credentials")
	}
	if !strings.Contains(err.Error(), "AWS_ACCESS_KEY_ID") {
		t.Errorf("error = %v, want credentials error", err)
	}
}

func TestEmbedding_BearerToken(t *testing.T) {
	var gotAuth string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		resp, _ := json.Marshal(map[string]any{
			"embedding":           []float64{0.5},
			"inputTextTokenCount": 1,
		})
		_, _ = w.Write(resp)
	}))
	defer server.Close()

	model := Embedding("amazon.titan-embed-text-v2:0",
		WithBearerToken("my-bearer-token"),
		WithBaseURL(server.URL),
	)

	_, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}

	if gotAuth != "Bearer my-bearer-token" {
		t.Errorf("Authorization = %q, want 'Bearer my-bearer-token'", gotAuth)
	}
}
