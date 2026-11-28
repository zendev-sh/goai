package bedrock

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance check.
var _ provider.EmbeddingModel = (*embeddingModel)(nil)

const (
	// maxEmbedResponseBytes bounds the size of a successful embedding response
	// body to defend against a malicious/misbehaving server returning an
	// unbounded payload.
	maxEmbedResponseBytes = 64 << 20 // 64 MiB
	// maxEmbedErrorBytes bounds the size of an error response body read solely
	// to extract the error message.
	maxEmbedErrorBytes = 1 << 20 // 1 MiB
)

// Embedding creates a Bedrock text embedding model for the given model ID.
//
// Supported models:
//   - amazon.titan-embed-text-v1: single input, 1024 dims.
//   - amazon.titan-embed-text-v2:0: single input; ProviderOptions: "dimensions" (256/512/1024),
//     "normalize" (bool, default true), "embeddingTypes" ([]string).
//   - amazon.titan-embed-image-v1: single input, text-only path;
//     ProviderOptions: "outputEmbeddingLength" (256/384/1024).
//   - amazon.nova-2-multimodal-embeddings-v1:0: single input, text-only path, 3072 dims default;
//     ProviderOptions: "embeddingPurpose", "embeddingDimension" (256/384/1024/3072), "truncationMode".
//   - cohere.embed-english-v3, cohere.embed-multilingual-v3: up to 96 texts/call, 1024 dims;
//     ProviderOptions: "input_type", "truncate".
//   - cohere.embed-v4:0: up to 96 texts/call, 1536 dims default;
//     ProviderOptions: "input_type", "truncate", "output_dimension", "embedding_types".
//   - twelvelabs.marengo-embed-2-7-v1:0: single input, text-only path, max 77 tokens;
//     ProviderOptions: "textTruncate" ("end"/"none").
//   - twelvelabs.marengo-embed-3-0-v1:0: single input, text-only path, max 500 tokens.
//
// Credentials are resolved in the same order as Chat: explicit options,
// then AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN env vars,
// or AWS_BEARER_TOKEN_BEDROCK for Bearer auth.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	o.region = cmp.Or(o.region, os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"), "us-east-1")
	o.bearerToken = cmp.Or(o.bearerToken, os.Getenv("AWS_BEARER_TOKEN_BEDROCK"))
	o.accessKey = cmp.Or(o.accessKey, os.Getenv("AWS_ACCESS_KEY_ID"))
	o.secretKey = cmp.Or(o.secretKey, os.Getenv("AWS_SECRET_ACCESS_KEY"))
	o.sessionToken = cmp.Or(o.sessionToken, os.Getenv("AWS_SESSION_TOKEN"))
	o.baseURL = cmp.Or(o.baseURL, os.Getenv("AWS_BEDROCK_BASE_URL"))
	return &embeddingModel{id: modelID, opts: o}
}

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

// MaxValuesPerCall returns the maximum batch size per InvokeModel call.
// Cohere models support up to 96 texts (AWS-confirmed). All other models are single-input.
func (m *embeddingModel) MaxValuesPerCall() int {
	if strings.Contains(m.id, "cohere") {
		return 96
	}
	return 1
}

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	if len(values) == 0 {
		return &provider.EmbedResult{}, nil
	}
	switch {
	case strings.Contains(m.id, "cohere"):
		return m.doCohereEmbed(ctx, values, params)
	case strings.Contains(m.id, "nova"):
		return m.doNovaEmbed(ctx, values[0], params)
	case strings.Contains(m.id, "twelvelabs") || strings.Contains(m.id, "marengo"):
		return m.doMarengoEmbed(ctx, values[0], params)
	case strings.HasPrefix(m.id, "amazon.titan-embed-image-v1"):
		return m.doTitanMultimodalEmbed(ctx, values[0], params)
	default:
		return m.doTitanEmbed(ctx, values[0], params)
	}
}

// doTitanEmbed handles Amazon Titan text embedding models (single input per call).
// V1 only sends inputText. V2 additionally supports normalize, dimensions, embeddingTypes.
func (m *embeddingModel) doTitanEmbed(ctx context.Context, value string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	body := map[string]any{"inputText": value}

	if isTitanV2(m.id) {
		body["normalize"] = true
		if params.ProviderOptions != nil {
			if v, ok := params.ProviderOptions["dimensions"]; ok {
				body["dimensions"] = v
			}
			if v, ok := params.ProviderOptions["normalize"]; ok {
				body["normalize"] = v
			}
			if v, ok := params.ProviderOptions["embeddingTypes"]; ok {
				body["embeddingTypes"] = v
			}
		}
	}

	resp, err := m.invokeModel(ctx, httpc.MustMarshalJSON(body))
	if err != nil {
		return nil, err
	}

	var result struct {
		Embedding           []float64       `json:"embedding"`
		EmbeddingsByType    json.RawMessage `json:"embeddingsByType"`
		InputTextTokenCount int             `json:"inputTextTokenCount"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	// Titan V2 also returns "embeddingsByType", the only place the vector lives
	// when embeddingTypes omits "float". Unlike Cohere's map it holds ONE flat
	// vector per type, never a matrix.
	embedding := result.Embedding
	if len(result.EmbeddingsByType) > 0 {
		embedding, err = parseTitanEmbeddingsByType(result.EmbeddingsByType)
		if err != nil {
			return nil, err
		}
	}

	return &provider.EmbedResult{
		Embeddings: [][]float64{embedding},
		Usage:      provider.Usage{InputTokens: result.InputTextTokenCount, TotalTokens: result.InputTextTokenCount},
		Response:   provider.ResponseMetadata{Model: m.id},
	}, nil
}

// parseTitanEmbeddingsByType parses Titan V2's "embeddingsByType":
// {"float": [0.1, ...], "binary": [0, 1, ...]}. Each value is a single flat
// vector (binary as one 0/1 integer per dimension); "float" is preferred.
func parseTitanEmbeddingsByType(raw json.RawMessage) ([]float64, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, errors.New("bedrock: unrecognised titan embeddingsByType format")
	}
	for _, key := range []string{"float", "binary"} {
		val, ok := m[key]
		if !ok || len(val) == 0 || string(val) == "null" {
			continue
		}
		var v []float64
		if err := json.Unmarshal(val, &v); err != nil {
			return nil, fmt.Errorf("bedrock: parsing titan %s embedding: %w", key, err)
		}
		return v, nil
	}
	return nil, errors.New("bedrock: no embeddings in titan embeddingsByType")
}

// isTitanV2 returns true for Titan V2 which supports normalize/dimensions/embeddingTypes.
func isTitanV2(modelID string) bool {
	return strings.Contains(modelID, "titan-embed-text-v2")
}

// doTitanMultimodalEmbed handles amazon.titan-embed-image-v1 using the text-only path.
// ProviderOptions: "outputEmbeddingLength" (256, 384, 1024).
func (m *embeddingModel) doTitanMultimodalEmbed(ctx context.Context, value string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	embedCfg := map[string]any{"outputEmbeddingLength": 1024}
	if params.ProviderOptions != nil {
		if v, ok := params.ProviderOptions["outputEmbeddingLength"]; ok {
			embedCfg["outputEmbeddingLength"] = v
		}
	}
	body := map[string]any{
		"inputText":       value,
		"embeddingConfig": embedCfg,
	}

	resp, err := m.invokeModel(ctx, httpc.MustMarshalJSON(body))
	if err != nil {
		return nil, err
	}

	var result struct {
		Embedding           []float64 `json:"embedding"`
		InputTextTokenCount int       `json:"inputTextTokenCount"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}
	return &provider.EmbedResult{
		Embeddings: [][]float64{result.Embedding},
		Usage:      provider.Usage{InputTokens: result.InputTextTokenCount, TotalTokens: result.InputTextTokenCount},
		Response:   provider.ResponseMetadata{Model: m.id},
	}, nil
}

// doNovaEmbed handles amazon.nova-2-multimodal-embeddings-v1:0 using the text-only path.
// ProviderOptions: "embeddingPurpose" (default "GENERIC_INDEX"),
// "embeddingDimension" (256/384/1024/3072), "truncationMode" ("START"/"END"/"NONE").
func (m *embeddingModel) doNovaEmbed(ctx context.Context, value string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	purpose := "GENERIC_INDEX"
	dimension := 3072
	truncation := "END"

	if params.ProviderOptions != nil {
		if v, ok := params.ProviderOptions["embeddingPurpose"].(string); ok && v != "" {
			purpose = v
		}
		switch v := params.ProviderOptions["embeddingDimension"].(type) {
		case int:
			dimension = v
		case float64:
			dimension = int(v)
		}
		if v, ok := params.ProviderOptions["truncationMode"].(string); ok && v != "" {
			truncation = v
		}
	}

	body := map[string]any{
		"schemaVersion": "nova-multimodal-embed-v1",
		"taskType":      "SINGLE_EMBEDDING",
		"singleEmbeddingParams": map[string]any{
			"embeddingPurpose":   purpose,
			"embeddingDimension": dimension,
			"text": map[string]any{
				"truncationMode": truncation,
				"value":          value,
			},
		},
	}

	resp, err := m.invokeModel(ctx, httpc.MustMarshalJSON(body))
	if err != nil {
		return nil, err
	}

	var result struct {
		Embeddings []struct {
			Embedding           []float64 `json:"embedding"`
			TruncatedCharLength int       `json:"truncatedCharLength"`
		} `json:"embeddings"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}
	if len(result.Embeddings) == 0 {
		return nil, errors.New("bedrock: nova returned no embeddings")
	}
	return &provider.EmbedResult{
		Embeddings: [][]float64{result.Embeddings[0].Embedding},
		Response:   provider.ResponseMetadata{Model: m.id},
	}, nil
}

// doMarengoEmbed handles TwelveLabs Marengo models using the text-only path.
// Marengo 2.7: {"inputType":"text","inputText":"...","textTruncate":"end"}
// Marengo 3.0: {"inputType":"text","text":{"inputText":"..."}}
func (m *embeddingModel) doMarengoEmbed(ctx context.Context, value string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	var body map[string]any
	if strings.Contains(m.id, "3-0") {
		body = map[string]any{
			"inputType": "text",
			"text":      map[string]any{"inputText": value},
		}
	} else {
		truncate := "end"
		if params.ProviderOptions != nil {
			if v, ok := params.ProviderOptions["textTruncate"].(string); ok && v != "" {
				truncate = v
			}
		}
		body = map[string]any{
			"inputType":    "text",
			"inputText":    value,
			"textTruncate": truncate,
		}
	}

	resp, err := m.invokeModel(ctx, httpc.MustMarshalJSON(body))
	if err != nil {
		return nil, err
	}

	embedding, err := parseMarengoEmbedding(resp, m.id)
	if err != nil {
		return nil, err
	}
	return &provider.EmbedResult{
		Embeddings: [][]float64{embedding},
		Response:   provider.ResponseMetadata{Model: m.id},
	}, nil
}

// parseMarengoEmbedding extracts the embedding from Marengo response.
// 2.7 returns {"embedding":[...]}, 3.0 returns {"data":{"embedding":[...]}}.
func parseMarengoEmbedding(resp []byte, modelID string) ([]float64, error) {
	if strings.Contains(modelID, "3-0") {
		var result struct {
			Data struct {
				Embedding []float64 `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(resp, &result); err != nil {
			return nil, fmt.Errorf("parsing response: %w", err)
		}
		return result.Data.Embedding, nil
	}
	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}
	return result.Embedding, nil
}

// doCohereEmbed handles Cohere embedding models on Bedrock (batch up to 96).
// Supports v3 (embed-english-v3, embed-multilingual-v3) and v4 (embed-v4:0).
// V4 returns embeddings nested under "embeddings.float" or "embeddings" flat;
// v3 always returns a flat array.
func (m *embeddingModel) doCohereEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	body := map[string]any{
		"texts":      values,
		"input_type": "search_document",
	}

	if params.ProviderOptions != nil {
		for _, key := range []string{"input_type", "truncate", "output_dimension", "embedding_types"} {
			if v, ok := params.ProviderOptions[key]; ok {
				body[key] = v
			}
		}
	}

	resp, err := m.invokeModel(ctx, httpc.MustMarshalJSON(body))
	if err != nil {
		return nil, err
	}

	var raw struct {
		Embeddings json.RawMessage `json:"embeddings"`
	}
	if err := json.Unmarshal(resp, &raw); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}
	embeddings, err := parseCohereEmbeddings(raw.Embeddings)
	if err != nil {
		return nil, fmt.Errorf("parsing embeddings: %w", err)
	}
	return &provider.EmbedResult{
		Embeddings: embeddings,
		Response:   provider.ResponseMetadata{Model: m.id},
	}, nil
}

// parseCohereEmbeddings handles both Cohere v3 (flat array) and v4
// ({"float": [...], "int8": [...], ...}) formats.
func parseCohereEmbeddings(raw json.RawMessage) ([][]float64, error) {
	var flat [][]float64
	if err := json.Unmarshal(raw, &flat); err == nil {
		return flat, nil
	}
	return parseTypedEmbeddings(raw)
}

// parseTypedEmbeddings parses a Bedrock embeddings map keyed by embedding type
// ("float", "int8", "uint8", "binary", "ubinary") — used by Cohere v4
// ({"embeddings": {...}}) and Titan V2 ({"embeddingsByType": {...}}). It returns
// the first present type's vectors, converted to float64.
func parseTypedEmbeddings(raw json.RawMessage) ([][]float64, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, errors.New("bedrock: unrecognised embeddings format")
	}
	for _, key := range []string{"float", "int8", "uint8", "binary", "ubinary"} {
		val, ok := m[key]
		if !ok || len(val) == 0 || string(val) == "null" {
			continue
		}
		switch key {
		case "float":
			var v [][]float64
			if err := json.Unmarshal(val, &v); err != nil {
				return nil, err
			}
			return v, nil
		case "int8":
			var v [][]int8
			if err := json.Unmarshal(val, &v); err != nil {
				return nil, err
			}
			return toFloat64Matrix(v), nil
		case "uint8":
			var v [][]uint8
			if err := json.Unmarshal(val, &v); err != nil {
				return nil, err
			}
			return toFloat64Matrix(v), nil
		case "binary", "ubinary":
			return parseBinaryRows(val)
		}
	}
	return nil, errors.New("bedrock: no embeddings in response (embedding_types may not include \"float\")")
}

// toFloat64Matrix converts an integer matrix to float64, preserving each value.
func toFloat64Matrix[T ~int8 | ~uint8](v [][]T) [][]float64 {
	out := make([][]float64, len(v))
	for i, row := range v {
		out[i] = make([]float64, len(row))
		for j, x := range row {
			out[i][j] = float64(x)
		}
	}
	return out
}

// parseBinaryRows decodes packed-bit binary embeddings. Bedrock returns them as
// either a single base64 string (single input) or an array of base64 strings
// (batched input). Each bit encodes one dimension as 0.0/1.0.
func parseBinaryRows(raw json.RawMessage) ([][]float64, error) {
	var single string
	if err := json.Unmarshal(raw, &single); err == nil && single != "" {
		vec, err := decodeBinaryVector(single)
		if err != nil {
			return nil, err
		}
		return [][]float64{vec}, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, errors.New("bedrock: unrecognised binary embedding format")
	}
	out := make([][]float64, 0, len(arr))
	for _, s := range arr {
		vec, err := decodeBinaryVector(s)
		if err != nil {
			return nil, err
		}
		out = append(out, vec)
	}
	return out, nil
}

// decodeBinaryVector decodes one base64-encoded packed-bit vector into a
// float64 slice of 0.0/1.0 values (one per bit, MSB-first).
func decodeBinaryVector(s string) ([]float64, error) {
	data, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("decoding binary embedding: %w", err)
	}
	vec := make([]float64, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			if b&(1<<uint(i)) != 0 {
				vec = append(vec, 1)
			} else {
				vec = append(vec, 0)
			}
		}
	}
	return vec, nil
}

// invokeModel sends a POST to the Bedrock InvokeModel endpoint and returns the raw response body.
func (m *embeddingModel) invokeModel(ctx context.Context, jsonBody []byte) ([]byte, error) {
	if m.opts.bearerToken == "" && (m.opts.accessKey == "" || m.opts.secretKey == "") {
		return nil, errors.New("bedrock: AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY required")
	}

	region := m.opts.region
	escapedID := url.PathEscape(m.id)

	var reqURL string
	if m.opts.baseURL != "" {
		reqURL = m.opts.baseURL + "/model/" + escapedID + "/invoke"
	} else {
		if !validRegion(region) {
			return nil, fmt.Errorf("bedrock: invalid AWS region %q", region)
		}
		reqURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke",
			region, escapedID)
	}

	req := httpc.MustNewRequest(ctx, "POST", reqURL, jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "goai/amazon-bedrock")

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}

	if m.opts.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+m.opts.bearerToken)
	} else {
		signAWSSigV4(req, jsonBody, m.opts.accessKey, m.opts.secretKey, m.opts.sessionToken, region, "bedrock")
	}

	httpClient := m.opts.httpClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpc.Do(httpClient, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxEmbedErrorBytes))
		return nil, goai.ParseHTTPErrorWithHeaders("bedrock", resp.StatusCode, body, resp.Header)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxEmbedResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	if len(body) > maxEmbedResponseBytes {
		return nil, fmt.Errorf("bedrock: embedding response body exceeds %d bytes", maxEmbedResponseBytes)
	}
	return body, nil
}
