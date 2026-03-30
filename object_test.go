package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// --- Test types ---

type simpleObject struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type nestedObject struct {
	Title  string       `json:"title"`
	Author authorObject `json:"author"`
	Tags   []string     `json:"tags"`
}

type authorObject struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

// --- GenerateObject tests ---

func TestGenerateObject_BasicStruct(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			capturedParams = params
			return &provider.GenerateResult{
				Text:         `{"name":"Alice","age":30}`,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 15, OutputTokens: 8},
			}, nil
		},
	}

	result, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate a person"))
	if err != nil {
		t.Fatal(err)
	}

	if result.Object.Name != "Alice" {
		t.Errorf("Object.Name = %q, want %q", result.Object.Name, "Alice")
	}
	if result.Object.Age != 30 {
		t.Errorf("Object.Age = %d, want 30", result.Object.Age)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.Usage.InputTokens != 15 || result.Usage.OutputTokens != 8 {
		t.Errorf("Usage = %+v, want InputTokens=15, OutputTokens=8", result.Usage)
	}

	// Verify ResponseFormat was set.
	if capturedParams.ResponseFormat == nil {
		t.Fatal("ResponseFormat not set on params")
	}
	if capturedParams.ResponseFormat.Name != "response" {
		t.Errorf("ResponseFormat.Name = %q, want %q", capturedParams.ResponseFormat.Name, "response")
	}
	// Verify schema is valid JSON with expected structure.
	var schema map[string]any
	if err := json.Unmarshal(capturedParams.ResponseFormat.Schema, &schema); err != nil {
		t.Fatalf("ResponseFormat.Schema is not valid JSON: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("schema type = %v, want object", schema["type"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema missing properties")
	}
	if _, ok := props["name"]; !ok {
		t.Error("schema missing 'name' property")
	}
	if _, ok := props["age"]; !ok {
		t.Error("schema missing 'age' property")
	}
}

func TestGenerateObject_NestedStruct(t *testing.T) {
	jsonResp := `{"title":"Go in Action","author":{"name":"Bob","email":"bob@example.com"},"tags":["go","programming"]}`
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         jsonResp,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 15},
			}, nil
		},
	}

	result, err := GenerateObject[nestedObject](t.Context(), model, WithPrompt("generate a book"))
	if err != nil {
		t.Fatal(err)
	}

	if result.Object.Title != "Go in Action" {
		t.Errorf("Object.Title = %q, want %q", result.Object.Title, "Go in Action")
	}
	if result.Object.Author.Name != "Bob" {
		t.Errorf("Object.Author.Name = %q, want %q", result.Object.Author.Name, "Bob")
	}
	if result.Object.Author.Email != "bob@example.com" {
		t.Errorf("Object.Author.Email = %q, want %q", result.Object.Author.Email, "bob@example.com")
	}
	if len(result.Object.Tags) != 2 || result.Object.Tags[0] != "go" || result.Object.Tags[1] != "programming" {
		t.Errorf("Object.Tags = %v, want [go, programming]", result.Object.Tags)
	}
}

func TestGenerateObject_WithExplicitSchema(t *testing.T) {
	customSchema := json.RawMessage(`{"type":"object","properties":{"custom":{"type":"string"}},"required":["custom"]}`)
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			capturedParams = params
			return &provider.GenerateResult{
				Text:         `{"name":"test","age":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithExplicitSchema(customSchema),
	)
	if err != nil {
		t.Fatal(err)
	}

	if capturedParams.ResponseFormat == nil {
		t.Fatal("ResponseFormat not set")
	}
	// The explicit schema should be used, not auto-generated.
	if string(capturedParams.ResponseFormat.Schema) != string(customSchema) {
		t.Errorf("ResponseFormat.Schema = %s, want %s", capturedParams.ResponseFormat.Schema, customSchema)
	}
}

func TestGenerateObject_WithSchemaName(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			capturedParams = params
			return &provider.GenerateResult{
				Text:         `{"name":"test","age":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithSchemaName("person"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if capturedParams.ResponseFormat == nil {
		t.Fatal("ResponseFormat not set")
	}
	if capturedParams.ResponseFormat.Name != "person" {
		t.Errorf("ResponseFormat.Name = %q, want %q", capturedParams.ResponseFormat.Name, "person")
	}
}

func TestGenerateObject_EmptyResponse(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err == nil {
		t.Fatal("expected error for empty response")
	}
	if err.Error() != "goai: empty response from model" {
		t.Errorf("error = %q, want 'goai: empty response from model'", err.Error())
	}
}

func TestGenerateObject_InvalidJSON(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         `{"name": "Alice", "age": not_a_number}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	// Error should contain context about parsing.
	if got := err.Error(); len(got) == 0 {
		t.Error("error message is empty")
	}
	// Should contain "parsing structured output" prefix.
	if got := err.Error(); !strContains(got, "parsing structured output") {
		t.Errorf("error = %q, want it to contain 'parsing structured output'", got)
	}
	// Should contain truncated raw text.
	if got := err.Error(); !strContains(got, "raw:") {
		t.Errorf("error = %q, want it to contain 'raw:'", got)
	}
}

func TestGenerateObject_ModelError(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, &APIError{Message: "internal server error", StatusCode: 500}
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err == nil {
		t.Fatal("expected error from model")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", apiErr.StatusCode)
	}
}

func TestGenerateObject_Hooks(t *testing.T) {
	model := &mockModel{
		id: "test-hooks",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         `{"name":"Alice","age":30}`,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	var requestCalled bool
	var responseCalled bool
	var capturedRequestInfo RequestInfo
	var capturedResponseInfo ResponseInfo

	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithOnRequest(func(info RequestInfo) {
			requestCalled = true
			capturedRequestInfo = info
		}),
		WithOnResponse(func(info ResponseInfo) {
			responseCalled = true
			capturedResponseInfo = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if !requestCalled {
		t.Error("OnRequest hook not called")
	}
	if !responseCalled {
		t.Error("OnResponse hook not called")
	}
	if capturedRequestInfo.Model != "test-hooks" {
		t.Errorf("RequestInfo.Model = %q, want %q", capturedRequestInfo.Model, "test-hooks")
	}
	if capturedRequestInfo.MessageCount != 1 {
		t.Errorf("RequestInfo.MessageCount = %d, want 1", capturedRequestInfo.MessageCount)
	}
	if capturedResponseInfo.Usage.InputTokens != 10 {
		t.Errorf("ResponseInfo.Usage.InputTokens = %d, want 10", capturedResponseInfo.Usage.InputTokens)
	}
	if capturedResponseInfo.FinishReason != provider.FinishStop {
		t.Errorf("ResponseInfo.FinishReason = %q, want %q", capturedResponseInfo.FinishReason, provider.FinishStop)
	}
	if capturedResponseInfo.Error != nil {
		t.Errorf("ResponseInfo.Error = %v, want nil", capturedResponseInfo.Error)
	}
	if capturedResponseInfo.Latency <= 0 {
		t.Error("ResponseInfo.Latency should be positive")
	}
}

func TestGenerateObject_Hooks_OnError(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, &APIError{Message: "bad request", StatusCode: 400}
		},
	}

	var capturedResponseInfo ResponseInfo
	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithOnResponse(func(info ResponseInfo) {
			capturedResponseInfo = info
		}),
	)
	if err == nil {
		t.Fatal("expected error")
	}

	if capturedResponseInfo.Error == nil {
		t.Error("ResponseInfo.Error should be non-nil on error")
	}
	if capturedResponseInfo.StatusCode != 400 {
		t.Errorf("ResponseInfo.StatusCode = %d, want 400", capturedResponseInfo.StatusCode)
	}
}

func TestGenerateObject_RetryOnTransientError(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return nil, &APIError{Message: "rate limited", StatusCode: 429, IsRetryable: true}
			}
			return &provider.GenerateResult{
				Text:         `{"name":"Alice","age":30}`,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	result, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxRetries(2),
	)
	if err != nil {
		t.Fatal(err)
	}

	if callCount != 2 {
		t.Errorf("model called %d times, want 2", callCount)
	}
	if result.Object.Name != "Alice" {
		t.Errorf("Object.Name = %q, want %q", result.Object.Name, "Alice")
	}
}

func TestGenerateObject_DefaultSchemaName(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			capturedParams = params
			return &provider.GenerateResult{
				Text:         `{"name":"test","age":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	// Default schema name should be "response".
	if capturedParams.ResponseFormat.Name != "response" {
		t.Errorf("default ResponseFormat.Name = %q, want %q", capturedParams.ResponseFormat.Name, "response")
	}
}

func TestGenerateObject_ParamsPassThrough(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			capturedParams = params
			return &provider.GenerateResult{
				Text:         `{"name":"test","age":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithSystem("be helpful"),
		WithMaxOutputTokens(200),
		WithTemperature(0.3),
	)
	if err != nil {
		t.Fatal(err)
	}

	if capturedParams.System != "be helpful" {
		t.Errorf("System = %q, want %q", capturedParams.System, "be helpful")
	}
	if capturedParams.MaxOutputTokens != 200 {
		t.Errorf("MaxOutputTokens = %d, want 200", capturedParams.MaxOutputTokens)
	}
	if capturedParams.Temperature == nil || *capturedParams.Temperature != 0.3 {
		t.Error("Temperature not passed through")
	}
}

// --- StreamObject tests ---

func TestStreamObject_PartialObjectStream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			// Verify ResponseFormat is set.
			if params.ResponseFormat == nil {
				t.Error("ResponseFormat not set on stream params")
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name"`},
				provider.StreamChunk{Type: provider.ChunkText, Text: `:"Ali`},
				provider.StreamChunk{Type: provider.ChunkText, Text: `ce","age"`},
				provider.StreamChunk{Type: provider.ChunkText, Text: `:30}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 8}},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate a person"))
	if err != nil {
		t.Fatal(err)
	}

	var partials []*simpleObject
	for partial := range stream.PartialObjectStream() {
		partials = append(partials, partial)
	}

	// Should have at least one partial object emitted.
	if len(partials) == 0 {
		t.Fatal("expected at least one partial object")
	}

	// Last partial should have the complete data.
	last := partials[len(partials)-1]
	if last.Name != "Alice" {
		t.Errorf("last partial Name = %q, want %q", last.Name, "Alice")
	}
	if last.Age != 30 {
		t.Errorf("last partial Age = %d, want 30", last.Age)
	}
}

func TestStreamObject_Result(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Bob","age":25}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 12, OutputTokens: 6}},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	// Consume partials first.
	for range stream.PartialObjectStream() {
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "Bob" {
		t.Errorf("Object.Name = %q, want %q", result.Object.Name, "Bob")
	}
	if result.Object.Age != 25 {
		t.Errorf("Object.Age = %d, want 25", result.Object.Age)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.Usage.InputTokens != 12 || result.Usage.OutputTokens != 6 {
		t.Errorf("Usage = %+v", result.Usage)
	}
}

func TestStreamObject_ResultWithoutConsumingPartials(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Carol","age":40}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 8, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	// Call Result() directly without consuming PartialObjectStream.
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "Carol" {
		t.Errorf("Object.Name = %q, want %q", result.Object.Name, "Carol")
	}
	if result.Object.Age != 40 {
		t.Errorf("Object.Age = %d, want 40", result.Object.Age)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.Usage.InputTokens != 8 {
		t.Errorf("Usage.InputTokens = %d, want 8", result.Usage.InputTokens)
	}
}

func TestStreamObject_NestedStruct(t *testing.T) {
	jsonResp := `{"title":"Go Book","author":{"name":"Dan","email":"dan@test.com"},"tags":["go"]}`
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: jsonResp},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[nestedObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Title != "Go Book" {
		t.Errorf("Object.Title = %q, want %q", result.Object.Title, "Go Book")
	}
	if result.Object.Author.Name != "Dan" {
		t.Errorf("Object.Author.Name = %q, want %q", result.Object.Author.Name, "Dan")
	}
	if len(result.Object.Tags) != 1 || result.Object.Tags[0] != "go" {
		t.Errorf("Object.Tags = %v, want [go]", result.Object.Tags)
	}
}

func TestStreamObject_ErrorFromProvider(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "rate limited", StatusCode: 429, IsRetryable: true}
		},
	}

	_, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err == nil {
		t.Fatal("expected error from provider")
	}
}

func TestStreamObject_EmptyStream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	// Empty stream: finalObject should be nil, so result should have zero-value Object.
	if result.Object.Name != "" || result.Object.Age != 0 {
		t.Errorf("expected zero-value Object for empty stream, got %+v", result.Object)
	}
}

func TestStreamObject_InvalidJSON(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `not valid json at all`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	_, err = stream.Result()
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strContains(err.Error(), "parsing structured output") {
		t.Errorf("error = %q, want it to contain 'parsing structured output'", err.Error())
	}
}

func TestStreamObject_WithSchemaName(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			capturedParams = params
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"test","age":1}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithSchemaName("my_schema"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Result(); err != nil {
		t.Fatal(err)
	}

	if capturedParams.ResponseFormat == nil {
		t.Fatal("ResponseFormat not set")
	}
	if capturedParams.ResponseFormat.Name != "my_schema" {
		t.Errorf("ResponseFormat.Name = %q, want %q", capturedParams.ResponseFormat.Name, "my_schema")
	}
}

func TestStreamObject_WithExplicitSchema(t *testing.T) {
	customSchema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"integer"}}}`)
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			capturedParams = params
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"test","age":1}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithExplicitSchema(customSchema),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Result(); err != nil {
		t.Fatal(err)
	}

	if capturedParams.ResponseFormat == nil {
		t.Fatal("ResponseFormat not set")
	}
	if string(capturedParams.ResponseFormat.Schema) != string(customSchema) {
		t.Errorf("ResponseFormat.Schema = %s, want %s", capturedParams.ResponseFormat.Schema, customSchema)
	}
}

func TestStreamObject_PartialObjectsProgressive(t *testing.T) {
	// Simulate chunks that build up JSON progressively.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":`},
				provider.StreamChunk{Type: provider.ChunkText, Text: `"Eve"`},
				provider.StreamChunk{Type: provider.ChunkText, Text: `,"age":2`},
				provider.StreamChunk{Type: provider.ChunkText, Text: `5}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{OutputTokens: 4}},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	var partials []*simpleObject
	for p := range stream.PartialObjectStream() {
		partials = append(partials, p)
	}

	if len(partials) == 0 {
		t.Fatal("expected partial objects")
	}

	// The final partial should be complete.
	last := partials[len(partials)-1]
	if last.Name != "Eve" {
		t.Errorf("last partial Name = %q, want %q", last.Name, "Eve")
	}
	if last.Age != 25 {
		t.Errorf("last partial Age = %d, want 25", last.Age)
	}

	// Result should also be valid.
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "Eve" {
		t.Errorf("Object.Name = %q, want %q", result.Object.Name, "Eve")
	}
	if result.Usage.OutputTokens != 4 {
		t.Errorf("Usage.OutputTokens = %d, want 4", result.Usage.OutputTokens)
	}
}

func TestStreamObject_ConcurrentResultCalls(t *testing.T) {
	// Ensure multiple calls to Result() are safe (consumeOnce).
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Safe","age":99}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	results := make([]*ObjectResult[simpleObject], 5)
	errs := make([]error, 5)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx], errs[idx] = stream.Result()
		}(i)
	}
	wg.Wait()

	for i, e := range errs {
		if e != nil {
			t.Fatalf("results[%d] error: %v", i, e)
		}
	}
	for i, r := range results {
		if r.Object.Name != "Safe" {
			t.Errorf("results[%d].Object.Name = %q, want %q", i, r.Object.Name, "Safe")
		}
	}
}

func TestStreamObject_ContextCancelDuringPartial(t *testing.T) {
	// Exercise the ctx.Done() path in consume: use an UNBUFFERED partialOut
	// so the first parseable partial blocks the send, then cancel.
	ctx, cancel := context.WithCancel(t.Context())

	source := make(chan provider.StreamChunk, 10)
	source <- provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"A","age":30}`}
	source <- provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop}
	close(source)

	os := &ObjectStream[simpleObject]{
		ctx:    ctx,
		source: source,
		doneCh: make(chan struct{}),
	}

	partialCh := make(chan *simpleObject) // unbuffered - send will block
	done := make(chan struct{})
	go func() {
		os.consume(partialCh)
		close(done)
	}()

	// consume reads the text chunk, parses partial successfully, tries to send
	// to unbuffered partialCh - blocks. Cancel context to unblock via ctx.Done().
	time.Sleep(10 * time.Millisecond) // let consume reach the blocking send
	cancel()
	<-done

	// Verify consume terminated: doneCh should be closed.
	select {
	case <-os.doneCh:
		// expected: consume closed doneCh
	default:
		t.Error("expected doneCh to be closed after consume returns")
	}

	for range partialCh {
	}
}

func TestStreamObject_RetryOnTransientError(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			callCount++
			if callCount == 1 {
				return nil, &APIError{Message: "service unavailable", StatusCode: 503, IsRetryable: true}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Retry","age":1}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxRetries(2),
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if callCount != 2 {
		t.Errorf("model called %d times, want 2", callCount)
	}
	if result.Object.Name != "Retry" {
		t.Errorf("Object.Name = %q, want %q", result.Object.Name, "Retry")
	}
}

func TestStreamObject_DefaultSchemaName(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			capturedParams = params
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"test","age":1}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Result(); err != nil {
		t.Fatal(err)
	}

	if capturedParams.ResponseFormat.Name != "response" {
		t.Errorf("default ResponseFormat.Name = %q, want %q", capturedParams.ResponseFormat.Name, "response")
	}
}

// --- Helper ---

func strContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// --- truncate tests ---

func TestTruncate_StringWithEllipsis(t *testing.T) {
	if got := truncate("hello", 10); got != "hello" {
		t.Errorf("truncate short = %q, want %q", got, "hello")
	}
	if got := truncate("hello world", 5); got != "hello..." {
		t.Errorf("truncate long = %q, want %q", got, "hello...")
	}
	if got := truncate("", 5); got != "" {
		t.Errorf("truncate empty = %q, want empty", got)
	}
}

// --- Verify model receives ResponseFormat in GenerateObject ---

func TestGenerateObject_ResponseFormatSchema(t *testing.T) {
	// Verify the auto-generated schema contains the expected fields for simpleObject.
	var capturedSchema json.RawMessage
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			if params.ResponseFormat != nil {
				capturedSchema = params.ResponseFormat.Schema
			}
			return &provider.GenerateResult{
				Text:         `{"name":"test","age":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}

	if capturedSchema == nil {
		t.Fatal("schema not captured")
	}

	var schema map[string]any
	if err := json.Unmarshal(capturedSchema, &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}

	// Should be strict mode: additionalProperties = false.
	if ap, ok := schema["additionalProperties"]; !ok || ap != false {
		t.Errorf("expected additionalProperties=false, got %v", schema["additionalProperties"])
	}

	// Should have required fields.
	required, ok := schema["required"].([]any)
	if !ok {
		t.Fatal("schema missing required array")
	}

	requiredNames := make(map[string]bool)
	for _, r := range required {
		requiredNames[fmt.Sprintf("%v", r)] = true
	}
	if !requiredNames["name"] {
		t.Error("'name' not in required")
	}
	if !requiredNames["age"] {
		t.Error("'age' not in required")
	}
}

func TestGenerateObject_WithTimeout(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         `{"name":"test","age":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}
	result, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("test"),
		WithTimeout(5*time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "test" {
		t.Errorf("got name %q, want %q", result.Object.Name, "test")
	}
}

func TestGenerateObject_OnResponseWithAPIError(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: false}
		},
	}
	var statusCode int
	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("test"),
		WithMaxRetries(0),
		WithOnResponse(func(info ResponseInfo) {
			statusCode = info.StatusCode
		}),
	)
	if err == nil {
		t.Fatal("expected error")
	}
	if statusCode != 429 {
		t.Errorf("statusCode = %d, want 429", statusCode)
	}
}

func TestStreamObject_Error_OnResponseFired(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "service unavailable", StatusCode: 503, IsRetryable: false}
		},
	}

	var capturedResponse ResponseInfo
	_, err := StreamObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxRetries(0),
		WithOnResponse(func(info ResponseInfo) {
			capturedResponse = info
		}),
	)
	if err == nil {
		t.Fatal("expected error from StreamObject")
	}
	if capturedResponse.Error == nil {
		t.Error("OnResponse not called or ResponseInfo.Error is nil")
	}
	if capturedResponse.StatusCode != 503 {
		t.Errorf("ResponseInfo.StatusCode = %d, want 503", capturedResponse.StatusCode)
	}
}

func TestStreamObject_WithTimeout(t *testing.T) {
	ch := make(chan provider.StreamChunk, 3)
	ch <- provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"hi","age":5}`}
	ch <- provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop}
	close(ch)

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return &provider.StreamResult{Stream: ch}, nil
		},
	}
	stream, err := StreamObject[simpleObject](t.Context(), model,
		WithPrompt("test"),
		WithTimeout(5*time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "hi" {
		t.Errorf("got name %q, want %q", result.Object.Name, "hi")
	}
}

func TestStreamObject_WithTimeout_ErrorCancelsContext(t *testing.T) {
	// StreamObject with Timeout: when DoStream fails, timeoutCancel must be called.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "bad request", StatusCode: 400}
		},
	}

	_, err := StreamObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithTimeout(5*time.Second),
	)
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", apiErr.StatusCode)
	}
}

func TestGenerateObject_InvalidJSONTruncated(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         `{"name": not valid json at all`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}
	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("test"),
	)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strContains(err.Error(), "parsing structured output") {
		t.Errorf("error = %q, want 'parsing structured output' prefix", err.Error())
	}
}

func TestObjectStream_PartialObjectStreamAfterResult(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Alice","age":30}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("test"))
	if err != nil {
		t.Fatal(err)
	}

	// Call Result() first -- this consumes the source.
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "Alice" {
		t.Errorf("Object.Name = %q, want Alice", result.Object.Name)
	}

	// Now call PartialObjectStream() -- should return a closed channel (0 items).
	ch := stream.PartialObjectStream()
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 items from PartialObjectStream after Result(), got %d", count)
	}
}

func TestStreamObject_ErrNilOnSuccess(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Alice","age":30}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}
	type Person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	stream, err := StreamObject[Person](context.Background(), model, WithPrompt("test"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if stream.Err() != nil {
		t.Errorf("expected nil Err(), got %v", stream.Err())
	}
}

func TestStreamObject_ErrReturnsStreamError(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Al`},
				provider.StreamChunk{Type: provider.ChunkError, Error: errors.New("stream broke")},
			), nil
		},
	}
	type Person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	stream, err := StreamObject[Person](context.Background(), model, WithPrompt("test"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = stream.Result()
	if stream.Err() == nil {
		t.Fatal("expected non-nil Err()")
	}
	if stream.Err().Error() != "stream broke" {
		t.Errorf("unexpected error: %v", stream.Err())
	}
}

// --- GenerateObject tool loop tests ---

// TestGenerateObject_ToolLoop verifies that GenerateObject runs intermediate tool
// steps and uses the results in the final structured-output step.
func TestGenerateObject_ToolLoop(t *testing.T) {
	callCount := 0
	var capturedMessages [][]provider.Message // messages at each DoGenerate call

	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			capturedMessages = append(capturedMessages, params.Messages)

			switch callCount {
			case 1:
				// Tool step: model requests a tool call.
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "call_1", Name: "get_age", Input: json.RawMessage(`{"name":"Alice"}`)},
					},
					Usage: provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			case 2:
				// Final step: structured output after tool results are in context.
				return &provider.GenerateResult{
					Text:         `{"name":"Alice","age":30}`,
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
				}, nil
			default:
				t.Fatalf("unexpected call count %d", callCount)
				return nil, nil
			}
		},
	}

	toolExecuted := false
	result, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate a person"),
		WithMaxSteps(2), // 1 tool step + 1 final structured-output step
		WithTools(Tool{
			Name:        "get_age",
			Description: "get age for a person",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				toolExecuted = true
				return "30", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if !toolExecuted {
		t.Error("tool was not executed")
	}
	if callCount != 2 {
		t.Errorf("DoGenerate called %d times, want 2", callCount)
	}
	if result.Object.Name != "Alice" || result.Object.Age != 30 {
		t.Errorf("Object = %+v, want {Alice 30}", result.Object)
	}

	// Usage should be the sum of both steps.
	if result.Usage.InputTokens != 30 || result.Usage.OutputTokens != 13 {
		t.Errorf("Usage = %+v, want InputTokens=30 OutputTokens=13", result.Usage)
	}

	// Steps: 1 tool step + 1 final step.
	if len(result.Steps) != 2 {
		t.Fatalf("Steps len = %d, want 2", len(result.Steps))
	}
	if result.Steps[0].Number != 1 || result.Steps[0].FinishReason != provider.FinishToolCalls {
		t.Errorf("Steps[0] = %+v", result.Steps[0])
	}
	if result.Steps[1].Number != 2 || result.Steps[1].FinishReason != provider.FinishStop {
		t.Errorf("Steps[1] = %+v", result.Steps[1])
	}

	// Tool results must be present in the second call's messages.
	if len(capturedMessages) != 2 {
		t.Fatalf("capturedMessages len = %d, want 2", len(capturedMessages))
	}
	lastMsgs := capturedMessages[1]
	var foundToolResult bool
	for _, msg := range lastMsgs {
		for _, part := range msg.Content {
			if part.Type == provider.PartToolResult {
				foundToolResult = true
			}
		}
	}
	if !foundToolResult {
		t.Error("tool result not present in final step messages")
	}

	// ResponseFormat is set on every call — the mock returns FinishToolCalls on
	// step 1 because real providers call tools even when ResponseFormat is present,
	// deferring JSON output until they have enough information.
}

// TestGenerateObject_ToolLoop_MaxSteps1_NoLoop verifies that MaxSteps=1 (the
// default) skips the tool loop entirely and goes straight to structured output.
func TestGenerateObject_ToolLoop_MaxSteps1_NoLoop(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if params.ResponseFormat == nil {
				t.Error("ResponseFormat must be set on the single call when MaxSteps=1")
			}
			return &provider.GenerateResult{
				Text:         `{"name":"Bob","age":25}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	toolExecuted := false
	result, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		// MaxSteps defaults to 1 — no explicit WithMaxSteps needed.
		WithTools(Tool{
			Name:        "unused_tool",
			Description: "never called",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				toolExecuted = true
				return "", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if toolExecuted {
		t.Error("tool should not have been executed with MaxSteps=1")
	}
	if callCount != 1 {
		t.Errorf("DoGenerate called %d times, want 1", callCount)
	}
	if result.Object.Name != "Bob" {
		t.Errorf("Object.Name = %q, want Bob", result.Object.Name)
	}
	if len(result.Steps) != 1 {
		t.Errorf("Steps len = %d, want 1", len(result.Steps))
	}
}

// TestGenerateObject_ToolLoop_StopsEarlyWhenNoToolCalls verifies that the loop
// exits immediately when the model returns finishReason "stop" without calling
// any tools, and that the text from that step is parsed as structured output.
func TestGenerateObject_ToolLoop_StopsEarlyWhenNoToolCalls(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if params.ResponseFormat == nil {
				t.Error("ResponseFormat must be set on every step")
			}
			if callCount > 1 {
				t.Fatalf("unexpected call count %d — should have stopped after step 1", callCount)
			}
			// Model decides it doesn't need tools, returns JSON directly.
			return &provider.GenerateResult{
				Text:         `{"name":"Carol","age":22}`,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5},
			}, nil
		},
	}

	result, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxSteps(5), // high limit — model stops on its own after 1 call
		WithTools(Tool{
			Name:        "some_tool",
			Description: "some tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "result", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if callCount != 1 {
		t.Errorf("DoGenerate called %d times, want 1", callCount)
	}
	if result.Object.Name != "Carol" {
		t.Errorf("Object.Name = %q, want Carol", result.Object.Name)
	}
	if len(result.Steps) != 1 {
		t.Errorf("Steps len = %d, want 1", len(result.Steps))
	}
}

// TestGenerateObject_ToolLoop_MaxStepsExhausted verifies that an error is
// returned when MaxSteps is reached with tool calls still pending.
func TestGenerateObject_ToolLoop_MaxStepsExhausted(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			// Always requests a tool call — never produces stop.
			return &provider.GenerateResult{
				FinishReason: provider.FinishToolCalls,
				ToolCalls:    []provider.ToolCall{{ID: "c", Name: "t", Input: json.RawMessage(`{}`)}},
			}, nil
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "t",
			Description: "t",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute:     func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err == nil {
		t.Fatal("expected error when max steps exhausted")
	}
	if !strings.Contains(err.Error(), "max steps") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestGenerateObject_ToolLoop_ErrorDuringToolStep verifies that an error in an
// intermediate step is propagated immediately.
func TestGenerateObject_ToolLoop_ErrorDuringToolStep(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			return nil, &APIError{Message: "upstream error", StatusCode: 500}
		},
	}

	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxSteps(3),
		WithMaxRetries(0),
		WithTools(Tool{
			Name:        "t",
			Description: "t",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute:     func(_ context.Context, _ json.RawMessage) (string, error) { return "", nil },
		}),
	)
	if err == nil {
		t.Fatal("expected error")
	}
	if callCount != 1 {
		t.Errorf("DoGenerate called %d times, want 1", callCount)
	}
}

// TestGenerateObject_ToolLoop_OnStepFinish verifies that OnStepFinish is called
// for every step including the final structured-output step.
func TestGenerateObject_ToolLoop_OnStepFinish(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls:    []provider.ToolCall{{ID: "c1", Name: "t", Input: json.RawMessage(`{}`)}},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         `{"name":"Dave","age":40}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	var stepNumbers []int
	_, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("generate"),
		WithMaxSteps(2), // 1 tool step + 1 final structured-output step
		WithOnStepFinish(func(s StepResult) { stepNumbers = append(stepNumbers, s.Number) }),
		WithTools(Tool{
			Name:        "t",
			Description: "t",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute:     func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(stepNumbers) != 2 || stepNumbers[0] != 1 || stepNumbers[1] != 2 {
		t.Errorf("stepNumbers = %v, want [1 2]", stepNumbers)
	}
}

// TestGenerateObject_ToolLoop_SingleStepHasOneStep verifies ObjectResult.Steps
// contains exactly one entry for the default single-step case.
func TestGenerateObject_ToolLoop_SingleStepHasOneStep(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         `{"name":"Eve","age":28}`,
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
			}, nil
		},
	}

	result, err := GenerateObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 1 {
		t.Errorf("Steps len = %d, want 1", len(result.Steps))
	}
	if result.Steps[0].Number != 1 {
		t.Errorf("Steps[0].Number = %d, want 1", result.Steps[0].Number)
	}
	if result.Usage.InputTokens != 5 || result.Usage.OutputTokens != 3 {
		t.Errorf("Usage = %+v, want InputTokens=5 OutputTokens=3", result.Usage)
	}
}

// TestGenerateObject_ToolLoop_ToolChoiceRequiredResets verifies that
// WithToolChoice("required") is cleared after the first tool step so the model
// can produce structured output on the next step instead of looping forever.
func TestGenerateObject_ToolLoop_ToolChoiceRequiredResets(t *testing.T) {
	step := 0
	var capturedToolChoices []string
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			step++
			capturedToolChoices = append(capturedToolChoices, params.ToolChoice)
			if step == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "c1", Name: "echo", Input: json.RawMessage(`{"msg":"hi"}`)},
					},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         `{"name":"Bob","age":25}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateObject[simpleObject](t.Context(), model,
		WithPrompt("go"),
		WithTools(Tool{
			Name: "echo", Description: "echo",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "hi", nil },
		}),
		WithToolChoice("required"),
		WithMaxSteps(5),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "Bob" {
		t.Errorf("Object.Name = %q, want Bob", result.Object.Name)
	}
	if len(capturedToolChoices) != 2 {
		t.Fatalf("steps = %d, want 2", len(capturedToolChoices))
	}
	if capturedToolChoices[0] != "required" {
		t.Errorf("step 1 tool_choice = %q, want required", capturedToolChoices[0])
	}
	if capturedToolChoices[1] != "" {
		t.Errorf("step 2 tool_choice = %q, want empty (reset)", capturedToolChoices[1])
	}
}

func TestStreamObject_ProviderMetadata(t *testing.T) {
	wantMeta := map[string]map[string]any{
		"openai": {"logprobs": true},
	}

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Alice","age":30}`},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
					Metadata: map[string]any{
						"providerMetadata": wantMeta,
					},
				},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("generate"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.PartialObjectStream() {
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil, want non-nil")
	}
	openaiMeta, ok := result.ProviderMetadata["openai"]
	if !ok {
		t.Fatal("missing 'openai' key in ProviderMetadata")
	}
	if openaiMeta["logprobs"] != true {
		t.Error("missing 'logprobs' in openai ProviderMetadata")
	}
}

func TestStreamObject_ProviderMetadata_ResponseLevel(t *testing.T) {
	// Anthropic convention: flat metadata keys go into Response.ProviderMetadata.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Eve","age":28}`},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Metadata: map[string]any{
						"iterations": 2,
						"container":  "xyz",
					},
				},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("gen"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.PartialObjectStream() {
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Response.ProviderMetadata == nil {
		t.Fatal("Response.ProviderMetadata is nil")
	}
	if result.Response.ProviderMetadata["iterations"] != 2 {
		t.Errorf("iterations = %v, want 2", result.Response.ProviderMetadata["iterations"])
	}
	if result.Response.ProviderMetadata["container"] != "xyz" {
		t.Errorf("container = %v, want xyz", result.Response.ProviderMetadata["container"])
	}
}

func TestStreamObject_ProviderMetadata_BothConventions(t *testing.T) {
	// Both nested (OpenAI) and flat (Anthropic) conventions in a single chunk.
	wantNested := map[string]map[string]any{
		"openai": {"logprobs": true},
	}

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: `{"name":"Zoe","age":22}`},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Metadata: map[string]any{
						"providerMetadata": wantNested,
						"iterations":       7,
					},
				},
			), nil
		},
	}

	stream, err := StreamObject[simpleObject](t.Context(), model, WithPrompt("gen"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.PartialObjectStream() {
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}

	// Nested convention → ObjectResult.ProviderMetadata
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	if _, ok := result.ProviderMetadata["openai"]; !ok {
		t.Error("missing 'openai' key in ProviderMetadata")
	}

	// Flat convention → Response.ProviderMetadata
	if result.Response.ProviderMetadata == nil {
		t.Fatal("Response.ProviderMetadata is nil")
	}
	if result.Response.ProviderMetadata["iterations"] != 7 {
		t.Errorf("iterations = %v, want 7", result.Response.ProviderMetadata["iterations"])
	}

	// "providerMetadata" key must NOT leak into flat.
	if _, ok := result.Response.ProviderMetadata["providerMetadata"]; ok {
		t.Error("providerMetadata key should not leak into Response.ProviderMetadata")
	}
}
