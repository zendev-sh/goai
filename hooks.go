package goai

import (
	"encoding/json"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// RequestInfo is passed to the OnRequest hook before a generation call.
type RequestInfo struct {
	// Model is the model ID.
	Model string

	// MessageCount is the number of messages in the request.
	MessageCount int

	// ToolCount is the number of tools available.
	ToolCount int

	// Timestamp is when the request was initiated.
	Timestamp time.Time
}

// ResponseInfo is passed to the OnResponse hook after a generation call completes.
type ResponseInfo struct {
	// Latency is the time from request to response.
	Latency time.Duration

	// Usage is the token consumption for this call.
	Usage provider.Usage

	// FinishReason indicates why generation stopped.
	FinishReason provider.FinishReason

	// Error is non-nil if the call failed.
	Error error

	// StatusCode is the HTTP status code (0 if not applicable).
	StatusCode int
}

// ToolCallInfo is passed to the OnToolCall hook after a tool executes.
type ToolCallInfo struct {
	// ToolName is the name of the tool that was called.
	ToolName string

	// Input is the raw JSON arguments passed to the tool.
	Input json.RawMessage

	// Output is the string result returned by the tool.
	// Empty if the tool returned an error or was not found.
	Output string

	// Duration is how long the tool execution took.
	Duration time.Duration

	// Error is non-nil if the tool execution failed.
	Error error
}

// WithOnStepFinish sets a callback invoked after each generation step completes.
func WithOnStepFinish(fn func(StepResult)) Option {
	return func(o *options) { o.OnStepFinish = fn }
}

// WithOnRequest sets a callback invoked before each model call.
func WithOnRequest(fn func(RequestInfo)) Option {
	return func(o *options) { o.OnRequest = fn }
}

// WithOnResponse sets a callback invoked after each model call completes.
func WithOnResponse(fn func(ResponseInfo)) Option {
	return func(o *options) { o.OnResponse = fn }
}

// WithOnToolCall sets a callback invoked after each tool execution.
func WithOnToolCall(fn func(ToolCallInfo)) Option {
	return func(o *options) { o.OnToolCall = fn }
}
