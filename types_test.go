package goai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestNewTool_SchemaAndExecute(t *testing.T) {
	type weatherIn struct {
		City string `json:"city" jsonschema:"description=City name"`
		Days int    `json:"days"`
	}

	var gotCity string
	var gotDays int
	tool := NewTool("get_weather", "Get the weather",
		func(_ context.Context, in weatherIn) (string, error) {
			gotCity = in.City
			gotDays = in.Days
			return "sunny in " + in.City, nil
		})

	if tool.Name != "get_weather" {
		t.Errorf("Name = %q, want get_weather", tool.Name)
	}
	if tool.Description != "Get the weather" {
		t.Errorf("Description = %q", tool.Description)
	}

	// InputSchema is generated from the struct.
	want := SchemaFrom[weatherIn]()
	if string(tool.InputSchema) != string(want) {
		t.Errorf("InputSchema = %s, want %s", tool.InputSchema, want)
	}
	if !strings.Contains(string(tool.InputSchema), `"city"`) {
		t.Errorf("InputSchema missing city property: %s", tool.InputSchema)
	}

	// Execute unmarshals raw JSON into the typed struct.
	out, err := tool.Execute(context.Background(), json.RawMessage(`{"city":"NYC","days":3}`))
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if out != "sunny in NYC" {
		t.Errorf("output = %q, want 'sunny in NYC'", out)
	}
	if gotCity != "NYC" || gotDays != 3 {
		t.Errorf("decoded input = {%q, %d}, want {NYC, 3}", gotCity, gotDays)
	}
}

func TestNewTool_InvalidInput(t *testing.T) {
	called := false
	tool := NewTool("t", "d", func(_ context.Context, _ struct {
		N int `json:"n"`
	}) (string, error) {
		called = true
		return "ok", nil
	})

	// "n" expects a number; a string fails to unmarshal.
	out, err := tool.Execute(context.Background(), json.RawMessage(`{"n":"not-a-number"}`))
	if err == nil {
		t.Fatal("expected error for invalid input, got nil")
	}
	if called {
		t.Error("execute must not run when input fails to unmarshal")
	}
	if out != "" {
		t.Errorf("output = %q, want empty", out)
	}
	if !strings.Contains(err.Error(), `tool "t"`) || !strings.Contains(err.Error(), "invalid input") {
		t.Errorf("error = %q, want it to name the tool and 'invalid input'", err)
	}
}

func TestNewTool_EmptyInput(t *testing.T) {
	// A no-parameter tool: empty/zero args leave In at its zero value and
	// execute still runs.
	for _, raw := range []json.RawMessage{nil, json.RawMessage(""), json.RawMessage("{}")} {
		ran := false
		tool := NewTool("noop", "no params", func(_ context.Context, _ struct{}) (string, error) {
			ran = true
			return "done", nil
		})
		out, err := tool.Execute(context.Background(), raw)
		if err != nil {
			t.Fatalf("raw=%q: unexpected error: %v", raw, err)
		}
		if !ran || out != "done" {
			t.Errorf("raw=%q: ran=%v out=%q, want true/done", raw, ran, out)
		}
	}
}

func TestNewTool_ExecuteErrorPropagates(t *testing.T) {
	sentinel := "boom from tool"
	tool := NewTool("t", "d", func(_ context.Context, _ struct{}) (string, error) {
		return "", &simpleErr{sentinel}
	})
	_, err := tool.Execute(context.Background(), json.RawMessage(`{}`))
	if err == nil || err.Error() != sentinel {
		t.Fatalf("err = %v, want %q (execute error forwarded verbatim)", err, sentinel)
	}
}

type simpleErr struct{ s string }

func (e *simpleErr) Error() string { return e.s }

func TestNewTool_WorksInGenerateLoop(t *testing.T) {
	// End-to-end: NewTool plugs into the auto tool loop like a hand-built Tool.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "echo", Input: json.RawMessage(`{"msg":"hi"}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	var seen string
	tool := NewTool("echo", "echo a message", func(_ context.Context, in struct {
		Msg string `json:"msg"`
	}) (string, error) {
		seen = in.Msg
		return "echoed: " + in.Msg, nil
	})

	res, err := GenerateText(t.Context(), model, WithPrompt("go"), WithMaxSteps(3), WithTools(tool))
	if err != nil {
		t.Fatal(err)
	}
	if seen != "hi" {
		t.Errorf("tool saw msg=%q, want hi", seen)
	}
	if res.Text != "done" {
		t.Errorf("Text = %q, want done", res.Text)
	}
}
