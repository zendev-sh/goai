//go:build ignore

// Example: multi-step agent loop with tool calls and step callbacks.
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run examples/agent-loop/main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	model := google.Chat("gemini-2.0-flash", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

	// Define multiple tools for a multi-step workflow.
	searchTool := goai.Tool{
		Name:        "search",
		Description: "Search for information on a topic.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {"type": "string", "description": "Search query"}
			},
			"required": ["query"]
		}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct{ Query string }
			_ = json.Unmarshal(input, &args)
			return fmt.Sprintf("Search results for %q: Go was created by Google in 2009.", args.Query), nil
		},
	}

	calculatorTool := goai.Tool{
		Name:        "calculate",
		Description: "Calculate: subtract second number from first. Input: {\"a\": 2026, \"b\": 2009}",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"a": {"type": "integer", "description": "First number"},
				"b": {"type": "integer", "description": "Second number"}
			},
			"required": ["a", "b"]
		}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct{ A, B int }
			_ = json.Unmarshal(input, &args)
			return fmt.Sprintf("%d", args.A-args.B), nil
		},
	}

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithSystem("You are a research assistant. Use tools to find information and calculate."),
		goai.WithPrompt("How old is the Go programming language? Calculate 2026 minus the year it was created."),
		goai.WithTools(searchTool, calculatorTool),
		goai.WithMaxSteps(5),
		goai.WithOnStepFinish(func(step goai.StepResult) {
			fmt.Printf("--- Step %d (finish: %s, tools: %d) ---\n",
				step.Number, step.FinishReason, len(step.ToolCalls))
		}),
		goai.WithOnToolCall(func(info goai.ToolCallInfo) {
			fmt.Printf("  Tool: %s (%d bytes input)\n", info.ToolName, len(info.Input))
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n=== Final Answer ===")
	fmt.Println(result.Text)
	fmt.Printf("Total steps: %d, Tokens: %d in, %d out\n",
		len(result.Steps), result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
