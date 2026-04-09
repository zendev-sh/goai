//go:build ignore

// Example: lifecycle hooks for tool interception, output transformation, and loop control.
//
// Demonstrates the three interceptor hooks:
//   - OnBeforeToolExecute: permission gate (skip dangerous tools)
//   - OnAfterToolExecute: secret scanning (redact sensitive output)
//   - OnBeforeStep: inject context between tool loop steps
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run examples/hooks/main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	model := google.Chat("gemini-2.0-flash", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

	// Define tools.
	readFile := goai.Tool{
		Name:        "read_file",
		Description: "Read the contents of a file.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path to read"}
			},
			"required": ["path"]
		}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct {
				Path string `json:"path"`
			}
			json.Unmarshal(input, &args)
			// Simulate returning sensitive content.
			return fmt.Sprintf("Contents of %s:\nAPI_KEY=sk-secret-12345\nDB_HOST=prod.example.com", args.Path), nil
		},
	}

	deleteFile := goai.Tool{
		Name:        "delete_file",
		Description: "Delete a file from disk.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path to delete"}
			},
			"required": ["path"]
		}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			return "deleted", nil
		},
	}

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithPrompt("Read the .env file, then delete the temp.log file."),
		goai.WithMaxSteps(5),
		goai.WithTools(readFile, deleteFile),

		// Hook 1: Permission gate -- block dangerous tools.
		// info.Ctx carries the tool execution context (with tool call ID).
		goai.WithOnBeforeToolExecute(func(info goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
			// Use info.Ctx for cancellation-aware work (e.g., spawning child agents).
			if info.Ctx.Err() != nil {
				return goai.BeforeToolExecuteResult{Skip: true, Error: info.Ctx.Err()}
			}
			if info.ToolName == "delete_file" {
				fmt.Printf("[PERMISSION] Blocked: %s (call %s)\n", info.ToolName, goai.ToolCallIDFromContext(info.Ctx))
				return goai.BeforeToolExecuteResult{
					Skip:   true,
					Result: "Permission denied: delete_file is not allowed in this session.",
				}
			}
			fmt.Printf("[PERMISSION] Allowed: %s\n", info.ToolName)
			// Can also override context or input before Execute:
			//   return goai.BeforeToolExecuteResult{Ctx: customCtx, Input: rewrittenInput}
			return goai.BeforeToolExecuteResult{} // proceed normally
		}),

		// Hook 2: Secret scanning --redact sensitive patterns in tool output.
		goai.WithOnAfterToolExecute(func(info goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
			if info.Error != nil {
				return goai.AfterToolExecuteResult{} // don't modify errors
			}
			redacted := info.Output
			// Simple redaction: mask anything that looks like a secret key.
			if strings.Contains(redacted, "sk-") {
				redacted = strings.ReplaceAll(redacted, "sk-secret-12345", "sk-***REDACTED***")
				fmt.Printf("[REDACT] Masked secret in %s output\n", info.ToolName)
			}
			return goai.AfterToolExecuteResult{
				Output: redacted,
				// Metadata flows to OnToolCall for observability (e.g., display title).
				Metadata: map[string]any{"redacted": true},
			}
		}),

		// Hook 3: Loop control -- fires before step 2+ only (not step 1).
		goai.WithOnBeforeStep(func(info goai.BeforeStepInfo) goai.BeforeStepResult {
			fmt.Printf("[STEP] Before step %d (2+ only) with %d messages\n", info.Step, len(info.Messages))
			// info.Ctx carries the generation context -- use for cancellation checks.
			if info.Ctx.Err() != nil {
				return goai.BeforeStepResult{Stop: true}
			}
			return goai.BeforeStepResult{}
		}),

		// Observability hooks (existing) --log tool execution.
		goai.WithOnToolCall(func(info goai.ToolCallInfo) {
			status := "OK"
			if info.Skipped {
				status = "SKIPPED"
			} else if info.Error != nil {
				status = "ERROR"
			}
			fmt.Printf("[TOOL] %s: %s (step %d, %s)\n", info.ToolName, status, info.Step, info.Duration)
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\n--- Result ---\n%s\n", result.Text)
	fmt.Printf("Steps: %d, Tokens: %d in / %d out\n",
		len(result.Steps), result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
