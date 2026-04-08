//go:build ignore

// Example: multi-turn conversation with tools using ResponseMessages.
//
// ResponseMessages captures the full assistant + tool message history from each
// generation call. Append it to your conversation to preserve tool call context
// across turns, preventing the model from re-executing tools unnecessarily.
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run examples/multi-turn/main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	model := google.Chat("gemini-2.0-flash", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

	weatherTool := goai.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"city": {"type": "string", "description": "City name"}
			},
			"required": ["city"]
		}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct{ City string }
			if err := json.Unmarshal(input, &args); err != nil {
				return "", err
			}
			return fmt.Sprintf("Weather in %s: 22C, sunny", args.City), nil
		},
	}

	ctx := context.Background()
	var messages []provider.Message

	// Turn 1: user asks about weather (triggers tool call).
	messages = append(messages, goai.UserMessage("What's the weather in Tokyo?"))

	result, err := goai.GenerateText(ctx, model,
		goai.WithSystem("You are a helpful assistant. Be concise."),
		goai.WithMessages(messages...),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(5),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Turn 1: %s\n", result.Text)
	fmt.Printf("  Steps: %d, ResponseMessages: %d\n", len(result.Steps), len(result.ResponseMessages))

	// Append ResponseMessages to preserve the full tool call history.
	messages = append(messages, result.ResponseMessages...)

	// Turn 2: user asks a follow-up (should NOT re-execute the weather tool).
	messages = append(messages, goai.UserMessage("Thanks! Is that warm enough for a picnic?"))

	result, err = goai.GenerateText(ctx, model,
		goai.WithSystem("You are a helpful assistant. Be concise."),
		goai.WithMessages(messages...),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(5),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Turn 2: %s\n", result.Text)
	fmt.Printf("  Steps: %d (should be 1 - no tool re-execution)\n", len(result.Steps))

	// Append again for further turns.
	messages = append(messages, result.ResponseMessages...)

	fmt.Printf("\nTotal messages in history: %d\n", len(messages))
}
