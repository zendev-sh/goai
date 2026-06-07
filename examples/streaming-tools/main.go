//go:build ignore

// Example: streaming text generation with multi-step tool loops.
//
// Usage:
//
//	export OPENAI_API_KEY=...
//	go run examples/streaming-tools/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/openai"
)

func main() {
	model := openai.Chat("gpt-4o-mini", openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

	weatherTool := goai.NewTool("get_weather", "Get the current weather for a city.",
		func(_ context.Context, args struct {
			City string `json:"city" jsonschema:"description=City name"`
		}) (string, error) {
			return fmt.Sprintf(`{"city": %q, "temp": "18°C", "condition": "partly cloudy"}`, args.City), nil
		})

	stream, err := goai.StreamText(context.Background(), model,
		goai.WithSystem("You are a helpful weather assistant. Be concise."),
		goai.WithPrompt("What's the weather in Tokyo and Paris?"),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(5),
		goai.WithOnToolCallStart(func(info goai.ToolCallStartInfo) {
			fmt.Printf("  [running %s...]\n", info.ToolName)
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Consume raw chunks to observe every event in the multi-step loop.
	for chunk := range stream.Stream() {
		switch chunk.Type {
		case provider.ChunkText:
			fmt.Print(chunk.Text)
		case provider.ChunkToolCall:
			fmt.Printf("\n  [calling %s]\n", chunk.ToolName)
		case provider.ChunkStepFinish:
			fmt.Printf("  [step done: %s]\n", chunk.FinishReason)
		case provider.ChunkFinish:
			fmt.Printf("\n  [finish: %d in, %d out]\n", chunk.Usage.InputTokens, chunk.Usage.OutputTokens)
		}
	}

	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}

	// Final result with aggregated steps and usage.
	result := stream.Result()
	fmt.Printf("\nTotal steps: %d, Tokens: %d in, %d out\n",
		len(result.Steps), result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
