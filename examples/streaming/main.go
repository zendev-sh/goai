//go:build ignore

// Example: streaming text generation with GoAI.
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run examples/streaming/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	model := google.Chat("gemini-2.0-flash", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

	stream, err := goai.StreamText(context.Background(), model,
		goai.WithSystem("You are a helpful assistant."),
		goai.WithPrompt("Write a haiku about Go programming."),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Stream text tokens as they arrive.
	for text := range stream.TextStream() {
		fmt.Print(text)
	}
	fmt.Println()

	// Check for errors that occurred during streaming.
	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}

	// Get the final result with usage stats.
	result := stream.Result()
	fmt.Printf("Tokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
