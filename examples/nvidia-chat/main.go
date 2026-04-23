//go:build ignore

// Example: NVIDIA NIM streaming chat with GoAI.
//
// Usage:
//
//	export NVIDIA_API_KEY=...
//	go run examples/nvidia-chat/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/nvidia"
)

func main() {
	model := nvidia.Chat("meta/llama-3.3-70b-instruct",
		nvidia.WithAPIKey(os.Getenv("NVIDIA_API_KEY")),
	)

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithSystem("You are a helpful assistant."),
		goai.WithPrompt("What is the capital of France?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("=== Non-streaming ===")
	fmt.Println(result.Text)
	fmt.Printf("Tokens: %d in, %d out\n\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)

	fmt.Println("=== Streaming ===")
	stream, err := goai.StreamText(context.Background(), model,
		goai.WithSystem("You are a helpful assistant."),
		goai.WithPrompt("Count from 1 to 5."),
	)
	if err != nil {
		log.Fatal(err)
	}

	for text := range stream.TextStream() {
		fmt.Print(text)
	}
	fmt.Println()

	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}

	result = stream.Result()
	fmt.Printf("Tokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}