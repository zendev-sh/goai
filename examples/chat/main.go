//go:build ignore

// Example: simple non-streaming text generation with GoAI.
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run examples/chat/main.go
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
	model := google.Chat("gemini-3-flash-preview", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithSystem("You are a helpful assistant. Be concise."),
		goai.WithPrompt("What is the capital of France?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
	fmt.Printf("Tokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
