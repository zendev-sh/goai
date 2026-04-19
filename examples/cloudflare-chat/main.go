//go:build ignore

// Example: Cloudflare Workers AI non-streaming chat.
//
// Usage:
//
//	export CLOUDFLARE_API_TOKEN=...
//	export CLOUDFLARE_ACCOUNT_ID=...
//	go run examples/cloudflare-chat/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/cloudflare"
)

func main() {
	model := cloudflare.Chat("@cf/meta/llama-3.1-8b-instruct")

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
