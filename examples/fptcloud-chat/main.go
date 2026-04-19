//go:build ignore

// Example: FPT Smart Cloud AI Marketplace non-streaming chat.
//
// Usage:
//
//	export FPT_API_KEY=...
//	# Optional: export FPT_REGION=jp    # default is "global"
//	go run examples/fptcloud-chat/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/fptcloud"
)

func main() {
	model := fptcloud.Chat("Qwen3-32B")

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithSystem("You are a helpful assistant. Be concise."),
		goai.WithPrompt("What is the capital of Vietnam?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
	fmt.Printf("Tokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
