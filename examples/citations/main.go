//go:build ignore

// Example: accessing citations and sources from AI responses.
//
// Sources are populated when the provider returns citations:
//   - Google Gemini: grounding metadata from Google Search
//   - OpenAI: URL citations from web browsing (Responses API)
//   - Cohere: inline document citations
//   - Perplexity/xAI: web search sources
//
// This example uses Google Gemini, but the Sources API is provider-agnostic.
//
// Usage:
//
//	export GEMINI_API_KEY=...
//	go run examples/citations/main.go
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
		goai.WithPrompt("What is the Go programming language? Give a brief history."),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
	fmt.Println()

	// Sources are always accessible -- empty when the provider doesn't ground.
	// When grounding is active, sources contain URL, title, and text offsets.
	if len(result.Sources) > 0 {
		fmt.Printf("Sources (%d):\n", len(result.Sources))
		for i, s := range result.Sources {
			fmt.Printf("  [%d] type=%s title=%q\n", i+1, s.Type, s.Title)
			if s.URL != "" {
				fmt.Printf("      url=%s\n", s.URL)
			}
			if s.StartIndex > 0 || s.EndIndex > 0 {
				fmt.Printf("      text[%d:%d]\n", s.StartIndex, s.EndIndex)
			}
		}
	} else {
		fmt.Println("No sources returned.")
		fmt.Println("(Sources are populated when the provider grounds responses, e.g.,")
		fmt.Println(" Google Search grounding, OpenAI web browsing, Perplexity search.)")
	}

	// Sources are also available per-step in multi-step tool loops.
	for i, step := range result.Steps {
		if len(step.Sources) > 0 {
			fmt.Printf("\nStep %d sources: %d\n", i+1, len(step.Sources))
		}
	}

	fmt.Printf("\nTokens: %d in, %d out\n",
		result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
