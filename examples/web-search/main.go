//go:build ignore

// Example: Web search across providers.
//
// Demonstrates provider-defined web search tools for OpenAI, Anthropic, and Google.
// Each provider has its own tool factory -- the model decides when to search.
//
// OpenAI web search (pick one auth method):
//
//	export OPENAI_API_KEY=...
//	go run examples/web-search/main.go openai
//
//	export AZURE_OPENAI_API_KEY=... AZURE_RESOURCE_NAME=...
//	go run examples/web-search/main.go openai-azure
//
// Anthropic web search (pick one auth method):
//
//	export ANTHROPIC_API_KEY=...
//	go run examples/web-search/main.go anthropic
//
//	export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=...
//	go run examples/web-search/main.go anthropic-bedrock
//
// Google search grounding:
//
//	export GEMINI_API_KEY=...
//	go run examples/web-search/main.go google
package main

import (
	"cmp"
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/anthropic"
	"github.com/zendev-sh/goai/provider/azure"
	"github.com/zendev-sh/goai/provider/bedrock"
	"github.com/zendev-sh/goai/provider/google"
	"github.com/zendev-sh/goai/provider/openai"
)

func main() {
	prov := "openai"
	if len(os.Args) > 1 {
		prov = os.Args[1]
	}

	var model provider.LanguageModel
	var tools []goai.Tool

	switch prov {
	// --- OpenAI via direct API key ---
	case "openai":
		model = openai.Chat("gpt-4.1-mini",
			openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
		def := openai.Tools.WebSearch()
		tools = toolsFromDef(def)

	// --- OpenAI via Azure ---
	case "openai-azure":
		model = azure.Chat("gpt-4.1-mini",
			azure.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")))
		def := openai.Tools.WebSearch()
		tools = toolsFromDef(def)

	// --- Anthropic via direct API key ---
	case "anthropic":
		model = anthropic.Chat("claude-sonnet-4-20250514",
			anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
		def := anthropic.Tools.WebSearch(anthropic.WithMaxUses(5))
		tools = toolsFromDef(def)

	// --- Anthropic via AWS Bedrock ---
	case "anthropic-bedrock":
		model = bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0",
			bedrock.WithAccessKey(os.Getenv("AWS_ACCESS_KEY_ID")),
			bedrock.WithSecretKey(os.Getenv("AWS_SECRET_ACCESS_KEY")),
			bedrock.WithRegion(os.Getenv("AWS_REGION")))
		def := anthropic.Tools.WebSearch(anthropic.WithMaxUses(5))
		tools = toolsFromDef(def)

	// --- Google via Gemini API key ---
	case "google":
		geminiKey := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY"))
		model = google.Chat("gemini-2.5-flash",
			google.WithAPIKey(geminiKey))
		def := google.Tools.GoogleSearch()
		tools = toolsFromDef(def)

	default:
		log.Fatalf("Unknown: %s (use openai, openai-azure, anthropic, anthropic-bedrock, google)", prov)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, model,
		goai.WithPrompt("What is the latest version of Go released in 2026?"),
		goai.WithMaxOutputTokens(500),
		goai.WithTools(tools...),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Answer:", result.Text)
	fmt.Printf("\nSources: %d\n", len(result.Sources))
	for i, s := range result.Sources {
		fmt.Printf("  [%d] %s - %s\n", i, s.Title, s.URL)
	}
	fmt.Printf("\nUsage: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}

func toolsFromDef(def provider.ToolDefinition) []goai.Tool {
	return []goai.Tool{{
		Name:                   def.Name,
		ProviderDefinedType:    def.ProviderDefinedType,
		ProviderDefinedOptions: def.ProviderDefinedOptions,
	}}
}
