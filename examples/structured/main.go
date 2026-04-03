//go:build ignore

// Example: structured output with GenerateObject and StreamObject.
//
// Usage:
//
//	GEMINI_API_KEY=... go run ./examples/structured/
package main

import (
	"cmp"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

type Recipe struct {
	Name        string   `json:"name" jsonschema:"description=Recipe name"`
	Ingredients []string `json:"ingredients" jsonschema:"description=List of ingredients"`
	Steps       []string `json:"steps" jsonschema:"description=Cooking steps"`
	PrepTimeMin int      `json:"prep_time_min" jsonschema:"description=Preparation time in minutes"`
	Difficulty  string   `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
}

func main() {
	apiKey := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY"))
	if apiKey == "" {
		log.Fatal("GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY must be set")
	}

	model := google.Chat("gemini-2.5-flash", google.WithAPIKey(apiKey))
	ctx := context.Background()

	// --- GenerateObject ---
	fmt.Println("=== GenerateObject ===")
	result, err := goai.GenerateObject[Recipe](ctx, model,
		goai.WithPrompt("Give me a recipe for chocolate chip cookies"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Recipe: %s\n", result.Object.Name)
	fmt.Printf("Difficulty: %s\n", result.Object.Difficulty)
	fmt.Printf("Prep time: %d min\n", result.Object.PrepTimeMin)
	fmt.Printf("Ingredients (%d):\n", len(result.Object.Ingredients))
	for _, ing := range result.Object.Ingredients {
		fmt.Printf("  - %s\n", ing)
	}
	fmt.Printf("Steps (%d):\n", len(result.Object.Steps))
	for i, step := range result.Object.Steps {
		fmt.Printf("  %d. %s\n", i+1, step)
	}
	fmt.Printf("Usage: %d input, %d output tokens\n", result.Usage.InputTokens, result.Usage.OutputTokens)

	// --- StreamObject ---
	fmt.Println("\n=== StreamObject ===")
	stream, err := goai.StreamObject[Recipe](ctx, model,
		goai.WithPrompt("Give me a recipe for pancakes"),
	)
	if err != nil {
		log.Fatal(err)
	}

	var lastPartial *Recipe
	for partial := range stream.PartialObjectStream() {
		lastPartial = partial
		if partial.Name != "" {
			fmt.Printf("\rStreaming: %s (%d ingredients so far)", partial.Name, len(partial.Ingredients))
		}
	}
	fmt.Println()

	if lastPartial != nil {
		fmt.Printf("Final: %s -- %s difficulty\n", lastPartial.Name, lastPartial.Difficulty)
	}

	final, err := stream.Result()
	if err != nil {
		log.Fatal("stream result:", err)
	}
	fmt.Printf("Usage: %d input, %d output tokens\n", final.Usage.InputTokens, final.Usage.OutputTokens)
}
