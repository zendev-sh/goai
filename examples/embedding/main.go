//go:build ignore

// Example: text embeddings with Embed and EmbedMany.
//
// Usage:
//
//	GEMINI_API_KEY=... go run ./examples/embedding/
package main

import (
	"cmp"
	"context"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	apiKey := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY"))
	if apiKey == "" {
		log.Fatal("GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY must be set")
	}

	model := google.Embedding("gemini-embedding-001", google.WithAPIKey(apiKey))
	ctx := context.Background()

	// --- Single embedding ---
	fmt.Println("=== Embed ===")
	result, err := goai.Embed(ctx, model, "The quick brown fox jumps over the lazy dog")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Embedding dimensions: %d\n", len(result.Embedding))
	fmt.Printf("First 5 values: %v\n", result.Embedding[:min(5, len(result.Embedding))])

	// --- Batch embeddings ---
	fmt.Println("\n=== EmbedMany ===")
	texts := []string{
		"I love programming",
		"Software engineering is great",
		"The weather is nice today",
		"Cats are wonderful pets",
	}
	manyResult, err := goai.EmbedMany(ctx, model, texts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Generated %d embeddings\n", len(manyResult.Embeddings))

	// --- Similarity comparison ---
	fmt.Println("\n=== Cosine Similarity ===")
	for i := range len(texts) {
		for j := i + 1; j < len(texts); j++ {
			sim := cosineSimilarity(manyResult.Embeddings[i], manyResult.Embeddings[j])
			fmt.Printf("  \"%s\" vs \"%s\": %.4f\n", texts[i], texts[j], sim)
		}
	}
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
