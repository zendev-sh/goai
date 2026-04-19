// Package goai is a Go SDK for building AI applications. One SDK, 24+ providers.
//
// Inspired by the Vercel AI SDK. The same clean abstractions, idiomatically
// adapted for Go with generics, interfaces, and functional options.
//
// # Core Functions
//
//   - [GenerateText]: non-streaming text generation
//   - [StreamText]: streaming text generation via channels
//   - [GenerateObject]: structured output with auto-generated JSON Schema
//   - [StreamObject]: streaming structured output with partial object emission
//   - [Embed]: single text embedding
//   - [EmbedMany]: batch text embeddings with auto-chunking
//   - [GenerateImage]: image generation from text prompts
//
// # Providers
//
// 24+ providers: OpenAI, Anthropic, Google, Bedrock, Azure, Vertex, Mistral,
// xAI, Groq, Cohere, DeepSeek, Fireworks, Together, DeepInfra, OpenRouter,
// Perplexity, Cerebras, MiniMax, Cloudflare Workers AI, FPT Smart Cloud,
// Ollama, vLLM, RunPod, and any OpenAI-compatible endpoint.
//
// Providers auto-resolve API keys from environment variables:
//
//	model := openai.Chat("gpt-4o") // reads OPENAI_API_KEY
//
// # Quick Start
//
//	result, err := goai.GenerateText(ctx, model,
//	    goai.WithPrompt("What is the capital of France?"),
//	)
//	fmt.Println(result.Text)
//
// # Streaming
//
//	stream, err := goai.StreamText(ctx, model, goai.WithPrompt("Hello"))
//	for text := range stream.TextStream() {
//	    fmt.Print(text)
//	}
//
// # Structured Output
//
//	type Recipe struct {
//	    Name       string   `json:"name"`
//	    Ingredients []string `json:"ingredients"`
//	}
//	result, err := goai.GenerateObject[Recipe](ctx, model,
//	    goai.WithPrompt("Give me a recipe for cookies"),
//	)
//
// # Tools
//
// Define tools with JSON Schema and an Execute handler. Set [WithMaxSteps]
// to enable the auto tool loop:
//
//	result, err := goai.GenerateText(ctx, model,
//	    goai.WithPrompt("What's the weather in Tokyo?"),
//	    goai.WithTools(weatherTool),
//	    goai.WithMaxSteps(3),
//	)
//
// See https://goai.sh for full documentation.
package goai
