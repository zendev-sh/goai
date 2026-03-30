//go:build ignore

// Example: Langfuse tracing for a multi-step agent with tool calls.
//
// Every run creates a Langfuse trace with:
//   - A root span wrapping the full run
//   - A generation per LLM step
//   - A span per tool execution
//
// Usage:
//
//	export LANGFUSE_PUBLIC_KEY=pk-lf-...
//	export LANGFUSE_SECRET_KEY=sk-lf-...
//	export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
//	export OPENAI_API_KEY=...
//	go run ./examples/langfuse/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/observability/langfuse"
	"github.com/zendev-sh/goai/provider/openai"
)

// WeatherReport is the structured output returned by the agent.
type WeatherReport struct {
	City        string `json:"city" jsonschema:"description=City name"`
	Temperature string `json:"temperature" jsonschema:"description=Current temperature with unit"`
	Summary     string `json:"summary" jsonschema:"description=One-sentence weather summary"`
}

func main() {
	model := openai.Chat("gpt-4o-mini", openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

	// Create a Hooks instance once. The HTTP client is shared across runs.
	// Credentials are read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.
	lf := langfuse.New(langfuse.Config{
		TraceName: "weather-agent",
		Version:   "1.0.0",
		OnFlushError: func(err error) {
			log.Printf("langfuse flush error: %v", err)
		},
	})

	// Tool that simulates fetching weather data.
	weatherTool := goai.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		InputSchema: goai.SchemaFrom[struct {
			City string `json:"city" jsonschema:"description=City name,required"`
		}](),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct{ City string }
			_ = json.Unmarshal(input, &args)
			// Simulate an API call.
			time.Sleep(50 * time.Millisecond)
			return fmt.Sprintf(`{"city":%q,"temp":"22°C","condition":"sunny"}`, args.City), nil
		},
	}

	ctx := context.Background()

	// --- Single run with Run() ---
	// Run() returns fresh options for a one-off call.
	fmt.Println("=== Single run ===")
	result, err := goai.GenerateObject[WeatherReport](ctx, model,
		append(
			lf.Run(),
			goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
			goai.WithPrompt("What's the weather in Tokyo?"),
			goai.WithTools(weatherTool),
			goai.WithMaxSteps(3),
		)...,
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("City: %s\nTemp: %s\nSummary: %s\n", result.Object.City, result.Object.Temperature, result.Object.Summary)
	fmt.Printf("Steps: %d | Tokens: %d in / %d out\n\n",
		len(result.Steps), result.Usage.InputTokens, result.Usage.OutputTokens)

	// --- Reusable factory with With() ---
	// With() bakes in base options and returns a factory you call per run.
	// Use this when you run the same agent multiple times (e.g. in a loop).
	fmt.Println("=== Reusable factory (two cities) ===")
	runAgent := lf.With(
		goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(3),
	)

	for _, city := range []string{"London", "Sydney"} {
		r, err := goai.GenerateObject[WeatherReport](ctx, model,
			append(runAgent(), goai.WithPrompt("What's the weather in "+city+"?"))...,
		)
		if err != nil {
			log.Printf("%s: %v", city, err)
			continue
		}
		fmt.Printf("%s: %s — %s\n", r.Object.City, r.Object.Temperature, r.Object.Summary)
	}

	// --- Advanced: per-run metadata and session grouping ---
	fmt.Println("\n=== With session and user ID ===")
	lfSession := langfuse.New(langfuse.Config{
		TraceName: "weather-agent",
		Version:   "1.0.0",
		SessionID: "session-abc123",
		UserID:    "user-42",
		Tags:      []string{"weather", "demo"},
	})
	r, err := goai.GenerateObject[WeatherReport](ctx, model,
		append(
			lfSession.Run(),
			goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
			goai.WithPrompt("What's the weather in Paris?"),
			goai.WithTools(weatherTool),
			goai.WithMaxSteps(3),
		)...,
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Paris: %s — %s\n", r.Object.Temperature, r.Object.Summary)
}
