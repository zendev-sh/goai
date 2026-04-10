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
//	export GEMINI_API_KEY=...
//	export LANGFUSE_PUBLIC_KEY=pk-lf-...
//	export LANGFUSE_SECRET_KEY=sk-lf-...
//	export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
//	go run examples/langfuse/main.go
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
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	model := google.Chat("gemini-2.0-flash", google.WithAPIKey(os.Getenv("GEMINI_API_KEY")))

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
			time.Sleep(50 * time.Millisecond)
			return fmt.Sprintf(`{"city":%q,"temp":"22°C","condition":"sunny"}`, args.City), nil
		},
	}

	ctx := context.Background()

	// --- WithTracing: simple one-liner ---
	fmt.Println("=== WithTracing (simple) ===")
	result, err := goai.GenerateText(ctx, model,
		langfuse.WithTracing(
			langfuse.TraceName("weather-agent"),
			langfuse.Version("1.0.0"),
		),
		goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
		goai.WithPrompt("What's the weather in Tokyo?"),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(5),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Result: %s\n", result.Text[:min(120, len(result.Text))])
	fmt.Printf("Steps: %d | Tokens: %d in / %d out\n\n",
		len(result.Steps), result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)

	// --- WithTracing: with session, user, and tags ---
	fmt.Println("=== WithTracing (with options) ===")
	r, err := goai.GenerateText(ctx, model,
		langfuse.WithTracing(
			langfuse.TraceName("weather-agent"),
			langfuse.Version("1.0.0"),
			langfuse.SessionID("session-abc123"),
			langfuse.UserID("user-42"),
			langfuse.Tags("weather", "demo"),
			langfuse.OnFlushError(func(err error) {
				log.Printf("langfuse flush error: %v", err)
			}),
		),
		goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
		goai.WithPrompt("What's the weather in Paris?"),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(5),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Paris: %s\n", r.Text[:min(120, len(r.Text))])
}
