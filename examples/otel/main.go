// Example: OpenTelemetry tracing for a multi-step agent with tool calls.
//
// Every run creates OTel spans with:
//   - A root span wrapping the full run
//   - A child span per LLM generation step
//   - A child span per tool execution
//
// Usage:
//
//	export OPENAI_API_KEY=...
//	cd examples/otel && go run .
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	goaiotel "github.com/zendev-sh/goai/observability/otel"
	"github.com/zendev-sh/goai/provider/openai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// WeatherReport is the structured output returned by the agent.
type WeatherReport struct {
	City        string `json:"city" jsonschema:"description=City name"`
	Temperature string `json:"temperature" jsonschema:"description=Current temperature with unit"`
	Summary     string `json:"summary" jsonschema:"description=One-sentence weather summary"`
}

func main() {
	// --- Exporter setup ---
	// This example uses stdout for local development.
	// For production, use OTLP HTTP exporter:
	//
	//   import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	//
	//   exporter, err := otlptracehttp.New(ctx,
	//       otlptracehttp.WithEndpoint("tempo.example.com:4318"),
	//       otlptracehttp.WithHeaders(map[string]string{
	//           "Authorization": "Bearer <API_KEY>",
	//       }),
	//   )
	//
	// Or set standard OTel env vars (SDK reads them automatically):
	//   OTEL_EXPORTER_OTLP_ENDPOINT=https://tempo.example.com:4318
	//   OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <API_KEY>"
	exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		log.Fatal(err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter))
	defer func() { _ = tp.Shutdown(context.Background()) }()
	otel.SetTracerProvider(tp)

	model := openai.Chat("gpt-4o-mini", openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

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

	// --- WithTracing: uses the global TracerProvider ---
	fmt.Println("=== WithTracing (simple) ===")
	result, err := goai.GenerateObject[WeatherReport](ctx, model,
		goaiotel.WithTracing(
			goaiotel.WithSpanName("weather-agent"),
		),
		goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
		goai.WithPrompt("What's the weather in Tokyo?"),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("City: %s\nTemp: %s\nSummary: %s\n", result.Object.City, result.Object.Temperature, result.Object.Summary)
	fmt.Printf("Steps: %d | Tokens: %d in / %d out\n\n",
		len(result.Steps), result.Usage.InputTokens, result.Usage.OutputTokens)

	// --- WithTracing: with explicit provider and custom attributes ---
	fmt.Println("\n=== WithTracing (with options) ===")
	r, err := goai.GenerateObject[WeatherReport](ctx, model,
		goaiotel.WithTracing(
			goaiotel.WithTracerProvider(tp),
			goaiotel.WithSpanName("weather-agent"),
			goaiotel.WithAttributes(
				attribute.String("user.id", "user-42"),
				attribute.String("session.id", "session-abc123"),
			),
			goaiotel.RecordInputMessages(true),
			goaiotel.RecordOutputMessages(true),
		),
		goai.WithSystem("You are a weather assistant. Always call get_weather before answering."),
		goai.WithPrompt("What's the weather in Paris?"),
		goai.WithTools(weatherTool),
		goai.WithMaxSteps(3),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Paris: %s - %s\n", r.Object.Temperature, r.Object.Summary)
}
