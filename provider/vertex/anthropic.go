package vertex

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/anthropic"
)

// AnthropicChat creates a Vertex AI language model that uses Anthropic's native
// Messages API via Google Cloud's rawPredict/streamRawPredict endpoints.
//
// Unlike the regular Vertex Chat (which uses the OpenAI-compatible endpoint),
// this provider speaks native Anthropic SSE, giving full support for prompt
// caching, cache token tracking, and extended thinking.
//
// Usage:
//
//	model := vertex.AnthropicChat("claude-sonnet-4-20250514",
//		vertex.WithProject("my-project"),
//		vertex.WithLocation("us-east5"),
//	)
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
func AnthropicChat(modelID string, opts ...Option) provider.LanguageModel {
	o := resolveOpts(opts)

	baseURL := anthropicBaseURL(o)
	ts := resolveAnthropicTokenSource(o)

	return anthropic.Chat(modelID,
		anthropic.WithTokenSource(ts),
		anthropic.WithBaseURL(baseURL),
		anthropic.WithAuthMode(anthropic.AuthBearer),
		anthropic.WithURLBuilder(anthropicURLBuilder(baseURL)),
		anthropic.WithBodyTransformer(anthropicBodyTransformer),
		anthropic.WithErrorProvider("vertex-anthropic"),
		anthropic.WithSkipEnvResolve(),
	)
}

// anthropicBaseURL builds the base URL for Vertex Anthropic endpoints.
//
// Pattern:
//
//	https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/anthropic/models
//
// When location is "global", no location prefix on the hostname:
//
//	https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/anthropic/models
func anthropicBaseURL(o options) string {
	if o.baseURL != "" {
		return strings.TrimRight(o.baseURL, "/")
	}

	project := cmp.Or(o.project, os.Getenv("GOOGLE_VERTEX_PROJECT"), os.Getenv("GOOGLE_CLOUD_PROJECT"), os.Getenv("GCLOUD_PROJECT"))
	location := cmp.Or(o.location, os.Getenv("GOOGLE_VERTEX_LOCATION"), os.Getenv("GOOGLE_CLOUD_LOCATION"), "us-central1")

	var host string
	if location == "global" {
		host = "aiplatform.googleapis.com"
	} else {
		host = location + "-aiplatform.googleapis.com"
	}

	return fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/publishers/anthropic/models",
		host, project, location)
}

// anthropicURLBuilder returns a URLBuilder that constructs Vertex rawPredict URLs.
//
//	Streaming:     {baseURL}/{modelID}:streamRawPredict
//	Non-streaming: {baseURL}/{modelID}:rawPredict
func anthropicURLBuilder(baseURL string) anthropic.URLBuilder {
	return func(_, modelID string, streaming bool) string {
		if streaming {
			return baseURL + "/" + modelID + ":streamRawPredict"
		}
		return baseURL + "/" + modelID + ":rawPredict"
	}
}

// anthropicBodyTransformer adapts the Anthropic request body for Vertex:
//   - removes "model" (model ID is in the URL path)
//   - adds "anthropic_version" field required by Vertex
func anthropicBodyTransformer(body map[string]any) map[string]any {
	delete(body, "model")
	body["anthropic_version"] = "vertex-2023-10-16"
	return body
}

// resolveAnthropicTokenSource resolves auth for Vertex Anthropic.
// Uses the explicit token source from options, or falls back to ADC.
func resolveAnthropicTokenSource(o options) provider.TokenSource {
	if o.tokenSource != nil {
		return o.tokenSource
	}
	// ADC auto-detect (same as regular Vertex).
	ts, err := ADCTokenSource(context.Background())
	if err != nil {
		return &failingTokenSource{err: err}
	}
	return ts
}
