---
title: Provider-Defined Tools
description: "Use built-in server-side tools from OpenAI, Anthropic, Google, xAI, and Groq including web search, code execution, and computer use."
---

# Provider-Defined Tools

Provider-defined tools are built-in capabilities that execute server-side at the provider. Unlike regular tools where you supply an `Execute` function, provider tools are handled entirely by the provider's API. The model decides when to use them, and the provider executes them without any round-trip to your code.

## How They Work

Provider-defined tools use two special fields on the tool definition:

- `ProviderDefinedType` - identifies the tool type for the provider's API
- `ProviderDefinedOptions` - provider-specific configuration

Each provider package exposes a `Tools` struct with factory functions that return correctly configured `provider.ToolDefinition` values. Pass these to `WithTools` by wrapping them in a `goai.Tool`:

```go
import (
    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What happened in tech news today?"),
    goai.WithTools(goai.Tool{
        Name:                   openai.Tools.WebSearch().Name,
        ProviderDefinedType:    openai.Tools.WebSearch().ProviderDefinedType,
        ProviderDefinedOptions: openai.Tools.WebSearch().ProviderDefinedOptions,
    }),
)
```

Or more concisely, build a helper:

```go
// Requires "github.com/zendev-sh/goai" and "github.com/zendev-sh/goai/provider".
func providerTool(td provider.ToolDefinition) goai.Tool {
    return goai.Tool{
        Name:                   td.Name,
        ProviderDefinedType:    td.ProviderDefinedType,
        ProviderDefinedOptions: td.ProviderDefinedOptions,
    }
}

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What happened in tech news today?"),
    goai.WithTools(providerTool(openai.Tools.WebSearch())),
)
```

## Available Tools

### Anthropic (10 tools)

Import: `github.com/zendev-sh/goai/provider/anthropic`

| Factory                                        | Description                                            |
| ---------------------------------------------- | ------------------------------------------------------ |
| `anthropic.Tools.Computer(opts)`               | Computer use - screenshot, mouse, keyboard (v20250124) |
| `anthropic.Tools.Computer_20251124(opts)`      | Computer use with zoom support (Opus 4.5+)             |
| `anthropic.Tools.Bash()`                       | Bash shell execution (v20250124)                       |
| `anthropic.Tools.TextEditor()`                 | Text editor - view, create, replace (v20250429)        |
| `anthropic.Tools.TextEditor_20250728(opts...)` | Text editor with optional maxCharacters (Sonnet 4+)    |
| `anthropic.Tools.WebSearch(opts...)`           | Web search via Brave Search (v20250305)                |
| `anthropic.Tools.WebSearch_20260209(opts...)`  | Web search (v20260209, requires beta header)           |
| `anthropic.Tools.WebFetch(opts...)`            | Fetch web content by URL (v20260209)                   |
| `anthropic.Tools.CodeExecution()`              | Python code execution in sandbox (v20260120)           |
| `anthropic.Tools.CodeExecution_20250825()`     | Code execution (v20250825, requires beta header)       |

```go
// Each factory returns a provider.ToolDefinition. Wrap with goai.Tool{} before passing to WithTools.

// Computer use
td := anthropic.Tools.Computer(anthropic.ComputerToolOptions{
    DisplayWidthPx:  1920,
    DisplayHeightPx: 1080,
})

// Web search with domain filtering
td := anthropic.Tools.WebSearch(
    anthropic.WithAllowedDomains("docs.go.dev", "pkg.go.dev"),
    anthropic.WithMaxUses(5),
)

// Code execution (no options needed)
td := anthropic.Tools.CodeExecution()
```

### OpenAI (4 tools)

Import: `github.com/zendev-sh/goai/provider/openai`

| Factory                                 | Description                           |
| --------------------------------------- | ------------------------------------- |
| `openai.Tools.WebSearch(opts...)`       | Web search via Responses API          |
| `openai.Tools.CodeInterpreter(opts...)` | Python code execution in sandbox      |
| `openai.Tools.FileSearch(opts...)`      | Semantic search over uploaded files   |
| `openai.Tools.ImageGeneration(opts...)` | Generate images within a conversation |

```go
// Each factory returns a provider.ToolDefinition. Wrap with goai.Tool{} before passing to WithTools.

// Web search with location
td := openai.Tools.WebSearch(
    openai.WithSearchContextSize("medium"),
    openai.WithUserLocation(openai.WebSearchLocation{
        Country:  "US",
        City:     "San Francisco",
        Timezone: "America/Los_Angeles",
    }),
)

// Code interpreter with container
td := openai.Tools.CodeInterpreter(
    openai.WithContainerID("container-id"),
)

// File search with vector stores
td := openai.Tools.FileSearch(
    openai.WithVectorStoreIDs("vs_abc123"),
    openai.WithMaxNumResults(10),
)

// Image generation
td := openai.Tools.ImageGeneration(
    openai.WithImageQuality("high"),
    openai.WithImageSize("1024x1024"),
)
```

**Additional WebSearch options:**

| Option                                         | Type                 | Description                                         |
| ---------------------------------------------- | -------------------- | --------------------------------------------------- |
| `WithSearchContextSize(size string)`           | `string`             | `"low"`, `"medium"` (default), `"high"` — controls context breadth and cost |
| `WithUserLocation(loc WebSearchLocation)`      | `WebSearchLocation`  | Location hint for geographically relevant results (`Country`, `City`, `Timezone`) |
| `WithSearchFilters(filters WebSearchFilters)`  | `WebSearchFilters`   | Domain allow-list for search results                |
| `WithExternalWebAccess(enabled bool)`          | `bool`               | `true` = live content (default), `false` = cached   |

**Additional CodeInterpreter options:**

| Option                                               | Type                         | Description                                       |
| ---------------------------------------------------- | ---------------------------- | ------------------------------------------------- |
| `WithContainerID(id string)`                         | `string`                     | Use an existing container by ID                   |
| `WithContainerFiles(container *CodeInterpreterContainer)` | `*CodeInterpreterContainer` | Auto-provisioned container with uploaded file IDs |

**Additional FileSearch options:**

| Option                                           | Type                 | Description                                              |
| ------------------------------------------------ | -------------------- | -------------------------------------------------------- |
| `WithRanking(ranking FileSearchRanking)`         | `FileSearchRanking`  | Ranking options: `Ranker` (string) and `ScoreThreshold` (0–1) |
| `WithFileSearchFilters(filters FileSearchFilter)` | `FileSearchFilter`  | Metadata filter; use `*FileSearchComparisonFilter` or `*FileSearchCompoundFilter` |

**Additional ImageGeneration options:**

| Option                                           | Type                    | Description                                                |
| ------------------------------------------------ | ----------------------- | ---------------------------------------------------------- |
| `WithBackground(bg string)`                     | `string`                | Background type for the generated image                    |
| `WithInputFidelity(fidelity string)`             | `string`                | `"low"` or `"high"` input processing fidelity              |
| `WithInputImageMask(mask ImageGenerationMask)`   | `ImageGenerationMask`   | Inpainting mask (`FileID` or `ImageURL`)                   |
| `WithImageModel(model string)`                   | `string`                | Image model to use (default: `"gpt-image-1"`)             |
| `WithModeration(mod string)`                     | `string`                | Moderation level (default: `"auto"`)                       |
| `WithOutputCompression(level int)`               | `int`                   | Output compression 0–100                                   |
| `WithOutputFormat(format string)`                | `string`                | `"png"`, `"jpeg"`, `"webp"`                                |
| `WithPartialImages(n int)`                       | `int`                   | Partial images in streaming (0–3)                          |
| `WithImageQuality(quality string)`               | `string`                | `"auto"`, `"low"`, `"medium"`, `"high"`                    |
| `WithImageSize(size string)`                     | `string`                | `"auto"`, `"1024x1024"`, `"1024x1536"`, `"1536x1024"`     |

### Google (3 tools)

Import: `github.com/zendev-sh/goai/provider/google`

| Factory                              | Description                             |
| ------------------------------------ | --------------------------------------- |
| `google.Tools.GoogleSearch(opts...)` | Grounding with Google Search            |
| `google.Tools.URLContext()`          | Fetch and process web content from URLs |
| `google.Tools.CodeExecution()`       | Python code execution in sandbox        |

```go
// Each factory returns a provider.ToolDefinition. Wrap with goai.Tool{} before passing to WithTools.

// Google Search with time range filter
td := google.Tools.GoogleSearch(
    google.WithTimeRange("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
)

// URL context (no options)
td := google.Tools.URLContext()

// Code execution (no options)
td := google.Tools.CodeExecution()
```

### xAI (2 tools)

Import: `github.com/zendev-sh/goai/provider/xai`

| Factory                        | Description                 |
| ------------------------------ | --------------------------- |
| `xai.Tools.WebSearch(opts...)` | Web search                  |
| `xai.Tools.XSearch(opts...)`   | Search posts on X (Twitter) |

```go
// Each factory returns a provider.ToolDefinition. Wrap with goai.Tool{} before passing to WithTools.

// Web search with domain filtering
td := xai.Tools.WebSearch(
    xai.WithAllowedDomains("go.dev", "github.com"),
)

// X search with date range and handle filtering
td := xai.Tools.XSearch(
    xai.WithAllowedXHandles("@golang"),
    xai.WithXSearchDateRange("2025-01-01", "2025-12-31"),
)
```

### Groq (1 tool)

Import: `github.com/zendev-sh/goai/provider/groq`

| Factory                      | Description                |
| ---------------------------- | -------------------------- |
| `groq.Tools.BrowserSearch()` | Interactive browser search |

```go
// Returns a provider.ToolDefinition. Wrap with goai.Tool{} before passing to WithTools.
td := groq.Tools.BrowserSearch()
```

## Beta Headers

Some Anthropic tools require beta headers. GoAI handles this automatically - when a provider-defined tool requires a beta header, the provider injects it into the request. No manual header management is needed.
