---
title: Structured Output
description: "Generate type-safe structured responses using Go generics. Auto-generate JSON Schema from structs with GenerateObject and StreamObject."
---

# Structured Output

GoAI can generate type-safe structured responses using Go generics. Define a Go struct, and GoAI auto-generates the JSON Schema, sends it to the model, and parses the response back into your type.

## GenerateObject

```go
package main

import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

type Sentiment struct {
    Text       string  `json:"text"`
    Sentiment  string  `json:"sentiment" jsonschema:"enum=positive|negative|neutral"`
    Confidence float64 `json:"confidence" jsonschema:"description=Confidence score from 0 to 1"`
}

func main() {
    model := openai.Chat("gpt-4o")

    result, err := goai.GenerateObject[Sentiment](context.Background(), model,
        goai.WithPrompt("Analyze the sentiment: I love this product!"),
    )
    if err != nil {
        panic(err)
    }
    fmt.Printf("%s (%.0f%% confidence)\n", result.Object.Sentiment, result.Object.Confidence*100)
}
```

The type parameter `[Sentiment]` tells GoAI to:

1. Generate a JSON Schema from the struct via `SchemaFrom[Sentiment]()`
2. Send the schema to the model as the required response format
3. Parse the JSON response into a `Sentiment` value

The result is an `*ObjectResult[Sentiment]` with the parsed object in `result.Object`.

## Struct Tags

GoAI reads two struct tags to build the schema:

- `json:"name"` - field name in JSON (standard Go convention)
- `jsonschema:"description=...,enum=a|b|c"` - description and allowed values

```go
type Recipe struct {
    Name        string   `json:"name" jsonschema:"description=Recipe name"`
    Ingredients []string `json:"ingredients"`
    Steps       []string `json:"steps"`
    PrepTime    int      `json:"prep_time" jsonschema:"description=Prep time in minutes"`
    Difficulty  string   `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
}
```

All exported fields are required by default. Use pointer types for optional (nullable) fields:

```go
type Profile struct {
    Name  string  `json:"name"`            // required
    Email *string `json:"email"`           // nullable
    Age   *int    `json:"age"`             // nullable
}
```

## SchemaFrom

To inspect the generated schema without making an API call:

```go
schema := goai.SchemaFrom[Recipe]()
fmt.Println(string(schema))
```

This returns the `json.RawMessage` that GoAI sends to the provider. Useful for debugging or logging.

## StreamObject

For long structured responses, stream partial objects as they arrive:

```go
type Article struct {
    Title    string   `json:"title"`
    Summary  string   `json:"summary"`
    Sections []string `json:"sections"`
    Tags     []string `json:"tags"`
}

stream, err := goai.StreamObject[Article](ctx, model,
    goai.WithPrompt("Write an article about Go concurrency patterns."),
)
if err != nil {
    panic(err)
}

for partial := range stream.PartialObjectStream() {
    fmt.Printf("\rTitle: %s (%d sections)", partial.Title, len(partial.Sections))
}
fmt.Println()

result, err := stream.Result()
if err != nil {
    panic(err)
}
fmt.Printf("Final: %s - %d sections\n", result.Object.Title, len(result.Object.Sections))
```

`PartialObjectStream()` returns a `<-chan *Article` that emits progressively populated objects as JSON tokens arrive. Early emissions may have zero-value fields that fill in over time.

After the channel closes, call `Result()` to get the final validated object.

## Explicit Schema

If reflection-based schema generation does not fit your use case, provide a raw JSON Schema directly with `WithExplicitSchema`:

```go
schema := json.RawMessage(`{
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"],
    "additionalProperties": false
}`)

type Response struct {
    Answer     string  `json:"answer"`
    Confidence float64 `json:"confidence"`
}

result, err := goai.GenerateObject[Response](ctx, model,
    goai.WithPrompt("What is 2+2?"),
    goai.WithExplicitSchema(schema),
)
```

The explicit schema is sent to the provider as-is, bypassing `SchemaFrom[T]()`. The response is still parsed into the type parameter, so the schema and struct must be compatible.

Use `WithSchemaName` to change the schema name sent to the provider (default is "response"):

```go
result, err := goai.GenerateObject[Recipe](ctx, model,
    goai.WithPrompt("Give me a cookie recipe."),
    goai.WithSchemaName("recipe"),
)
```

## Provider Compatibility

Structured output works with any provider that supports JSON Schema response format. Tested with OpenAI, Anthropic, and Google. Other OpenAI-compatible providers generally support it for models that accept JSON Schema.
