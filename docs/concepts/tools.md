---
title: Tool Calling
description: "Define custom tools for LLM function calling in Go. Configure auto tool loops, tool choice, step callbacks, and multi-step agent workflows."
---

# Tools

Tools let the model call functions defined in your code. The model decides when to use a tool based on the conversation, generates structured input, and GoAI executes the function and feeds the result back.

## Defining a Tool

A `goai.Tool` has a name, description, JSON Schema for input, and an `Execute` function:

```go
weatherTool := goai.Tool{
    Name:        "get_weather",
    Description: "Get the current weather for a city.",
    InputSchema: json.RawMessage(`{
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }`),
    Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
        var params struct {
            City string `json:"city"`
        }
        if err := json.Unmarshal(input, &params); err != nil {
            return "", err
        }
        // Call your weather API here.
        return fmt.Sprintf("72F and sunny in %s", params.City), nil
    },
}
```

Pass tools to `GenerateText` or `StreamText` with `WithTools`:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What's the weather in Tokyo?"),
    goai.WithTools(weatherTool),
)
```

## Single Step vs Auto Loop

By default, `MaxSteps` is 1. The model can request tool calls, but GoAI does not execute them automatically. The tool calls appear in `result.ToolCalls` for you to handle.

Set `WithMaxSteps(n)` to enable the auto tool loop. GoAI will:

1. Send the prompt to the model
2. If the model requests tool calls, execute each tool's `Execute` function
3. Append the tool results to the conversation
4. Send the updated conversation back to the model
5. Repeat until the model stops requesting tools, or `n` steps are reached

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What's the weather in Tokyo and London?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
fmt.Printf("Steps: %d\n", len(result.Steps))
```

## Tool Choice

Control how the model selects tools with `WithToolChoice`:

```go
// Model decides whether to use tools (default)
goai.WithToolChoice("auto")

// Model must use at least one tool
goai.WithToolChoice("required")

// Model must not use any tools
goai.WithToolChoice("none")

// Model must use a specific tool
goai.WithToolChoice("get_weather")
```

## Step Callbacks

`WithOnStepFinish` is called after each step completes, including after tool execution. Use it for logging or progress tracking:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Research the weather in 3 cities."),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(10),
    goai.WithOnStepFinish(func(step goai.StepResult) {
        fmt.Printf("Step %d: %d tool calls, finish=%s\n",
            step.Number, len(step.ToolCalls), step.FinishReason)
    }),
)
```

## Tool Call Hooks

`WithOnToolCall` is called after each individual tool execution:

```go
import "time"

// ...

goai.WithOnToolCall(func(info goai.ToolCallInfo) {
    fmt.Printf("[step %d] tool %s took %v\n", info.Step, info.ToolName, info.Duration)
    fmt.Printf("  input:  %s\n", info.Input)
    fmt.Printf("  output: %s\n", info.Output)
    if info.Error != nil {
        fmt.Printf("  error: %v\n", info.Error)
    }
})
```

### ToolCallInfo Fields

| Field          | Type               | Description                                                                 |
| -------------- | ------------------ | --------------------------------------------------------------------------- |
| `ToolCallID`   | `string`           | Provider-assigned identifier for this tool call                             |
| `ToolName`     | `string`           | Name of the tool that was executed                                          |
| `Step`         | `int`              | 1-based index of the generation step in which this tool was called          |
| `Input`        | `json.RawMessage`  | Raw JSON arguments passed to the tool                                       |
| `Output`       | `string`           | String result returned by the tool (empty if the tool errored)              |
| `OutputObject` | `any`              | Parsed JSON value of Output when valid JSON; nil otherwise                  |
| `StartTime`    | `time.Time`        | When the tool execution began                                               |
| `Duration`     | `time.Duration`    | Time taken to execute the tool                                              |
| `Error`        | `error`            | Error returned by the tool, if any                                          |

## Tools Without Execute

If a tool has no `Execute` function, GoAI sends the tool definition to the model but does not participate in the auto loop. The model can still request the tool call, and it appears in `result.ToolCalls`. This is useful when you manage the tool loop yourself.

## StepResult

Each step in a multi-step tool loop produces a `goai.StepResult`:

| Field              | Type                        | Description                          |
| ------------------ | --------------------------- | ------------------------------------ |
| `Number`           | `int`                       | 1-based step index                   |
| `Text`             | `string`                    | Text generated in this step          |
| `ToolCalls`        | `[]provider.ToolCall`       | Tool calls requested in this step    |
| `FinishReason`     | `provider.FinishReason`     | Why this step stopped                |
| `Usage`            | `provider.Usage`            | Token usage for this step            |
| `Response`         | `provider.ResponseMetadata` | Provider metadata for this step      |
| `ProviderMetadata` | `map[string]map[string]any` | Provider-specific data for this step |
| `Sources`          | `[]provider.Source`         | Citations from this step             |
