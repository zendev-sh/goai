---
title: Prompt Caching
description: "Reduce cost and latency with prompt caching in GoAI. Enable cache control for Anthropic-family system prompts with a single option."
---

# Prompt Caching

Prompt caching reduces cost and latency by reusing previously processed prompt content across requests. Enable it with a single option:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithMessages(goai.SystemMessage("You are a helpful assistant with a very long system prompt...")),
    goai.WithPrompt("Hello"),
    goai.WithPromptCaching(true),
)
```

## What It Does

When enabled, GoAI marks the last content part of each system message with `CacheControl: "ephemeral"`.

- Anthropic uses this hint to send `cache_control: {"type": "ephemeral"}` on the final system content block.
- Bedrock uses a Bedrock-specific `cachePoint` block appended to the system prompt.
- Other providers may ignore the option and/or print a warning.

This targets system-role entries in `WithMessages(...)` only. The separate `WithSystem(...)` field is passed through unchanged, and conversation/tool messages are not cache-marked.

## Provider Support

- **Anthropic** - uses `cache_control: {"type": "ephemeral"}` on content blocks
- **Bedrock** - supported via a Bedrock `cachePoint` on the system prompt
- **MiniMax** - supported when delegating to Anthropic
- **Other providers** - the flag is passed through and the provider may ignore it (GoAI prints a warning to stderr in providers that explicitly do not support it)

## Usage Tracking

Cache hit/miss information appears in `Usage`:

```go
fmt.Printf("Cache read tokens:  %d\n", result.TotalUsage.CacheReadTokens)
fmt.Printf("Cache write tokens: %d\n", result.TotalUsage.CacheWriteTokens)
```

`CacheReadTokens > 0` means the provider served part of the prompt from cache. `CacheWriteTokens > 0` means the provider stored new content in the cache for future requests.
