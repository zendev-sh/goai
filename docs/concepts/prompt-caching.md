---
title: Prompt Caching
description: "Reduce cost and latency with prompt caching in GoAI. Enable cache control for Anthropic and OpenAI system prompts with a single option."
---

# Prompt Caching

Prompt caching reduces cost and latency by reusing previously processed prompt content across requests. Enable it with a single option:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithSystem("You are a helpful assistant with a very long system prompt..."),
    goai.WithPrompt("Hello"),
    goai.WithPromptCaching(true),
)
```

## What It Does

When enabled, GoAI marks the last content part of each system message with `CacheControl: "ephemeral"`. Providers that support prompt caching (Anthropic, OpenAI) use this hint to cache the processed system prompt and reuse it across requests.

This targets system messages only. Conversation messages and tool results are not cached, because system prompts are the most effective target for caching in typical usage patterns.

## Provider Support

- **Anthropic** - uses `cache_control: {"type": "ephemeral"}` on content blocks
- **OpenAI** - uses the equivalent caching mechanism
- **Other providers** - the flag is passed through but may be ignored if the provider does not support it

## Usage Tracking

Cache hit/miss information appears in `Usage`:

```go
fmt.Printf("Cache read tokens:  %d\n", result.TotalUsage.CacheReadTokens)
fmt.Printf("Cache write tokens: %d\n", result.TotalUsage.CacheWriteTokens)
```

`CacheReadTokens > 0` means the provider served part of the prompt from cache. `CacheWriteTokens > 0` means the provider stored new content in the cache for future requests.
