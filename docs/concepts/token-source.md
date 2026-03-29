---
title: TokenSource
description: "Manage LLM authentication with GoAI's TokenSource interface. Support static API keys, OAuth tokens, Azure Managed Identity, and GCP credentials."
---

# TokenSource

Most providers authenticate with a static API key. But some environments use expiring tokens: Azure Managed Identity, Google service accounts, OAuth flows. `TokenSource` abstracts over both patterns so provider code never needs to know the difference.

## The Interface

```go
type TokenSource interface {
    Token(ctx context.Context) (string, error)
}
```

Most providers accept a `TokenSource` via `WithTokenSource` (exceptions: Bedrock uses AWS credential options; Ollama requires no auth). The provider calls `Token()` before each API request to get the current credential.

## Static API Keys

For static keys, use `StaticToken`:

```go
import "github.com/zendev-sh/goai/provider"

model := openai.Chat("gpt-4o",
    openai.WithTokenSource(provider.StaticToken("sk-...")),
)
```

`WithAPIKey` is shorthand for the same thing:

```go
model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))
```

If neither is provided, providers read from environment variables automatically.

## CachedTokenSource

For expiring tokens, use `CachedTokenSource`. It wraps a fetch function that returns a `*Token` with an optional expiry. The token is cached and reused until it expires, then a fresh one is fetched.

```go
import "github.com/zendev-sh/goai/provider"

ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
    // Fetch a fresh token from your identity provider.
    tok, exp, err := myIdentityProvider.GetToken(ctx)
    if err != nil {
        return nil, err
    }
    return &provider.Token{
        Value:     tok,
        ExpiresAt: exp,
    }, nil
})

model := azure.Chat("gpt-4o",
    azure.WithTokenSource(ts),
)
```

Key properties:

- **Thread-safe.** Multiple goroutines can call `Token()` concurrently.
- **Lazy.** The fetch function is not called until the first `Token()` call.
- **TTL-based.** The cached token is reused as long as `time.Now()` is before `ExpiresAt`. Set `ExpiresAt` to zero for tokens that never expire.
- **Non-blocking refresh.** The fetch function is called outside any lock. Concurrent callers during a refresh are never blocked waiting on each other's network call (brief double-fetch is acceptable).

## Token Invalidation

`CachedTokenSource` also implements `InvalidatingTokenSource`:

```go
type InvalidatingTokenSource interface {
    TokenSource
    Invalidate()
}
```

Calling `Invalidate()` clears the cached token, forcing the next `Token()` call to re-fetch. This is useful for application-level handling when a token is rejected (e.g., clearing a cached token after a 401 response in your own retry logic).

> **Note:** `CachedTokenSource` returns the `TokenSource` interface, so use a type assertion to access `Invalidate()`:
> ```go
> ts := provider.CachedTokenSource(fetchFunc)
> // ... later, to invalidate:
> ts.(provider.InvalidatingTokenSource).Invalidate()
> ```

## Examples

### Azure Managed Identity

```go
ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
    cred, err := azidentity.NewDefaultAzureCredential(nil)
    if err != nil {
        return nil, err
    }
    token, err := cred.GetToken(ctx, policy.TokenRequestOptions{
        Scopes: []string{"https://cognitiveservices.azure.com/.default"},
    })
    if err != nil {
        return nil, err
    }
    return &provider.Token{
        Value:     token.Token,
        ExpiresAt: token.ExpiresOn,
    }, nil
})

model := azure.Chat("gpt-4o", azure.WithTokenSource(ts))
```

### Google Service Account

> **Note:** The Google Gemini provider (`provider/google`) sends all tokens via the `x-goog-api-key` header, which only accepts API keys — not OAuth2 Bearer tokens. For OAuth2 or service account authentication with Google, use the Vertex AI provider (`provider/vertex`) instead, which supports standard Bearer auth.

```go
ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
    tokenSource, err := google.DefaultTokenSource(ctx,
        "https://www.googleapis.com/auth/cloud-platform",
    )
    if err != nil {
        return nil, err
    }
    tok, err := tokenSource.Token()
    if err != nil {
        return nil, err
    }
    return &provider.Token{
        Value:     tok.AccessToken,
        ExpiresAt: tok.Expiry,
    }, nil
})

model := vertex.Chat("gemini-2.0-flash", vertex.WithTokenSource(ts))
```

### Custom OAuth

```go
ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
    resp, err := http.PostForm("https://auth.example.com/token", url.Values{
        "grant_type":    {"client_credentials"},
        "client_id":     {clientID},
        "client_secret": {clientSecret},
    })
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var body struct {
        AccessToken string `json:"access_token"`
        ExpiresIn   int    `json:"expires_in"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
        return nil, err
    }
    return &provider.Token{
        Value:     body.AccessToken,
        ExpiresAt: time.Now().Add(time.Duration(body.ExpiresIn) * time.Second),
    }, nil
})
```
