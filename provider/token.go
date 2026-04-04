package provider

import (
	"context"
	"sync"
	"time"
)

// TokenSource provides authentication tokens for API requests.
// Providers accept a TokenSource to support dynamic credentials
// (OAuth, service accounts, device flow) beyond static API keys.
type TokenSource interface {
	// Token returns a valid token. Implementations must be safe for concurrent use.
	Token(ctx context.Context) (string, error)
}

// InvalidatingTokenSource is a TokenSource whose cached token can be cleared,
// forcing a fresh fetch on the next call. Supports application-level retry-on-401 logic.
type InvalidatingTokenSource interface {
	TokenSource
	Invalidate()
}

// Token represents an authentication token with optional expiry.
type Token struct {
	// Value is the token string (API key, OAuth access token, etc.).
	Value string

	// ExpiresAt is when the token expires. Zero value means no expiry.
	ExpiresAt time.Time
}

// TokenFetchFunc fetches a fresh token. Used by CachedTokenSource.
type TokenFetchFunc func(ctx context.Context) (*Token, error)

// StaticToken creates a TokenSource that always returns the given key.
// Use this for simple API key authentication.
func StaticToken(key string) TokenSource {
	return &staticToken{key: key}
}

var _ TokenSource = (*staticToken)(nil)

type staticToken struct {
	key string
}

func (s *staticToken) Token(_ context.Context) (string, error) {
	return s.key, nil
}

// CachedTokenSource creates a TokenSource that caches tokens until expiry.
// The fetchFn is called lazily on first use and again when the cached token expires.
// It is safe for concurrent use.
//
// The returned TokenSource also implements InvalidatingTokenSource,
// allowing application-level retry-on-401 logic to force a token refresh.
func CachedTokenSource(fetchFn TokenFetchFunc) TokenSource {
	return &cachedTokenSource{fetch: fetchFn}
}

var _ InvalidatingTokenSource = (*cachedTokenSource)(nil)

type cachedTokenSource struct {
	fetch  TokenFetchFunc
	mu     sync.RWMutex
	cached *Token
}

func (c *cachedTokenSource) Token(ctx context.Context) (string, error) {
	c.mu.RLock()
	if c.cached != nil && (c.cached.ExpiresAt.IsZero() || time.Now().Before(c.cached.ExpiresAt)) {
		val := c.cached.Value
		c.mu.RUnlock()
		return val, nil
	}
	c.mu.RUnlock()

	// Fetch outside lock to avoid blocking concurrent callers during network calls.
	// Brief double-fetch is acceptable for token refresh.
	token, err := c.fetch(ctx)
	if err != nil {
		return "", err
	}

	c.mu.Lock()
	// Only write back if cache is still empty or expired - a concurrent
	// goroutine may have already written a fresher token (e.g. after Invalidate).
	if c.cached == nil || (!c.cached.ExpiresAt.IsZero() && !time.Now().Before(c.cached.ExpiresAt)) {
		c.cached = token
	}
	c.mu.Unlock()
	return token.Value, nil
}

func (c *cachedTokenSource) Invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cached = nil
}
