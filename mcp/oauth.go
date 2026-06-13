package mcp

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"

	"github.com/zendev-sh/goai/provider"
)

// ErrAuthorizationRequired is returned by an OAuth [provider.TokenSource] when
// no usable token is stored and no [OAuthConfig.Authorize] callback was
// provided to run the interactive authorization-code flow.
var ErrAuthorizationRequired = errors.New("mcp: oauth authorization required but no Authorize callback configured")

// AuthServerMetadata holds the OAuth 2.0 Authorization Server endpoints
// discovered for an MCP server, per RFC 8414 (Authorization Server Metadata).
type AuthServerMetadata struct {
	// Issuer is the authorization server's issuer identifier.
	Issuer string `json:"issuer,omitempty"`

	// AuthorizationEndpoint is the URL the user-agent is sent to in order to
	// obtain an authorization grant (RFC 6749 §3.1).
	AuthorizationEndpoint string `json:"authorization_endpoint"`

	// TokenEndpoint is the URL used to exchange a grant for tokens and to
	// refresh access tokens (RFC 6749 §3.2).
	TokenEndpoint string `json:"token_endpoint"`

	// RegistrationEndpoint is the optional Dynamic Client Registration endpoint
	// (RFC 7591). Empty if the server does not advertise one.
	RegistrationEndpoint string `json:"registration_endpoint,omitempty"`
}

// DiscoverAuth resolves the OAuth Authorization Server for an MCP server URL by
// following the MCP authorization discovery chain:
//
//	Protected Resource Metadata (RFC 9728): the document naming the
//	authorization server is located, in order, by
//	  1. the resource_metadata pointer in the WWW-Authenticate header of an
//	     unauthenticated probe (RFC 9728 §5.1), and
//	  2. the well-known URIs {origin}/.well-known/oauth-protected-resource{/path}
//	     (path-aware) and {origin}/.well-known/oauth-protected-resource (RFC 9728
//	     §3.1).
//	Its authorization_servers[0] is the Authorization Server base URL.
//
//	Authorization Server Metadata: the endpoints are fetched from the
//	.well-known/oauth-authorization-server (RFC 8414) and
//	.well-known/openid-configuration (OpenID Connect Discovery) documents, each
//	tried in host-insert and path-append form.
//
// If no authorization server is found, the MCP server's own origin is tried as a
// legacy fallback (some servers skip the indirection). hc may be nil, in which
// case http.DefaultClient is used.
func DiscoverAuth(ctx context.Context, serverURL string, hc *http.Client) (*AuthServerMetadata, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	origin, err := originOf(serverURL)
	if err != nil {
		return nil, err
	}

	// RFC 9728 — find the Authorization Server URL (best-effort).
	asURL := discoverAuthServerURL(ctx, hc, serverURL, origin)

	// RFC 8414 / OpenID Connect Discovery — fetch AS metadata.
	for _, candidate := range authServerMetadataURLs(asURL, origin) {
		md, err := getJSON(ctx, hc, candidate)
		if err != nil {
			continue
		}
		authEndpoint, _ := md["authorization_endpoint"].(string)
		if authEndpoint == "" {
			continue
		}
		return &AuthServerMetadata{
			Issuer:                strFromJSON(md["issuer"]),
			AuthorizationEndpoint: authEndpoint,
			TokenEndpoint:         strFromJSON(md["token_endpoint"]),
			RegistrationEndpoint:  strFromJSON(md["registration_endpoint"]),
		}, nil
	}

	return nil, fmt.Errorf("mcp: no OAuth authorization server metadata found for %s", serverURL)
}

// resourceMetadataRe extracts the resource_metadata pointer from a
// WWW-Authenticate challenge (RFC 9728 §5.1).
var resourceMetadataRe = regexp.MustCompile(`resource_metadata="([^"]+)"`)

// discoverAuthServerURL resolves authorization_servers[0] from the Protected
// Resource Metadata (RFC 9728): it tries the WWW-Authenticate pointer first,
// then the well-known URIs. Returns "" when none is found.
func discoverAuthServerURL(ctx context.Context, hc *http.Client, serverURL, origin string) string {
	candidates := make([]string, 0, 3)
	if ptr := resourceMetadataPointer(ctx, hc, serverURL); ptr != "" {
		candidates = append(candidates, ptr)
	}
	candidates = append(candidates, protectedResourceWellKnown(serverURL, origin)...)

	for _, u := range candidates {
		md, err := getJSON(ctx, hc, u)
		if err != nil {
			continue
		}
		servers, ok := md["authorization_servers"].([]any)
		if !ok || len(servers) == 0 {
			continue
		}
		if as, _ := servers[0].(string); as != "" {
			return as
		}
	}
	return ""
}

// resourceMetadataPointer probes the MCP server unauthenticated and returns the
// resource_metadata URL advertised in the WWW-Authenticate header of its 401
// challenge, or "" when the server does not advertise one.
func resourceMetadataPointer(ctx context.Context, hc *http.Client, serverURL string) string {
	body := strings.NewReader(fmt.Sprintf(
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":%q,"capabilities":{},"clientInfo":{"name":"goai","version":"1.0"}}}`,
		DefaultProtocolVersion,
	))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, serverURL, body)
	if err != nil {
		return ""
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := hc.Do(req)
	if err != nil {
		return ""
	}
	defer func() { _ = resp.Body.Close() }()
	if m := resourceMetadataRe.FindStringSubmatch(resp.Header.Get("WWW-Authenticate")); len(m) == 2 {
		return m[1]
	}
	return ""
}

// protectedResourceWellKnown returns the RFC 9728 §3.1 well-known URIs for the
// Protected Resource Metadata of serverURL: the path-aware form first (the
// resource path is inserted after the well-known segment, e.g. a server at
// /mcp/ publishes at /.well-known/oauth-protected-resource/mcp/), then the bare
// form at the origin.
func protectedResourceWellKnown(serverURL, origin string) []string {
	const seg = "/.well-known/oauth-protected-resource"
	out := make([]string, 0, 2)
	if path := resourcePath(serverURL); path != "" {
		out = append(out, origin+seg+path)
	}
	return append(out, origin+seg)
}

// authServerMetadataURLs lists the candidate AS metadata documents for an
// authorization server (asURL), with the MCP origin appended as a legacy
// fallback. It covers both RFC 8414 (oauth-authorization-server) and OpenID
// Connect Discovery (openid-configuration), each in host-insert form (the
// well-known segment inserted between host and path, RFC 8414 §3.1) and
// path-append form (appended to the issuer path, OIDC Discovery).
func authServerMetadataURLs(asURL, origin string) []string {
	bases := make([]string, 0, 2)
	if asURL != "" {
		bases = append(bases, asURL)
	}
	bases = append(bases, origin)

	out := make([]string, 0, 8)
	seen := map[string]bool{}
	add := func(u string) {
		if u != "" && !seen[u] {
			seen[u] = true
			out = append(out, u)
		}
	}
	for _, base := range bases {
		b := strings.TrimRight(base, "/")
		bOrigin, bPath := originAndPath(b)
		for _, seg := range []string{"/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"} {
			if bPath != "" {
				add(bOrigin + seg + bPath)
			}
			add(b + seg)
		}
	}
	return out
}

// ClientRegistration describes a public OAuth client to register dynamically.
type ClientRegistration struct {
	// ClientName is a human-readable name for the client.
	ClientName string

	// RedirectURIs are the allowed redirect URIs for the authorization-code flow.
	RedirectURIs []string

	// Scopes is the optional set of scopes the client will request.
	Scopes []string
}

// RegisterClient performs OAuth 2.0 Dynamic Client Registration (RFC 7591)
// against registrationEndpoint and returns the server-assigned client_id.
//
// The client is registered as a public client (token_endpoint_auth_method
// "none") supporting the authorization_code and refresh_token grants.
//
// hc may be nil, in which case http.DefaultClient is used.
func RegisterClient(ctx context.Context, registrationEndpoint string, reg ClientRegistration, hc *http.Client) (clientID string, err error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	body := map[string]any{
		"client_name":                reg.ClientName,
		"redirect_uris":              reg.RedirectURIs,
		"grant_types":                []string{"authorization_code", "refresh_token"},
		"response_types":             []string{"code"},
		"token_endpoint_auth_method": "none",
	}
	if len(reg.Scopes) > 0 {
		body["scope"] = strings.Join(reg.Scopes, " ")
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("mcp: marshal registration request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, registrationEndpoint, strings.NewReader(string(payload)))
	if err != nil {
		return "", fmt.Errorf("mcp: create registration request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := hc.Do(req)
	if err != nil {
		return "", fmt.Errorf("mcp: dynamic client registration: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("mcp: dynamic client registration: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var parsed struct {
		ClientID string `json:"client_id"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", fmt.Errorf("mcp: parse registration response: %w", err)
	}
	if parsed.ClientID == "" {
		return "", errors.New("mcp: registration response missing client_id")
	}
	return parsed.ClientID, nil
}

// GeneratePKCE returns a PKCE code_verifier and its S256 code_challenge per
// RFC 7636, where challenge = BASE64URL(SHA-256(verifier)).
//
// The token sources in this package generate PKCE values internally; this
// helper is exposed for callers that drive the authorization flow themselves.
func GeneratePKCE() (verifier, challenge string) {
	verifier = oauth2.GenerateVerifier()
	return verifier, oauth2.S256ChallengeFromVerifier(verifier)
}

// TokenStore persists OAuth tokens keyed by MCP server URL. Implementations
// must be safe for concurrent use. The in-memory default ([NewMemoryTokenStore])
// is suitable for a single process; persist tokens (e.g. encrypted on disk) by
// supplying a custom implementation via [OAuthConfig.Store].
type TokenStore interface {
	// Get returns the stored token for key and whether one was found.
	Get(key string) (*oauth2.Token, bool)

	// Set stores tok for key, replacing any existing token.
	Set(key string, tok *oauth2.Token) error

	// Delete removes any stored token for key.
	Delete(key string) error
}

// NewMemoryTokenStore returns an in-memory [TokenStore].
func NewMemoryTokenStore() TokenStore {
	return &memoryTokenStore{tokens: make(map[string]*oauth2.Token)}
}

type memoryTokenStore struct {
	mu     sync.RWMutex
	tokens map[string]*oauth2.Token
}

func (s *memoryTokenStore) Get(key string) (*oauth2.Token, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	tok, ok := s.tokens[key]
	return tok, ok
}

func (s *memoryTokenStore) Set(key string, tok *oauth2.Token) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tokens[key] = tok
	return nil
}

func (s *memoryTokenStore) Delete(key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.tokens, key)
	return nil
}

// AuthorizeFunc runs the interactive OAuth authorization-code step. It receives
// the authorization URL to present to the user and must return the redirect URL
// the authorization server sent back (which carries the "code" and "state"
// query parameters). Returning the full redirect URL lets this package validate
// the state parameter; returning the bare authorization code is also accepted,
// in which case state validation is skipped.
type AuthorizeFunc func(ctx context.Context, authURL string) (redirectURL string, err error)

// OAuthConfig configures an OAuth [provider.TokenSource] for an MCP server.
type OAuthConfig struct {
	// ServerURL is the MCP server URL. It is used both as the discovery target
	// and as the key under which tokens are stored.
	ServerURL string

	// Metadata, when set, supplies the authorization server endpoints directly
	// and skips discovery. When nil, [DiscoverAuth] is called for ServerURL.
	Metadata *AuthServerMetadata

	// ClientID is the OAuth client_id. When empty, dynamic client registration
	// (RFC 7591) is attempted using the server's registration_endpoint.
	ClientID string

	// ClientName is the human-readable client name used for dynamic registration.
	ClientName string

	// RedirectURI is the redirect URI used in the authorization-code flow. It
	// must match a URI the client is registered with.
	RedirectURI string

	// Scopes are the OAuth scopes to request.
	Scopes []string

	// Store persists tokens across calls. When nil, an in-memory store is used.
	Store TokenStore

	// HTTPClient is used for discovery, registration, and token requests. When
	// nil, http.DefaultClient is used.
	HTTPClient *http.Client

	// Authorize runs the interactive authorization step when no usable token is
	// available. When nil, the token source returns [ErrAuthorizationRequired]
	// instead of starting a flow (useful when only a stored token is expected).
	Authorize AuthorizeFunc
}

// NewOAuthTokenSource builds a [provider.TokenSource] that authenticates to an
// MCP server using OAuth 2.0 with PKCE. It discovers the authorization server
// (unless Metadata is supplied), performs dynamic client registration (unless
// ClientID is supplied), and returns a token source that:
//
//   - returns the stored access token while it is valid;
//   - refreshes it using the refresh token when expired;
//   - runs the interactive authorization-code flow via [OAuthConfig.Authorize]
//     when no usable token exists.
//
// The returned token source implements [provider.InvalidatingTokenSource]; its
// Invalidate method forces a refresh on the next call, which [NewOAuthHTTPClient]
// uses to recover from HTTP 401 responses.
func NewOAuthTokenSource(ctx context.Context, cfg OAuthConfig) (provider.TokenSource, error) {
	if cfg.ServerURL == "" {
		return nil, errors.New("mcp: OAuthConfig.ServerURL is required")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}

	md := cfg.Metadata
	if md == nil {
		var err error
		md, err = DiscoverAuth(ctx, cfg.ServerURL, hc)
		if err != nil {
			return nil, err
		}
	}

	clientID := cfg.ClientID
	if clientID == "" {
		if md.RegistrationEndpoint == "" {
			return nil, errors.New("mcp: no ClientID provided and server has no registration_endpoint")
		}
		var err error
		clientID, err = RegisterClient(ctx, md.RegistrationEndpoint, ClientRegistration{
			ClientName:   cfg.ClientName,
			RedirectURIs: []string{cfg.RedirectURI},
			Scopes:       cfg.Scopes,
		}, hc)
		if err != nil {
			return nil, err
		}
	}

	store := cfg.Store
	if store == nil {
		store = NewMemoryTokenStore()
	}

	return &oauthTokenSource{
		cfg: &oauth2.Config{
			ClientID:    clientID,
			RedirectURL: cfg.RedirectURI,
			Scopes:      cfg.Scopes,
			Endpoint: oauth2.Endpoint{
				AuthURL:  md.AuthorizationEndpoint,
				TokenURL: md.TokenEndpoint,
			},
		},
		store:     store,
		key:       cfg.ServerURL,
		authorize: cfg.Authorize,
		hc:        hc,
	}, nil
}

var _ provider.InvalidatingTokenSource = (*oauthTokenSource)(nil)

type oauthTokenSource struct {
	cfg       *oauth2.Config
	store     TokenStore
	key       string
	authorize AuthorizeFunc
	hc        *http.Client

	mu    sync.Mutex
	force bool // forces a refresh on the next Token call, even if not yet expired
}

func (s *oauthTokenSource) Token(ctx context.Context) (string, error) {
	s.mu.Lock()
	force := s.force
	s.force = false
	s.mu.Unlock()

	// Route oauth2's internal HTTP calls through the configured client.
	ctx = context.WithValue(ctx, oauth2.HTTPClient, s.hc)

	if tok, ok := s.store.Get(s.key); ok && tok.AccessToken != "" {
		if !force && tok.Valid() {
			return tok.AccessToken, nil
		}
		if tok.RefreshToken != "" {
			refreshFrom := tok
			if force {
				refreshFrom = expired(tok)
			}
			// oauth2's TokenSource refreshes lock-free and only when needed.
			newTok, err := s.cfg.TokenSource(ctx, refreshFrom).Token()
			if err == nil {
				_ = s.store.Set(s.key, newTok)
				return newTok.AccessToken, nil
			}
			// Refresh failed (e.g. refresh token revoked) — fall through to
			// a fresh authorization flow.
		}
	}

	return s.authorizeAndStore(ctx)
}

func (s *oauthTokenSource) Invalidate() {
	s.mu.Lock()
	s.force = true
	s.mu.Unlock()
}

func (s *oauthTokenSource) authorizeAndStore(ctx context.Context) (string, error) {
	if s.authorize == nil {
		return "", ErrAuthorizationRequired
	}

	verifier := oauth2.GenerateVerifier()
	state, err := randomState()
	if err != nil {
		return "", err
	}
	authURL := s.cfg.AuthCodeURL(state, oauth2.S256ChallengeOption(verifier))

	redirect, err := s.authorize(ctx, authURL)
	if err != nil {
		return "", fmt.Errorf("mcp: authorization: %w", err)
	}
	code, err := codeFromRedirect(redirect, state)
	if err != nil {
		return "", err
	}

	tok, err := s.cfg.Exchange(ctx, code, oauth2.VerifierOption(verifier))
	if err != nil {
		return "", fmt.Errorf("mcp: token exchange: %w", err)
	}
	if err := s.store.Set(s.key, tok); err != nil {
		return "", err
	}
	return tok.AccessToken, nil
}

// NewOAuthHTTPClient returns an *http.Client that attaches the token from ts as
// a Bearer Authorization header to every request. When a request receives an
// HTTP 401 response and ts implements [provider.InvalidatingTokenSource], the
// token is invalidated and the request is retried once with a fresh token.
//
// Pass the result to [WithHTTPClient] or [WithSSEHTTPClient] to authenticate a
// remote MCP transport:
//
//	ts, _ := mcp.NewOAuthTokenSource(ctx, cfg)
//	hc := mcp.NewOAuthHTTPClient(ts, nil)
//	transport := mcp.NewHTTPTransport(serverURL, mcp.WithHTTPClient(hc))
//
// base may be nil, in which case http.DefaultClient is used as the underlying
// client.
func NewOAuthHTTPClient(ts provider.TokenSource, base *http.Client) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	baseTransport := base.Transport
	if baseTransport == nil {
		baseTransport = http.DefaultTransport
	}
	c := *base
	c.Transport = &oauthRoundTripper{base: baseTransport, ts: ts}
	return &c
}

type oauthRoundTripper struct {
	base http.RoundTripper
	ts   provider.TokenSource
}

func (rt *oauthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := rt.ts.Token(req.Context())
	if err != nil {
		return nil, err
	}
	resp, err := rt.base.RoundTrip(withBearer(req, token))
	if err != nil {
		return resp, err
	}

	if resp.StatusCode != http.StatusUnauthorized {
		return resp, nil
	}
	inv, ok := rt.ts.(provider.InvalidatingTokenSource)
	if !ok {
		return resp, nil
	}

	// Drop the unauthorized response and retry once with a refreshed token.
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	inv.Invalidate()

	token, err = rt.ts.Token(req.Context())
	if err != nil {
		return nil, err
	}
	retryReq, err := cloneWithBody(req)
	if err != nil {
		return nil, err
	}
	return rt.base.RoundTrip(withBearer(retryReq, token))
}

// withBearer clones req and sets the Authorization header, leaving the original
// request (and its headers) untouched.
func withBearer(req *http.Request, token string) *http.Request {
	r := req.Clone(req.Context())
	r.Header.Set("Authorization", "Bearer "+token)
	return r
}

// cloneWithBody clones req with a fresh body for a retry. Requests built with
// net/http carry a GetBody for in-memory bodies (bytes/strings readers), which
// the MCP transports use, so the retry can re-send the payload.
func cloneWithBody(req *http.Request) (*http.Request, error) {
	r := req.Clone(req.Context())
	if req.Body != nil && req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return nil, fmt.Errorf("mcp: rewind request body for retry: %w", err)
		}
		r.Body = body
	}
	return r, nil
}

// codeFromRedirect extracts the authorization code from the value returned by an
// [AuthorizeFunc]. If the value parses as a URL carrying query parameters, the
// "error" parameter is surfaced and "state" is validated against want. If it
// carries no recognizable query parameters, it is treated as the bare code.
func codeFromRedirect(redirect, want string) (string, error) {
	if redirect == "" {
		return "", errors.New("mcp: authorization returned empty redirect")
	}
	// Treat the value as a redirect URL only when it clearly is one (scheme and
	// host present); otherwise it is the bare authorization code.
	u, err := url.Parse(redirect)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return redirect, nil
	}
	q := u.Query()
	if e := q.Get("error"); e != "" {
		if desc := q.Get("error_description"); desc != "" {
			return "", fmt.Errorf("mcp: authorization error %q: %s", e, desc)
		}
		return "", fmt.Errorf("mcp: authorization error %q", e)
	}
	if got := q.Get("state"); got != want {
		return "", errors.New("mcp: authorization state mismatch")
	}
	code := q.Get("code")
	if code == "" {
		return "", errors.New("mcp: authorization redirect missing code")
	}
	return code, nil
}

// expired returns a shallow copy of tok with its expiry set in the past, so
// oauth2's TokenSource treats it as invalid and performs a refresh.
func expired(tok *oauth2.Token) *oauth2.Token {
	clone := *tok
	clone.Expiry = time.Now().Add(-time.Hour)
	return &clone
}

// randomState returns a cryptographically random, URL-safe CSRF state token.
func randomState() (string, error) {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("mcp: generate state: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// originOf returns the scheme://host[:port] origin of a URL.
func originOf(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("mcp: parse server URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("mcp: invalid server URL %q", raw)
	}
	return u.Scheme + "://" + u.Host, nil
}

// resourcePath returns the path component of raw (e.g. "/mcp/"), or "" when it
// is empty or the bare root.
func resourcePath(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Path == "" || u.Path == "/" {
		return ""
	}
	return u.Path
}

// originAndPath splits raw into its origin and trailing-slash-trimmed path. On a
// parse failure or a URL without scheme/host it returns (raw, "").
func originAndPath(raw string) (origin, path string) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return raw, ""
	}
	return u.Scheme + "://" + u.Host, strings.TrimRight(u.Path, "/")
}

// getJSON performs a GET and decodes a JSON object body. Non-200 responses and
// non-object bodies return an error.
func getJSON(ctx context.Context, hc *http.Client, url string) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("mcp: GET %s: HTTP %d", url, resp.StatusCode)
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// strFromJSON returns v as a string when it is one, else "".
func strFromJSON(v any) string {
	s, _ := v.(string)
	return s
}
