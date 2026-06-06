package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/oauth2"

	"github.com/zendev-sh/goai/provider"
)

// oauthServer is a configurable mock that serves the discovery, registration,
// and token endpoints needed by the OAuth token source.
type oauthServer struct {
	mu sync.Mutex

	srv *httptest.Server

	// toggles
	protectedResource bool // serve RFC 9728 pointing at this server
	asMetadata        bool // serve RFC 8414
	registration      bool // serve registration endpoint

	// captured / canned values
	issuedClientID   string
	lastTokenForm    url.Values
	refreshCount     int32
	accessToken      string
	refreshToken     string
	tokenExpiresIn   int
	registrationCode int  // optional override status for registration
	failRefresh      bool // reject refresh_token grants with HTTP 400
	failExchange     bool // reject authorization_code grants with HTTP 400
}

func newOAuthServer() *oauthServer {
	o := &oauthServer{
		protectedResource: true,
		asMetadata:        true,
		registration:      true,
		issuedClientID:    "dyn-client-123",
		accessToken:       "access-1",
		refreshToken:      "refresh-1",
		tokenExpiresIn:    3600,
	}
	o.srv = httptest.NewServer(http.HandlerFunc(o.handle))
	return o
}

func (o *oauthServer) URL() string { return o.srv.URL + "/mcp" }
func (o *oauthServer) Close()      { o.srv.Close() }

func (o *oauthServer) handle(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/.well-known/oauth-protected-resource":
		if !o.protectedResource {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, map[string]any{
			"authorization_servers": []string{o.srv.URL},
		})
	case "/.well-known/oauth-authorization-server":
		if !o.asMetadata {
			http.NotFound(w, r)
			return
		}
		md := map[string]any{
			"issuer":                 o.srv.URL,
			"authorization_endpoint": o.srv.URL + "/authorize",
			"token_endpoint":         o.srv.URL + "/token",
		}
		if o.registration {
			md["registration_endpoint"] = o.srv.URL + "/register"
		}
		writeJSON(w, md)
	case "/register":
		if o.registrationCode != 0 {
			w.WriteHeader(o.registrationCode)
			return
		}
		writeJSON(w, map[string]any{"client_id": o.issuedClientID})
	case "/token":
		_ = r.ParseForm()
		o.mu.Lock()
		o.lastTokenForm = r.PostForm
		isRefresh := r.PostForm.Get("grant_type") == "refresh_token"
		if isRefresh {
			atomic.AddInt32(&o.refreshCount, 1)
		}
		failRefresh := o.failRefresh
		access := o.accessToken
		refresh := o.refreshToken
		expiresIn := o.tokenExpiresIn
		o.mu.Unlock()
		if isRefresh && failRefresh {
			http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
			return
		}
		o.mu.Lock()
		failExchange := o.failExchange
		o.mu.Unlock()
		if !isRefresh && failExchange {
			http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"access_token":  access,
			"token_type":    "Bearer",
			"refresh_token": refresh,
		}
		if expiresIn > 0 {
			resp["expires_in"] = expiresIn
		}
		writeJSON(w, resp)
	default:
		http.NotFound(w, r)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// ── DiscoverAuth ────────────────────────────────────────────────────────────

func TestDiscoverAuth_Chain(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	md, err := DiscoverAuth(context.Background(), o.URL(), nil)
	if err != nil {
		t.Fatalf("DiscoverAuth: %v", err)
	}
	if md.AuthorizationEndpoint != o.srv.URL+"/authorize" {
		t.Errorf("AuthorizationEndpoint = %q", md.AuthorizationEndpoint)
	}
	if md.TokenEndpoint != o.srv.URL+"/token" {
		t.Errorf("TokenEndpoint = %q", md.TokenEndpoint)
	}
	if md.RegistrationEndpoint != o.srv.URL+"/register" {
		t.Errorf("RegistrationEndpoint = %q", md.RegistrationEndpoint)
	}
	if md.Issuer != o.srv.URL {
		t.Errorf("Issuer = %q", md.Issuer)
	}
}

func TestDiscoverAuth_FallbackToOrigin(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()
	o.protectedResource = false // no RFC 9728; must fall back to origin RFC 8414

	md, err := DiscoverAuth(context.Background(), o.URL(), o.srv.Client())
	if err != nil {
		t.Fatalf("DiscoverAuth: %v", err)
	}
	if md.AuthorizationEndpoint == "" {
		t.Error("expected authorization endpoint via origin fallback")
	}
}

func TestDiscoverAuth_NotFound(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()
	o.protectedResource = false
	o.asMetadata = false

	if _, err := DiscoverAuth(context.Background(), o.URL(), nil); err == nil {
		t.Fatal("expected error when no metadata is served")
	}
}

func TestDiscoverAuth_InvalidURL(t *testing.T) {
	if _, err := DiscoverAuth(context.Background(), "not-a-url", nil); err == nil {
		t.Fatal("expected error for URL without scheme/host")
	}
}

// ── RegisterClient ──────────────────────────────────────────────────────────

func TestRegisterClient(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	id, err := RegisterClient(context.Background(), o.srv.URL+"/register", ClientRegistration{
		ClientName:   "goai-test",
		RedirectURIs: []string{"http://localhost/cb"},
		Scopes:       []string{"a", "b"},
	}, nil)
	if err != nil {
		t.Fatalf("RegisterClient: %v", err)
	}
	if id != o.issuedClientID {
		t.Errorf("client_id = %q, want %q", id, o.issuedClientID)
	}
}

func TestRegisterClient_HTTPError(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()
	o.registrationCode = http.StatusBadRequest

	if _, err := RegisterClient(context.Background(), o.srv.URL+"/register", ClientRegistration{}, nil); err == nil {
		t.Fatal("expected error on 400 registration response")
	}
}

func TestRegisterClient_MissingClientID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{})
	}))
	defer srv.Close()

	if _, err := RegisterClient(context.Background(), srv.URL, ClientRegistration{}, nil); err == nil {
		t.Fatal("expected error when response omits client_id")
	}
}

// ── PKCE ────────────────────────────────────────────────────────────────────

func TestGeneratePKCE(t *testing.T) {
	verifier, challenge := GeneratePKCE()
	if verifier == "" || challenge == "" {
		t.Fatal("empty PKCE values")
	}
	sum := sha256.Sum256([]byte(verifier))
	want := base64.RawURLEncoding.EncodeToString(sum[:])
	if challenge != want {
		t.Errorf("challenge = %q, want %q", challenge, want)
	}
	if v2, _ := GeneratePKCE(); v2 == verifier {
		t.Error("expected distinct verifiers across calls")
	}
}

// ── MemoryTokenStore ────────────────────────────────────────────────────────

func TestMemoryTokenStore(t *testing.T) {
	s := NewMemoryTokenStore()
	if _, ok := s.Get("k"); ok {
		t.Fatal("expected miss on empty store")
	}
	tok := &oauth2.Token{AccessToken: "x"}
	if err := s.Set("k", tok); err != nil {
		t.Fatal(err)
	}
	got, ok := s.Get("k")
	if !ok || got.AccessToken != "x" {
		t.Fatalf("Get = %v, %v", got, ok)
	}
	if err := s.Delete("k"); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("k"); ok {
		t.Fatal("expected miss after Delete")
	}
}

// ── NewOAuthTokenSource: full authorization flow ────────────────────────────

func TestOAuthTokenSource_AuthorizeFlow(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	var seenAuthURL string
	cfg := OAuthConfig{
		ServerURL:   o.URL(),
		ClientName:  "goai-test",
		RedirectURI: "http://localhost:0/cb",
		Scopes:      []string{"read"},
		HTTPClient:  o.srv.Client(),
		Authorize: func(_ context.Context, authURL string) (string, error) {
			seenAuthURL = authURL
			// Echo back a redirect carrying the state the SDK generated.
			u, _ := url.Parse(authURL)
			state := u.Query().Get("state")
			return "http://localhost:0/cb?code=the-code&state=" + url.QueryEscape(state), nil
		},
	}

	ts, err := NewOAuthTokenSource(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	tok, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok != o.accessToken {
		t.Errorf("token = %q, want %q", tok, o.accessToken)
	}

	// Authorization URL must carry PKCE S256 + client_id from dynamic registration.
	au, _ := url.Parse(seenAuthURL)
	q := au.Query()
	if q.Get("code_challenge_method") != "S256" || q.Get("code_challenge") == "" {
		t.Error("authorization URL missing PKCE challenge")
	}
	if q.Get("client_id") != o.issuedClientID {
		t.Errorf("client_id = %q, want %q", q.Get("client_id"), o.issuedClientID)
	}
	// Token endpoint must have received the code_verifier.
	if o.lastTokenForm.Get("code_verifier") == "" {
		t.Error("token exchange missing code_verifier")
	}

	// Second call returns the cached token without re-authorizing.
	o.mu.Lock()
	o.accessToken = "should-not-be-used"
	o.mu.Unlock()
	tok2, err := ts.Token(context.Background())
	if err != nil || tok2 != "access-1" {
		t.Fatalf("cached Token = %q, %v", tok2, err)
	}
}

func TestOAuthTokenSource_NoAuthorizeCallback(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		HTTPClient: o.srv.Client(),
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	if _, err := ts.Token(context.Background()); !errors.Is(err, ErrAuthorizationRequired) {
		t.Fatalf("err = %v, want ErrAuthorizationRequired", err)
	}
}

func TestOAuthTokenSource_RefreshOnExpiry(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	store := NewMemoryTokenStore()
	_ = store.Set(o.URL(), &oauth2.Token{
		AccessToken:  "old",
		RefreshToken: "refresh-1",
		Expiry:       time.Now().Add(-time.Minute), // already expired
	})

	o.mu.Lock()
	o.accessToken = "refreshed"
	o.mu.Unlock()

	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		Metadata:   &AuthServerMetadata{AuthorizationEndpoint: o.srv.URL + "/authorize", TokenEndpoint: o.srv.URL + "/token"},
		Store:      store,
		HTTPClient: o.srv.Client(),
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	tok, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok != "refreshed" {
		t.Errorf("token = %q, want refreshed", tok)
	}
	if atomic.LoadInt32(&o.refreshCount) != 1 {
		t.Errorf("refresh count = %d, want 1", o.refreshCount)
	}
}

func TestOAuthTokenSource_InvalidateForcesRefresh(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	store := NewMemoryTokenStore()
	_ = store.Set(o.URL(), &oauth2.Token{
		AccessToken:  "valid-but-rejected",
		RefreshToken: "refresh-1",
		Expiry:       time.Now().Add(time.Hour), // still valid by clock
	})

	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		Metadata:   &AuthServerMetadata{AuthorizationEndpoint: o.srv.URL + "/authorize", TokenEndpoint: o.srv.URL + "/token"},
		Store:      store,
		HTTPClient: o.srv.Client(),
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}

	inv, ok := ts.(provider.InvalidatingTokenSource)
	if !ok {
		t.Fatal("token source must implement InvalidatingTokenSource")
	}
	inv.Invalidate()

	if _, err := ts.Token(context.Background()); err != nil {
		t.Fatalf("Token after Invalidate: %v", err)
	}
	if atomic.LoadInt32(&o.refreshCount) != 1 {
		t.Errorf("refresh count = %d, want 1 (Invalidate should force refresh)", o.refreshCount)
	}
}

// ── NewOAuthHTTPClient round tripper ────────────────────────────────────────

func TestNewOAuthHTTPClient_InjectsBearerAndRetriesOn401(t *testing.T) {
	var calls int32
	var seenTokens []string
	resource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenTokens = append(seenTokens, r.Header.Get("Authorization"))
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		// Echo body to verify it was re-sent on retry.
		_, _ = w.Write([]byte("ok"))
	}))
	defer resource.Close()

	ts := &fakeInvalidatingTS{tokens: []string{"first", "second"}}
	hc := NewOAuthHTTPClient(ts, nil)

	req, _ := http.NewRequest(http.MethodPost, resource.URL, strings.NewReader(`{"jsonrpc":"2.0"}`))
	resp, err := hc.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2 (one 401 + one retry)", calls)
	}
	if seenTokens[0] != "Bearer first" || seenTokens[1] != "Bearer second" {
		t.Errorf("tokens = %v, want [Bearer first, Bearer second]", seenTokens)
	}
	if !ts.invalidated {
		t.Error("expected Invalidate to be called on 401")
	}
}

func TestNewOAuthHTTPClient_NoRetryWhenNotInvalidating(t *testing.T) {
	var calls int32
	resource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer resource.Close()

	// Plain TokenSource (no Invalidate) must not trigger a retry.
	hc := NewOAuthHTTPClient(provider.StaticToken("tok"), nil)
	req, _ := http.NewRequest(http.MethodGet, resource.URL, nil)
	resp, err := hc.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	_ = resp.Body.Close()
	if calls != 1 {
		t.Errorf("calls = %d, want 1 (no retry without InvalidatingTokenSource)", calls)
	}
}

func TestNewOAuthHTTPClient_TokenError(t *testing.T) {
	hc := NewOAuthHTTPClient(&fakeErrTS{}, nil)
	req, _ := http.NewRequest(http.MethodGet, "http://example.invalid", nil)
	if _, err := hc.Do(req); err == nil {
		t.Fatal("expected error when token source fails")
	}
}

// ── codeFromRedirect ────────────────────────────────────────────────────────

func TestCodeFromRedirect(t *testing.T) {
	tests := []struct {
		name     string
		redirect string
		want     string
		wantState string
		wantErr  bool
	}{
		{name: "full url", redirect: "http://cb?code=abc&state=s1", wantState: "s1", want: "abc"},
		{name: "bare code", redirect: "abc", want: "abc"},
		{name: "state mismatch", redirect: "http://cb?code=abc&state=other", wantState: "s1", wantErr: true},
		{name: "error param", redirect: "http://cb?error=access_denied&error_description=no", wantState: "s1", wantErr: true},
		{name: "missing code", redirect: "http://cb?state=s1", wantState: "s1", wantErr: true},
		{name: "empty", redirect: "", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := codeFromRedirect(tc.redirect, tc.wantState)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("code = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestOAuthTokenSource_AuthorizeError(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()

	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		HTTPClient: o.srv.Client(),
		Authorize: func(_ context.Context, _ string) (string, error) {
			return "", errors.New("user cancelled")
		},
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	if _, err := ts.Token(context.Background()); err == nil || !strings.Contains(err.Error(), "user cancelled") {
		t.Fatalf("err = %v, want authorization error", err)
	}
}

func TestOAuthTokenSource_RefreshFailsFallsBackToAuthorize(t *testing.T) {
	o := newOAuthServer()
	o.failRefresh = true // refresh grant is rejected, forcing re-authorization
	o.accessToken = "after-reauth"
	defer o.Close()

	store := NewMemoryTokenStore()
	_ = store.Set(o.URL(), &oauth2.Token{
		AccessToken:  "old",
		RefreshToken: "revoked",
		Expiry:       time.Now().Add(-time.Minute),
	})

	var authorized bool
	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL: o.URL(),
		ClientID:  "static",
		Metadata: &AuthServerMetadata{
			AuthorizationEndpoint: o.srv.URL + "/authorize",
			TokenEndpoint:         o.srv.URL + "/token",
		},
		Store:      store,
		HTTPClient: o.srv.Client(),
		Authorize: func(_ context.Context, authURL string) (string, error) {
			authorized = true
			u, _ := url.Parse(authURL)
			return "http://cb?code=fresh-code&state=" + url.QueryEscape(u.Query().Get("state")), nil
		},
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	tok, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if !authorized {
		t.Error("expected fallback to interactive authorization after refresh failure")
	}
	if tok != "after-reauth" {
		t.Errorf("token = %q, want after-reauth", tok)
	}
}

func TestNewOAuthTokenSource_DiscoveryFails(t *testing.T) {
	o := newOAuthServer()
	o.protectedResource = false
	o.asMetadata = false
	defer o.Close()

	if _, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static", // skip registration; isolate discovery failure
		HTTPClient: o.srv.Client(),
	}); err == nil {
		t.Fatal("expected discovery failure error")
	}
}

func TestNewOAuthTokenSource_RegistrationFails(t *testing.T) {
	o := newOAuthServer()
	o.registrationCode = http.StatusInternalServerError
	defer o.Close()

	if _, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		HTTPClient: o.srv.Client(),
	}); err == nil {
		t.Fatal("expected registration failure error")
	}
}

func TestNewOAuthTokenSource_Validation(t *testing.T) {
	if _, err := NewOAuthTokenSource(context.Background(), OAuthConfig{}); err == nil {
		t.Fatal("expected error when ServerURL is empty")
	}

	// No ClientID and no registration endpoint → error.
	o := newOAuthServer()
	defer o.Close()
	o.registration = false
	if _, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		HTTPClient: o.srv.Client(),
	}); err == nil {
		t.Fatal("expected error when no client_id and no registration endpoint")
	}
}

// ── error-path coverage ─────────────────────────────────────────────────────

func TestDiscoverAuth_NetworkError(t *testing.T) {
	// Dead host: every getJSON GET fails, DiscoverAuth returns not-found.
	if _, err := DiscoverAuth(context.Background(), "http://127.0.0.1:1/mcp", nil); err == nil {
		t.Fatal("expected error against dead host")
	}
}

func TestDiscoverAuth_NonJSONProtectedResource(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/.well-known/oauth-protected-resource":
			_, _ = w.Write([]byte("not json")) // getJSON decode error (swallowed)
		case "/.well-known/oauth-authorization-server":
			writeJSON(w, map[string]any{"authorization_endpoint": r.Host + "/a", "token_endpoint": "/t"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	if _, err := DiscoverAuth(context.Background(), srv.URL+"/mcp", srv.Client()); err != nil {
		t.Fatalf("expected fallback to succeed, got %v", err)
	}
}

func TestDiscoverAuth_ASMetadataMissingAuthEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-authorization-server" {
			writeJSON(w, map[string]any{"token_endpoint": "/t"}) // no authorization_endpoint -> continue
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	if _, err := DiscoverAuth(context.Background(), srv.URL+"/mcp", srv.Client()); err == nil {
		t.Fatal("expected error when AS metadata lacks authorization_endpoint")
	}
}

func TestOriginOf_ParseError(t *testing.T) {
	// A control character makes url.Parse fail outright.
	if _, err := DiscoverAuth(context.Background(), "http://exa\x7fmple", nil); err == nil {
		t.Fatal("expected parse error for malformed URL")
	}
}

func TestRegisterClient_NetworkError(t *testing.T) {
	if _, err := RegisterClient(context.Background(), "http://127.0.0.1:1/register", ClientRegistration{}, nil); err == nil {
		t.Fatal("expected network error")
	}
}

func TestNewOAuthTokenSource_NilHTTPClient(t *testing.T) {
	// Metadata + ClientID provided: constructor needs no network and exercises
	// the default-HTTPClient branch.
	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL: "https://example.com/mcp",
		ClientID:  "static",
		Metadata:  &AuthServerMetadata{AuthorizationEndpoint: "https://as/authorize", TokenEndpoint: "https://as/token"},
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	if _, err := ts.Token(context.Background()); !errors.Is(err, ErrAuthorizationRequired) {
		t.Fatalf("err = %v, want ErrAuthorizationRequired", err)
	}
}

func TestOAuthTokenSource_BadRedirectFromAuthorize(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()
	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		HTTPClient: o.srv.Client(),
		Authorize: func(_ context.Context, _ string) (string, error) {
			return "http://cb?error=access_denied", nil // codeFromRedirect error
		},
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	if _, err := ts.Token(context.Background()); err == nil {
		t.Fatal("expected error from bad redirect")
	}
}

func TestOAuthTokenSource_ExchangeError(t *testing.T) {
	o := newOAuthServer()
	o.failExchange = true
	defer o.Close()
	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		HTTPClient: o.srv.Client(),
		Authorize: func(_ context.Context, authURL string) (string, error) {
			u, _ := url.Parse(authURL)
			return "http://cb?code=c&state=" + url.QueryEscape(u.Query().Get("state")), nil
		},
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	if _, err := ts.Token(context.Background()); err == nil {
		t.Fatal("expected token exchange error")
	}
}

func TestOAuthTokenSource_StoreSetError(t *testing.T) {
	o := newOAuthServer()
	defer o.Close()
	ts, err := NewOAuthTokenSource(context.Background(), OAuthConfig{
		ServerURL:  o.URL(),
		ClientID:   "static",
		Store:      failingStore{},
		HTTPClient: o.srv.Client(),
		Authorize: func(_ context.Context, authURL string) (string, error) {
			u, _ := url.Parse(authURL)
			return "http://cb?code=c&state=" + url.QueryEscape(u.Query().Get("state")), nil
		},
	})
	if err != nil {
		t.Fatalf("NewOAuthTokenSource: %v", err)
	}
	if _, err := ts.Token(context.Background()); err == nil {
		t.Fatal("expected store.Set error")
	}
}

func TestNewOAuthHTTPClient_TransportError(t *testing.T) {
	// base.RoundTrip fails (connection refused).
	hc := NewOAuthHTTPClient(provider.StaticToken("tok"), nil)
	req, _ := http.NewRequest(http.MethodGet, "http://127.0.0.1:1", nil)
	if _, err := hc.Do(req); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestNewOAuthHTTPClient_RetryTokenError(t *testing.T) {
	resource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer resource.Close()
	// Errors on the second Token() call (the post-401 refresh).
	ts := &fakeInvalidatingTS{tokens: []string{"first"}, errAfter: 1}
	hc := NewOAuthHTTPClient(ts, nil)
	req, _ := http.NewRequest(http.MethodGet, resource.URL, nil)
	if _, err := hc.Do(req); err == nil {
		t.Fatal("expected error when retry token fetch fails")
	}
}

func TestNewOAuthHTTPClient_CloneBodyError(t *testing.T) {
	resource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer resource.Close()
	ts := &fakeInvalidatingTS{tokens: []string{"a", "b"}}
	hc := NewOAuthHTTPClient(ts, nil)
	req, _ := http.NewRequest(http.MethodPost, resource.URL, strings.NewReader("body"))
	req.GetBody = func() (io.ReadCloser, error) { return nil, errors.New("rewind failed") }
	if _, err := hc.Do(req); err == nil {
		t.Fatal("expected clone body error on retry")
	}
}

// ── test doubles ────────────────────────────────────────────────────────────

type fakeInvalidatingTS struct {
	tokens      []string
	idx         int
	calls       int
	errAfter    int // when >0, Token returns an error on call number errAfter+
	invalidated bool
}

func (f *fakeInvalidatingTS) Token(_ context.Context) (string, error) {
	f.calls++
	if f.errAfter > 0 && f.calls > f.errAfter {
		return "", errors.New("token fetch failed")
	}
	tok := f.tokens[f.idx]
	if f.idx < len(f.tokens)-1 {
		f.idx++
	}
	return tok, nil
}

func (f *fakeInvalidatingTS) Invalidate() { f.invalidated = true }

type fakeErrTS struct{}

func (fakeErrTS) Token(_ context.Context) (string, error) {
	return "", errors.New("boom")
}

// failingStore returns an error from Set to exercise persistence error paths.
type failingStore struct{}

func (failingStore) Get(string) (*oauth2.Token, bool) { return nil, false }
func (failingStore) Set(string, *oauth2.Token) error  { return errors.New("store failed") }
func (failingStore) Delete(string) error              { return nil }
