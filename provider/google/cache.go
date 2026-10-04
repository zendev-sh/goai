package google

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// CacheClient creates and manages Gemini explicit context-caching resources
// (cachedContents). It reuses the same auth, endpoint routing and tool
// serialization as chat requests (see buildToolsAndConfig), so a cached prefix
// is byte-identical to what a generate request would otherwise send — there is
// no second, drifting serialization of system/tools/toolConfig.
//
// A cachedContents resource stores only the immutable prefix
// (model + systemInstruction + tools + toolConfig). The conversation history is
// never part of it: it travels in each request's contents and references the
// resource by name via the cachedContent field.
type CacheClient struct {
	opts options
}

// NewCacheClient builds a CacheClient. It accepts the same options as Chat
// (WithAPIKey/WithTokenSource/WithVertex/WithBaseURL/...) so caching
// authenticates and routes exactly like generation for the same credentials.
func NewCacheClient(opts ...Option) *CacheClient {
	return &CacheClient{opts: resolveOptions(opts...)}
}

// CachedContentInput is the immutable prefix to store. ToolChoice and
// ProviderOptions are serialized identically to a generate request so the cached
// toolConfig matches; History is intentionally absent (it is never cached).
type CachedContentInput struct {
	Model           string
	System          string
	Tools           []provider.ToolDefinition
	ToolChoice      string
	ProviderOptions map[string]any
	TTL             time.Duration
}

// CachedContent is the stored resource as returned by the API. Name is the
// resource name to pass back as the cachedContent of a request (and to Renew):
// "cachedContents/<id>" on the direct API, or the full
// "projects/.../cachedContents/<id>" path on Vertex. Every output field of the
// resource is mapped; the input-only ones (contents, tools, systemInstruction,
// toolConfig, ttl) are never returned. Optional fields absent from the
// response keep their zero value.
type CachedContent struct {
	Name        string
	DisplayName string
	Model       string
	CreatedAt   time.Time
	UpdatedAt   time.Time
	ExpiresAt   time.Time
	Usage       CachedContentUsage
}

// CachedContentUsage is the resource's usageMetadata. TotalTokens is the size
// of the cached prefix, which storage is billed on. The per-modality counts
// are only reported by Vertex.
type CachedContentUsage struct {
	TotalTokens          int
	TextCount            int
	ImageCount           int
	VideoDurationSeconds int
	AudioDurationSeconds int
}

// Create stores a new cachedContents resource and returns its name + expiry.
func (c *CacheClient) Create(ctx context.Context, in CachedContentInput) (CachedContent, error) {
	if in.TTL <= 0 {
		return CachedContent{}, fmt.Errorf("google: cachedContents TTL must be positive, got %s", in.TTL)
	}
	tools, toolConfig, err := buildToolsAndConfig(in.Tools, in.ToolChoice, in.ProviderOptions)
	if err != nil {
		return CachedContent{}, err
	}

	body := map[string]any{
		"model": cacheModelName(c.opts, in.Model),
		"ttl":   fmt.Sprintf("%ds", int(in.TTL.Seconds())),
	}
	if si := buildSystemInstruction(in.System); si != nil {
		body["systemInstruction"] = si
	}
	if tools != nil {
		body["tools"] = tools
	}
	if toolConfig != nil {
		body["toolConfig"] = toolConfig
	}

	reqURL, err := c.createURL()
	if err != nil {
		return CachedContent{}, err
	}
	return c.send(ctx, http.MethodPost, reqURL, body)
}

// Renew extends the TTL of an existing resource via PATCH and returns the new
// expiry. name is the value returned by Create.
func (c *CacheClient) Renew(ctx context.Context, name string, ttl time.Duration) (CachedContent, error) {
	if ttl <= 0 {
		return CachedContent{}, fmt.Errorf("google: cachedContents TTL must be positive, got %s", ttl)
	}
	reqURL, err := c.resourceURL(name, "?updateMask=ttl")
	if err != nil {
		return CachedContent{}, err
	}
	body := map[string]any{"ttl": fmt.Sprintf("%ds", int(ttl.Seconds()))}
	return c.send(ctx, http.MethodPatch, reqURL, body)
}

// maxCacheResponseBytes bounds a cachedContents response body to defend against
// a malicious or misbehaving server returning an unbounded payload.
const maxCacheResponseBytes = 1 << 20 // 1 MiB

func (c *CacheClient) send(ctx context.Context, method, url string, body any) (CachedContent, error) {
	resp, err := doGoogleJSON(ctx, c.opts, method, url, body, nil)
	if err != nil {
		return CachedContent{}, err
	}
	defer func() { _ = resp.Body.Close() }()

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxCacheResponseBytes+1))
	if err != nil {
		return CachedContent{}, fmt.Errorf("reading cachedContents response: %w", err)
	}
	if len(data) > maxCacheResponseBytes {
		return CachedContent{}, fmt.Errorf("google: cachedContents response exceeds %d bytes", maxCacheResponseBytes)
	}
	var out struct {
		Name          string `json:"name"`
		DisplayName   string `json:"displayName"`
		Model         string `json:"model"`
		CreateTime    string `json:"createTime"`
		UpdateTime    string `json:"updateTime"`
		ExpireTime    string `json:"expireTime"`
		UsageMetadata struct {
			TotalTokenCount      int `json:"totalTokenCount"`
			TextCount            int `json:"textCount"`
			ImageCount           int `json:"imageCount"`
			VideoDurationSeconds int `json:"videoDurationSeconds"`
			AudioDurationSeconds int `json:"audioDurationSeconds"`
		} `json:"usageMetadata"`
	}
	if uerr := json.Unmarshal(data, &out); uerr != nil {
		return CachedContent{}, fmt.Errorf("parsing cachedContents response: %w", uerr)
	}
	if out.Name == "" {
		return CachedContent{}, fmt.Errorf("cachedContents response missing name: %s", string(data))
	}
	expires, err := time.Parse(time.RFC3339, out.ExpireTime)
	if err != nil {
		return CachedContent{}, fmt.Errorf("parsing cachedContents expireTime %q: %w", out.ExpireTime, err)
	}
	created, err := parseOptionalTime("createTime", out.CreateTime)
	if err != nil {
		return CachedContent{}, err
	}
	updated, err := parseOptionalTime("updateTime", out.UpdateTime)
	if err != nil {
		return CachedContent{}, err
	}
	u := out.UsageMetadata
	return CachedContent{
		Name:        out.Name,
		DisplayName: out.DisplayName,
		Model:       out.Model,
		CreatedAt:   created,
		UpdatedAt:   updated,
		ExpiresAt:   expires,
		Usage: CachedContentUsage{
			TotalTokens:          u.TotalTokenCount,
			TextCount:            u.TextCount,
			ImageCount:           u.ImageCount,
			VideoDurationSeconds: u.VideoDurationSeconds,
			AudioDurationSeconds: u.AudioDurationSeconds,
		},
	}, nil
}

// parseOptionalTime parses an RFC 3339 timestamp field that the API may omit;
// an absent field yields the zero time.
func parseOptionalTime(field, value string) (time.Time, error) {
	if value == "" {
		return time.Time{}, nil
	}
	t, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return time.Time{}, fmt.Errorf("parsing cachedContents %s %q: %w", field, value, err)
	}
	return t, nil
}

// cacheModelName is the model reference accepted in the cachedContents body:
// the bare "models/<id>" on the direct API, or the fully-qualified publisher
// path on Vertex.
func cacheModelName(o options, modelID string) string {
	if o.isVertex {
		return fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", o.project, o.location, modelID)
	}
	return "models/" + modelID
}

// createURL is the collection endpoint for POSTing a new resource.
func (c *CacheClient) createURL() (string, error) {
	if !c.opts.isVertex {
		return cacheBaseURL(c.opts) + "/cachedContents", nil
	}
	if err := validateVertex(c.opts); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s/projects/%s/locations/%s/cachedContents", cacheBaseURL(c.opts), c.opts.project, c.opts.location), nil
}

// resourceURL is the item endpoint for an existing resource (PATCH/GET/DELETE).
// name is the resource name returned by Create.
func (c *CacheClient) resourceURL(name, query string) (string, error) {
	if c.opts.isVertex {
		if err := validateVertex(c.opts); err != nil {
			return "", err
		}
	}
	return cacheBaseURL(c.opts) + "/" + name + query, nil
}

// cacheBaseURL is the versioned API root for cachedContents. A WithBaseURL
// override wins (testing/proxy), in Vertex mode too.
func cacheBaseURL(o options) string {
	if o.isVertex {
		if base := strings.TrimRight(o.baseURL, "/"); base != defaultBaseURL && base != "" {
			return base + "/v1beta1"
		}
		return "https://" + vertexHost(o.location) + "/v1beta1"
	}
	return strings.TrimRight(o.baseURL, "/") + "/v1beta"
}

// validateVertex guards project/location before hostname interpolation
// (anti-SSRF), mirroring endpointURL.
func validateVertex(o options) error {
	if !validGCPIdentifier(o.project) {
		return fmt.Errorf("google: invalid vertex project %q", o.project)
	}
	if !validGCPIdentifier(o.location) {
		return fmt.Errorf("google: invalid vertex location %q", o.location)
	}
	return nil
}
