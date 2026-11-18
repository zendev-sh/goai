package bedrock

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/anthropic"
)

// AnthropicChat creates a Bedrock language model that speaks Anthropic's
// native Messages API via Bedrock's InvokeModel / InvokeModelWithResponseStream
// endpoints (rather than the cross-vendor Converse API).
//
// Use this when you need Anthropic features that Converse does not expose:
// notably server-executed tools (web_search, code_execution, web_fetch),
// extended thinking with per-request thinking config, or full prompt caching.
//
// Usage:
//
//	model := bedrock.AnthropicChat(modelID,
//		bedrock.WithAccessKey(os.Getenv("AWS_ACCESS_KEY_ID")),
//		bedrock.WithSecretKey(os.Getenv("AWS_SECRET_ACCESS_KEY")),
//		bedrock.WithRegion(os.Getenv("AWS_REGION")),
//	)
//
// All Bedrock options apply (region inference, bearer / SigV4 auth, custom
// HTTP client). The returned model inherits the anthropic provider's parsing
// (including server-tool round-trip), so behaviour matches direct Anthropic.
func AnthropicChat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	o.region = cmp.Or(o.region, os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"), "us-east-1")
	if r := inferRegionFromModel(modelID); r != "" && !regionMatchesGeo(o.region, modelID) {
		o.region = r
	}
	o.bearerToken = cmp.Or(o.bearerToken, os.Getenv("AWS_BEARER_TOKEN_BEDROCK"))
	o.accessKey = cmp.Or(o.accessKey, os.Getenv("AWS_ACCESS_KEY_ID"))
	o.secretKey = cmp.Or(o.secretKey, os.Getenv("AWS_SECRET_ACCESS_KEY"))
	o.sessionToken = cmp.Or(o.sessionToken, os.Getenv("AWS_SESSION_TOKEN"))
	o.baseURL = cmp.Or(o.baseURL, os.Getenv("AWS_BEDROCK_BASE_URL"))

	baseURL := o.baseURL
	if baseURL == "" {
		baseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", o.region)
	}
	baseURL = strings.TrimRight(baseURL, "/")

	base := o.httpClient
	if base == nil {
		base = http.DefaultClient
	}
	rt := base.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	signing := &http.Client{
		Transport: &bedrockAnthropicTransport{
			base:         rt,
			region:       o.region,
			accessKey:    o.accessKey,
			secretKey:    o.secretKey,
			sessionToken: o.sessionToken,
			bearerToken:  o.bearerToken,
		},
	}

	anthropicOpts := []anthropic.Option{
		anthropic.WithBaseURL(baseURL),
		anthropic.WithURLBuilder(bedrockAnthropicURLBuilder),
		anthropic.WithBodyTransformer(bedrockAnthropicBodyTransformer),
		anthropic.WithErrorProvider("bedrock-anthropic"),
		anthropic.WithHTTPClient(signing),
		anthropic.WithSkipEnvResolve(),
		// Bedrock documents a narrower structured-output set than the direct
		// Claude API (Opus 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5, Haiku 4.5).
		anthropic.WithNativeOutputFormatSupport(anthropic.NativeOutputFormatBedrock),
		// Auto-streaming switches DoGenerate to InvokeModelWithResponseStream,
		// which needs a different IAM permission than InvokeModel. Keep it
		// opt-in so existing deployments do not regress by default.
		anthropic.WithAutoStreaming(false),
	}
	// Provide a placeholder token source so the anthropic provider's auth
	// resolution is satisfied. The transport below replaces the request's
	// auth headers with SigV4 (or a Bedrock bearer token) before sending.
	anthropicOpts = append(anthropicOpts,
		anthropic.WithTokenSource(provider.StaticToken("bedrock-anthropic")),
	)
	if len(o.headers) > 0 {
		anthropicOpts = append(anthropicOpts, anthropic.WithHeaders(o.headers))
	}
	return anthropic.Chat(modelID, anthropicOpts...)
}

// bedrockAnthropicURLBuilder maps Anthropic's Messages endpoint shape onto
// Bedrock's per-model invoke endpoints.
//
//	Streaming:     {baseURL}/model/{modelID}/invoke-with-response-stream
//	Non-streaming: {baseURL}/model/{modelID}/invoke
func bedrockAnthropicURLBuilder(baseURL, modelID string, streaming bool) string {
	endpoint := "invoke"
	if streaming {
		endpoint = "invoke-with-response-stream"
	}
	return baseURL + "/model/" + url.PathEscape(modelID) + "/" + endpoint
}

// bedrockAnthropicBodyTransformer rewrites the request body to Bedrock's
// expected shape for Anthropic models on InvokeModel:
//
//   - "model" must be omitted (it is in the URL path).
//   - "anthropic_version" must be "bedrock-2023-05-31".
//
// The "stream" field is read by the anthropic provider to pick the URL
// (invoke vs invoke-with-response-stream); the transport strips it before
// sending so Bedrock's body validator does not see it. The anthropic-beta
// header is moved into the body's "anthropic_beta" field by the transport
// (Bedrock InvokeModel does not honour the header).
func bedrockAnthropicBodyTransformer(body map[string]any) map[string]any {
	delete(body, "model")
	body["anthropic_version"] = "bedrock-2023-05-31"
	return body
}

// bedrockAnthropicTransport signs each request with SigV4 (or bearer) and
// translates Bedrock's EventStream binary streaming responses into a plain
// SSE byte stream that the anthropic provider's parseSSE can consume.
type bedrockAnthropicTransport struct {
	base                                                    http.RoundTripper
	region, accessKey, secretKey, sessionToken, bearerToken string
}

func (t *bedrockAnthropicTransport) CanReleaseReplay() bool { return httpc.CanReleaseReplay(t.base) }

func (t *bedrockAnthropicTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Keep the original headers and replay body intact for client redirects.
	req = req.Clone(req.Context())
	// Strip Anthropic-specific auth/version headers and capture the beta
	// header so we can fold it into the JSON body (Bedrock requires that).
	beta := req.Header.Get("anthropic-beta")
	req.Header.Del("x-api-key")
	req.Header.Del("Authorization")
	req.Header.Del("anthropic-version")
	req.Header.Del("anthropic-beta")

	var body []byte
	if req.Body != nil {
		b, err := io.ReadAll(req.Body)
		_ = req.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("bedrock-anthropic: reading request body: %w", err)
		}
		body = b
		// Strip the "stream" field (the anthropic provider sets it for URL
		// selection but Bedrock InvokeModel rejects unknown body keys), and
		// translate anthropic-beta from header to body. Bedrock accepts only
		// a subset of Anthropic's beta flags; the anthropic provider bakes
		// in flags that Bedrock rejects, so we filter to a known-good list.
		var bodyMap map[string]any
		if err := json.Unmarshal(body, &bodyMap); err == nil {
			// Reject Anthropic server-executed tools that Bedrock does not
			// support (web_search / code_execution / web_fetch) before sending.
			if err := rejectUnsupportedServerTools(bodyMap); err != nil {
				return nil, err
			}
			delete(bodyMap, "stream")
			if beta != "" {
				if filtered := filterBedrockBetas(splitBetas(beta)); len(filtered) > 0 {
					bodyMap["anthropic_beta"] = filtered
				} else {
					delete(bodyMap, "anthropic_beta")
				}
			} else {
				delete(bodyMap, "anthropic_beta")
			}
			if reBody, err := json.Marshal(bodyMap); err == nil {
				body = reBody
			}
		}
		httpc.SetBody(req, body)
	}

	if t.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+t.bearerToken)
	} else {
		signAWSSigV4(req, body, t.accessKey, t.secretKey, t.sessionToken, t.region, "bedrock")
	}

	resp, err := httpc.DoRoundTrip(t.base, req)
	if err != nil {
		return resp, err
	}

	if resp.StatusCode == http.StatusOK && strings.HasSuffix(req.URL.Path, "invoke-with-response-stream") {
		// Translate AWS EventStream → SSE so anthropic.parseSSE can read it.
		resp.Body = newEventStreamSSEReader(resp.Body)
		// The anthropic SSE scanner switches behaviour on Content-Type;
		// expose plain SSE so its scanner does not look for JSON.
		resp.Header.Set("Content-Type", "text/event-stream")
	}
	return resp, nil
}

// unsupportedBedrockServerTools are Anthropic server-executed tool types that
// AWS Bedrock does not support. AWS returns "web_search not available on
// Bedrock" (and likewise for code_execution / web_fetch); forwarding them with
// a beta header causes a validation error. We reject them up-front with a clear
// message instead of letting the request fail opaquely server-side.
//
// Computer-use tools (computer_*/bash_*/text_editor_*) ARE supported on Bedrock
// and are deliberately not gated here.
var unsupportedBedrockServerTools = map[string]bool{
	"web_search_20250305":     true,
	"web_search_20260209":     true,
	"web_fetch_20260209":      true,
	"code_execution_20250522": true,
	"code_execution_20250825": true,
	"code_execution_20260120": true,
}

// rejectUnsupportedServerTools scans the outgoing request body's "tools" array
// for Anthropic server-executed tool types that Bedrock does not support and
// returns a descriptive error if any are present.
func rejectUnsupportedServerTools(body map[string]any) error {
	tools, _ := body["tools"].([]any)
	for _, t := range tools {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		typ, _ := tm["type"].(string)
		if unsupportedBedrockServerTools[typ] {
			return fmt.Errorf(
				"bedrock-anthropic: server tool %q is not supported on AWS Bedrock "+
					"(web_search, web_fetch and code_execution are Anthropic-direct-only); "+
					"remove it or replace it with a custom tool", typ)
		}
	}
	return nil
}

func splitBetas(header string) []string {
	var out []string
	for _, b := range strings.Split(header, ",") {
		b = strings.TrimSpace(b)
		if b != "" {
			out = append(out, b)
		}
	}
	return out
}

// bedrockSupportedBetas lists Anthropic beta flags Bedrock InvokeModel accepts.
// Flags not in this set (some that the upstream provider sets by default)
// cause Bedrock to return "The provided request is not valid".
var bedrockSupportedBetas = map[string]bool{
	"web-search-2025-03-05":         true,
	"computer-use-2024-10-22":       true,
	"computer-use-2025-01-24":       true,
	"computer-use-2025-11-24":       true,
	"code-execution-2025-08-25":     true,
	"code-execution-2025-05-22":     true,
	"context-management-2025-06-27": true,
	"prompt-caching-2024-07-31":     true,
}

func filterBedrockBetas(in []string) []string {
	var out []string
	for _, b := range in {
		if bedrockSupportedBetas[b] {
			out = append(out, b)
		}
	}
	return out
}

// eventStreamSSEReader wraps a Bedrock EventStream response body and
// re-emits each chunk's payload as a plain SSE "data: ...\n\n" line.
//
// Bedrock InvokeModelWithResponseStream wraps each native model SSE event
// inside an EventStream frame whose JSON payload looks like:
//
//	{"bytes": "<base64 of the original SSE event JSON>"}
//
// We decode the base64 and re-emit it as a plain SSE frame so downstream
// providers (here, the anthropic provider) can parse it unchanged.
type eventStreamSSEReader struct {
	src     io.ReadCloser
	decoder *eventStreamDecoder
	pending bytes.Buffer
	err     error
}

func newEventStreamSSEReader(src io.ReadCloser) io.ReadCloser {
	return &eventStreamSSEReader{
		src:     src,
		decoder: newEventStreamDecoder(src),
	}
}

func (r *eventStreamSSEReader) Read(p []byte) (int, error) {
	for r.pending.Len() == 0 && r.err == nil {
		frame, err := r.decoder.Next()
		if err != nil {
			r.err = err
			break
		}
		switch frame.MessageType {
		case "event":
			var ev struct {
				Bytes []byte `json:"bytes"`
			}
			if json.Unmarshal(frame.Payload, &ev) == nil && len(ev.Bytes) > 0 {
				r.pending.WriteString("data: ")
				r.pending.Write(ev.Bytes)
				r.pending.WriteString("\n\n")
			}
		case "exception", "error":
			// Surface as an error event so parseSSE classifies/forwards it.
			r.pending.WriteString("event: error\ndata: ")
			r.pending.Write(frame.Payload)
			r.pending.WriteString("\n\n")
		}
	}
	if r.pending.Len() > 0 {
		return r.pending.Read(p)
	}
	return 0, r.err
}

func (r *eventStreamSSEReader) Close() error {
	return r.src.Close()
}
