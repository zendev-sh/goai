package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

const maxGoogleFileErrorBytes int64 = 1 << 20

// maxGoogleFileResponseBytes bounds the size of a file-upload success body.
const maxGoogleFileResponseBytes int64 = 64 << 20

type fileUploader struct {
	opts options
}

const filePollInterval = 100 * time.Millisecond

type googleFile struct {
	Name           string    `json:"name"`
	URI            string    `json:"uri"`
	MimeType       string    `json:"mimeType"`
	SizeBytes      SizeBytes `json:"sizeBytes"`
	ExpirationTime string    `json:"expirationTime"`
	DisplayName    string    `json:"displayName"`
	State          string    `json:"state"`
	Error          *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// SizeBytes is the byte size of a Google file. The GenAI API serializes the
// proto3 int64 field as a JSON string (e.g. "123456789"), but some servers
// emit a plain JSON number; this type accepts both wire forms.
type SizeBytes int64

// UnmarshalJSON accepts sizeBytes as either a JSON string or a JSON number.
func (s *SizeBytes) UnmarshalJSON(data []byte) error {
	if data[0] == '"' {
		var str string
		if err := json.Unmarshal(data, &str); err != nil {
			return fmt.Errorf("google: decoding sizeBytes string: %w", err)
		}
		n, err := strconv.ParseInt(str, 10, 64)
		if err != nil {
			return fmt.Errorf("google: parsing sizeBytes %q: %w", str, err)
		}
		*s = SizeBytes(n)
		return nil
	}
	var num int64
	if err := json.Unmarshal(data, &num); err != nil {
		return fmt.Errorf("google: decoding sizeBytes number: %w", err)
	}
	*s = SizeBytes(num)
	return nil
}

// UploadFile uploads a file using the Gemini Files API resumable protocol.
//
// The upload runs in two phases driven by the X-Goog-Upload-* headers:
//
//  1. START: an initial POST to /upload/v1beta/files carrying the JSON
//     metadata and the X-Goog-Upload-Protocol/Command/Header-* headers. The
//     response returns the session upload URL in the X-Goog-Upload-URL header.
//  2. UPLOAD+FINALIZE: a PUT to that URL with the raw file bytes and
//     X-Goog-Upload-Command: upload, finalize.
//
// This replaces the former single multipart POST, which the Gemini API does
// not accept for the Files API.
func (u *fileUploader) UploadFile(ctx context.Context, upload provider.FileUpload) (*provider.RemoteFileRef, error) {
	data, err := io.ReadAll(upload.Reader)
	if err != nil {
		return nil, fmt.Errorf("reading file: %w", err)
	}

	mediaType := upload.MediaType
	if mediaType == "" {
		mediaType = http.DetectContentType(data)
	}

	metadata := map[string]any{
		"file": map[string]any{
			"displayName": upload.Filename,
			"mimeType":    mediaType,
		},
	}
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("marshaling metadata: %w", err)
	}

	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}
	client := u.startHTTPClient()

	// Phase 1: start a resumable upload session.
	initReq, err := httpc.NewRequest(ctx, http.MethodPost, u.opts.baseURL+"/upload/v1beta/files", metaJSON)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	initReq.Header.Set("Content-Type", "application/json")
	initReq.Header.Set("X-Goog-Upload-Protocol", "resumable")
	initReq.Header.Set("X-Goog-Upload-Command", "start")
	initReq.Header.Set("X-Goog-Upload-Header-Content-Length", strconv.FormatInt(int64(len(data)), 10))
	initReq.Header.Set("X-Goog-Upload-Header-Content-Type", mediaType)
	// Caller headers first, credential LAST so auth cannot be overridden.
	for k, v := range u.opts.headers {
		initReq.Header.Set(k, v)
	}
	initReq.Header.Set("x-goog-api-key", token)

	initResp, err := httpc.Do(client, initReq)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	uploadURL := initResp.Header.Get("X-Goog-Upload-URL")
	if initResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(initResp.Body, maxGoogleFileErrorBytes))
		_ = initResp.Body.Close()
		return nil, goai.ParseHTTPErrorWithHeaders("google", initResp.StatusCode, respBody, initResp.Header)
	}
	_ = initResp.Body.Close()
	if uploadURL == "" {
		return nil, fmt.Errorf("google: resumable upload response missing X-Goog-Upload-URL header")
	}
	// SSRF defense: only PUT the file bytes to the configured API origin. A
	// compromised/malicious server cannot redirect the upload (and the
	// x-goog-api-key it carries) to an attacker-controlled host.
	if !sameOrigin(uploadURL, u.opts.baseURL) {
		return nil, fmt.Errorf("google: resumable upload URL must use the configured API origin")
	}

	// Phase 2: upload the raw bytes and finalize in one shot. The URL has
	// already been parsed and validated by sameOrigin above, so
	// NewRequestWithContext cannot fail here.
	uploadReq, _ := httpc.NewRequest(ctx, http.MethodPut, uploadURL, data)
	uploadReq.Header.Set("Content-Type", mediaType)
	uploadReq.Header.Set("Content-Length", strconv.FormatInt(int64(len(data)), 10))
	uploadReq.Header.Set("X-Goog-Upload-Protocol", "resumable")
	uploadReq.Header.Set("X-Goog-Upload-Command", "upload, finalize")
	uploadReq.Header.Set("X-Goog-Upload-Offset", "0")
	// Caller headers first, credential LAST so auth cannot be overridden.
	for k, v := range u.opts.headers {
		uploadReq.Header.Set(k, v)
	}
	uploadReq.Header.Set("x-goog-api-key", token)

	uploadResp, err := httpc.Do(u.uploadHTTPClient(), uploadReq)
	if err != nil {
		return nil, fmt.Errorf("sending upload request: %w", err)
	}
	defer func() { _ = uploadResp.Body.Close() }()

	if uploadResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(uploadResp.Body, maxGoogleFileErrorBytes))
		return nil, goai.ParseHTTPErrorWithHeaders("google", uploadResp.StatusCode, respBody, uploadResp.Header)
	}

	var result struct {
		File googleFile `json:"file"`
	}
	respBody, err := io.ReadAll(io.LimitReader(uploadResp.Body, maxGoogleFileResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	if int64(len(respBody)) > maxGoogleFileResponseBytes {
		return nil, fmt.Errorf("google: file upload response body exceeds %d bytes", maxGoogleFileResponseBytes)
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	file, err := u.waitForFile(ctx, result.File)
	if err != nil {
		return nil, err
	}
	return u.remoteFileRef(file, data), nil
}

func (u *fileUploader) httpClient() *http.Client {
	if u.opts.httpClient != nil {
		return u.opts.httpClient
	}
	return http.DefaultClient
}

// startHTTPClient returns a client whose redirect policy refuses all
// redirects, so the x-goog-api-key (and provider headers) carried by the
// phase-1 START request are never forwarded to a redirect target. Mirrors the
// redirect-refusing client used by the sibling video provider for its
// operation requests.
func (u *fileUploader) startHTTPClient() *http.Client {
	client := *u.httpClient()
	client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &client
}

// uploadHTTPClient returns a client whose redirect policy refuses cross-origin
// redirects and strips the API key (and provider headers) before following any
// redirect that leaves the configured API origin. Mirrors the defense used by
// the sibling video provider so the file bytes and x-goog-api-key are never
// sent to an untrusted host.
func (u *fileUploader) uploadHTTPClient() *http.Client {
	client := *u.httpClient()
	previousCheck := client.CheckRedirect
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if !sameOrigin(req.URL.String(), u.opts.baseURL) {
			req.Header.Del("x-goog-api-key")
			for key := range u.opts.headers {
				req.Header.Del(key)
			}
			return errors.New("google file upload refused a cross-origin redirect")
		}
		if previousCheck != nil {
			return previousCheck(req, via)
		}
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}
	return &client
}

// validateResourceName rejects resource names that could traverse outside the
// /v1beta/ path (e.g. a server- or caller-supplied "files/../../admin") before
// they are interpolated into a request URL.
//
// Real Gemini resource names contain "/" (e.g. "files/abc123",
// "operations/123"), so "/" cannot be rejected outright. Instead we reject the
// constructs that enable traversal: a leading "/", percent-encoding ("%",
// which defeats naive ../ checks via forms like "..%2f" or "..%2e%2e%2f"), and
// any ".." segment. Callers must additionally escape the name with
// escapeResourcePath before interpolating it into a URL.
func validateResourceName(name string) error {
	if name == "" {
		return fmt.Errorf("google: empty resource name")
	}
	if strings.HasPrefix(name, "/") {
		return fmt.Errorf("google: unsafe resource name %q: must not start with '/'", name)
	}
	if strings.Contains(name, "%") {
		return fmt.Errorf("google: unsafe resource name %q: must not contain percent-encoding", name)
	}
	if strings.Contains(name, "..") {
		return fmt.Errorf("google: unsafe resource name %q: must not contain path traversal", name)
	}
	return nil
}

// escapeResourcePath escapes each path segment of a resource name so that
// server-controlled names cannot smuggle path separators, query strings, or
// fragments into the request URL. The collection/id separator ("/") is
// preserved so names like "files/abc123" keep their meaning after escaping.
func escapeResourcePath(name string) string {
	parts := strings.Split(name, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func (u *fileUploader) remoteFileRef(file googleFile, data []byte) *provider.RemoteFileRef {
	var expiresAt time.Time
	if file.ExpirationTime != "" {
		if t, err := time.Parse(time.RFC3339, file.ExpirationTime); err == nil {
			expiresAt = t
		}
	}
	return &provider.RemoteFileRef{
		Provider:  "google",
		ID:        file.Name,
		URI:       file.URI,
		Filename:  file.DisplayName,
		MediaType: file.MimeType,
		ExpiresAt: expiresAt,
		Data:      data,
	}
}

func (u *fileUploader) waitForFile(ctx context.Context, file googleFile) (googleFile, error) {
	for {
		switch file.State {
		case "ACTIVE":
			return file, nil
		case "FAILED":
			if file.Error != nil && file.Error.Message != "" {
				return googleFile{}, fmt.Errorf("google: file processing failed: %s", file.Error.Message)
			}
			return googleFile{}, fmt.Errorf("google: file processing failed")
		case "PROCESSING", "STATE_UNSPECIFIED", "":
			// Continue polling until the resource reaches a terminal state.
		default:
			return googleFile{}, fmt.Errorf("google: unknown file state %q", file.State)
		}

		timer := time.NewTimer(filePollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return googleFile{}, ctx.Err()
		case <-timer.C:
		}

		token, err := u.opts.tokenSource.Token(ctx)
		if err != nil {
			return googleFile{}, fmt.Errorf("resolving auth token: %w", err)
		}
		if err := validateResourceName(file.Name); err != nil {
			return googleFile{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.opts.baseURL+"/v1beta/"+escapeResourcePath(file.Name), nil)
		if err != nil {
			return googleFile{}, fmt.Errorf("creating request: %w", err)
		}
		// Caller headers first, credential LAST so auth cannot be overridden.
		for k, v := range u.opts.headers {
			req.Header.Set(k, v)
		}
		req.Header.Set("x-goog-api-key", token)
		client := u.opts.httpClient
		if client == nil {
			client = http.DefaultClient
		}
		resp, err := client.Do(req)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return googleFile{}, ctxErr
			}
			return googleFile{}, fmt.Errorf("sending request: %w", err)
		}
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxGoogleFileErrorBytes))
		_ = resp.Body.Close()
		if readErr != nil {
			return googleFile{}, fmt.Errorf("reading response: %w", readErr)
		}
		if resp.StatusCode != http.StatusOK {
			return googleFile{}, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, body, resp.Header)
		}
		if err := json.Unmarshal(body, &file); err != nil {
			return googleFile{}, fmt.Errorf("decoding response: %w", err)
		}
	}
}

func (u *fileUploader) DeleteFile(ctx context.Context, ref provider.RemoteFileRef) error {
	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return fmt.Errorf("resolving auth token: %w", err)
	}

	if err := validateResourceName(ref.ID); err != nil {
		return err
	}
	reqURL := u.opts.baseURL + "/v1beta/" + escapeResourcePath(ref.ID)
	req, err := http.NewRequestWithContext(ctx, "DELETE", reqURL, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	// Caller headers first, credential LAST so auth cannot be overridden.
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("x-goog-api-key", token)

	client := u.opts.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxGoogleFileErrorBytes))
		return goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	return nil
}

// hasRemoteRef returns true if any message part contains a RemoteRef.
func hasRemoteRef(msgs []provider.Message) bool {
	for _, msg := range msgs {
		for _, part := range msg.Content {
			if part.RemoteRef != nil {
				return true
			}
		}
	}
	return false
}

// filePartToContent converts a PartFile to a Gemini content part.
// Uses fileData for remote references, inlineData for inline files.
func filePartToContent(part provider.Part) map[string]any {
	if part.RemoteRef != nil {
		return map[string]any{
			"fileData": map[string]any{
				"fileUri":  part.RemoteRef.URI,
				"mimeType": part.RemoteRef.MediaType,
			},
		}
	}
	mediaType, data, ok := httpc.ParseDataURL(part.URL)
	if ok {
		return map[string]any{
			"inlineData": map[string]any{
				"mimeType": mediaType,
				"data":     data,
			},
		}
	}
	return nil
}
