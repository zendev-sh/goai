package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

const filesBetaHeader = "files-2024-09-04"

// maxFileResponseBytes bounds the size of a file-upload success response body.
const maxFileResponseBytes = 64 << 20 // 64 MiB

// fileIDAllowlist matches Anthropic file IDs (e.g. "file_01J..."), which are
// server-generated and consist solely of ASCII letters, digits, "_", "." and
// "-". We validate against a strict allowlist before interpolating the ID into
// the DELETE URL path: legitimate IDs never contain "/" (so escaping is
// unnecessary), and the allowlist rejects "/", "\", percent-encoded separators
// ("%2f"), and any other character that could alter the path.
var fileIDAllowlist = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

type fileUploader struct {
	opts options
}

func (u *fileUploader) UploadFile(ctx context.Context, upload provider.FileUpload) (*provider.RemoteFileRef, error) {
	data, err := io.ReadAll(upload.Reader)
	if err != nil {
		return nil, fmt.Errorf("reading file: %w", err)
	}

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Anthropic's Files API only accepts the "file" part (plus an optional
	// "expires_in_seconds"). There is no official "purpose" field, so we do not
	// send one -- sending it causes the API to reject the upload.
	fw, err := w.CreateFormFile("file", upload.Filename)
	if err != nil {
		return nil, fmt.Errorf("creating form file: %w", err)
	}
	if _, err := fw.Write(data); err != nil {
		return nil, fmt.Errorf("writing file data: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("closing multipart writer: %w", err)
	}

	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	req, err := httpc.NewRequest(ctx, "POST", u.opts.baseURL+"/v1/files", buf.Bytes())
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("anthropic-version", apiVersion)
	req.Header.Set("anthropic-beta", filesBetaHeader)
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}
	// Auth is applied last so a caller-supplied header named "x-api-key"
	// cannot override the real credential.
	req.Header.Set("x-api-key", token)

	client := u.opts.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := httpc.Do(client, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		// Cap the error body at 1 MiB so a runaway/hostile error response
		// cannot exhaust memory (matches the anthropic.go error-read pattern).
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, goai.ParseHTTPErrorWithHeaders("anthropic", resp.StatusCode, respBody, resp.Header)
	}

	// The Files API returns created_at as an RFC3339 timestamp string, and
	// reports the file size as size_bytes and the MIME type as mime_type
	// (not bytes / purpose).
	var result struct {
		ID        string `json:"id"`
		Type      string `json:"type"`
		SizeBytes int64  `json:"size_bytes"`
		MimeType  string `json:"mime_type"`
		CreatedAt string `json:"created_at"`
		Filename  string `json:"filename"`
	}
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxFileResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	if len(respBody) > maxFileResponseBytes {
		return nil, fmt.Errorf("anthropic: file upload response body exceeds %d bytes", maxFileResponseBytes)
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	// Prefer the MIME type reported by the API, then the caller's, then sniff.
	mediaType := result.MimeType
	if mediaType == "" {
		mediaType = upload.MediaType
	}
	if mediaType == "" {
		mediaType = http.DetectContentType(data)
	}

	return &provider.RemoteFileRef{
		Provider:  "anthropic",
		ID:        result.ID,
		URI:       "",
		Filename:  result.Filename,
		MediaType: mediaType,
		ExpiresAt: time.Time{},
		Data:      data,
	}, nil
}

func (u *fileUploader) DeleteFile(ctx context.Context, ref provider.RemoteFileRef) error {
	// Guard against same-origin path traversal: ref.ID is server-controlled and
	// interpolated into the URL path. Reject any ID that could climb out of the
	// /v1/files/ prefix -- slashes, backslashes, "..", or percent-encoded
	// separators. A strict allowlist is the safest approach: legitimate
	// Anthropic file IDs never contain "/" (e.g. "file_01J..."), so escaping is
	// unnecessary. The extra ".." check catches a bare parent-directory segment,
	// which the allowlist would otherwise permit (dots are valid in IDs).
	if !fileIDAllowlist.MatchString(ref.ID) || strings.Contains(ref.ID, "..") {
		return fmt.Errorf("anthropic: refusing to delete file with unsafe ID %q", ref.ID)
	}

	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return fmt.Errorf("resolving auth token: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "DELETE", u.opts.baseURL+"/v1/files/"+ref.ID, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("anthropic-version", apiVersion)
	req.Header.Set("anthropic-beta", filesBetaHeader)
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}
	// Auth is applied last so a caller-supplied header named "x-api-key"
	// cannot override the real credential.
	req.Header.Set("x-api-key", token)

	client := u.opts.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		// Cap the error body at 1 MiB so a runaway/hostile error response
		// cannot exhaust memory (matches the anthropic.go error-read pattern).
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return goai.ParseHTTPErrorWithHeaders("anthropic", resp.StatusCode, respBody, resp.Header)
	}

	return nil
}
