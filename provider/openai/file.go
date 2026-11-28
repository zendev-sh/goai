package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

type fileUploader struct {
	opts options
}

// maxFileErrorBytes bounds the size of an error response body read solely to
// extract the error message.
const maxFileErrorBytes = 1 << 20 // 1 MiB

// maxFileResponseBytes bounds the size of a successful file-upload response
// body decoded into the RemoteFileRef. A well-formed OpenAI response is tiny;
// anything larger indicates a misbehaving or hostile server.
const maxFileResponseBytes = 64 << 20 // 64 MiB

func (u *fileUploader) UploadFile(ctx context.Context, upload provider.FileUpload) (*provider.RemoteFileRef, error) {
	data, err := io.ReadAll(upload.Reader)
	if err != nil {
		return nil, fmt.Errorf("reading file: %w", err)
	}

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	if err := w.WriteField("purpose", upload.Purpose); err != nil {
		return nil, fmt.Errorf("writing purpose field: %w", err)
	}

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

	req, err := httpc.NewRequest(ctx, "POST", u.opts.baseURL+"/files", buf.Bytes())
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}
	// Auth is applied last so a caller-supplied header named "Authorization"
	// cannot override the real credential.
	req.Header.Set("Authorization", "Bearer "+token)

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
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxFileErrorBytes))
		return nil, goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
	}

	var result struct {
		ID        string `json:"id"`
		Bytes     int64  `json:"bytes"`
		CreatedAt int64  `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	}
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxFileResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	if len(respBody) > maxFileResponseBytes {
		return nil, fmt.Errorf("openai: file upload response body exceeds %d bytes", maxFileResponseBytes)
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	mediaType := upload.MediaType
	if mediaType == "" {
		mediaType = http.DetectContentType(data)
	}

	return &provider.RemoteFileRef{
		Provider:  "openai",
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
	// /files/ prefix (slashes, "..", or percent-encoded variants thereof).
	if strings.ContainsAny(ref.ID, `/\`) || strings.Contains(ref.ID, "..") || strings.Contains(ref.ID, "%") {
		return fmt.Errorf("openai: refusing to delete file with unsafe ID %q", ref.ID)
	}

	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return fmt.Errorf("resolving auth token: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "DELETE", u.opts.baseURL+"/files/"+ref.ID, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}
	// Auth is applied last so a caller-supplied header named "Authorization"
	// cannot override the real credential.
	req.Header.Set("Authorization", "Bearer "+token)

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
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxFileErrorBytes))
		return goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
	}

	return nil
}
