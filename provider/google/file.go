package google

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
	"github.com/zendev-sh/goai/provider"
)

type fileUploader struct {
	opts options
}

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

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Metadata part.
	metaPart, err := w.CreatePart(map[string][]string{
		"Content-Type": {"application/json; charset=UTF-8"},
	})
	if err != nil {
		return nil, fmt.Errorf("creating metadata part: %w", err)
	}
	if _, err := metaPart.Write(metaJSON); err != nil {
		return nil, fmt.Errorf("writing metadata: %w", err)
	}

	// File data part.
	filePart, err := w.CreatePart(map[string][]string{
		"Content-Type": {mediaType},
	})
	if err != nil {
		return nil, fmt.Errorf("creating file part: %w", err)
	}
	if _, err := filePart.Write(data); err != nil {
		return nil, fmt.Errorf("writing file data: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("closing multipart writer: %w", err)
	}

	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	reqURL := u.opts.baseURL + "/upload/v1beta/files"
	req, err := http.NewRequestWithContext(ctx, "POST", reqURL, &buf)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("x-goog-api-key", token)
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}

	client := u.opts.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	var result struct {
		File struct {
			Name           string `json:"name"`
			URI            string `json:"uri"`
			MimeType       string `json:"mimeType"`
			SizeBytes      string `json:"sizeBytes"`
			ExpirationTime string `json:"expirationTime"`
			DisplayName    string `json:"displayName"`
		} `json:"file"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	var expiresAt time.Time
	if result.File.ExpirationTime != "" {
		if t, err := time.Parse(time.RFC3339, result.File.ExpirationTime); err == nil {
			expiresAt = t
		}
	}

	return &provider.RemoteFileRef{
		Provider:  "google",
		ID:        result.File.Name,
		URI:       result.File.URI,
		Filename:  result.File.DisplayName,
		MediaType: result.File.MimeType,
		ExpiresAt: expiresAt,
		Data:      data,
	}, nil
}

func (u *fileUploader) DeleteFile(ctx context.Context, ref provider.RemoteFileRef) error {
	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return fmt.Errorf("resolving auth token: %w", err)
	}

	reqURL := u.opts.baseURL + "/v1beta/" + ref.ID
	req, err := http.NewRequestWithContext(ctx, "DELETE", reqURL, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("x-goog-api-key", token)
	for k, v := range u.opts.headers {
		req.Header.Set(k, v)
	}

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
		respBody, _ := io.ReadAll(resp.Body)
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
	mediaType, data, ok := httpcParseDataURL(part.URL)
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

// httpcParseDataURL is an alias for httpc.ParseDataURL to avoid importing httpc.
// It parses data: URLs into media type and base64 data.
var httpcParseDataURL = func(url string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", "", false
	}
	rest := url[5:]
	semicolon := strings.Index(rest, ";base64,")
	if semicolon < 0 {
		return "", "", false
	}
	return rest[:semicolon], rest[semicolon+8:], true
}