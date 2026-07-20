package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
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

	req, err := http.NewRequestWithContext(ctx, "POST", u.opts.baseURL+"/files", &buf)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+token)
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
		return nil, goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
	}

	var result struct {
		ID        string `json:"id"`
		Bytes     int64  `json:"bytes"`
		CreatedAt int64  `json:"created_at"`
		Filename  string `json:"filename"`
		Purpose   string `json:"purpose"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
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
	token, err := u.opts.tokenSource.Token(ctx)
	if err != nil {
		return fmt.Errorf("resolving auth token: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "DELETE", u.opts.baseURL+"/files/"+ref.ID, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
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

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
	}

	return nil
}