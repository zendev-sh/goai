package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

var _ provider.VideoModel = (*videoModel)(nil)

const (
	maxGoogleVideoOperationBytes int64 = 10 << 20
	maxGoogleVideoErrorBytes     int64 = 1 << 20
	maxGoogleVideoDownloadBytes  int64 = 512 << 20
)

// Video creates a Google Veo video model using the Gemini API.
func Video(modelID string, opts ...Option) provider.VideoModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY")); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("GOOGLE_GENERATIVE_AI_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return &videoModel{id: modelID, opts: o}
}

type videoModel struct {
	id   string
	opts options
}

func (m *videoModel) ModelID() string { return m.id }

func (m *videoModel) DoGenerate(ctx context.Context, params provider.VideoParams) (*provider.VideoResult, error) {
	if err := validateNonChatOptions(m.opts); err != nil {
		return nil, err
	}
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	instance := map[string]any{"prompt": params.Prompt}
	if len(params.FrameImages) > 0 && len(params.InputReferences) > 0 {
		return nil, errors.New("google Gemini video generation cannot combine frame images with input references")
	}
	if params.Image != nil {
		encoded, err := encodeVideoMedia(*params.Image)
		if err != nil {
			return nil, err
		}
		instance["image"] = encoded
	}
	hasFirstFrame := params.Image != nil
	seenFirstFrame := false
	seenLastFrame := false
	for _, frame := range params.FrameImages {
		switch frame.Type {
		case provider.VideoFrameFirst:
			if seenFirstFrame {
				return nil, errors.New("google Gemini video generation received a duplicate first frame")
			}
			seenFirstFrame = true
			hasFirstFrame = true
		case provider.VideoFrameLast:
			if seenLastFrame {
				return nil, errors.New("google Gemini video generation received a duplicate last frame")
			}
			seenLastFrame = true
		default:
			return nil, fmt.Errorf("google Gemini video generation received unsupported frame type %q", frame.Type)
		}
		encoded, err := encodeVideoMedia(frame.Image)
		if err != nil {
			return nil, err
		}
		switch frame.Type {
		case provider.VideoFrameFirst:
			instance["image"] = encoded
		case provider.VideoFrameLast:
			instance["lastFrame"] = encoded
		}
	}
	if seenLastFrame && !hasFirstFrame {
		return nil, errors.New("google Gemini video generation last frame requires an initial image or first frame")
	}

	parameters := map[string]any{"sampleCount": params.N}
	if params.AspectRatio != "" {
		parameters["aspectRatio"] = params.AspectRatio
	}
	if params.Resolution != "" {
		resolution := map[string]string{
			"1280x720":  "720p",
			"1920x1080": "1080p",
			"3840x2160": "4k",
		}[params.Resolution]
		parameters["resolution"] = cmp.Or(resolution, params.Resolution)
	}
	if params.Duration > 0 {
		parameters["durationSeconds"] = int(params.Duration / time.Second)
	}
	if params.FPS > 0 {
		return nil, errors.New("google Gemini video generation does not support fps")
	}
	if params.Seed != nil {
		parameters["seed"] = *params.Seed
	}
	if params.GenerateAudio != nil {
		parameters["generateAudio"] = *params.GenerateAudio
	}
	if gopts, ok := params.ProviderOptions["google"].(map[string]any); ok {
		for key, value := range gopts {
			parameters[key] = value
		}
	}
	if len(params.InputReferences) > 0 {
		references := make([]map[string]any, 0, len(params.InputReferences))
		for _, reference := range params.InputReferences {
			if !strings.HasPrefix(reference.MediaType, "image/") {
				return nil, errors.New("google Gemini video generation only supports image references")
			}
			encoded, err := encodeVideoMedia(reference)
			if err != nil {
				return nil, err
			}
			references = append(references, map[string]any{
				"image":         encoded,
				"referenceType": "asset",
			})
		}
		instance["referenceImages"] = references
	}

	body := map[string]any{
		"instances":  []map[string]any{instance},
		"parameters": parameters,
	}
	reqURL := fmt.Sprintf("%s/v1beta/models/%s:predictLongRunning", m.opts.baseURL, url.PathEscape(m.id))
	operation, err := m.startOperation(ctx, token, reqURL, body, params.MaxRetries)
	if err != nil {
		return nil, err
	}
	if operation.Name == "" {
		return nil, errors.New("google video generation returned an empty operation name")
	}

	operation, err = m.pollOperation(ctx, token, operation, params.PollInterval, params.PollTimeout)
	if err != nil {
		return nil, err
	}

	videos, err := m.downloadVideos(ctx, token, operation.videoFiles())
	if err != nil {
		return nil, err
	}
	if len(videos) == 0 {
		return nil, errors.New("google video generation returned no videos")
	}

	return &provider.VideoResult{
		Videos: videos,
		ProviderMetadata: map[string]map[string]any{
			"google": {"operationName": operation.Name},
		},
		Response: provider.ResponseMetadata{ID: operation.Name, Model: m.id},
	}, nil
}

func encodeVideoMedia(media provider.MediaData) (map[string]any, error) {
	if media.URL != "" {
		return nil, errors.New("google Gemini video generation requires inline image data")
	}
	if len(media.Data) == 0 {
		return nil, errors.New("google Gemini video generation received empty image data")
	}
	if !strings.HasPrefix(media.MediaType, "image/") {
		return nil, errors.New("google Gemini video generation requires an image media type")
	}
	if media.MediaType != "image/png" && media.MediaType != "image/jpeg" {
		return nil, errors.New("google Gemini video generation requires a PNG or JPEG image")
	}
	return map[string]any{
		"bytesBase64Encoded": base64.StdEncoding.EncodeToString(media.Data),
		"mimeType":           media.MediaType,
	}, nil
}

type googleVideoFile struct {
	URI                string `json:"uri"`
	MimeType           string `json:"mimeType"`
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	VideoBytes         string `json:"videoBytes"`
}

type googleGeneratedVideo struct {
	Video googleVideoFile `json:"video"`
}

type googleVideoOperation struct {
	Name  string `json:"name"`
	Done  bool   `json:"done"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Response struct {
		GenerateVideoResponse struct {
			GeneratedSamples []googleGeneratedVideo `json:"generatedSamples"`
			GeneratedVideos  []googleGeneratedVideo `json:"generatedVideos"`
		} `json:"generateVideoResponse"`
		GeneratedSamples []googleGeneratedVideo `json:"generatedSamples"`
		GeneratedVideos  []googleGeneratedVideo `json:"generatedVideos"`
	} `json:"response"`
}

func (o googleVideoOperation) videoFiles() []googleVideoFile {
	groups := [][]googleGeneratedVideo{
		o.Response.GenerateVideoResponse.GeneratedSamples,
		o.Response.GenerateVideoResponse.GeneratedVideos,
		o.Response.GeneratedSamples,
		o.Response.GeneratedVideos,
	}
	for _, group := range groups {
		if len(group) == 0 {
			continue
		}
		files := make([]googleVideoFile, len(group))
		for i, generated := range group {
			files[i] = generated.Video
		}
		return files
	}
	return nil
}

func (m *videoModel) startOperation(ctx context.Context, token, reqURL string, body map[string]any, maxRetries int) (googleVideoOperation, error) {
	if maxRetries < 0 {
		maxRetries = 0
	}
	for attempt := 0; ; attempt++ {
		operation, err := m.requestOperation(ctx, token, http.MethodPost, reqURL, body)
		if err == nil {
			return operation, nil
		}

		var apiErr *goai.APIError
		if !errors.As(err, &apiErr) || !safeVideoStartRetry(apiErr) || attempt >= maxRetries {
			if apiErr != nil && safeVideoStartRetry(apiErr) && maxRetries > 0 && attempt >= maxRetries {
				return googleVideoOperation{}, fmt.Errorf("goai: %d retries exhausted: %w", maxRetries, err)
			}
			return googleVideoOperation{}, err
		}
		if err := waitForVideoRetry(ctx, videoRetryDelay(apiErr, attempt)); err != nil {
			return googleVideoOperation{}, err
		}
	}
}

func safeVideoStartRetry(apiErr *goai.APIError) bool {
	return apiErr.IsRetryable && apiErr.StatusCode == http.StatusTooManyRequests
}

func videoRetryDelay(apiErr *goai.APIError, attempt int) time.Duration {
	if serverDelay := videoRetryAfter(apiErr); serverDelay > 0 {
		return serverDelay
	}
	return videoExponentialDelay(attempt)
}

func videoPollRetryDelay(apiErr *goai.APIError, attempt int) time.Duration {
	serverDelay := min(videoRetryAfter(apiErr), 60*time.Second)
	return max(serverDelay, videoExponentialDelay(attempt))
}

func videoRetryAfter(apiErr *goai.APIError) time.Duration {
	if apiErr.ResponseHeaders != nil {
		if value := apiErr.ResponseHeaders["retry-after-ms"]; value != "" {
			if milliseconds, err := strconv.ParseInt(value, 10, 64); err == nil && milliseconds > 0 {
				return time.Duration(milliseconds) * time.Millisecond
			}
		}
		if value := apiErr.ResponseHeaders["retry-after"]; value != "" {
			if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds > 0 {
				return time.Duration(seconds) * time.Second
			}
		}
	}
	return 0
}

func videoExponentialDelay(attempt int) time.Duration {
	delay := 2 * time.Second * time.Duration(1<<min(attempt, 5))
	return min(delay, 60*time.Second)
}

func waitForVideoRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *videoModel) requestOperation(ctx context.Context, token, method, reqURL string, body map[string]any) (googleVideoOperation, error) {
	if !sameOrigin(reqURL, m.opts.baseURL) {
		return googleVideoOperation{}, errors.New("google video operation URL must use the configured API origin")
	}
	var payload []byte
	if body != nil {
		payload = httpc.MustMarshalJSON(body)
	}
	req := httpc.MustNewRequest(ctx, method, reqURL, payload)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	m.setHeaders(req, token)

	resp, err := httpc.Do(m.operationHTTPClient(), req)
	if err != nil {
		return googleVideoOperation{}, fmt.Errorf("sending video request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := readLimited(resp.Body, maxGoogleVideoOperationBytes)
	if err != nil {
		return googleVideoOperation{}, fmt.Errorf("reading video response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return googleVideoOperation{}, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	var operation googleVideoOperation
	if err := json.Unmarshal(respBody, &operation); err != nil {
		return googleVideoOperation{}, fmt.Errorf("parsing video operation: %w", err)
	}
	return operation, nil
}

func (m *videoModel) pollOperation(ctx context.Context, token string, operation googleVideoOperation, interval, timeout time.Duration) (googleVideoOperation, error) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	deadline := time.Now().Add(timeout)
	pollCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	reqURL, err := operationURL(m.opts.baseURL, operation.Name)
	if err != nil {
		return googleVideoOperation{}, err
	}
	nextDelay := interval
	pollFailures := 0

	for !operation.Done {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return googleVideoOperation{}, fmt.Errorf("google video generation timed out after %s", timeout)
		}
		delay := nextDelay
		nextDelay = interval
		timer := time.NewTimer(delay)
		select {
		case <-pollCtx.Done():
			timer.Stop()
			if ctx.Err() != nil {
				return googleVideoOperation{}, ctx.Err()
			}
			return googleVideoOperation{}, fmt.Errorf("google video generation timed out after %s", timeout)
		case <-timer.C:
		}

		var err error
		operation, err = m.requestOperation(pollCtx, token, http.MethodGet, reqURL, nil)
		if err != nil {
			if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
				return googleVideoOperation{}, fmt.Errorf("google video generation timed out after %s", timeout)
			}
			var apiErr *goai.APIError
			if errors.As(err, &apiErr) && apiErr.IsRetryable {
				nextDelay = videoPollRetryDelay(apiErr, pollFailures)
				if remaining := time.Until(deadline); nextDelay >= remaining {
					nextDelay = remaining / 2
				}
				nextDelay = max(interval, nextDelay)
				pollFailures++
				continue
			}
			return googleVideoOperation{}, err
		}
		pollFailures = 0
	}
	if operation.Error != nil {
		return googleVideoOperation{}, fmt.Errorf("google video generation failed: %s (code %d)", operation.Error.Message, operation.Error.Code)
	}
	return operation, nil
}

func operationURL(baseURL, name string) (string, error) {
	if strings.HasPrefix(name, "https://") || strings.HasPrefix(name, "http://") {
		if !sameOrigin(name, baseURL) {
			return "", errors.New("google video operation URL must use the configured API origin")
		}
		return name, nil
	}
	return strings.TrimRight(baseURL, "/") + "/v1beta/" + strings.TrimLeft(name, "/"), nil
}

func decodeVideoData(encoded string, remainingBytes int64, index int) ([]byte, error) {
	if int64(base64.StdEncoding.DecodedLen(len(encoded))) > remainingBytes {
		return nil, fmt.Errorf("video data exceeds %d byte download limit", maxGoogleVideoDownloadBytes)
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("decoding video %d: %w", index, err)
	}
	return data, nil
}

func (m *videoModel) downloadVideos(ctx context.Context, token string, files []googleVideoFile) ([]provider.VideoData, error) {
	videos := make([]provider.VideoData, 0, len(files))
	remainingBytes := maxGoogleVideoDownloadBytes
	for i, file := range files {
		encoded := cmp.Or(file.BytesBase64Encoded, file.VideoBytes)
		if encoded != "" {
			data, err := decodeVideoData(encoded, remainingBytes, i)
			if err != nil {
				return nil, err
			}
			remainingBytes -= int64(len(data))
			videos = append(videos, provider.VideoData{Data: data, MediaType: cmp.Or(file.MimeType, "video/mp4")})
			continue
		}
		if file.URI == "" {
			return nil, fmt.Errorf("video %d has no data or URI", i)
		}
		if !sameOrigin(file.URI, m.opts.baseURL) {
			return nil, fmt.Errorf("video %d URL must use the configured API origin", i)
		}

		req := httpc.MustNewRequest(ctx, http.MethodGet, file.URI, nil)
		m.setHeaders(req, token)
		resp, err := m.downloadHTTPClient().Do(req)
		if err != nil {
			return nil, fmt.Errorf("downloading video %d: %w", i, err)
		}
		if resp.StatusCode != http.StatusOK {
			data, readErr := io.ReadAll(io.LimitReader(resp.Body, maxGoogleVideoErrorBytes))
			_ = resp.Body.Close()
			if readErr != nil {
				return nil, fmt.Errorf("reading video %d error response: %w", i, readErr)
			}
			return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, data, resp.Header)
		}
		if resp.ContentLength > remainingBytes {
			_ = resp.Body.Close()
			return nil, fmt.Errorf("video data exceeds %d byte download limit", maxGoogleVideoDownloadBytes)
		}
		data, readErr := readLimited(resp.Body, remainingBytes)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("reading video %d: %w", i, readErr)
		}
		remainingBytes -= int64(len(data))
		mediaType := resp.Header.Get("Content-Type")
		if mediaType == "" {
			mediaType = cmp.Or(file.MimeType, "video/mp4")
		}
		videos = append(videos, provider.VideoData{Data: data, MediaType: mediaType})
	}
	return videos, nil
}

func readLimited(reader io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("response exceeds %d byte limit", limit)
	}
	return data, nil
}

func (m *videoModel) operationHTTPClient() *http.Client {
	client := *m.httpClient()
	client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &client
}

func (m *videoModel) downloadHTTPClient() *http.Client {
	client := *m.httpClient()
	previousCheck := client.CheckRedirect
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if !sameOrigin(req.URL.String(), m.opts.baseURL) {
			req.Header.Del("x-goog-api-key")
			for key := range m.opts.headers {
				req.Header.Del(key)
			}
			return errors.New("google video request refused a cross-origin redirect")
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

func sameOrigin(rawURL, baseURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	base, err := url.Parse(baseURL)
	if err != nil {
		return false
	}
	if u.Scheme != base.Scheme || u.Host != base.Host {
		return false
	}
	// Reject userinfo (e.g. "https://evil@api.google.com"), which can be used
	// to disguise a foreign authority as the configured origin.
	if u.User != nil || base.User != nil {
		return false
	}
	return true
}

func (m *videoModel) setHeaders(req *http.Request, token string) {
	// Apply caller headers first, then the credential last so it cannot be
	// overridden by a caller-supplied x-goog-api-key.
	for key, value := range m.opts.headers {
		req.Header.Set(key, value)
	}
	req.Header.Set("x-goog-api-key", token)
}

func (m *videoModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("goai: no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *videoModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
