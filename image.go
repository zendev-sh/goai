package goai

import (
	"context"
	"errors"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// GenerateImage generates images from a text prompt using the given model.
//
// Note: GenerateImage uses ImageOption, not the general-purpose Option type.
// Use WithImagePrompt, WithImageTimeout, WithImageMaxRetries, etc.
// These cannot be mixed with Option values from GenerateText/Embed calls.
func GenerateImage(ctx context.Context, model provider.ImageModel, opts ...ImageOption) (*ImageResult, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := imageOptions{
		n:          1,
		maxRetries: 2,
	}
	for _, opt := range opts {
		opt(&o)
	}

	if o.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.timeout)
		defer cancel()
	}

	params := provider.ImageParams{
		Prompt:          o.prompt,
		N:               o.n,
		Size:            o.size,
		AspectRatio:     o.aspectRatio,
		ProviderOptions: o.providerOptions,
	}

	result, err := withRetry(ctx, o.maxRetries, func() (*provider.ImageResult, error) {
		return model.DoGenerate(ctx, params)
	})
	if err != nil {
		return nil, err
	}

	return &ImageResult{
		Images:           result.Images,
		ProviderMetadata: result.ProviderMetadata,
		Usage:            result.Usage,
		Response:         result.Response,
	}, nil
}

// ImageResult contains the generated images.
type ImageResult struct {
	Images []provider.ImageData

	// ProviderMetadata contains provider-specific response data.
	ProviderMetadata map[string]map[string]any

	// Usage tracks token or operation consumption (if reported by the provider).
	Usage provider.Usage

	// Response contains provider-specific response metadata (ID, model, headers).
	Response provider.ResponseMetadata
}

// ImageOption configures a GenerateImage call.
//
// GenerateImage uses ImageOption (not the general-purpose Option type) because
// image generation has a distinct parameter set that does not overlap with text
// generation options. Common cross-cutting options like timeout and retries have
// dedicated WithImage* equivalents: WithImageTimeout, WithImageMaxRetries.
type ImageOption func(*imageOptions)

type imageOptions struct {
	prompt          string
	n               int
	size            string
	aspectRatio     string
	providerOptions map[string]any
	maxRetries      int
	timeout         time.Duration
}

// WithImagePrompt sets the text prompt for image generation.
func WithImagePrompt(prompt string) ImageOption {
	return func(o *imageOptions) {
		o.prompt = prompt
	}
}

// WithImageCount sets the number of images to generate.
// Values below 1 are clamped to 1 (minimum one image).
func WithImageCount(n int) ImageOption {
	return func(o *imageOptions) {
		if n < 1 {
			n = 1
		}
		o.n = n
	}
}

// WithImageSize sets the image size (e.g. "1024x1024", "512x512").
func WithImageSize(size string) ImageOption {
	return func(o *imageOptions) {
		o.size = size
	}
}

// WithAspectRatio sets the aspect ratio (e.g. "16:9", "1:1").
func WithAspectRatio(ratio string) ImageOption {
	return func(o *imageOptions) {
		o.aspectRatio = ratio
	}
}

// WithImageProviderOptions sets provider-specific options.
// Values must be JSON-serializable (no channels, functions, or unsafe pointers).
func WithImageProviderOptions(opts map[string]any) ImageOption {
	validateProviderOptions("WithImageProviderOptions", opts)
	return func(o *imageOptions) {
		o.providerOptions = opts
	}
}

// WithImageMaxRetries sets the maximum number of retries for transient errors.
// Values below 0 are clamped to 0 (no retries).
func WithImageMaxRetries(n int) ImageOption {
	return func(o *imageOptions) {
		if n < 0 {
			n = 0
		}
		o.maxRetries = n
	}
}

// WithImageTimeout sets a timeout for the image generation request.
func WithImageTimeout(d time.Duration) ImageOption {
	return func(o *imageOptions) {
		o.timeout = d
	}
}
