package goai

import (
	"context"
	"errors"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// GenerateImage generates images from a text prompt.
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
		Images: result.Images,
	}, nil
}

// ImageResult contains the generated images.
type ImageResult struct {
	Images []provider.ImageData
}

// ImageOption configures image generation.
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
func WithImageCount(n int) ImageOption {
	return func(o *imageOptions) {
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
func WithImageProviderOptions(opts map[string]any) ImageOption {
	return func(o *imageOptions) {
		o.providerOptions = opts
	}
}

// WithImageMaxRetries sets the maximum number of retries for transient errors.
func WithImageMaxRetries(n int) ImageOption {
	return func(o *imageOptions) {
		o.maxRetries = n
	}
}

// WithImageTimeout sets a timeout for the image generation request.
func WithImageTimeout(d time.Duration) ImageOption {
	return func(o *imageOptions) {
		o.timeout = d
	}
}
