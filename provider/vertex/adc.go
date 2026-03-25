package vertex

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/oauth2/google"

	"github.com/zendev-sh/goai/provider"
)

// ADCTokenSource creates a TokenSource that uses Google Application Default Credentials.
//
// It auto-detects credentials from (in order):
//  1. GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON file)
//  2. gcloud CLI credentials (~/.config/gcloud/application_default_credentials.json)
//  3. GCE metadata service (when running on Google Cloud)
//
// This is the Go equivalent of Vercel's google-auth-library auto-detection.
// Credentials are resolved once at construction time; only token refresh happens
// on subsequent calls.
//
// Usage:
//
//	ts, err := vertex.ADCTokenSource(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//	model := vertex.Chat("gemini-2.5-pro",
//		vertex.WithTokenSource(ts),
//		vertex.WithProject("my-project"),
//	)
func ADCTokenSource(ctx context.Context, scopes ...string) (provider.TokenSource, error) {
	if len(scopes) == 0 {
		scopes = []string{"https://www.googleapis.com/auth/cloud-platform"}
	}
	creds, err := google.FindDefaultCredentials(ctx, scopes...)
	if err != nil {
		return nil, fmt.Errorf("vertex: finding default credentials: %w", err)
	}
	return provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
		tok, err := creds.TokenSource.Token()
		if err != nil {
			return nil, err
		}
		return &provider.Token{
			Value:     tok.AccessToken,
			ExpiresAt: tok.Expiry.Add(-30 * time.Second), // refresh 30s early
		}, nil
	}), nil
}
