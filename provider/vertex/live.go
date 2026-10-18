package vertex

import (
	"context"
	"errors"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/google"
)

var _ provider.RealtimeModel = (*invalidRealtimeModel)(nil)

// Live creates a Gemini Live realtime model on Vertex AI, such as
// "gemini-3.8-live". Project, location, and credentials resolve like Chat: the
// project is required and authentication defaults to Application Default
// Credentials. WithBaseURL replaces the regional endpoint host.
//
// Experimental: see provider.RealtimeModel.
func Live(modelID string, opts ...Option) provider.RealtimeModel {
	o := resolveOpts(opts)
	if err := validateNonChatOptions(o, "Live"); err != nil {
		return &invalidRealtimeModel{id: modelID, err: err}
	}
	if o.project == "" {
		return &invalidRealtimeModel{id: modelID, err: errors.New("vertex: Live requires a project (WithProject or GOOGLE_CLOUD_PROJECT)")}
	}
	gopts := []google.Option{
		google.WithVertex(o.project, o.location),
		google.WithHeaders(o.headers),
		google.WithHTTPClient(o.httpClient),
	}
	if o.tokenSource != nil {
		gopts = append(gopts, google.WithTokenSource(o.tokenSource))
	}
	if o.baseURL != "" {
		gopts = append(gopts, google.WithVertexBaseURL(o.baseURL))
	}
	return google.Live(modelID, gopts...)
}

// invalidRealtimeModel reports a configuration error when the session opens.
type invalidRealtimeModel struct {
	id  string
	err error
}

func (m *invalidRealtimeModel) ModelID() string { return m.id }

func (m *invalidRealtimeModel) OpenRealtime(context.Context, provider.RealtimeParams) (provider.RealtimeConnection, error) {
	return nil, m.err
}
