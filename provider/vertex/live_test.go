package vertex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

func TestLive_UsesVertexEndpointAndOAuth(t *testing.T) {
	ignore := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, ignore) })

	type handshake struct{ path, auth, model string }
	got := make(chan handshake, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		ctx := r.Context()
		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		var msg struct {
			Setup struct {
				Model string `json:"model"`
			} `json:"setup"`
		}
		_ = json.Unmarshal(data, &msg)
		got <- handshake{r.URL.Path, r.Header.Get("Authorization"), msg.Setup.Model}
		_ = conn.Write(ctx, websocket.MessageText, []byte(`{"setupComplete":{}}`))
		for {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(server.Close)

	conn, err := Live("gemini-3.8-live",
		WithProject("my-project"),
		WithLocation("europe-west4"),
		WithTokenSource(provider.StaticToken("oauth-token")),
		WithBaseURL(server.URL),
	).OpenRealtime(t.Context(), provider.RealtimeParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()

	h := <-got
	if !strings.Contains(h.path, "LlmBidiService/BidiGenerateContent") {
		t.Errorf("path = %s", h.path)
	}
	if h.auth != "Bearer oauth-token" {
		t.Errorf("Authorization = %q", h.auth)
	}
	if want := "projects/my-project/locations/europe-west4/publishers/google/models/gemini-3.8-live"; h.model != want {
		t.Errorf("model = %s, want %s", h.model, want)
	}
}

func TestLive_ConfigurationErrors(t *testing.T) {
	for _, env := range []string{"GOOGLE_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"} {
		t.Setenv(env, "")
	}
	tests := map[string]provider.RealtimeModel{
		"missing project": Live("m", WithTokenSource(provider.StaticToken("t"))),
		"native gemini":   Live("m", WithProject("p"), WithTokenSource(provider.StaticToken("t")), WithNativeGemini()),
	}
	for name, model := range tests {
		t.Run(name, func(t *testing.T) {
			if model.ModelID() != "m" {
				t.Fatalf("ModelID = %q", model.ModelID())
			}
			if _, err := model.OpenRealtime(context.Background(), provider.RealtimeParams{}); err == nil {
				t.Fatal("expected a configuration error")
			}
		})
	}
}
