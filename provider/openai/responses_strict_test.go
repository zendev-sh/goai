package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestResponsesFunctionToolStrict(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts map[string]any
		want any
	}{
		{name: "unset"},
		{name: "false", opts: map[string]any{"strict": false}, want: false},
		{name: "true", opts: map[string]any{"strict": true}, want: true},
		{name: "invalid", opts: map[string]any{"strict": "false"}},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, streaming), func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					if _, present := body["strict"]; present {
						t.Error("strict must not appear at the request root")
					}
					tools := body["tools"].([]any)
					function := tools[0].(map[string]any)
					got, present := function["strict"]
					if got != tc.want || present != (tc.want != nil) {
						t.Errorf("function strict = %v (present=%t), want %v", got, present, tc.want)
					}
					if _, present := tools[1].(map[string]any)["strict"]; present {
						t.Error("strict must not be added to provider-defined tools")
					}
					format := body["text"].(map[string]any)["format"].(map[string]any)
					if format["strict"] != false {
						t.Error("tool strict must not change structured-output strictness")
					}
					response := `{"id":"resp_1","status":"completed","output":[]}`
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":%s}\n\n", response)
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, response)
					}
				}))
				defer server.Close()
				model := Chat("test-model", WithAPIKey("test-key"), WithBaseURL(server.URL))
				params := provider.GenerateParams{
					ProviderOptions: tc.opts,
					Tools: []provider.ToolDefinition{
						{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`)},
						{ProviderDefinedType: "web_search"},
					},
					ResponseFormat: &provider.ResponseFormat{Name: "answer", Schema: json.RawMessage(`{"type":"object","properties":{}}`)},
				}
				if streaming {
					result, err := model.DoStream(t.Context(), params)
					if err != nil {
						t.Fatal(err)
					}
					for chunk := range result.Stream {
						if chunk.Error != nil {
							t.Fatal(chunk.Error)
						}
					}
				} else if _, err := model.DoGenerate(t.Context(), params); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}
