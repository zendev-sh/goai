package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/zendev-sh/goai"
)

// No request-side store option: the service's actual storage state must drive
// automatic continuation, even when a stateless service uses resp_ IDs.
func TestResponsesStorageContinuation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		stores []string // empty means omitted, "null" means unknown
	}{
		{"deepseek-stateless", []string{"false"}},
		{"openai-stored", []string{"true"}},
		{"gateway-omitted", []string{""}},
		{"gateway-null", []string{"null"}},
		{"stored-then-stateless", []string{"true", "false"}},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", tc.name, streaming), func(t *testing.T) {
				requests := make(chan map[string]any, len(tc.stores)+1)
				n := 0
				reasoning := map[string]any{"type": "reasoning", "id": "rs_1", "summary": []any{}, "content": []any{map[string]any{"type": "reasoning_text", "text": "Check the tool."}}}
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						http.Error(w, "invalid request", http.StatusBadRequest)
						return
					}
					requests <- body
					n++
					output := []map[string]any{replayMessage("msg_final", "final_answer", "Done.")}
					response := map[string]any{"id": fmt.Sprintf("resp_%d", n), "status": "completed", "model": "test"}
					if n <= len(tc.stores) {
						item := maps.Clone(reasoning)
						item["id"] = fmt.Sprintf("rs_%d", n)
						output = []map[string]any{item, {"type": "function_call", "id": fmt.Sprintf("fc_%d", n), "call_id": fmt.Sprintf("call_%d", n), "name": "lookup", "arguments": "{}"}}
						if store := tc.stores[n-1]; store != "" {
							var value any
							if err := json.Unmarshal([]byte(store), &value); err != nil {
								t.Error(err)
								return
							}
							response["store"] = value
						}
					}
					response["output"] = output
					if !streaming {
						if err := json.NewEncoder(w).Encode(response); err != nil {
							t.Error(err)
						}
						return
					}
					w.Header().Set("Content-Type", "text/event-stream")
					emit := func(event any) {
						data, err := json.Marshal(event)
						if err != nil {
							t.Error(err)
							return
						}
						fmt.Fprintf(w, "data: %s\n\n", data)
					}
					if n <= len(tc.stores) {
						emit(map[string]any{"type": "response.output_item.added", "output_index": 1, "item": output[1]})
						emit(map[string]any{"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{}"})
						emit(map[string]any{"type": "response.function_call_arguments.done", "output_index": 1})
					}
					emit(map[string]any{"type": "response.completed", "response": response})
				}))
				defer server.Close()
				model := Chat("test", WithAPIKey("test"), WithBaseURL(server.URL))
				opts := []goai.Option{
					goai.WithPrompt("check"), goai.WithMaxSteps(len(tc.stores) + 1),
					goai.WithTools(goai.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
				}
				if streaming {
					stream, err := goai.StreamText(t.Context(), model, opts...)
					if err != nil {
						t.Fatal(err)
					}
					stream.Result()
					if err := stream.Err(); err != nil {
						t.Fatal(err)
					}
				} else if _, err := goai.GenerateText(t.Context(), model, opts...); err != nil {
					t.Fatal(err)
				}
				for step := 0; step <= len(tc.stores); step++ {
					var body map[string]any
					select {
					case body = <-requests:
					default:
						t.Fatalf("missing request %d", step+1)
					}
					if _, ok := body["store"]; ok {
						t.Fatal("test must not rely on a request-side store option")
					}
					if step == 0 {
						continue
					}
					input := body["input"].([]any)
					if tc.stores[step-1] != "false" {
						if body["previous_response_id"] != "resp_1" || len(input) != 1 || input[0].(map[string]any)["type"] != "function_call_output" {
							t.Fatalf("stored/unknown continuation changed: %#v", body)
						}
						continue
					}
					if _, ok := body["previous_response_id"]; ok {
						t.Fatalf("stateless response enabled server continuation: %#v", body)
					}
					if len(input) != 1+3*step || input[0].(map[string]any)["role"] != "user" {
						t.Fatalf("lost full history: %#v", input)
					}
					for i := range step {
						want := maps.Clone(reasoning)
						want["id"] = fmt.Sprintf("rs_%d", i+1)
						if !reflect.DeepEqual(input[1+3*i], want) || input[2+3*i].(map[string]any)["call_id"] != fmt.Sprintf("call_%d", i+1) || input[3+3*i].(map[string]any)["call_id"] != fmt.Sprintf("call_%d", i+1) {
							t.Fatalf("lost reasoning/tool history: %#v", input)
						}
					}
				}
			})
		}
	}
}
