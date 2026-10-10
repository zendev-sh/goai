package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

const reasoningNumericItem = `{"type":"reasoning","id":"rs_numeric","summary":[],"content":[{"type":"reasoning_text","text":"Think."}],"extension":{"large":1e1000,"integer":9007199254740993,"small":1e-1000}}`

// Both APIs require reasoning items, not the display text reconstructed as a
// summary. Exercise the next HTTP request, including persisted SDK history.
func TestResponsesReasoningReplay(t *testing.T) {
	for _, tc := range []struct {
		name, model, item string
	}{
		{"deepseek", "deepseek-flash", `{"type":"reasoning","id":"rs_raw","status":"completed","summary":[],"content":[{"type":"reasoning_text","text":"First."},{"type":"reasoning_text","text":""},{"type":"reasoning_text","text":"Second."}]}`},
		{"openai", "gpt-5", `{"type":"reasoning","id":"rs_summary","status":"completed","summary":[{"type":"summary_text","text":"First."},{"type":"summary_text","text":""},{"type":"summary_text","text":"Second."}],"encrypted_content":"opaque-completed-state"}`},
		{"openai-encrypted-only", "gpt-5", `{"type":"reasoning","id":"rs_encrypted","summary":[],"encrypted_content":"opaque-completed-state"}`},
		{"openai-schema-mixed", "gpt-5", `{"type":"reasoning","id":"rs_mixed","summary":[{"type":"summary_text","text":"Summary."}],"content":[{"type":"reasoning_text","text":"Raw."}],"encrypted_content":"opaque-completed-state"}`},
		{"empty", "gpt-5", `{"type":"reasoning","id":"rs_empty","summary":[],"content":[]}`},
		{"numeric-extension", "deepseek-flash", reasoningNumericItem},
	} {
		for _, mode := range []string{"generate", "stream-terminal", "stream-item-done"} {
			for _, toolLoop := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/tools=%v", tc.name, mode, toolLoop), func(t *testing.T) {
					var reasoning map[string]any
					decoder := json.NewDecoder(strings.NewReader(tc.item))
					decoder.UseNumber()
					if err := decoder.Decode(&reasoning); err != nil {
						t.Fatal(err)
					}
					output := []map[string]any{reasoning, replayMessage("msg_before", "commentary", "Checking.")}
					if toolLoop {
						output = append(output, map[string]any{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": "{}"})
					}
					requests := make(chan map[string]any, 3)
					n := 0
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						var body map[string]any
						decoder := json.NewDecoder(r.Body)
						decoder.UseNumber()
						if err := decoder.Decode(&body); err != nil {
							t.Error(err)
							http.Error(w, "invalid request", http.StatusBadRequest)
							return
						}
						requests <- body
						n++
						items := output
						if n > 1 {
							items = []map[string]any{replayMessage("msg_final", "final_answer", "Done.")}
						}
						if mode == "stream-terminal" && body["stream"] == true {
							w.Header().Set("Content-Type", "text/event-stream")
							emit := func(event any) {
								data, err := json.Marshal(event)
								if err != nil {
									t.Error(err)
									return
								}
								fmt.Fprintf(w, "data: %s\n\n", data)
							}
							// The completed snapshot is authoritative, even if an earlier
							// added item carried incomplete encrypted state.
							emit(map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"type": "reasoning", "id": "rs_pending", "encrypted_content": "partial"}})
							emit(map[string]any{"type": "response.output_item.done", "output_index": 0, "item": map[string]any{"type": "reasoning", "id": "rs_pending", "encrypted_content": "older-snapshot"}})
							if toolLoop && n == 1 {
								emit(map[string]any{"type": "response.output_item.added", "output_index": 2, "item": output[2]})
								emit(map[string]any{"type": "response.function_call_arguments.delta", "output_index": 2, "delta": "{}"})
								emit(map[string]any{"type": "response.function_call_arguments.done", "output_index": 2})
							}
							emit(map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_test", "model": tc.model, "output": items}})
						} else {
							emitReplayResponse(w, items, body["stream"] == true, false)
						}
					}))
					defer server.Close()
					model := Chat(tc.model, WithAPIKey("test"), WithBaseURL(server.URL))
					opts := []goai.Option{goai.WithPrompt("check"), goai.WithProviderOptions(map[string]any{"store": false})}
					if toolLoop {
						opts = append(opts, goai.WithMaxSteps(2), goai.WithTools(goai.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}))
					}
					var result *goai.TextResult
					var err error
					if mode == "generate" {
						result, err = goai.GenerateText(t.Context(), model, opts...)
					} else {
						stream, streamErr := goai.StreamText(t.Context(), model, opts...)
						if streamErr != nil {
							t.Fatal(streamErr)
						}
						result, err = stream.Result(), stream.Err()
					}
					if err != nil {
						t.Fatal(err)
					}
					<-requests
					if !toolLoop {
						saved, err := json.Marshal(result.ResponseMessages)
						if err != nil {
							t.Fatal(err)
						}
						var messages []provider.Message
						if err := json.Unmarshal(saved, &messages); err != nil {
							t.Fatal(err)
						}
						messages = append([]provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "check"}}}}, messages...)
						messages = append(messages, provider.Message{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "continue"}}})
						if _, err := model.DoGenerate(t.Context(), provider.GenerateParams{Messages: messages, ProviderOptions: map[string]any{"store": false}}); err != nil {
							t.Fatal(err)
						}
					}
					var replay map[string]any
					select {
					case replay = <-requests:
					default:
						t.Fatal("missing continuation request")
					}
					if _, ok := replay["previous_response_id"]; ok {
						t.Fatal("stateless replay must not use previous_response_id")
					}
					input := replay["input"].([]any)
					wantLen := 4
					if toolLoop {
						wantLen = 5
					}
					if len(input) != wantLen {
						t.Fatalf("input = %#v, want %d items", input, wantLen)
					}
					if !reflect.DeepEqual(input[1], reasoning) {
						t.Fatalf("reasoning replay = %#v, want %s", input[1], tc.item)
					}
					if input[2].(map[string]any)["phase"] != "commentary" {
						t.Fatalf("lost adjacent message order/phase: %#v", input)
					}
					if toolLoop && (input[3].(map[string]any)["type"] != "function_call" || input[4].(map[string]any)["type"] != "function_call_output") {
						t.Fatalf("lost tool round trip: %#v", input)
					}
				})
			}
		}
	}
}

func TestResponsesReasoningReplayState(t *testing.T) {
	for _, wire := range []string{
		`{"type":"reasoning","id":"rs_1","summary":[],"content":[]}`,
		`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":""}],"content":[{"type":"reasoning_text","text":""}]}`,
		`{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}`,
		`{"type":"reasoning","id":"rs_1","summary":null,"content":null,"encrypted_content":null}`,
	} {
		t.Run(wire, func(t *testing.T) {
			result, err := parseResponsesResult([]byte(`{"output":[` + wire + `]}`))
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Content) != 1 || len(result.ReasoningParts) != 1 {
				t.Fatalf("reasoning state was dropped: %#v", result)
			}
			// A display edit must not rewrite the protocol payload.
			part := result.ReasoningParts[0]
			part.Text = "display only"
			before, err := json.Marshal(part)
			if err != nil {
				t.Fatal(err)
			}
			item, ok := reasoningInputItem(part)
			if !ok {
				t.Fatal("missing replay item")
			}
			var want map[string]any
			if err := json.Unmarshal([]byte(wire), &want); err != nil {
				t.Fatal(err)
			}
			// OpenAI requires a summary array even for encrypted-only items.
			if want["summary"] == nil {
				want["summary"] = []map[string]any{}
			}
			gotJSON, err := json.Marshal(item)
			if err != nil {
				t.Fatal(err)
			}
			wantJSON, err := json.Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			if string(gotJSON) != string(wantJSON) {
				t.Fatalf("replay = %s, want %s", gotJSON, wantJSON)
			}
			item["id"] = "different" // serializer must return its own map
			after, err := json.Marshal(part)
			if err != nil || string(after) != string(before) {
				t.Fatalf("replay mutated its input: before=%s, after=%s, err=%v", before, after, err)
			}
		})
	}
}

func TestResponsesReasoningReplayLegacyJSON(t *testing.T) {
	for _, raw := range []any{nil, map[string]any(nil), 42} {
		part := openAIReasoningPart("rs_legacy", "Summary.", "opaque")
		part.ProviderOptions["openai"].(map[string]any)["rawItem"] = raw
		for _, persisted := range []bool{false, true} {
			if persisted {
				data, err := json.Marshal(part)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(data, &part); err != nil {
					t.Fatal(err)
				}
			}
			item, ok := reasoningInputItem(part)
			if !ok {
				t.Fatal("old reasoning metadata is no longer replayable")
			}
			data, err := json.Marshal(item)
			want := `{"encrypted_content":"opaque","id":"rs_legacy","summary":[{"text":"Summary.","type":"summary_text"}],"type":"reasoning"}`
			if err != nil || string(data) != want {
				t.Fatalf("legacy replay = %s, err=%v", data, err)
			}
		}
	}
}

func TestResponsesReasoningReplayPreservesNumericState(t *testing.T) {
	result, err := parseResponsesResult([]byte(`{"output":[` + reasoningNumericItem + `,{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Answer."}]}],"usage":{"input_tokens":5,"output_tokens":7}}`))
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Answer." || result.Reasoning != "Think." || result.Usage.TotalTokens != 12 {
		t.Fatalf("usable result lost: %#v", result)
	}
	if len(result.Content) != 2 || result.Content[0].Type != provider.PartReasoning || result.Content[1].Type != provider.PartText || len(result.ReasoningParts) != 1 {
		t.Fatalf("ordered replay lost: %#v", result.Content)
	}
	// Use ordinary JSON persistence, not UseNumber at the caller boundary.
	saved, err := json.Marshal(result.ReasoningParts[0])
	if err != nil {
		t.Fatal(err)
	}
	var part provider.Part
	if err := json.Unmarshal(saved, &part); err != nil {
		t.Fatal(err)
	}
	item, ok := reasoningInputItem(part)
	if !ok {
		t.Fatal("reasoning item is no longer replayable")
	}
	extension, ok := item["extension"].(map[string]any)
	if !ok {
		t.Fatalf("extension lost: %#v", item)
	}
	for key, want := range map[string]json.Number{"large": "1e1000", "integer": "9007199254740993", "small": "1e-1000"} {
		if got := extension[key]; got != want {
			t.Errorf("extension[%s] = %#v, want %s", key, got, want)
		}
	}
	if _, err := json.Marshal(item); err != nil {
		t.Fatalf("cannot serialize replay request: %v", err)
	}
}

func TestResponsesReasoningReplayInvalidSavedItem(t *testing.T) {
	for _, tc := range []struct {
		name, text, encrypted, want string
	}{
		{"display", "display", "", `[{"id":"rs_1","summary":[{"text":"display","type":"summary_text"}],"type":"reasoning"}]`},
		{"encrypted only", "", "opaque", `[{"encrypted_content":"opaque","id":"rs_1","summary":[],"type":"reasoning"}]`},
		{"display and encrypted", "display", "opaque", `[{"encrypted_content":"opaque","id":"rs_1","summary":[{"text":"display","type":"summary_text"}],"type":"reasoning"}]`},
	} {
		for _, raw := range []string{"", "not json", `{"type":"reasoning"`, "null", "[]"} {
			t.Run(fmt.Sprintf("%s/%q", tc.name, raw), func(t *testing.T) {
				part := openAIReasoningPart("rs_1", tc.text, tc.encrypted)
				part.ProviderOptions["openai"].(map[string]any)["rawItem"] = raw
				// Assert the actual input items, not just the helper's boolean:
				// neither assistant output_text nor dropping encrypted state is OK.
				input := convertToResponsesInput([]provider.Message{{Role: provider.RoleAssistant, Content: []provider.Part{part}}})
				data, err := json.Marshal(input)
				if err != nil || string(data) != tc.want {
					t.Fatalf("damaged history replay = %s, want %s, err=%v", data, tc.want, err)
				}
			})
		}
	}
}

func TestResponsesReasoningTextDisplayParity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		deltas [][]string
		want   string
	}{
		{"one part, multiple deltas", [][]string{{"A", "B"}}, "AB"},
		{"two parts", [][]string{{"A"}, {"B"}}, "A\n\nB"},
		{"empty parts and deltas", [][]string{{""}, {"A", "", "1"}, {""}, {"B", "2"}, {""}}, "A1\n\nB2"},
		{"text does not define boundaries", [][]string{{"A\n\n"}, {"B"}}, "A\n\n\n\nB"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var content []any
			for _, deltas := range tc.deltas {
				content = append(content, map[string]any{"type": "reasoning_text", "text": strings.Join(deltas, "")})
			}
			item := map[string]any{"type": "reasoning", "id": "rs_display", "summary": []any{}, "content": content}
			response := map[string]any{"id": "resp_display", "status": "completed", "store": false, "output": []any{item}}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Stream bool `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					http.Error(w, "invalid request", http.StatusBadRequest)
					return
				}
				if !request.Stream {
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
				emit(map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"type": "reasoning", "id": "rs_display"}})
				for index, deltas := range tc.deltas {
					for _, delta := range deltas {
						emit(map[string]any{"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_display", "content_index": index, "delta": delta})
					}
				}
				emit(map[string]any{"type": "response.output_item.done", "output_index": 0, "item": item})
				emit(map[string]any{"type": "response.completed", "response": response})
			}))
			defer server.Close()
			model := Chat("deepseek-flash", WithAPIKey("test"), WithBaseURL(server.URL))
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("stream=%v", streaming), func(t *testing.T) {
					var result *goai.TextResult
					var err error
					if streaming {
						stream, streamErr := goai.StreamText(t.Context(), model, goai.WithPrompt("think"))
						if streamErr != nil {
							t.Fatal(streamErr)
						}
						result, err = stream.Result(), stream.Err()
					} else {
						result, err = goai.GenerateText(t.Context(), model, goai.WithPrompt("think"))
					}
					if err != nil {
						t.Fatal(err)
					}
					if result.Reasoning != tc.want || len(result.Steps) != 1 || result.Steps[0].Reasoning != tc.want {
						t.Fatalf("reasoning = %q, steps = %#v, want %q", result.Reasoning, result.Steps, tc.want)
					}
					if parts := result.Steps[0].Content; len(parts) != 1 || parts[0].Text != tc.want {
						t.Fatalf("snapshot = %#v, want text %q", parts, tc.want)
					}
					// Display separators must never leak into the original part texts.
					replay := convertToResponsesInput(result.ResponseMessages)
					if len(replay) != 1 || !reflect.DeepEqual(replay[0], item) {
						t.Fatalf("replay = %#v, want original item %#v", replay, item)
					}
				})
			}
		})
	}
}
