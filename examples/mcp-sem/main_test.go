//go:build ignore

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/mcp"
)

// Exercise the real CLI and stdio handshake without npm, network, or Sem installed.
func TestMainCLI(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the example's CLI fixture uses a POSIX shell; documented platforms are macOS/Linux")
	}
	dir := t.TempDir()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("SEM_TEST_EXE", exe)
	t.Setenv("SEM_TEST_HELPER", "1")
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := os.WriteFile(filepath.Join(dir, "npx"), []byte("#!/bin/sh\nexec \"$SEM_TEST_EXE\" -test.run=TestStdioHelper\n"), 0700); err != nil {
		t.Fatal(err)
	}
	out, err := os.CreateTemp(dir, "output")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	args, flags, stdout := os.Args, flag.CommandLine, os.Stdout
	defer func() { os.Args, flag.CommandLine, os.Stdout = args, flags, stdout }()
	os.Args = []string{"mcp-sem", "-repo", dir, "-query", "parse_config"}
	flag.CommandLine = flag.NewFlagSet("mcp-sem", flag.ExitOnError)
	os.Stdout = out
	main()
	data, err := os.ReadFile(out.Name())
	if err != nil || !strings.Contains(string(data), "fixture source") {
		t.Fatalf("output=%q err=%v", data, err)
	}
}

func TestStdioHelper(t *testing.T) {
	if os.Getenv("SEM_TEST_HELPER") != "1" {
		return
	}
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		var msg mcp.JSONRPCMessage
		if json.Unmarshal(scanner.Bytes(), &msg) != nil {
			os.Exit(2)
		}
		var result any
		switch msg.Method {
		case "notifications/initialized":
			continue
		case "initialize":
			result = map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]any{"name": "fixture", "version": "1"}}
		case "tools/call":
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "fixture source"}}}
		default:
			os.Exit(3)
		}
		if json.NewEncoder(os.Stdout).Encode(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": result}) != nil {
			os.Exit(4)
		}
	}
	os.Exit(0)
}

func TestSemTransport(t *testing.T) {
	dir := t.TempDir()
	tr, err := semTransport(dir, "parse_config")
	if err != nil {
		t.Fatal(err)
	}
	if tr.Command != "npx" || !reflect.DeepEqual(tr.Args, []string{"-y", "@ataraxy-labs/sem@0.27.0", "mcp"}) || tr.Dir != dir {
		t.Fatalf("unexpected command configuration: %+v", tr)
	}
	if !reflect.DeepEqual(tr.Env, map[string]string{"SEM_REPO": dir, "SEM_NO_NETWORK": "1", "SEM_NO_TELEMETRY": "1"}) {
		t.Fatal(tr.Env)
	}
	file := filepath.Join(dir, "file")
	if err := os.WriteFile(file, nil, 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range [][2]string{{"", "f"}, {dir, ""}, {dir + "\n", "f"}, {file, "f"}, {filepath.Join(dir, "missing"), "f"}} {
		if _, err := semTransport(tc[0], tc[1]); err == nil {
			t.Errorf("accepted invalid input %q", tc)
		}
	}
}

func TestCommandMissingFlags(t *testing.T) {
	args, flags := os.Args, flag.CommandLine
	defer func() { os.Args, flag.CommandLine = args, flags }()
	os.Args = []string{"mcp-sem"}
	flag.CommandLine = flag.NewFlagSet("mcp-sem", flag.ContinueOnError)
	if err := command(); err == nil {
		t.Fatal("expected required flags error")
	}
}

func TestRun(t *testing.T) {
	for _, mode := range []string{"success", "tool-error", "rpc-error", "output-error"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					w.WriteHeader(http.StatusMethodNotAllowed)
					return
				}
				var msg mcp.JSONRPCMessage
				if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
					t.Error(err)
					return
				}
				var result any
				switch msg.Method {
				case "initialize":
					result = map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]any{"name": "fixture", "version": "1"}}
				case "notifications/initialized":
					w.WriteHeader(http.StatusAccepted)
					return
				case "tools/call":
					calls++
					var p struct {
						Name      string         `json:"name"`
						Arguments map[string]any `json:"arguments"`
					}
					if err := json.Unmarshal(msg.Params, &p); err != nil {
						t.Error(err)
					}
					if p.Name != "sem_find" || !reflect.DeepEqual(p.Arguments, map[string]any{"query": "parse_config", "mode": "context", "token_budget": float64(1200)}) {
						t.Errorf("unexpected call: %+v", p)
					}
					result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "config.py: parse_config"}}, "isError": mode == "tool-error"}
				default:
					t.Errorf("unexpected method: %s", msg.Method)
				}
				w.Header().Set("Content-Type", "application/json")
				response := map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": result}
				if mode == "rpc-error" && msg.Method == "tools/call" {
					delete(response, "result")
					response["error"] = map[string]any{"code": -32603, "message": "fixture failure"}
				}
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Error(err)
				}
			}))
			defer srv.Close()
			var out bytes.Buffer
			var writer io.Writer = &out
			if mode == "output-error" {
				writer = brokenWriter{}
			}
			err := run(context.Background(), mcp.NewHTTPTransport(srv.URL), "parse_config", writer)
			if mode == "success" {
				if err != nil || !strings.Contains(out.String(), "parse_config") {
					t.Fatalf("output=%q err=%v", out.String(), err)
				}
			} else if err == nil {
				t.Fatal("expected error")
			}
			if calls != 1 {
				t.Fatalf("got %d calls", calls)
			}
		})
	}
}

type brokenWriter struct{}

func (brokenWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestRunConnectionFailure(t *testing.T) {
	err := run(context.Background(), mcp.NewStdioTransport(filepath.Join(t.TempDir(), "missing"), nil), "f", io.Discard)
	if err == nil {
		t.Fatal("expected connection error")
	}
}
