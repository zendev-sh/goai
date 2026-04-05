// Package langfuse provides Langfuse tracing hooks for goai.
//
// Basic usage -- credentials come from env vars (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
// LANGFUSE_HOST or LANGFUSE_BASE_URL):
//
//	result, err := goai.GenerateText(ctx, model,
//	    langfuse.WithTracing(),
//	    goai.WithPrompt("hello"),
//	)
//
// With options:
//
//	result, err := goai.GenerateText(ctx, model,
//	    langfuse.WithTracing(
//	        langfuse.TraceName("my-agent"),
//	        langfuse.UserID("user-123"),
//	        langfuse.Tags("prod", "v2"),
//	    ),
//	    goai.WithPrompt("hello"),
//	)
//
// Each call creates a fresh trace with isolated state. Concurrent calls are safe.
//
// Observation hierarchy:
//
//	Trace
//	└── Span("agent")            — wraps the entire run
//	    ├── Generation("step-1") — LLM call, child of agent span
//	    ├── Span("tool-name")    — tool execution, sibling of generation
//	    └── Generation("step-2") — final LLM call
package langfuse

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// TracingOption configures WithTracing.
type TracingOption func(*Config)

// TraceName sets the trace name (default "agent").
func TraceName(name string) TracingOption { return func(c *Config) { c.TraceName = name } }

// UserID sets the Langfuse user ID on the trace.
func UserID(id string) TracingOption { return func(c *Config) { c.UserID = id } }

// SessionID sets the Langfuse session ID for grouping traces.
func SessionID(id string) TracingOption { return func(c *Config) { c.SessionID = id } }

// Tags sets tags on the trace.
func Tags(tags ...string) TracingOption { return func(c *Config) { c.Tags = tags } }

// Metadata sets metadata on the trace.
func Metadata(m any) TracingOption { return func(c *Config) { c.Metadata = m } }

// Release sets the release identifier on the trace.
func Release(r string) TracingOption { return func(c *Config) { c.Release = r } }

// Version sets the version identifier on the trace.
func Version(v string) TracingOption { return func(c *Config) { c.Version = v } }

// Environment sets the environment (overrides LANGFUSE_ENV).
func Environment(env string) TracingOption { return func(c *Config) { c.Environment = env } }

// PromptName sets the Langfuse prompt name for linking generations.
func PromptName(name string) TracingOption { return func(c *Config) { c.PromptName = name } }

// PromptVersion sets the Langfuse prompt version for linking generations.
func PromptVersion(v int) TracingOption { return func(c *Config) { c.PromptVersion = v } }

// OnFlushError sets a callback for flush errors (default: silently discard).
func OnFlushError(fn func(error)) TracingOption { return func(c *Config) { c.OnFlushError = fn } }

// PublicKey overrides the LANGFUSE_PUBLIC_KEY env var.
func PublicKey(key string) TracingOption { return func(c *Config) { c.PublicKey = key } }

// SecretKey overrides the LANGFUSE_SECRET_KEY env var.
func SecretKey(key string) TracingOption { return func(c *Config) { c.SecretKey = key } }

// Host overrides the LANGFUSE_HOST / LANGFUSE_BASE_URL env var.
func Host(host string) TracingOption { return func(c *Config) { c.Host = host } }

// WithTracing returns a goai.Option that enables Langfuse tracing for a single call.
// Credentials are read from env vars unless overridden via PublicKey/SecretKey/Host options.
// Each invocation creates a fresh trace -- safe for concurrent use.
//
// Each call allocates a small amount of state for trace isolation. The underlying
// HTTP connections are pooled by Go's default transport, so creating a new
// http.Client per call does not waste TCP connections.
//
// If neither LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars nor PublicKey/SecretKey
// options are set, WithTracing logs a warning to stderr and returns a no-op option.
func WithTracing(opts ...TracingOption) goai.Option {
	cfg := Config{}
	for _, o := range opts {
		o(&cfg)
	}

	// Resolve credentials from env vars if not set via options.
	pub := cfg.PublicKey
	if pub == "" {
		pub = os.Getenv("LANGFUSE_PUBLIC_KEY")
	}
	sec := cfg.SecretKey
	if sec == "" {
		sec = os.Getenv("LANGFUSE_SECRET_KEY")
	}
	if pub == "" || sec == "" {
		fmt.Fprintln(os.Stderr, "langfuse: warning: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set, tracing disabled")
		return goai.WithOptions() // no-op
	}

	h := New(cfg)
	return goai.WithOptions(h.Run()...)
}

// Config configures Langfuse tracing. Credential fields override the corresponding
// env vars (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST).
type Config struct {
	PublicKey string // overrides LANGFUSE_PUBLIC_KEY
	SecretKey string // overrides LANGFUSE_SECRET_KEY
	Host      string // overrides LANGFUSE_HOST / LANGFUSE_BASE_URL

	TraceName   string // defaults to "agent"
	UserID      string
	SessionID   string
	Tags        []string
	Metadata    any
	Release     string
	Version     string
	Environment string // falls back to LANGFUSE_ENV

	PromptName    string
	PromptVersion int

	// OnFlushError is called when the HTTP flush to Langfuse fails.
	// If nil, flush errors are silently discarded (tracing must not crash the app).
	OnFlushError func(error)
}

// Hooks holds the shared HTTP client and config.
// Create once; call Run() to get a fresh set of options per agent run.
//
// Deprecated: Use [WithTracing] instead.
type Hooks struct {
	cfg Config
	mu  sync.Mutex
	lc  *client // lazily initialised, shared across runs
}

// New returns a Hooks instance. The HTTP client is initialised lazily on the first Run() call.
//
// Deprecated: Use [WithTracing] instead.
func New(cfg Config) *Hooks {
	if cfg.TraceName == "" {
		cfg.TraceName = "agent"
	}
	if cfg.Environment == "" {
		cfg.Environment = os.Getenv("LANGFUSE_ENV")
	}
	return &Hooks{cfg: cfg}
}

// client returns the shared HTTP client, initialising it on first call.
func (h *Hooks) client() *client {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.lc == nil {
		pub := h.cfg.PublicKey
		if pub == "" {
			pub = os.Getenv("LANGFUSE_PUBLIC_KEY")
		}
		sec := h.cfg.SecretKey
		if sec == "" {
			sec = os.Getenv("LANGFUSE_SECRET_KEY")
		}
		host := h.cfg.Host
		if host == "" {
			host = os.Getenv("LANGFUSE_HOST")
		}
		if host == "" {
			host = os.Getenv("LANGFUSE_BASE_URL")
		}
		if host == "" {
			host = "https://cloud.langfuse.com"
		}
		h.lc = newClient(host, pub, sec)
	}
	return h.lc
}

// With returns a per-run options factory that includes langfuse tracing plus
// any additional options. Use this as the Options field on an agent:
//
//	Options: lf.With()                      // tracing only
//	Options: lf.With(goai.WithMaxSteps(5))  // tracing + extra options
//
// Deprecated: Use [WithTracing] instead.
func (h *Hooks) With(opts ...goai.Option) func() []goai.Option {
	return func() []goai.Option {
		return append(h.Run(), opts...)
	}
}

// Run returns a fresh set of goai options scoped to a single agent run.
// Safe to call concurrently — each call gets completely isolated state.
// OnToolCall may fire from parallel goroutines, so a mutex guards shared state.
//
// Deprecated: Use [WithTracing] instead.
func (h *Hooks) Run() []goai.Option {
	cfg := h.cfg
	lc := h.client()

	// Per-run state. Isolated per Run() call. Most hooks are called
	// sequentially by goai, but OnToolCall may fire from parallel
	// goroutines when multiple tool calls execute concurrently, so
	// mu guards writes to obs and lastObsEnd.
	var (
		traceID     string
		agentSpanID string
		agentStart  time.Time
		traceInput  any
		gen         *pendingGen
		step        int
		lastObsEnd  time.Time
		obs         []ingestionEvent
		mu          sync.Mutex
	)

	// end is declared before opts so WithOnStepFinish can reference it.
	var end func(result any)

	// OnToolCallStart is intentionally not registered here because
	// OnToolCall.StartTime already provides accurate timing for tool spans.
	opts := []goai.Option{
		goai.WithOnRequest(func(info goai.RequestInfo) {
			input := messagesToInput(info.Messages)

			if traceID == "" {
				traceID = newID()
				agentSpanID = newID()
				agentStart = time.Now()
				traceInput = input
			}

			// Finalise the previous generation with its tool-call output.
			if gen != nil {
				for i := len(info.Messages) - 1; i >= 0; i-- {
					if info.Messages[i].Role == provider.RoleAssistant {
						gen.output = lastAssistantOutput(info.Messages[i])
						break
					}
				}
				obs = append(obs, gen.toEvent(traceID, agentSpanID, cfg))
				gen = nil
			}

			step++

			// Offset the generation start time so it appears after preceding tool spans
			// in Langfuse's timeline. Langfuse sorts observations by startTime, so if
			// a generation starts at the same wall-clock instant as the last tool span
			// ends (common when tool execution is very fast), they render in wrong order.
			now := time.Now()
			if !lastObsEnd.IsZero() {
				if floor := lastObsEnd.Add(10 * time.Millisecond); !now.After(floor) {
					now = floor
				}
			}

			genMeta := map[string]any{}
			if cfg.Environment != "" {
				genMeta["environment"] = cfg.Environment
			}

			gen = &pendingGen{
				id:        newID(),
				name:      fmt.Sprintf("step-%d", step),
				model:     info.Model,
				startTime: now,
				input:     input,
				metadata:  genMeta,
			}
		}),

		goai.WithOnResponse(func(info goai.ResponseInfo) {
			if gen == nil {
				return
			}
			gen.latency = info.Latency
			if info.Error != nil {
				// Run failed (e.g. context cancelled, max retries exhausted).
				// Flush whatever partial trace we have so it's not silently lost.
				gen.level = levelError
				gen.statusMsg = info.Error.Error()
				end(nil)
				return
			}
			if info.FinishReason == provider.FinishLength {
				gen.level = levelWarning
			}
			if info.FinishReason != "" {
				gen.statusMsg = string(info.FinishReason)
			}
			gen.usage = usageBody{
				Input:  info.Usage.InputTokens,
				Output: info.Usage.OutputTokens,
				Total:  info.Usage.InputTokens + info.Usage.OutputTokens,
				Unit:   unitTokens,
			}
			if info.Usage.ReasoningTokens > 0 {
				gen.metadata["reasoning_tokens"] = info.Usage.ReasoningTokens
			}
			if info.Usage.CacheReadTokens > 0 {
				gen.metadata["cache_read_tokens"] = info.Usage.CacheReadTokens
			}
			if info.Usage.CacheWriteTokens > 0 {
				gen.metadata["cache_write_tokens"] = info.Usage.CacheWriteTokens
			}
		}),

		goai.WithOnStepFinish(func(step goai.StepResult) {
			if step.FinishReason == provider.FinishToolCalls {
				return // intermediate step — more tool calls follow
			}
			var output any
			if step.Text != "" {
				_ = json.Unmarshal([]byte(step.Text), &output)
			}
			end(output)
		}),

		goai.WithOnToolCall(func(info goai.ToolCallInfo) {
			if traceID == "" {
				return
			}

			start := info.StartTime
			endTime := info.StartTime.Add(info.Duration)

			var inputVal any = info.Input
			if len(info.Input) > 0 {
				var parsed any
				if json.Unmarshal(info.Input, &parsed) == nil {
					inputVal = parsed
				}
			}
			outputVal := any(info.Output)
			if info.OutputObject != nil {
				outputVal = info.OutputObject
			}

			level := ""
			statusMsg := ""
			if info.Error != nil {
				level = levelError
				statusMsg = info.Error.Error()
			}

			mu.Lock()
			obs = append(obs, ingestionEvent{
				// ingestionEvent.ID is the deduplication key for the ingestion envelope;
				// spanBody.ID is the observation ID used for parent-child relationships.
				// Langfuse requires them to be distinct.
				ID:        newID(),
				Type:      eventSpan,
				Timestamp: formatTime(start),
				Body: spanBody{
					ID:                  newID(),
					TraceID:             traceID,
					ParentObservationID: agentSpanID,
					Name:                info.ToolName,
					StartTime:           formatTime(start),
					EndTime:             formatTime(endTime),
					Input:               inputVal,
					Output:              outputVal,
					Version:             cfg.Version,
					Metadata: map[string]any{
						"tool_call_id": info.ToolCallID,
						"step":         info.Step,
					},
					Level:         level,
					StatusMessage: statusMsg,
				},
			})
			if endTime.After(lastObsEnd) {
				lastObsEnd = endTime
			}
			mu.Unlock()
		}),
	}

	end = func(result any) {
		if traceID == "" {
			return
		}

		if gen != nil {
			gen.output = result
			obs = append(obs, gen.toEvent(traceID, agentSpanID, cfg))
			gen = nil
		}

		now := time.Now()

		traceMeta := cfg.Metadata
		if cfg.Environment != "" {
			traceMeta = mergeMeta(traceMeta, map[string]any{"environment": cfg.Environment})
		}

		// Build the complete batch in order:
		// 1. trace-create
		// 2. agent span-create (parent of all observations)
		// 3. all buffered observations (generations + tool spans)
		batch := make([]ingestionEvent, 0, 2+len(obs))
		batch = append(batch,
			ingestionEvent{
				ID:        newID(),
				Type:      eventTrace,
				Timestamp: formatTime(agentStart),
				Body: traceBody{
					ID:        traceID,
					Name:      cfg.TraceName,
					UserID:    cfg.UserID,
					SessionID: cfg.SessionID,
					Tags:      cfg.Tags,
					Metadata:  traceMeta,
					Release:   cfg.Release,
					Version:   cfg.Version,
					Input:     traceInput,
					Output:    result,
				},
			},
			ingestionEvent{
				ID:        newID(),
				Type:      eventSpan,
				Timestamp: formatTime(agentStart),
				Body: spanBody{
					ID:        agentSpanID,
					TraceID:   traceID,
					Name:      cfg.TraceName,
					StartTime: formatTime(agentStart),
					EndTime:   formatTime(now),
					Input:     traceInput,
					Output:    result,
					Version:   cfg.Version,
				},
			},
		)
		batch = append(batch, obs...)

		// Clear run state — the run is complete.
		traceID = ""
		agentSpanID = ""
		obs = nil

		lc.appendEvents(batch)
		if err := lc.flush(context.Background()); err != nil && cfg.OnFlushError != nil {
			cfg.OnFlushError(err)
		}
	}

	return opts
}

// --- Langfuse API body types -----------------------------------------------

type traceBody struct {
	ID        string   `json:"id"`
	Name      string   `json:"name,omitempty"`
	UserID    string   `json:"userId,omitempty"`
	SessionID string   `json:"sessionId,omitempty"`
	Tags      []string `json:"tags,omitempty"`
	Metadata  any      `json:"metadata,omitempty"`
	Release   string   `json:"release,omitempty"`
	Version   string   `json:"version,omitempty"`
	Input     any      `json:"input,omitempty"`
	Output    any      `json:"output,omitempty"`
}

type spanBody struct {
	ID                  string `json:"id"`
	TraceID             string `json:"traceId"`
	ParentObservationID string `json:"parentObservationId,omitempty"`
	Name                string `json:"name,omitempty"`
	StartTime           string `json:"startTime,omitempty"`
	EndTime             string `json:"endTime,omitempty"`
	Input               any    `json:"input,omitempty"`
	Output              any    `json:"output,omitempty"`
	Version             string `json:"version,omitempty"`
	Metadata            any    `json:"metadata,omitempty"`
	Level               string `json:"level,omitempty"`
	StatusMessage       string `json:"statusMessage,omitempty"`
}

type generationBody struct {
	ID                  string     `json:"id"`
	TraceID             string     `json:"traceId"`
	ParentObservationID string     `json:"parentObservationId,omitempty"`
	Name                string     `json:"name,omitempty"`
	StartTime           string     `json:"startTime,omitempty"`
	EndTime             string     `json:"endTime,omitempty"`
	Input               any        `json:"input,omitempty"`
	Output              any        `json:"output,omitempty"`
	Version             string     `json:"version,omitempty"`
	Metadata            any        `json:"metadata,omitempty"`
	Level               string     `json:"level,omitempty"`
	StatusMessage       string     `json:"statusMessage,omitempty"`
	Model               string     `json:"model,omitempty"`
	Usage               *usageBody `json:"usage,omitempty"`
	PromptName          string     `json:"promptName,omitempty"`
	PromptVersion       int        `json:"promptVersion,omitempty"`
}

type usageBody struct {
	Input  int    `json:"input,omitempty"`
	Output int    `json:"output,omitempty"`
	Total  int    `json:"total,omitempty"`
	Unit   string `json:"unit,omitempty"`
}

const (
	eventTrace      = "trace-create"
	eventSpan       = "span-create"
	eventGeneration = "generation-create"
	levelWarning    = "WARNING"
	levelError      = "ERROR"
	unitTokens      = "TOKENS"
)

// pendingGen accumulates data for a single LLM generation step across
// OnRequest → OnResponse → lazy finalisation.
type pendingGen struct {
	id        string
	name      string
	model     string
	startTime time.Time
	input     any
	metadata  map[string]any

	// populated by OnResponse
	latency   time.Duration
	level     string
	statusMsg string
	usage     usageBody

	// populated lazily (next OnRequest or end)
	output any
}

func (g *pendingGen) toEvent(traceID, parentID string, cfg Config) ingestionEvent {
	endTime := g.startTime.Add(g.latency)
	body := generationBody{
		ID:                  g.id,
		TraceID:             traceID,
		ParentObservationID: parentID,
		Name:                g.name,
		Model:               g.model,
		StartTime:           formatTime(g.startTime),
		EndTime:             formatTime(endTime),
		Input:               g.input,
		Output:              g.output,
		Version:             cfg.Version,
		PromptName:          cfg.PromptName,
		PromptVersion:       cfg.PromptVersion,
		Metadata:            g.metadata,
		Level:               g.level,
		StatusMessage:       g.statusMsg,
	}
	if g.usage.Input > 0 || g.usage.Output > 0 || g.usage.Total > 0 {
		body.Usage = &g.usage
	}
	return ingestionEvent{
		ID:        newID(),
		Type:      eventGeneration,
		Timestamp: formatTime(g.startTime),
		Body:      body,
	}
}

func mergeMeta(base any, extra map[string]any) map[string]any {
	result := make(map[string]any, len(extra))
	if m, ok := base.(map[string]any); ok {
		for k, v := range m {
			result[k] = v
		}
	}
	for k, v := range extra {
		result[k] = v
	}
	return result
}

func toolCallMap(p provider.Part) map[string]any {
	tc := map[string]any{"id": p.ToolCallID, "name": p.ToolName}
	if len(p.ToolInput) > 0 {
		var parsed any
		if json.Unmarshal(p.ToolInput, &parsed) == nil {
			tc["input"] = parsed
		} else {
			tc["input"] = string(p.ToolInput)
		}
	}
	return tc
}

func messagesToInput(msgs []provider.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		entry := map[string]any{"role": string(m.Role)}
		var textParts []string
		var toolCalls []map[string]any
		for _, p := range m.Content {
			switch p.Type {
			case provider.PartText:
				if p.Text != "" {
					textParts = append(textParts, p.Text)
				}
			case provider.PartToolCall:
				toolCalls = append(toolCalls, toolCallMap(p))
			case provider.PartToolResult:
				entry["tool_call_id"] = p.ToolCallID
				if p.ToolOutput != "" {
					var parsed any
					if json.Unmarshal([]byte(p.ToolOutput), &parsed) == nil {
						entry["content"] = parsed
					} else {
						entry["content"] = p.ToolOutput
					}
				}
			}
		}
		if len(toolCalls) > 0 {
			entry["tool_calls"] = toolCalls
		} else if len(textParts) > 0 {
			if _, set := entry["content"]; !set {
				if len(textParts) == 1 {
					entry["content"] = textParts[0]
				} else {
					entry["content"] = textParts
				}
			}
		}
		result = append(result, entry)
	}
	return result
}

func lastAssistantOutput(m provider.Message) any {
	var toolCalls []map[string]any
	var textParts []string
	for _, p := range m.Content {
		switch p.Type {
		case provider.PartToolCall:
			toolCalls = append(toolCalls, toolCallMap(p))
		case provider.PartText:
			if p.Text != "" {
				textParts = append(textParts, p.Text)
			}
		}
	}
	if len(toolCalls) > 0 {
		return toolCalls
	}
	if len(textParts) == 1 {
		return textParts[0]
	}
	return textParts
}
