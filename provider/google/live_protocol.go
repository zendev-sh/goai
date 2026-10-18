package google

import "encoding/json"

// Wire types of the Gemini Live API (BidiGenerateContent over WebSocket). They
// stay private: callers only see provider.RealtimeModel and its events.

// liveClientMessage carries exactly one of its fields per WebSocket message.
type liveClientMessage struct {
	Setup         map[string]any     `json:"setup,omitempty"`
	RealtimeInput *liveRealtimeInput `json:"realtimeInput,omitempty"`
	ToolResponse  *liveToolResponse  `json:"toolResponse,omitempty"`
}

// liveBlob is inline media; encoding/json writes Data as standard base64.
type liveBlob struct {
	MimeType string `json:"mimeType"`
	Data     []byte `json:"data"`
}

type liveRealtimeInput struct {
	Audio         *liveBlob `json:"audio,omitempty"`
	Text          string    `json:"text,omitempty"`
	ActivityStart *struct{} `json:"activityStart,omitempty"`
	ActivityEnd   *struct{} `json:"activityEnd,omitempty"`
}

type liveToolResponse struct {
	FunctionResponses []liveFunctionResponse `json:"functionResponses"`
}

type liveFunctionResponse struct {
	ID         string          `json:"id,omitempty"`
	Name       string          `json:"name"`
	Response   json.RawMessage `json:"response"`
	Scheduling string          `json:"scheduling,omitempty"`
}

// liveServerMessage holds usageMetadata/voiceActivity plus one message type.
type liveServerMessage struct {
	SetupComplete           *struct{}                    `json:"setupComplete"`
	ServerContent           *liveServerContent           `json:"serverContent"`
	ToolCall                *liveToolCall                `json:"toolCall"`
	ToolCallCancellation    *liveToolCallCancellation    `json:"toolCallCancellation"`
	GoAway                  *liveGoAway                  `json:"goAway"`
	SessionResumptionUpdate *liveSessionResumptionUpdate `json:"sessionResumptionUpdate"`
	UsageMetadata           *liveUsageMetadata           `json:"usageMetadata"`
	VoiceActivity           *liveVoiceActivity           `json:"voiceActivity"`
}

type liveServerContent struct {
	ModelTurn           *liveContent       `json:"modelTurn"`
	TurnComplete        bool               `json:"turnComplete"`
	GenerationComplete  bool               `json:"generationComplete"`
	Interrupted         bool               `json:"interrupted"`
	InputTranscription  *liveTranscription `json:"inputTranscription"`
	OutputTranscription *liveTranscription `json:"outputTranscription"`
}

type liveContent struct {
	Role  string     `json:"role"`
	Parts []livePart `json:"parts"`
}

type livePart struct {
	Text       string    `json:"text"`
	Thought    bool      `json:"thought"`
	InlineData *liveBlob `json:"inlineData"`
}

type liveTranscription struct {
	Text string `json:"text"`
}

type liveToolCall struct {
	FunctionCalls []liveFunctionCall `json:"functionCalls"`
}

type liveFunctionCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type liveToolCallCancellation struct {
	IDs []string `json:"ids"`
}

type liveGoAway struct {
	TimeLeft string `json:"timeLeft"`
}

type liveSessionResumptionUpdate struct {
	NewHandle string `json:"newHandle"`
	Resumable bool   `json:"resumable"`
}

type liveUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ResponseTokenCount      int `json:"responseTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
}

type liveVoiceActivity struct {
	Type string `json:"type"`
}
