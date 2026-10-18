package provider

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func TestErrFileUploadUnsupported(t *testing.T) {
	if !errors.Is(ErrFileUploadUnsupported, ErrFileUploadUnsupported) {
		t.Error("sentinel error should match itself")
	}
}

func TestFileUploadStruct(t *testing.T) {
	u := FileUpload{
		Filename:  "test.pdf",
		MediaType: "application/pdf",
		Purpose:   "assistants",
	}
	if u.Filename != "test.pdf" {
		t.Errorf("Filename = %q", u.Filename)
	}
	if u.MediaType != "application/pdf" {
		t.Errorf("MediaType = %q", u.MediaType)
	}
	if u.Purpose != "assistants" {
		t.Errorf("Purpose = %q", u.Purpose)
	}
}

func TestRemoteFileRefStruct(t *testing.T) {
	now := time.Now()
	ref := &RemoteFileRef{
		Provider:  "openai",
		ID:        "file-abc123",
		URI:       "",
		Filename:  "doc.pdf",
		MediaType: "application/pdf",
		ExpiresAt: now,
		Data:      []byte("pdf-content"),
	}
	if ref.Provider != "openai" {
		t.Errorf("Provider = %q", ref.Provider)
	}
	if ref.ID != "file-abc123" {
		t.Errorf("ID = %q", ref.ID)
	}
	if ref.Filename != "doc.pdf" {
		t.Errorf("Filename = %q", ref.Filename)
	}
	if ref.MediaType != "application/pdf" {
		t.Errorf("MediaType = %q", ref.MediaType)
	}
	if !ref.ExpiresAt.Equal(now) {
		t.Error("ExpiresAt mismatch")
	}
	if string(ref.Data) != "pdf-content" {
		t.Errorf("Data = %q", string(ref.Data))
	}
}

func TestPartRemoteRef(t *testing.T) {
	ref := &RemoteFileRef{ID: "file-xyz"}
	p := Part{
		Type:      PartFile,
		RemoteRef: ref,
		Filename:  "doc.pdf",
		MediaType: "application/pdf",
	}
	if p.RemoteRef != ref {
		t.Error("RemoteRef should be set")
	}
	if p.RemoteRef.ID != "file-xyz" {
		t.Errorf("RemoteRef.ID = %q", p.RemoteRef.ID)
	}
}

func TestModelCapabilitiesFileUpload(t *testing.T) {
	caps := ModelCapabilities{
		FileUpload: true,
	}
	if !caps.FileUpload {
		t.Error("FileUpload should be true")
	}

	caps2 := ModelCapabilities{}
	if caps2.FileUpload {
		t.Error("FileUpload should default to false")
	}
}

func TestFileUploadCapableModelInterface(t *testing.T) {
	var m interface{ FileUploader() FileUploader }
	m = &mockFileUploadModel{}
	if m == nil {
		t.Error("mock should implement FileUploadCapableModel")
	}
}

type mockFileUploadModel struct{}

func (m *mockFileUploadModel) FileUploader() FileUploader {
	return &mockFileUploader{}
}

type mockFileUploader struct{}

func (u *mockFileUploader) UploadFile(_ context.Context, _ FileUpload) (*RemoteFileRef, error) {
	return &RemoteFileRef{ID: "mock-file"}, nil
}

func (u *mockFileUploader) DeleteFile(_ context.Context, _ RemoteFileRef) error {
	return nil
}

func TestMockFileUploader(t *testing.T) {
	m := &mockFileUploadModel{}
	uploader := m.FileUploader()
	ref, err := uploader.UploadFile(t.Context(), FileUpload{Filename: "test.txt"})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.ID != "mock-file" {
		t.Errorf("ref.ID = %q", ref.ID)
	}
	if err := uploader.DeleteFile(t.Context(), *ref); err != nil {
		t.Fatalf("DeleteFile error: %v", err)
	}
}

func TestRealtimeSentinelErrors(t *testing.T) {
	for _, err := range []error{
		ErrRealtimeUnsupported,
		ErrRealtimeSessionClosed,
		ErrUnsupportedAudioFormat,
		ErrRealtimeBackpressure,
		ErrInvalidRealtimeConfig,
	} {
		if !errors.Is(err, err) {
			t.Errorf("sentinel error %v should match itself", err)
		}
	}
}

func TestAudioFormatValidate(t *testing.T) {
	tests := []struct {
		name    string
		format  AudioFormat
		wantErr bool
	}{
		{name: "zero value allowed"},
		{name: "pcm16", format: AudioFormat{Encoding: AudioPCM16, SampleRate: 16000, Channels: 1}},
		{name: "missing encoding", format: AudioFormat{SampleRate: 16000, Channels: 1}, wantErr: true},
		{name: "invalid rate", format: AudioFormat{Encoding: AudioPCM16, Channels: 1}, wantErr: true},
		{name: "invalid channels", format: AudioFormat{Encoding: AudioPCM16, SampleRate: 16000}, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.format.Validate()
			if tt.wantErr && !errors.Is(err, ErrUnsupportedAudioFormat) {
				t.Fatalf("Validate() error = %v, want ErrUnsupportedAudioFormat", err)
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
		})
	}
}

func TestRealtimeParamsValidate(t *testing.T) {
	valid := RealtimeParams{
		InputAudio:    AudioFormat{Encoding: AudioPCM16, SampleRate: 16000, Channels: 1},
		OutputAudio:   AudioFormat{Encoding: AudioPCM16, SampleRate: 24000, Channels: 1},
		TurnDetection: TurnDetection{Mode: TurnModeProviderVAD},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid params rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*RealtimeParams)
		alsoIs error
	}{
		{
			name: "bad input audio",
			mutate: func(p *RealtimeParams) {
				p.InputAudio.SampleRate = 0
			},
			alsoIs: ErrUnsupportedAudioFormat,
		},
		{
			name: "unknown turn mode",
			mutate: func(p *RealtimeParams) {
				p.TurnDetection.Mode = "unknown"
			},
		},
		{
			name: "negative max tokens",
			mutate: func(p *RealtimeParams) {
				p.MaxOutputTokens = -1
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := valid
			tt.mutate(&params)
			err := params.Validate()
			if !errors.Is(err, ErrInvalidRealtimeConfig) {
				t.Fatalf("Validate() error = %v, want ErrInvalidRealtimeConfig", err)
			}
			if tt.alsoIs != nil && !errors.Is(err, tt.alsoIs) {
				t.Fatalf("Validate() error = %v, want wrapped %v", err, tt.alsoIs)
			}
		})
	}
}

func TestRealtimeStructuredTypes(t *testing.T) {
	input := json.RawMessage(`{"month":"last"}`)
	output := json.RawMessage(`{"report_id":"r-1"}`)
	event := RealtimeEvent{
		Type:       RealtimeToolCallEvent,
		SessionID:  "s-1",
		ResponseID: "r-1",
		ToolCall: &RealtimeToolCall{
			ID:    "call-1",
			Name:  "sales_report",
			Input: input,
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5},
	}
	if event.ToolCall == nil || event.ToolCall.ID != "call-1" || event.ToolCall.Name != "sales_report" {
		t.Fatalf("unexpected tool event: %+v", event)
	}
	if string(event.ToolCall.Input) != string(input) {
		t.Fatalf("tool input = %s", event.ToolCall.Input)
	}

	clientEvent := RealtimeClientEvent{
		Type: RealtimeToolResultEvent,
		ToolResult: &RealtimeToolResult{
			ToolCallID: "call-1",
			Output:     output,
		},
	}
	if clientEvent.ToolResult == nil || string(clientEvent.ToolResult.Output) != string(output) {
		t.Fatalf("unexpected client tool result: %+v", clientEvent)
	}
}

func TestRealtimeCapabilitiesClone(t *testing.T) {
	original := RealtimeCapabilities{
		DirectClientTransports: []string{"websocket", "webrtc"},
		SupportedInputAudio: []AudioFormat{{
			Encoding: AudioPCM16, SampleRate: 16000, Channels: 1,
		}},
		SupportedOutputAudio: []AudioFormat{{
			Encoding: AudioPCM16, SampleRate: 24000, Channels: 1,
		}},
	}
	clone := original.Clone()
	clone.DirectClientTransports[0] = "changed"
	clone.SupportedInputAudio[0].SampleRate = 1
	clone.SupportedOutputAudio[0].SampleRate = 2

	if original.DirectClientTransports[0] != "websocket" {
		t.Fatalf("transport slice aliased: %+v", original.DirectClientTransports)
	}
	if original.SupportedInputAudio[0].SampleRate != 16000 {
		t.Fatalf("input audio slice aliased: %+v", original.SupportedInputAudio)
	}
	if original.SupportedOutputAudio[0].SampleRate != 24000 {
		t.Fatalf("output audio slice aliased: %+v", original.SupportedOutputAudio)
	}
}
