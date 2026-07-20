package provider

import (
	"context"
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