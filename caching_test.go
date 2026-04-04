package goai

import (
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestApplyCaching_SystemMessages(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleSystem,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "You are helpful."},
				{Type: provider.PartText, Text: "Be concise."},
			},
		},
		{
			Role:    provider.RoleUser,
			Content: []provider.Part{{Type: provider.PartText, Text: "Hello"}},
		},
	}

	result := applyCaching(msgs)

	// First system part should not be marked.
	if result[0].Content[0].CacheControl != "" {
		t.Errorf("first part CacheControl = %q, want empty", result[0].Content[0].CacheControl)
	}
	// Last system part should be marked.
	if result[0].Content[1].CacheControl != cacheControlEphemeral {
		t.Errorf("last part CacheControl = %q, want ephemeral", result[0].Content[1].CacheControl)
	}
	// User message should not be marked.
	if result[1].Content[0].CacheControl != "" {
		t.Errorf("user CacheControl = %q, want empty", result[1].Content[0].CacheControl)
	}
}

func TestApplyCaching_NoSystemMessages(t *testing.T) {
	msgs := []provider.Message{
		{
			Role:    provider.RoleUser,
			Content: []provider.Part{{Type: provider.PartText, Text: "Hello"}},
		},
	}

	result := applyCaching(msgs)

	if result[0].Content[0].CacheControl != "" {
		t.Errorf("CacheControl = %q, want empty", result[0].Content[0].CacheControl)
	}
}

func TestApplyCaching_EmptyContent(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleSystem, Content: nil},
	}

	// Should not panic.
	result := applyCaching(msgs)
	if len(result[0].Content) != 0 {
		t.Error("expected empty content")
	}
}

func TestApplyCaching_MultipleSystemMessages(t *testing.T) {
	msgs := []provider.Message{
		{
			Role:    provider.RoleSystem,
			Content: []provider.Part{{Type: provider.PartText, Text: "First system"}},
		},
		{
			Role:    provider.RoleSystem,
			Content: []provider.Part{{Type: provider.PartText, Text: "Second system"}},
		},
	}

	result := applyCaching(msgs)

	if result[0].Content[0].CacheControl != cacheControlEphemeral {
		t.Errorf("first system CacheControl = %q", result[0].Content[0].CacheControl)
	}
	if result[1].Content[0].CacheControl != cacheControlEphemeral {
		t.Errorf("second system CacheControl = %q", result[1].Content[0].CacheControl)
	}
}
