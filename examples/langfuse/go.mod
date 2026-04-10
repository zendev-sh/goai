module github.com/zendev-sh/goai/examples/langfuse

go 1.25.0

require (
	github.com/zendev-sh/goai v0.0.0
	github.com/zendev-sh/goai/observability/langfuse v0.0.0
)

require github.com/google/uuid v1.6.0 // indirect

replace (
	github.com/zendev-sh/goai => ../..
	github.com/zendev-sh/goai/observability/langfuse => ../../observability/langfuse
)
