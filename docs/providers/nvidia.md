---
title: NVIDIA NIM
description: "NVIDIA NIM (NVIDIA Inference Microservices) provider for GoAI SDK"
---

# NVIDIA NIM

[NVIDIA NIM](https://docs.nvidia.com/nim/) provides easy-to-deploy AI microservices for inference. GoAI supports chat and embedding models via the OpenAI-compatible API.

## Installation

```bash
go get github.com/zendev-sh/goai@latest
```

## Chat

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/nvidia"
)

func main() {
	model := nvidia.Chat("nvidia/llama-3.1-nemotron-70b-instruct")

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithPrompt("What is the capital of France?"),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Text)
}
```

### Available Models

NVIDIA NIM offers many models. Common choices include:
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `nvidia/llama-3.3-70b-instruct`
- `nvidia/nemotron-mini-4b-instruct`

## Embeddings

```go
model := nvidia.Embedding("nvidia/nv-embedqa-e5-v5")

result, err := goai.Embed(context.Background(), model,
	goai.WithPrompt("Hello world"),
)
if err != nil {
	log.Fatal(err)
}
fmt.Printf("Dimensions: %d\n", len(result.Embedding))
```

### Available Embedding Models

- `nvidia/nv-embedqa-e5-v5`
- `nvidia/nv-embed-v1`

## Authentication

The provider automatically reads from:

| Environment Variable | Description |
| -------------------- | ------------ |
| `NVIDIA_API_KEY`     | Your NVIDIA NGC API key |
| `NVIDIA_BASE_URL`    | Override the default endpoint |

Or pass explicitly:

```go
model := nvidia.Chat("nvidia/llama-3.1-nemotron-70b-instruct",
	nvidia.WithAPIKey("nvapi-..."),
	nvidia.WithBaseURL("https://custom.endpoint.com/v1"),
)
```

## Options

| Option | Description |
| ------ | ----------- |
| `WithAPIKey(key)` | Set static API key |
| `WithTokenSource(ts)` | Dynamic auth (TokenSource) |
| `WithBaseURL(url)` | Override API endpoint |
| `WithHeaders(h)` | Custom HTTP headers |
| `WithHTTPClient(c)` | Custom HTTP client |

## Self-Hosted NIM

Deploy NIM containers locally or in your infrastructure:

```go
model := nvidia.Chat("meta/llama-3.1-70b-instruct",
	nvidia.WithBaseURL("http://localhost:8000/v1"),
)
```

See [NVIDIA NIM documentation](https://docs.nvidia.com/nim/) for deployment instructions.