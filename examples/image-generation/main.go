//go:build ignore

// Example: image generation (GenerateImage and provider-defined image tool).
//
// This example demonstrates both approaches:
//  1. goai.GenerateImage() for direct image generation.
//  2. OpenAI ImageGeneration provider-defined tool inside GenerateText.
//
// Via OpenAI direct:
//
//	export OPENAI_API_KEY=...
//	go run examples/image-generation/main.go openai
//
// Via Azure OpenAI (requires image model on the same resource,
// and x-ms-oai-image-generation-deployment header):
//
//	export AZURE_IMAGE_API_KEY=... AZURE_IMAGE_RESOURCE_NAME=...
//	go run examples/image-generation/main.go azure
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/azure"
	"github.com/zendev-sh/goai/provider/openai"
)

func main() {
	auth := "azure"
	if len(os.Args) > 1 {
		auth = os.Args[1]
	}

	var model provider.LanguageModel
	switch auth {
	case "openai":
		model = openai.Chat("gpt-5",
			openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	case "azure":
		// Azure requires the image model deployment on the same resource.
		// The header tells Azure which deployment to use for image generation.
		// Use the image-specific resource (AZURE_IMAGE_*) which has both
		// the text model (gpt-5) and image model (gpt-image-1.5) deployed.
		model = azure.Chat("gpt-5",
			azure.WithAPIKey(os.Getenv("AZURE_IMAGE_API_KEY")),
			azure.WithEndpoint("https://"+os.Getenv("AZURE_IMAGE_RESOURCE_NAME")+".openai.azure.com"),
			azure.WithHeaders(map[string]string{
				"x-ms-oai-image-generation-deployment": "gpt-image-1.5",
			}),
		)
	default:
		log.Fatalf("Unknown: %s (use openai or azure)", auth)
	}

	def := openai.Tools.ImageGeneration(
		openai.WithImageQuality("low"),
		openai.WithImageSize("1024x1024"),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, model,
		goai.WithPrompt("Generate a logo for a Go SDK called GoAI. Modern, minimal, blue gradient."),
		goai.WithMaxOutputTokens(4096),
		goai.WithTools(goai.Tool{
			Name:                   def.Name,
			ProviderDefinedType:    def.ProviderDefinedType,
			ProviderDefinedOptions: def.ProviderDefinedOptions,
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	imgModel := openai.Image("gpt-image-1", openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	if auth == "azure" {
		imgModel = azure.Image("gpt-image-1",
			azure.WithAPIKey(os.Getenv("AZURE_IMAGE_API_KEY")),
			azure.WithEndpoint("https://"+os.Getenv("AZURE_IMAGE_RESOURCE_NAME")+".openai.azure.com"),
			azure.WithHeaders(map[string]string{
				"x-ms-oai-image-generation-deployment": "gpt-image-1.5",
			}),
		)
	}

	directImage, err := goai.GenerateImage(ctx, imgModel,
		goai.WithImagePrompt("Minimal blue gopher mascot logo for a Go SDK named GoAI"),
		goai.WithImageCount(1),
		goai.WithImageSize("1024x1024"),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Direct GenerateImage: %d image(s)\n", len(directImage.Images))

	fmt.Println("Text:", result.Text)
	fmt.Printf("Usage: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)

	if result.TotalUsage.OutputTokens > 0 {
		fmt.Println("Image generated successfully (embedded in response output)")
	}
}
