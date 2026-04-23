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