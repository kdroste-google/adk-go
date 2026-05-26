package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session/vertexai"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/loadmemorytool"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	location := "us-central1"
	projectId := "kdroste-adk-2025-12"
	reasoningEngine := "5008482722461515776" // mb03

	// model, err := gemini.NewModel(ctx, "gemini-3.1-flash-lite", &genai.ClientConfig{
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Location: location,
		Project:  projectId,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	// Initialise Vertex AI Engine Session Service
	sessionService, err := vertexai.NewSessionService(ctx, vertexai.VertexAIServiceConfig{
		ProjectID:       projectId,
		Location:        location,
		ReasoningEngine: reasoningEngine,
	})
	if err != nil {
		log.Fatalf("failed to create session service")
	}

	// Create new agent
	adkAgent, err := llmagent.New(llmagent.Config{
		Name:        "agent",
		Model:       model,
		Description: "Description",
		Instruction: "You are a helpful assistant..",
		Tools: []tool.Tool{
			loadmemorytool.New(),
		},
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	// Configure launcher
	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(adkAgent),
		SessionService: sessionService,
		MemoryService:  memory.InMemoryService(),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}

}
