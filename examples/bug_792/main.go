package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

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
	"google.golang.org/protobuf/types/known/structpb"
)

// toAnySlice converts any typed slice to []any by round-tripping through JSON.
// structpb.NewValue only accepts []any (not typed slices like []memory.Entry)
// and map[string]any (not arbitrary structs).
func toAnySlice[T any](in []T) []any {
	b, err := json.Marshal(in)
	if err != nil {
		panic(fmt.Errorf("toAnySlice marshal: %w", err))
	}
	var out []any
	if err := json.Unmarshal(b, &out); err != nil {
		panic(fmt.Errorf("toAnySlice unmarshal: %w", err))
	}
	return out
}

func main() {

	log.Printf("Time: %v", time.Now().Format("2006-01-02T15:04:05.000Z"))
	entries := []memory.Entry{
		{ID: "1",
			Content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Hi  there"},
				},
				Role: "user",
			},
			Author:    "user1",
			Timestamp: time.Now(),
		},
	}
	a, err := structpb.NewValue(toAnySlice(entries))
	if err != nil {
		log.Fatalf("failed to create NewValue: %v", err)
	}
	log.Println(a)

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
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(cc agent.CallbackContext) (*genai.Content, error) {
				v := []memory.Entry{}
				cc.State().Set("aaa", v)
				return nil, nil
			}},
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
