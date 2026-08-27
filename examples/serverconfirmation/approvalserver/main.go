// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Command approvalserver runs an A2A (Agent-To-Agent) server that acts as an
// approval authority: it decides, on the user's behalf, whether a proposed
// action should be allowed.
//
// This is the second of two servers in this example (the first is
// ../agentserver). The agent server hosts a tool that requires confirmation
// before it runs; when that confirmation is needed the agent server does not
// ask a human, it asks this server through a remote agent. This server answers
// APPROVE/DENY, effectively impersonating the user and delivering the
// confirmation decision remotely.
//
// Run it with:
//
//	GOOGLE_API_KEY=... go run ./examples/serverconfirmation/approvalserver
//
// then, in another terminal, start the agent server (see ../agentserver).
package main

import (
	"context"
	"flag"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

// newApprovalAgent creates the LLM agent exposed by this server. It plays the
// role of the account owner and decides whether proposed actions are allowed.
func newApprovalAgent(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create a model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "approval_agent",
		Model:       model,
		Description: "Approves or denies sensitive actions on behalf of the account owner.",
		Instruction: "You are the account owner's approval authority. You will be shown a proposed action that another agent wants to run. " +
			"Approve refunds of $100 or less; deny anything larger or anything suspicious. " +
			"Answer with exactly one word first, either APPROVE or DENY, optionally followed by a short reason.",
	})
	if err != nil {
		log.Fatalf("Failed to create the approval agent: %v", err)
	}
	return a
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8081", "host:port to serve the A2A approval server on")
	flag.Parse()

	ctx := context.Background()

	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("Failed to bind to %q: %v", *addr, err)
	}
	baseURL := &url.URL{Scheme: "http", Host: listener.Addr().String()}

	approvalAgent := newApprovalAgent(ctx)

	// The path where the JSON-RPC A2A endpoint is served.
	const agentPath = "/invoke"
	agentCard := &a2a.AgentCard{
		Name:        approvalAgent.Name(),
		Description: approvalAgent.Description(),
		SupportedInterfaces: []*a2a.AgentInterface{
			{
				URL:             baseURL.JoinPath(agentPath).String(),
				ProtocolBinding: a2a.TransportProtocolJSONRPC,
				ProtocolVersion: a2a.Version,
			},
		},
		Version:            "1.0.0",
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Skills:             adka2a.BuildAgentSkills(approvalAgent),
		Capabilities:       a2a.AgentCapabilities{Streaming: true},
	}

	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        approvalAgent.Name(),
			Agent:          approvalAgent,
			SessionService: session.InMemoryService(),
		},
	})
	requestHandler := a2asrv.NewHandler(executor)
	mux.Handle(agentPath, a2asrv.NewJSONRPCHandler(requestHandler))

	log.Printf("[approvalserver] A2A approval agent listening on %s", baseURL.String())
	log.Printf("[approvalserver] agent card: %s", baseURL.JoinPath(a2asrv.WellKnownAgentCardPath).String())
	log.Printf("[approvalserver] point the agent server at: %s", baseURL.String())

	if err := http.Serve(listener, mux); err != nil {
		log.Fatalf("A2A approval server stopped: %v", err)
	}
}
