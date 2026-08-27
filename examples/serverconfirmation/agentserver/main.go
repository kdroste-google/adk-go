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

// Command agentserver runs an HTTP server that hosts an LLM agent whose
// issue_refund tool requires confirmation before it executes. This is the first
// of two servers in this example (the second is ../approvalserver).
//
// When a request comes in the agent tries to run its issue_refund tool, but the
// tool is configured with RequireConfirmation, so the run suspends with an
// "adk_request_confirmation" function call instead of executing. Rather than ask
// a human, this server forwards the proposed action to the remote approval agent
// (../approvalserver) over A2A, reads its APPROVE/DENY decision, and feeds that
// decision back as the confirmation response — resuming the run. Only then does
// the issue_refund tool execute, here on the agent server.
//
// The scenario this server drives, per HTTP request:
//
//  1. A caller sends a prompt asking for a refund (curl the /run endpoint).
//  2. The local agent calls its issue_refund tool, which requires confirmation,
//     so the run suspends with an "adk_request_confirmation" function call.
//  3. Instead of asking a human, this server sends the proposed action to the
//     remote approval agent over A2A and reads its APPROVE/DENY decision.
//  4. This server feeds that decision back as the confirmation response (as if
//     the user had answered), resuming the run.
//  5. The issue_refund tool executes on this server.
//
// Run the approval server first (see ../approvalserver), then:
//
//	GOOGLE_API_KEY=... go run ./examples/serverconfirmation/agentserver \
//	    -approver http://127.0.0.1:8081
//
// and, in a third terminal, drive it with curl:
//
//	curl "http://127.0.0.1:8080/run"
//	curl "http://127.0.0.1:8080/run?q=Please+refund+\$500+for+order+B-9."
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync/atomic"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

const (
	appName          = "server-confirmation-agent"
	approverAppName  = "server-confirmation-approver"
	userID           = "user"
	approveDecision  = "APPROVE"
	denyDecision     = "DENY"
	defaultUserQuery = "Please issue a refund of $42 for order A-100."
)

// refundArgs are the inputs of the issue_refund tool.
type refundArgs struct {
	OrderID string  `json:"order_id"`
	Amount  float64 `json:"amount"`
}

// refundResult is the output of the issue_refund tool.
type refundResult struct {
	Status    string `json:"status"`
	Reference string `json:"reference"`
}

// issueRefund is the tool handler. It runs on this server, and only after the
// confirmation has been granted. RequireConfirmation on the tool short circuits
// the first invocation to raise a confirmation request; this handler runs once
// that request has been approved.
func issueRefund(ctx agent.Context, args refundArgs) (refundResult, error) {
	log.Printf("[agentserver] executing issue_refund: $%.2f for order %q", args.Amount, args.OrderID)
	return refundResult{
		Status:    "refunded",
		Reference: "RF-" + args.OrderID,
	}, nil
}

// server holds the runners shared across HTTP requests. The mainRunner drives
// the refund agent whose tool requires confirmation; the approverRunner reaches
// the remote approval agent that supplies the decision.
type server struct {
	mainRunner     *runner.Runner
	approverRunner *runner.Runner
	seq            atomic.Int64
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8080", "host:port to serve this agent server on")
	approverURL := flag.String("approver", "http://127.0.0.1:8081", "base URL of the remote A2A approval server (see ../approvalserver)")
	flag.Parse()

	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create a model: %v", err)
	}

	// The tool that runs on this server and requires confirmation.
	refundTool, err := functiontool.New(
		functiontool.Config{
			Name:                "issue_refund",
			Description:         "Issue a monetary refund for an order. Requires confirmation before it runs.",
			RequireConfirmation: true,
		},
		issueRefund,
	)
	if err != nil {
		log.Fatalf("Failed to create the refund tool: %v", err)
	}

	// The agent that owns and executes the tool.
	refundAgent, err := llmagent.New(llmagent.Config{
		Name:        "refund_agent",
		Model:       model,
		Description: "Handles refunds by running the issue_refund tool.",
		Instruction: "You process refunds. When the user asks for a refund, call the issue_refund tool with the order ID and amount.",
		Tools:       []tool.Tool{refundTool},
	})
	if err != nil {
		log.Fatalf("Failed to create the refund agent: %v", err)
	}

	mainRunner, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             refundAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("Failed to create the runner: %v", err)
	}

	// The approver: a remote agent that reaches the approval server over A2A.
	// This server uses it to obtain a confirmation decision on the user's behalf.
	approverAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              "approval_agent",
		Description:       "Remote approval authority that decides whether an action is allowed.",
		AgentCardProvider: remoteagent.NewAgentCardProvider(*approverURL),
	})
	if err != nil {
		log.Fatalf("Failed to create the approver agent: %v", err)
	}
	approverRunner, err := runner.New(runner.Config{
		AppName:           approverAppName,
		Agent:             approverAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("Failed to create the approver runner: %v", err)
	}

	srv := &server{mainRunner: mainRunner, approverRunner: approverRunner}

	mux := http.NewServeMux()
	mux.HandleFunc("/run", srv.handleRun)

	log.Printf("[agentserver] refund agent listening on http://%s", *addr)
	log.Printf("[agentserver] remote approver: %s", *approverURL)
	log.Printf("[agentserver] try: curl \"http://%s/run\"", *addr)

	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("Agent server stopped: %v", err)
	}
}

// handleRun runs one refund conversation for the incoming request. It streams a
// transcript of the flow back to the caller so the confirmation round-trip is
// visible.
func (s *server) handleRun(w http.ResponseWriter, r *http.Request) {
	prompt := r.URL.Query().Get("q")
	if prompt == "" {
		prompt = defaultUserQuery
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")

	n := s.seq.Add(1)
	mainSessionID := fmt.Sprintf("main-%d", n)
	approvalSessionID := fmt.Sprintf("approval-%d", n)

	emit := func(format string, args ...any) {
		line := fmt.Sprintf(format, args...)
		_, _ = fmt.Fprintln(w, line)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		log.Print(line)
	}

	emit("[agentserver] request: %q", prompt)
	if err := s.runConversation(r.Context(), emit, mainSessionID, approvalSessionID, prompt); err != nil {
		emit("[agentserver] error: %v", err)
	}
}

// runConversation sends the initial prompt to the refund agent and keeps the
// conversation going until there are no more pending confirmation requests.
// Whenever the tool asks for confirmation, the decision is fetched from the
// remote approval server.
func (s *server) runConversation(ctx context.Context, emit func(string, ...any), mainSessionID, approvalSessionID, prompt string) error {
	msg := genai.NewContentFromText(prompt, genai.RoleUser)
	for turn := 1; ; turn++ {
		emit("\n=== Turn %d ===", turn)

		var pending *genai.FunctionCall
		for event, err := range s.mainRunner.Run(ctx, userID, mainSessionID, msg, agent.RunConfig{}) {
			if err != nil {
				return err
			}
			printEvent(emit, "agentserver", event)
			if fc := confirmationCall(event); fc != nil {
				pending = fc
			}
		}

		if pending == nil {
			// No confirmation was requested this turn: the conversation is done.
			return nil
		}

		// Ask the remote approval server to confirm on the user's behalf.
		approved, err := s.askRemoteForApproval(ctx, emit, approvalSessionID, pending)
		if err != nil {
			return fmt.Errorf("failed to get remote approval: %w", err)
		}
		emit("\n[agentserver] remote decision: confirmed=%v; resuming tool", approved)
		msg = confirmationResponse(pending.ID, approved)
	}
}

// askRemoteForApproval forwards the proposed action to the remote approval agent
// and returns its decision. The remote agent, acting as the user, answers
// APPROVE or DENY.
func (s *server) askRemoteForApproval(ctx context.Context, emit func(string, ...any), approvalSessionID string, call *genai.FunctionCall) (bool, error) {
	original, err := toolconfirmation.OriginalCallFrom(call)
	if err != nil {
		return false, fmt.Errorf("failed to read original call: %w", err)
	}

	question := fmt.Sprintf(
		"An agent wants to run the tool %q with arguments %v. As the account owner, do you approve? Answer APPROVE or DENY.",
		original.Name, original.Args,
	)
	emit("\n[agentserver] asking remote approver: %s", question)

	var reply strings.Builder
	msg := genai.NewContentFromText(question, genai.RoleUser)
	for event, err := range s.approverRunner.Run(ctx, userID, approvalSessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return false, err
		}
		printEvent(emit, "approver", event)
		if event.Content != nil {
			for _, part := range event.Content.Parts {
				reply.WriteString(part.Text)
			}
		}
	}

	decision := strings.ToUpper(reply.String())
	approved := strings.Contains(decision, approveDecision) && !strings.Contains(decision, denyDecision)
	return approved, nil
}

// confirmationCall returns the "adk_request_confirmation" function call in the
// event, or nil if the event does not carry one.
func confirmationCall(event *session.Event) *genai.FunctionCall {
	if event.Content == nil {
		return nil
	}
	for _, part := range event.Content.Parts {
		if fc := part.FunctionCall; fc != nil && fc.Name == toolconfirmation.FunctionCallName {
			return fc
		}
	}
	return nil
}

// confirmationResponse builds the FunctionResponse that answers a confirmation
// request. It is authored as a user message: the remote decision is relayed
// back as though the user had made it.
func confirmationResponse(callID string, approved bool) *genai.Content {
	return &genai.Content{
		Role: string(genai.RoleUser),
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				Name: toolconfirmation.FunctionCallName,
				ID:   callID,
				Response: map[string]any{
					"confirmed": approved,
				},
			},
		}},
	}
}

// printEvent renders the interesting parts of an event so the flow is visible.
func printEvent(emit func(string, ...any), side string, event *session.Event) {
	if event.ErrorMessage != "" {
		emit("[%s:%s] error: %s", side, event.Author, event.ErrorMessage)
	}
	if event.Content == nil {
		return
	}
	for _, part := range event.Content.Parts {
		switch {
		case part.Text != "":
			emit("[%s:%s] %s", side, event.Author, part.Text)
		case part.FunctionCall != nil:
			emit("[%s:%s] -> call %s(%v) id=%s", side, event.Author, part.FunctionCall.Name, part.FunctionCall.Args, part.FunctionCall.ID)
		case part.FunctionResponse != nil:
			emit("[%s:%s] <- response %s(%v) id=%s", side, event.Author, part.FunctionResponse.Name, part.FunctionResponse.Response, part.FunctionResponse.ID)
		}
	}
}
