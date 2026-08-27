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

// Command client runs a local LLM agent whose issue_refund tool executes
// *locally on the client*, but whose confirmation is answered *remotely by the
// server* over A2A.
//
// This is the second half of a paired example (see ../server). It simulates a
// situation where the human user is not the one confirming a sensitive action.
// Instead, when the local tool asks for confirmation, the client forwards the
// proposed action to a remote approval agent, and feeds that remote decision
// back as the confirmation response — the server impersonates the user.
//
// The scenario this program drives:
//
//  1. The user asks the local agent to issue a refund.
//  2. The local agent calls its local issue_refund tool, which requires
//     confirmation, so the run suspends with an "adk_request_confirmation"
//     function call.
//  3. Instead of asking a human, the client sends the proposed action to the
//     remote approval agent over A2A and reads its APPROVE/DENY decision.
//  4. The client feeds that decision back as the confirmation response (as if
//     the user had answered), resuming the run.
//  5. The local issue_refund tool executes on the client.
//
// Run the server first (see ../server), then:
//
//	GOOGLE_API_KEY=... go run ./examples/a2aconfirmation/client \
//	    -server http://127.0.0.1:8080
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

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
	appName          = "a2a-confirmation-client"
	userID           = "user"
	mainSessionID    = "session-1"
	approvalSession  = "approval-session"
	approverAppName  = "a2a-confirmation-approver"
	approveDecision  = "APPROVE"
	denyDecision     = "DENY"
	mainAgentName    = "refund_agent"
	approverAgentNm  = "approval_agent"
	defaultUserQuery = "Please issue a refund of $42 for order A-100."
)

// refundArgs are the inputs of the local issue_refund tool.
type refundArgs struct {
	OrderID string  `json:"order_id"`
	Amount  float64 `json:"amount"`
}

// refundResult is the output of the local issue_refund tool.
type refundResult struct {
	Status    string `json:"status"`
	Reference string `json:"reference"`
}

// issueRefund is the tool handler. It runs *locally on the client*, and only
// after the confirmation has been granted. RequireConfirmation on the tool
// short circuits the first invocation to raise a confirmation request; this
// handler runs once that request has been approved.
func issueRefund(ctx agent.Context, args refundArgs) (refundResult, error) {
	log.Printf("[client] executing issue_refund LOCALLY: $%.2f for order %q", args.Amount, args.OrderID)
	return refundResult{
		Status:    "refunded",
		Reference: "RF-" + args.OrderID,
	}, nil
}

func main() {
	serverURL := flag.String("server", "http://127.0.0.1:8080", "base URL of the A2A approval server (see ../server)")
	prompt := flag.String("prompt", defaultUserQuery, "message to send to the agent")
	flag.Parse()

	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create a model: %v", err)
	}

	// The local tool that runs on the client and requires confirmation.
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

	// The local agent that owns and executes the tool.
	localAgent, err := llmagent.New(llmagent.Config{
		Name:        mainAgentName,
		Model:       model,
		Description: "Handles refunds by running the local issue_refund tool.",
		Instruction: "You process refunds. When the user asks for a refund, call the issue_refund tool with the order ID and amount.",
		Tools:       []tool.Tool{refundTool},
	})
	if err != nil {
		log.Fatalf("Failed to create the local agent: %v", err)
	}

	mainRunner, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             localAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("Failed to create the runner: %v", err)
	}

	// The approver: a remote agent that reaches the server over A2A. The client
	// uses it to obtain a confirmation decision on the user's behalf.
	approverAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              approverAgentNm,
		Description:       "Remote approval authority that decides whether an action is allowed.",
		AgentCardProvider: remoteagent.NewAgentCardProvider(*serverURL),
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

	if err := runConversation(ctx, mainRunner, approverRunner, *prompt); err != nil {
		log.Fatalf("Conversation failed: %v", err)
	}
}

// runConversation sends the initial prompt and keeps the conversation going
// until there are no more pending confirmation requests. Whenever the local
// tool asks for confirmation, the decision is fetched from the remote server.
func runConversation(ctx context.Context, mainRunner, approverRunner *runner.Runner, prompt string) error {
	msg := genai.NewContentFromText(prompt, genai.RoleUser)
	for turn := 1; ; turn++ {
		fmt.Printf("\n=== Turn %d ===\n", turn)

		var pending *genai.FunctionCall
		for event, err := range mainRunner.Run(ctx, userID, mainSessionID, msg, agent.RunConfig{}) {
			if err != nil {
				return err
			}
			printEvent("client", event)
			if fc := confirmationCall(event); fc != nil {
				pending = fc
			}
		}

		if pending == nil {
			// No confirmation was requested this turn: the conversation is done.
			return nil
		}

		// Ask the server to confirm on the user's behalf.
		approved, err := askServerForApproval(ctx, approverRunner, pending)
		if err != nil {
			return fmt.Errorf("failed to get remote approval: %w", err)
		}
		fmt.Printf("\n[client] remote decision: confirmed=%v; resuming local tool\n", approved)
		msg = confirmationResponse(pending.ID, approved)
	}
}

// askServerForApproval forwards the proposed action to the remote approval
// agent and returns its decision. The server, acting as the user, answers
// APPROVE or DENY.
func askServerForApproval(ctx context.Context, approverRunner *runner.Runner, confirmationCall *genai.FunctionCall) (bool, error) {
	original, err := toolconfirmation.OriginalCallFrom(confirmationCall)
	if err != nil {
		return false, fmt.Errorf("failed to read original call: %w", err)
	}

	question := fmt.Sprintf(
		"An agent wants to run the tool %q with arguments %v. As the account owner, do you approve? Answer APPROVE or DENY.",
		original.Name, original.Args,
	)
	fmt.Printf("\n[client] asking server to confirm: %s\n", question)

	var reply strings.Builder
	msg := genai.NewContentFromText(question, genai.RoleUser)
	for event, err := range approverRunner.Run(ctx, userID, approvalSession, msg, agent.RunConfig{}) {
		if err != nil {
			return false, err
		}
		printEvent("server", event)
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
func printEvent(side string, event *session.Event) {
	if event.ErrorMessage != "" {
		fmt.Printf("[%s:%s] error: %s\n", side, event.Author, event.ErrorMessage)
	}
	if event.Content == nil {
		return
	}
	for _, part := range event.Content.Parts {
		switch {
		case part.Text != "":
			fmt.Printf("[%s:%s] %s\n", side, event.Author, part.Text)
		case part.FunctionCall != nil:
			fmt.Printf("[%s:%s] -> call %s(%v) id=%s\n", side, event.Author, part.FunctionCall.Name, part.FunctionCall.Args, part.FunctionCall.ID)
		case part.FunctionResponse != nil:
			fmt.Printf("[%s:%s] <- response %s(%v) id=%s\n", side, event.Author, part.FunctionResponse.Name, part.FunctionResponse.Response, part.FunctionResponse.ID)
		}
	}
}
