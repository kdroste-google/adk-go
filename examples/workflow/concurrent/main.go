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

package main

import (
	"context"
	"log"
	"os"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/workflow"
)

func main() {
	ctx := context.Background()

	nodeConfig := workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	}

	initNode := workflow.NewFunctionNode("init", func(ctx agent.CallbackContext, _ string) (string, error) {
		ctx.State().Set("a", "init")
		return "init done", nil
	}, nodeConfig)

	workerA := workflow.NewFunctionNode("workerA", func(ctx agent.CallbackContext, _ string) (string, error) {
		i := 100
		for i > 0 {
			v, err := ctx.State().Get("a")
			log.Printf("WorkerA: v, err= %v, %v", v, err)
			time.Sleep(100 * time.Millisecond)
			i--
		}
		return "WorkerA done", nil
	}, nodeConfig)

	workerB := workflow.NewFunctionNode("workerB", func(ctx agent.CallbackContext, _ string) (string, error) {
		i := 100
		for i > 0 {
			if i == 50 {
				ctx.State().Set("a", "workerB")
			}
			v, err := ctx.State().Get("a")
			log.Printf("WorkerB: v, err= %v, %v", v, err)
			time.Sleep(57 * time.Millisecond)
			i--
		}
		return "WorkerB done", nil
	}, nodeConfig)

	finalNode := workflow.NewJoinNode("finalNode")

	// // 1. Define functions for nodes
	// // The first node will receive the user message as input (string).
	// upperFn := func(ctx agent.CallbackContext, input string) (string, error) {
	// 	if input == "" {
	// 		ctx.State().Set("input", "NONE!!")
	// 		return "No input received", nil
	// 	}
	// 	ctx.State().Set("input", input)

	// 	return strings.ToUpper(input), nil
	// }

	// suffixFn := func(ctx agent.CallbackContext, input string) (string, error) {
	// 	v, err := ctx.State().Get("input")
	// 	info := ""
	// 	if err == nil {
	// 		info = fmt.Sprintf("'input' found: %+v", v)
	// 	} else {
	// 		info = "no 'input' found"
	// 	}

	// 	return input + " IS AWESOME! " + info, nil
	// }

	// 2. Create Nodes

	// nodeA := workflow.NewFunctionNode("upper", upperFn, nodeConfig)
	// nodeB := workflow.NewFunctionNode("suffix", suffixFn, nodeConfig)

	// 3. Define flow (Edges)
	eb := workflow.NewEdgeBuilder()
	eb.Add(workflow.Start, initNode)
	eb.Add(initNode, workerA)
	eb.Add(initNode, workerB)
	eb.AddFanIn(finalNode, workerA, workerB)

	// 4. Create Workflow Agent
	myWorkflow, err := workflowagent.New(workflowagent.Config{
		Name:        "simple_sequence_workflow",
		Description: "Converts string to uppercase and appends a suffix",
		Edges:       eb.Build(),
	})
	if err != nil {
		log.Fatalf("failed to create workflow: %v", err)
	}

	log.Printf("Successfully created root agent: %s", myWorkflow.Name())

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(myWorkflow),
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
