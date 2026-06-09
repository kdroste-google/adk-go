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

package workflow

import (
	"encoding/json"
	"fmt"
	"iter"
	"log"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// FunctionNode wraps a custom function.
type FunctionNode struct {
	BaseNode
	fn func(ctx agent.CallbackContext, input any) (any, error)
}

// NewFunctionNode creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNode[IN, OUT any](name string, fn func(ctx agent.CallbackContext, input IN) (OUT, error), cfg NodeConfig) *FunctionNode {
	defer debugExit(debugEnter("NewFunctionNode"))
	return newFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, nil, nil, cfg)
}

// NewFunctionNodeWithSchema creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNodeWithSchema[IN, OUT any](name string, fn func(ctx agent.CallbackContext, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*FunctionNode, error) {
	defer debugExit(debugEnter("NewFunctionNodeWithSchema"))
	var ischema *jsonschema.Resolved
	var err error
	if inputSchema != nil {
		ischema, err = inputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving input schema: %w", err)
		}
	}

	var oschema *jsonschema.Resolved
	if outputSchema != nil {
		oschema, err = outputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving output schema: %w", err)
		}
	}

	return newFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, ischema, oschema, cfg), nil
}

// newFunctionNodeWithResolvedSchemas is an internal constructor that consumes already resolved schemas.
func newFunctionNodeWithResolvedSchemas[IN, OUT any](name string, fn func(ctx agent.CallbackContext, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Resolved, cfg NodeConfig) *FunctionNode {
	defer debugExit(debugEnter("newFunctionNodeWithResolvedSchemas"))
	wrappedFn := func(ctx agent.CallbackContext, input any) (any, error) {
		defer debugExit(debugEnter("FunctionNode.wrappedFn"))
		var output OUT
		var err error
		if input == nil {
			var zero IN
			output, err = fn(ctx, zero)
		} else {
			typedInput, ok := input.(IN)
			if !ok {
				// Fallback to the json-like input types that cannot be converted by the standard type assertion.
				// E.g. tool nodes return map[string]any as input and user may define a struct as the target type.
				typedInput, err = typeutil.ConvertToWithJSONSchema[any, IN](input, inputSchema)
				if err != nil {
					return nil, fmt.Errorf("new function node: invalid input type, expected %T: %w", new(IN), err)
				}
			}
			output, err = fn(ctx, typedInput)
		}

		if err != nil {
			return output, err
		}

		if outputSchema != nil {
			validateErr := outputSchema.Validate(output)
			if validateErr != nil {
				return nil, fmt.Errorf("function node %s: validation failed for output %T: %w", name, new(OUT), validateErr)
			}
		}

		return output, nil
	}

	return &FunctionNode{
		BaseNode: NewBaseNodeWithSchemas(name, "", cfg, inputSchema, outputSchema),
		fn:       wrappedFn,
	}
}

// Run executes the function node with the given input and returns an iterator over events.
func (n *FunctionNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	defer debugExit(debugEnter("FunctionNode.Run"))
	return func(yield func(*session.Event, error) bool) {
		defer debugExit(debugEnter("FunctionNode.Run.iter"))

		actions := &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
		callbackCtx := agent.NewCallbackContext(ctx, actions)
		output, err := n.fn(callbackCtx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Output = output
		event.Actions = *actions

		log.Printf("%+v\n", actions.StateDelta)

		if s, ok := output.(string); ok {
			event.Content = &genai.Content{
				Parts: []*genai.Part{{Text: s}},
			}
		}
		dumpEvent("FunctionNode "+n.Name(), event)
		yield(event, nil)
	}
}

func dumpEvent(msg string, event *session.Event) {
	b, err := json.Marshal(event)
	if err != nil {
		log.Printf("error marshalling event: %v", err)
		return
	}
	s := fmt.Sprintf("event from %s: %s", msg, string(b))
	debugPrint(s)
}
