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

package context

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

type AdkSpan interface {
	PluginManager() PluginManager
	State() session.State
	AgentName() string
	FunctionCallID() string
}

type adkSpan struct {
	pluginManager  PluginManager
	state          session.State
	actions        session.EventActions
	agentName      string
	functionCallID string
}

func NewToolSpan() AdkSpan {
	return &adkSpan{}
}

func NewCallbackSpan() AdkSpan {
	return &adkSpan{}
}

func NewInvocationSpan(invocationCtx agent.InvocationContext) AdkSpan {
	return &adkSpan{}
}

var _ AdkSpan = (*adkSpan)(nil)

func (a *adkSpan) PluginManager() PluginManager {
	return a.pluginManager
}

func (a *adkSpan) State() session.State {
	return a.state
}

func (a *adkSpan) AgentName() string {
	return a.agentName
}

func (a *adkSpan) FunctionCallID() string {
	return a.functionCallID
}

func (a *adkSpan) Actions() *session.EventActions {
	return &a.actions
}

// type UniContext struct {
// }

// // Actions implements [tool.Context].
// func (u *UniContext) Actions() *session.EventActions {
// 	panic("unimplemented")
// }

// // FunctionCallID implements [tool.Context].
// func (u *UniContext) FunctionCallID() string {
// 	panic("unimplemented")
// }

// // RequestConfirmation implements [tool.Context].
// func (u *UniContext) RequestConfirmation(hint string, payload any) error {
// 	panic("unimplemented")
// }

// // SearchMemory implements [tool.Context].
// func (u *UniContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
// 	panic("unimplemented")
// }

// // ToolConfirmation implements [tool.Context].
// func (u *UniContext) ToolConfirmation() *toolconfirmation.ToolConfirmation {
// 	panic("unimplemented")
// }

// // AgentName implements [agent.CallbackContext].
// func (u *UniContext) AgentName() string {
// 	panic("unimplemented")
// }

// // AppName implements [agent.CallbackContext].
// func (u *UniContext) AppName() string {
// 	panic("unimplemented")
// }

// // Artifacts implements [agent.CallbackContext].
// func (u *UniContext) Artifacts() agent.Artifacts {
// 	panic("unimplemented")
// }

// // Branch implements [agent.CallbackContext].
// func (u *UniContext) Branch() string {
// 	panic("unimplemented")
// }

// // InvocationID implements [agent.CallbackContext].
// func (u *UniContext) InvocationID() string {
// 	panic("unimplemented")
// }

// // ReadonlyState implements [agent.CallbackContext].
// func (u *UniContext) ReadonlyState() session.ReadonlyState {
// 	panic("unimplemented")
// }

// // SessionID implements [agent.CallbackContext].
// func (u *UniContext) SessionID() string {
// 	panic("unimplemented")
// }

// // State implements [agent.CallbackContext].
// func (u *UniContext) State() session.State {
// 	panic("unimplemented")
// }

// // UserContent implements [agent.CallbackContext].
// func (u *UniContext) UserContent() *genai.Content {
// 	panic("unimplemented")
// }

// // UserID implements [agent.CallbackContext].
// func (u *UniContext) UserID() string {
// 	panic("unimplemented")
// }

// // Deadline implements [context.Context].
// func (u *UniContext) Deadline() (deadline time.Time, ok bool) {
// 	panic("unimplemented")
// }

// // Done implements [context.Context].
// func (u *UniContext) Done() <-chan struct{} {
// 	panic("unimplemented")
// }

// // Err implements [context.Context].
// func (u *UniContext) Err() error {
// 	panic("unimplemented")
// }

// // Value implements [context.Context].
// func (u *UniContext) Value(key any) any {
// 	panic("unimplemented")
// }

// var _ context.Context = (*UniContext)(nil)
// var _ agent.CallbackContext = (*UniContext)(nil)
// var _ tool.Context = (*UniContext)(nil)
