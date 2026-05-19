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
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

type PluginManager interface {
	RunBeforeModelCallback(cctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)
	RunAfterModelCallback(cctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)
	RunOnModelErrorCallback(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)
	RunBeforeToolCallback(pureCtx PureContext, adkSpan AdkSpan, t tool.Tool, args map[string]any) (map[string]any, error)
	RunAfterToolCallback(pureCtx PureContext, adkSpan AdkSpan, t tool.Tool, args, result map[string]any, err error) (map[string]any, error)
	RunOnToolErrorCallback(pureCtx PureContext, adkSpan AdkSpan, t tool.Tool, args map[string]any, err error) (map[string]any, error)
}
