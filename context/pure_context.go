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
	"context"
	"fmt"
	"time"
)

type PureContext interface {
	context.Context
}

type pureContext struct {
	context.Context
}

func NewPureContext(ctx context.Context) PureContext {
	return &pureContext{Context: ctx}
}

// Deadline implements [context.Context].
func (p *pureContext) Deadline() (deadline time.Time, ok bool) {
	return p.Context.Deadline()
}

// Done implements [context.Context].
func (p *pureContext) Done() <-chan struct{} {
	return p.Context.Done()
}

// Err implements [context.Context].
func (p *pureContext) Err() error {
	return p.Context.Err()
}

// Value implements [context.Context].
func (p *pureContext) Value(key any) any {
	panic(fmt.Sprintf("pureContext.Value should not be used. Called with key: %v", key))
}

var _ context.Context = (PureContext)(nil)
