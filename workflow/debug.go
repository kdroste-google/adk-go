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
	"fmt"
	"os"
	"strings"
	"sync/atomic"
)

// debugEnabled reports whether function entry/exit tracing is on.
// It is enabled by setting the ADK_WORKFLOW_DEBUG environment variable
// to a non-empty, non-"0", non-"false" value before the program starts.
var debugEnabled = func() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("ADK_WORKFLOW_DEBUG")))
	return v != "" && v != "0" && v != "false"
}()

// debugDepth is a process-global indentation counter.
//
// Note: it is shared across goroutines, so under concurrency the rendered
// indentation may not perfectly nest. This is acceptable for the intended
// human-readable trace use case and avoids the overhead of per-goroutine
// tracking.
var debugDepth int64

const debugIndentUnit = "  " // two spaces per level

// debugEnter logs a function entry and returns the function name so it can
// be paired with debugExit via:
//
//	defer debugExit(debugEnter("MyFunc"))
//
// When tracing is disabled, both helpers are nearly free (a single atomic
// load via the package-level flag check) and produce no output.
func debugEnter(name string) string {
	if !debugEnabled {
		return name
	}
	depth := atomic.AddInt64(&debugDepth, 1) - 1
	fmt.Fprintf(os.Stderr, "%s-> %s\n", strings.Repeat(debugIndentUnit, int(depth)), name)
	return name
}

// debugExit logs a function exit. It is intended to be used together with
// debugEnter via a deferred call.
func debugExit(name string) {
	if !debugEnabled {
		return
	}
	depth := atomic.AddInt64(&debugDepth, -1)
	if depth < 0 {
		// Defensive: keep the counter non-negative if something gets out of sync.
		atomic.StoreInt64(&debugDepth, 0)
		depth = 0
	}
	fmt.Fprintf(os.Stderr, "%s<- %s\n", strings.Repeat(debugIndentUnit, int(depth)), name)
}
