// Copyright 2025 Google LLC
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

// Package agentengine provides a sublauncher that provides web interface as required by Agent Engine
package agentengine

import (
	"flag"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	weblauncher "google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/server/agentengine"
)

// agentEngineConfig contains parameters for launching ADK Agent Engine server
type agentEngineConfig struct {
	pathPrefix string
}

type agentEngineLauncher struct {
	flags  *flag.FlagSet // flags are used to parse command-line arguments
	config *agentEngineConfig
	router *mux.Router
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewLauncher() weblauncher.Sublauncher {
	config := &agentEngineConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.pathPrefix, "path_prefix", "/api", "ADK REST API path prefix. Default is '/api'.")

	return &agentEngineLauncher{
		config: config,
		flags:  fs,
	}
}

// CommandLineSyntax implements web.Sublauncher. Returns the command-line syntax for the agentEngine launcher.
func (a *agentEngineLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
}

// SimpleDescription implements web.Sublauncher
func (a *agentEngineLauncher) SimpleDescription() string {
	// TODO(kdroste) description
	return fmt.Sprintf("starts AgentEngine server which handles ??????????")
}

// UserMessage implements web.Sublauncher.
func (a *agentEngineLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	// TODO(kdroste) description
	printer(fmt.Sprintf("       agentEngine:  you can access this server ????????????????????????????????: %s", webUrl))
}

// SetupSubrouters adds the API router to the parent router.
func (a *agentEngineLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	// Create the ADK AgentEngine API handler
	apiHandler := agentengine.NewHandler(config, 60*time.Second)

	router.Methods("POST").
		PathPrefix(a.config.pathPrefix).
		Handler(http.StripPrefix(a.config.pathPrefix, apiHandler))

	return nil
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for A2A launcher.
func (a *agentEngineLauncher) Keyword() string {
	return "agentengine"
}

var _ weblauncher.Sublauncher = &agentEngineLauncher{}

// Parse parses the command-line arguments for the API launcher.
func (a *agentEngineLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse agent engine flags: %v", err)
	}
	p := a.config.pathPrefix
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	a.config.pathPrefix = strings.TrimSuffix(p, "/")

	restArgs := a.flags.Args()
	return restArgs, nil
}

// func (a *agentEngineLauncher) handleApiReasoningEngine(w http.ResponseWriter, r *http.Request) {
// 	fmt.Fprintf(w, "Hello from handleApiReasoningEngine: Not implemented")
// }

// type payload struct {
// 	ClassMethod string `json:"class_method"`
// 	Input       any    `json:"input"`
// }

// func logReq(req *http.Request, body []byte) {
// 	rb, err := json.Marshal(struct {
// 		Method  string
// 		URL     string
// 		Headers any
// 		Body    any
// 	}{
// 		Method:  req.Method,
// 		URL:     req.URL.String(),
// 		Body:    string(body),
// 		Headers: req.Header})

// 	if err != nil {
// 		log.Printf("json.Marshal() failed: %v", err)
// 		return
// 	}
// 	log.Printf("request: %v\n", string(rb))
// }

// type FakeWriter struct{}

// // Header implements [http.ResponseWriter].
// func (f *FakeWriter) Header() http.Header {
// 	res := make(http.Header)
// 	res.Set("Content-Type", "application/json")
// 	return res
// }

// // WriteHeader implements [http.ResponseWriter].
// func (f *FakeWriter) WriteHeader(statusCode int) {
// 	log.Printf("WriteHeader(): statusCode= %v", statusCode)
// }

// var _ http.ResponseWriter = &FakeWriter{}

// func (f *FakeWriter) Write(p []byte) (n int, err error) {

// 	log.Printf("Write(): p= %v", p)
// 	s := string(p)
// 	log.Printf("Write(): s= %v", s)

// 	return len(p), nil
// }

// func (a *agentEngineLauncher) handleApiSteamReasoningEngine(w http.ResponseWriter, r *http.Request) {
// 	fmt.Fprintf(w, "Hello from handleApiSteamReasoningEngine")

// 	if r == nil {
// 		log.Printf("req is nil")
// 		return
// 	}
// 	if r.Body == nil {
// 		log.Printf("req.Body is nil")
// 		return
// 	}
// 	body, err := io.ReadAll(r.Body)
// 	if err != nil {
// 		log.Printf("io.ReadAll() failed: %v", err)
// 		return
// 	}

// 	log.Printf("Body string: %v", string(body))

// 	// {"class_method":"create_session","input":{"user_id":"u_123"}}
// 	var p payload
// 	err = json.Unmarshal(body, &p)
// 	if err != nil {
// 		log.Printf("json.Unmarshal() failed: %v", err)
// 		return
// 	}
// 	log.Printf("Payload: %+v", p)

// 	logReq(r, body)

// 	switch {
// 	case p.ClassMethod == "create_session":
// 		log.Printf("Hello from create_session")
// 		req, err := http.NewRequest("POST", "/api/apps/{app_name}/users/{user_id}/sessions", nil)
// 		if err != nil {
// 			log.Printf("http.NewRequest() failed: %v", err)
// 			return
// 		}

// 		fw := &FakeWriter{}
// 		a.router.ServeHTTP(fw, req)
// 		// a.router

// 		// crr := &session.CreateRequest{
// 		// 	AppName: "app",
// 		// 	UserID:  "user",
// 		// }
// 		// ressession, err := s.ss.Create(r.Context(), crr)
// 		// if err != nil {
// 		// 	log.Printf("s.ss.Create() failed: %v", err)
// 		// 	return
// 		// }
// 		// log.Printf("ressession: %+v", ressession)
// 		// resp := fmt.Sprintf(`{"session_id":"%v}`, ressession.Session.ID())
// 		// w.Write([]byte(resp))

// 	default:
// 		err = fmt.Errorf("unrecognized class method: %v", p.ClassMethod)
// 	}

// }

// // SetupSubrouters implements the web.Sublauncher interface.
// func (a *agentEngineLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
// 	router.HandleFunc("/api/reasoning_engine", a.handleApiReasoningEngine)
// 	router.HandleFunc("/api/stream_reasoning_engine", a.handleApiSteamReasoningEngine)

// 	// sublaunchers are hidden in a.router
// 	for _, l := range a.sublaunchers {
// 		if _, isActive := a.activeSublaunchers[l.Keyword()]; isActive {
// 			if err := l.SetupSubrouters(a.router, config); err != nil {
// 				return fmt.Errorf("%s subrouter setup failed: %v", l.Keyword(), err)
// 			}
// 		}
// 	}
// 	return nil
// }
