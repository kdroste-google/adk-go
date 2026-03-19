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

// Package agentengine handles command line parameters and execution logic for agentengine deployment.
package agentengine

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"google.golang.org/adk/cmd/adkgo/internal/deploy"
	"google.golang.org/adk/internal/cli/util"
)

type gCloudFlags struct {
	region      string
	projectName string
}

type agentEngineServiceFlags struct {
	name            string
	serverPort      int
	a2aAgentCardURL string
	a2a             bool // enable a2a or not
	api             bool // enable api or not
	webui           bool // enable webui or not
}

type localProxyFlags struct {
	port int
}

type buildFlags struct {
	tempDir             string
	execPath            string
	execFile            string
	dockerfileBuildPath string
}

type sourceFlags struct {
	srcBasePath    string
	entryPointPath string
}

type deployAgentEngineFlags struct {
	gcloud      gCloudFlags
	agentEngine agentEngineServiceFlags
	proxy       localProxyFlags
	build       buildFlags
	source      sourceFlags
}

var flags deployAgentEngineFlags

// agentEngineCmd represents the agentEngine command
var agentEngineCmd = &cobra.Command{
	Use:   "agentengine",
	Short: "Deploys the application to Agent Engine.",
	// TODO(kdroste): add description
	Long: `????????????????????????????????????????????????????????????????????????????
	Local proxy adding authentication is started. 
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return flags.deployOnagentEngine()
	},
}

// init creates flags and adds subcommand to parent
func init() {
	deploy.DeployCmd.AddCommand(agentEngineCmd)

	agentEngineCmd.PersistentFlags().StringVarP(&flags.gcloud.region, "region", "r", "", "GCP Region")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.gcloud.projectName, "project_name", "p", "", "GCP Project Name")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.agentEngine.name, "name", "s", "", "Agent Engine name")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.build.tempDir, "temp_dir", "t", "", "Temp dir for build, defaults to os.TempDir() if not specified")
	agentEngineCmd.PersistentFlags().IntVar(&flags.proxy.port, "proxy_port", 8081, "Local proxy port")
	agentEngineCmd.PersistentFlags().IntVar(&flags.agentEngine.serverPort, "server_port", 8080, "agentEngine server port")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.source.entryPointPath, "entry_point_path", "e", "", "Path to an entry point (go 'main')")
	agentEngineCmd.PersistentFlags().BoolVar(&flags.agentEngine.a2a, "a2a", true, "Enable A2A")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.agentEngine.a2aAgentCardURL, "a2a_agent_url", "a", "http://127.0.0.1:8081", "A2A agent card URL as advertised in the public agent card")
	agentEngineCmd.PersistentFlags().BoolVar(&flags.agentEngine.api, "api", true, "Enable API")
	agentEngineCmd.PersistentFlags().BoolVar(&flags.agentEngine.webui, "webui", true, "Enable Web UI")
}

// computeFlags uses command line arguments to create a full config
func (f *deployAgentEngineFlags) computeFlags() error {
	return util.LogStartStop("Computing flags & preparing temp",
		func(p util.Printer) error {
			absp, err := filepath.Abs(flags.source.entryPointPath)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.source.entryPointPath, err)
			}
			f.source.entryPointPath = absp

			if flags.build.tempDir == "" {
				flags.build.tempDir = os.TempDir()
			}
			absp, err = filepath.Abs(flags.build.tempDir)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.build.tempDir, err)
			}
			f.build.tempDir, err = os.MkdirTemp(absp, "agentEngine_"+time.Now().Format("20060102_150405__")+"*")
			if err != nil {
				return fmt.Errorf("cannot create a temporary sub directory in '%v': %w", absp, err)
			}
			p("Using temp dir:", f.build.tempDir)

			// come up with a executable name based on entry point path
			dir, file := path.Split(f.source.entryPointPath)
			f.source.srcBasePath = dir
			f.source.entryPointPath = file
			if f.build.execPath == "" {
				exec, err := util.StripExtension(f.source.entryPointPath, ".go")
				if err != nil {
					return fmt.Errorf("cannot strip '.go' extension from entry point path '%v': %w", f.source.entryPointPath, err)
				}
				f.build.execFile = exec
				f.build.execPath = path.Join(f.build.tempDir, exec)
			}
			f.build.dockerfileBuildPath = path.Join(f.build.tempDir, "Dockerfile")

			return nil
		})
}

func (f *deployAgentEngineFlags) cleanTemp() error {
	return util.LogStartStop("Cleaning temp",
		func(p util.Printer) error {
			p("Clean temp starting with", f.build.tempDir)
			// err := os.RemoveAll(f.build.tempDir)
			// if err != nil {
			// 	return fmt.Errorf("failed to clean temp directory %v: %w", f.build.tempDir, err)
			// }
			return nil
		})
}

// compileEntryPoint builds locally the server using flags and environment variables in order to be run in agentEngine containter
func (f *deployAgentEngineFlags) compileEntryPoint() error {
	return util.LogStartStop("Compiling server",
		func(p util.Printer) error {
			p("Using", f.source.entryPointPath, "as entry point")
			// for help on ldflags you can run go build -ldflags="--help" ./examples/quickstart/main.go
			//    -s    disable symbol table
			//    -w    disable DWARF generation
			//   using those flags reduces the size of an executable
			cmd := exec.Command("go", "build", "-ldflags", "-s -w", "-o", f.build.execPath, f.source.entryPointPath)

			cmd.Dir = f.source.srcBasePath
			// build using staticallly linked libs, for linux/amd64
			cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux", "GOARCH=amd64")
			return util.LogCommand(cmd, p)
		})
}

// prepareDockerfile creates a temporary Dockerfile which will be executed by agentEngine
func (f *deployAgentEngineFlags) prepareDockerfile() error {
	return util.LogStartStop("Preparing Dockerfile",
		func(p util.Printer) error {
			p("Writing:", f.build.dockerfileBuildPath)

			var b strings.Builder
			b.WriteString(`
FROM gcr.io/distroless/static-debian11

COPY ` + f.build.execFile + `  /app/` + f.build.execFile + `
EXPOSE ` + strconv.Itoa(flags.agentEngine.serverPort) + `
# Command to run the executable when the container starts

FROM gcr.io/distroless/static-debian11

COPY ` + f.build.execFile + `  /app/` + f.build.execFile + `
EXPOSE ` + strconv.Itoa(flags.agentEngine.serverPort) + `
# Command to run the executable when the container starts
CMD ["/app/` + f.build.execFile + `", "agentEngine", "-port", "` + strconv.Itoa(flags.agentEngine.serverPort) + `"`)

			if flags.agentEngine.api {
				b.WriteString(`, "api", "-webui_address", "127.0.0.1:` + strconv.Itoa(f.proxy.port) + `"`)
			}
			if flags.agentEngine.a2a {
				b.WriteString(`, "a2a", "--a2a_agent_url", "` + flags.agentEngine.a2aAgentCardURL + `"`)
			}
			if flags.agentEngine.webui {
				b.WriteString(`, "webui", "--api_server_address", "http://127.0.0.1:` + strconv.Itoa(f.proxy.port) + `/api"]
				`)
			}
			return os.WriteFile(f.build.dockerfileBuildPath, []byte(b.String()), 0o600)
		})
}

// gcloudDeployToAgentEngine invokes gcloud to deploy source on agentEngine
func (f *deployAgentEngineFlags) gcloudDeployToAgentEngine() error {
	return util.LogStartStop("Deploying to Agent Engine",
		func(p util.Printer) error {
			// params := []string{
			// 	"run", "deploy", f.agentEngine.name,
			// 	"--source", ".",
			// 	"--set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest",
			// 	"--region", f.gcloud.region,
			// 	"--project", f.gcloud.projectName,
			// 	"--ingress", "all",
			// 	"--no-allow-unauthenticated",
			// }

			// cmd := exec.Command("gcloud", params...)

			// cmd.Dir = f.build.tempDir
			// return util.LogCommand(cmd, p)
			return nil
		})
}

// runGcloudProxy invokes gcloud to create a proxy which will add authentication headers to requests
func (f *deployAgentEngineFlags) runGcloudProxy() error {
	return util.LogStartStop("Running local gcloud authenticating proxy",
		func(p util.Printer) error {
			targetWidth := 80

			p(strings.Repeat("-", targetWidth))
			p(util.CenterString("", targetWidth))
			p(util.CenterString("Running ADK Web UI on http://127.0.0.1:"+strconv.Itoa(f.proxy.port)+"/ui/    <-- open this", targetWidth))
			p(util.CenterString("ADK REST API on http://127.0.0.1:"+strconv.Itoa(f.proxy.port)+"/api/         ", targetWidth))
			p(util.CenterString("", targetWidth))
			p(util.CenterString("Press Ctrl-C to stop", targetWidth))
			p(util.CenterString("", targetWidth))
			p(strings.Repeat("-", targetWidth))

			cmd := exec.Command("gcloud", "run", "services", "proxy", f.agentEngine.name, "--project", f.gcloud.projectName, "--port", strconv.Itoa(f.proxy.port), "--region", f.gcloud.region)
			return util.LogCommand(cmd, p)
		})
}

// deployOnagentEngine executes the sequence of actions preparing and deploying the agent to agentEngine. Then runs authenticating proxy to newly deployed service
func (f *deployAgentEngineFlags) deployOnagentEngine() error {
	fmt.Println(flags)

	err := f.computeFlags()
	if err != nil {
		return err
	}
	err = f.compileEntryPoint()
	if err != nil {
		return err
	}
	err = f.prepareDockerfile()
	if err != nil {
		return err
	}
	// err = f.gcloudDeployToAgentEngine()
	// if err != nil {
	// 	return err
	// }
	// err = f.cleanTemp()
	// if err != nil {
	// 	return err
	// }
	// err = f.runGcloudProxy()
	// if err != nil {
	// 	return err
	// }

	return nil
}

// ctx := context.Background()
// parent := "projects/kdroste-adk-2025-12/locations/us-central1"

// client, err := aiplatform.NewReasoningEngineRESTClient(ctx, option.WithEndpoint("https://us-central1-aiplatform.googleapis.com"))
// if err != nil {
// 	log.Fatalf("cannot create ReasoningEngineClient: %v", err)
// }
// defer client.Close()

// listRE := client.ListReasoningEngines(ctx, &aiplatformpb.ListReasoningEnginesRequest{
// 	Parent: parent,
// })
// if listRE == nil {
// 	log.Fatalf("client.ListReasoningEngines() returned nil")
// }

// for o := range listRE.All() {
// 	if o == nil {
// 		log.Printf("client.ListReasoningEngines() returned nil object")
// 		continue
// 	}
// 	log.Printf("ListRE: %+v", o.DisplayName)
// }

// dockerContent, err := os.ReadFile("/usr/local/google/home/kdroste/Projects/agentEngine/dockerImage/simpleService/simpleServiceDockerImage.tar")
// if err != nil {
// 	log.Fatalf("ioutil.ReadFile() failed: %v", err)
// }
// _ = dockerContent

// sourceContent, err := os.ReadFile("/usr/local/google/home/kdroste/Projects/agentEngine/dockerImage/custom06/res/a.tgz")
// // sourceContent, err := os.ReadFile("/usr/local/google/home/kdroste/Projects/agentEngine/dockerImage/custom02/res/a.tgz")
// // sourceContent, err := os.ReadFile("/usr/local/google/home/kdroste/Projects/agentEngine/dockerImage/custom01/res/a.tar")
// if err != nil {
// 	log.Fatalf("ioutil.ReadFile() failed to read source code: %v", err)
// }
// _ = sourceContent

// dateTimeString := time.Now().Format(time.RFC3339)

// req2 := &aiplatformpb.CreateReasoningEngineRequest{
// 	Parent: parent,
// 	ReasoningEngine: &aiplatformpb.ReasoningEngine{
// 		// Name:        "projects/example-project/locations/us-central1/reasoningEngines/my-engine",
// 		DisplayName: "Simple GO echo: " + dateTimeString,
// 		Description: "An engine with all fields populated",
// 		Spec: &aiplatformpb.ReasoningEngineSpec{
// 			DeploymentSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_{
// 				SourceCodeSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec{
// 					Source: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource_{
// 						InlineSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource{
// 							SourceArchive: sourceContent,
// 						},
// 					},
// 					// Source: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_DeveloperConnectSource_{
// 					// 	DeveloperConnectSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_DeveloperConnectSource{
// 					// 		Config: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_DeveloperConnectConfig{
// 					// 			GitRepositoryLink: "projects/p/locations/l/connections/c/gitRepositoryLink/g",
// 					// 			Dir:               "src",
// 					// 			Revision:          "main",
// 					// 		},
// 					// 	},
// 					// },

// 					LanguageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec_{
// 						ImageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec{},
// 					},
// 					// LanguageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_PythonSpec_{
// 					// 	PythonSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_PythonSpec{
// 					// 		Version:          "3.11",
// 					// 		EntrypointModule: "app.main",
// 					// 		EntrypointObject: "agent",
// 					// 		RequirementsFile: "requirements.txt",
// 					// 	},
// 					// },
// 				},
// 			},
// 			AgentFramework: "google-adk",
// 			DeploymentSpec: &aiplatformpb.ReasoningEngineSpec_DeploymentSpec{
// 				Env: []*aiplatformpb.EnvVar{
// 					{Name: "GOOGLE_CLOUD_REGION", Value: "us-central1"},
// 					{Name: "NUM_WORKERS", Value: "1"},
// 					{Name: "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", Value: "true"},
// 					{Name: "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", Value: "true"},
// 				},
// 			},
// 			// ServiceAccount: protoimpl.X.String("service-account@example.iam.gserviceaccount.com"),
// 			// PackageSpec: &aiplatformpb.ReasoningEngineSpec_PackageSpec{
// 			// 	PickleObjectGcsUri:    "gs://bucket/object.pkl",
// 			// 	DependencyFilesGcsUri: "gs://bucket/deps.tar.gz",
// 			// 	RequirementsGcsUri:    "gs://bucket/requirements.txt",
// 			// 	PythonVersion:         "3.11",
// 			// },
// 			// DeploymentSpec: &aiplatformpb.ReasoningEngineSpec_DeploymentSpec{
// 			// 	Env: []*aiplatformpb.EnvVar{
// 			// 		{Name: "ENV_VAR", Value: "ENV_VAR_VALUE"},
// 			// 	},
// 			// },
// 		},
// 	},
// }

// // req := &aiplatformpb.CreateReasoningEngineRequest{
// // 	Parent: "projects/kdroste-adk-2025-12/locations/us-central1",
// // 	ReasoningEngine: &aiplatformpb.ReasoningEngine{
// // 		DisplayName: "testDisplayName",
// // 		Description: "testDescription",
// // 		Name:        "",
// // 		ContextSpec: &aiplatformpb.ReasoningEngineContextSpec{
// // 			MemoryBankConfig: &aiplatformpb.ReasoningEngineContextSpec_MemoryBankConfig{
// // 				GenerationConfig: &aiplatformpb.ReasoningEngineContextSpec_MemoryBankConfig_GenerationConfig{
// // 					Model: "",
// // 				},
// // 				SimilaritySearchConfig: &aiplatformpb.ReasoningEngineContextSpec_MemoryBankConfig_SimilaritySearchConfig{
// // 					EmbeddingModel: "",
// // 				},
// // 				TtlConfig: &aiplatformpb.ReasoningEngineContextSpec_MemoryBankConfig_TtlConfig{
// // 					Ttl: &aiplatformpb.ReasoningEngineContextSpec_MemoryBankConfig_TtlConfig_DefaultTtl{
// // 						DefaultTtl: durationpb.New(time.Hour),
// // 					},
// // 				},
// // 			},
// // 		},

// // 		Spec: &aiplatformpb.ReasoningEngineSpec{
// // 			ServiceAccount: "",
// // 			DeploymentSpec: &aiplatformpb.ReasoningEngineSpec_DeploymentSpec{
// // 				Env:                []*aiplatformpb.EnvVar{&aiplatformpb.EnvVar{Name: "", Value: ""}},
// // 				SecretEnv:          []*aiplatformpb.SecretEnvVar{&aiplatformpb.SecretEnvVar{Name: "", SecretRef: &aiplatformpb.SecretRef{Secret: "", Version: ""}}},
// // 				PscInterfaceConfig: &aiplatformpb.PscInterfaceConfig{},
// // 			},
// // 			DeploymentSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_{
// // 				SourceCodeSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec{
// // 					LanguageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec_{
// // 						ImageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec{
// // 							BuildArgs: map[string]string{},
// // 						},
// // 					},
// // 					Source: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource_{
// // 						InlineSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource{
// // 							SourceArchive: dockerContent,
// // 						},
// // 					},
// // 				},
// // 			},
// // 			AgentFramework: "custom",
// // 			PackageSpec: &aiplatformpb.ReasoningEngineSpec_PackageSpec{
// // 				PickleObjectGcsUri:    "gs://kdroste-agent-engine-bucket/agent_engine/agent_engine.pkl",
// // 				DependencyFilesGcsUri: "gs://kdroste-agent-engine-bucket/agent_engine/dependencies.tar.gz",
// // 				RequirementsGcsUri:    "gs://kdroste-agent-engine-bucket/agent_engine/requirements.txt",
// // 				PythonVersion:         "3.13",
// // 			},
// // 			// DeploymentSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_{
// // 			// 	SourceCodeSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec{
// // 			// 		Source: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource_{
// // 			// 			InlineSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource{
// // 			// 				SourceArchive: dockerContent,
// // 			// 			},
// // 			// 		},
// // 			// 		LanguageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec_{
// // 			// 			ImageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec{
// // 			// 				BuildArgs: map[string]string{},
// // 			// 			},
// // 			// 		},
// // 			// 	},
// // 			// },
// // 		},
// // 	},
// // }

// // req := &aiplatformpb.CreateReasoningEngineRequest{
// // 	Parent: "projects/kdroste-adk-2025-12/locations/global",
// // 	ReasoningEngine: &aiplatformpb.ReasoningEngine{
// // 		Name:        "testName",
// // 		DisplayName: "testDisplayName",
// // 		Description: "testDescription",

// // 		Spec: &aiplatformpb.ReasoningEngineSpec{
// // 			DeploymentSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_{
// // 				SourceCodeSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec{
// // 					Source: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource_{
// // 						InlineSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource{
// // 							SourceArchive: dockerContent,
// // 						},
// // 					},
// // 					LanguageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec_{
// // 						ImageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec{
// // 							BuildArgs: map[string]string{"aaa": "bbb"},
// // 						},
// // 					},
// // 				},
// // 			},
// // 			// DeploymentSource: &aiplatformpb.ReasoningEngineSpec_ContainerSpec{
// // 			// 	ContainerSpec: &aiplatformpb.ContainerSpec{
// // 			// 		ImageUri: "testImageUri",
// // 			// 	},
// // 			// },
// // 		},
// // 	},
// // }
// log.Printf("Will create Reasoning Engine")
// op, err := client.CreateReasoningEngine(ctx, req2)
// if err != nil {
// 	log.Fatalf("client.CreateReasoningEngine() failed: %v", err)
// }
// log.Printf("client.CreateReasoningEngine returned: %+v", op)
// log.Printf("client.CreateReasoningEngine returned details: %+v", op.Name())

// var re *aiplatformpb.ReasoningEngine

// // Poll the Long Running Operation (LRO) for completion
// for !op.Done() {
// 	log.Printf("Waiting for operation to complete...")
// 	time.Sleep(10 * time.Second)

// 	re, err = op.Poll(ctx)
// 	//re, err = op.Poll(ctx)
// 	if err != nil {
// 		log.Fatalf("Failed to get operation status: %v", err)
// 	}
// }
// log.Printf("op.Poll() returned: %+v", re)

// // lroName := res.Name()
// // op, err := client.GetOperation(ctx, &longrunningpb.GetOperationRequest{Name: lroName})
// // log.Printf("client.GetOperation() returned: %+v, %+v", op, err)
// // if err != nil {
// // 	log.Fatalf("client.GetOperation() failed: %v", err)
// // }
// // log.Printf("client.GetOperation() returned: %+v", op)

// // resEng, err := res.Wait(ctx)
// // if err != nil {
// // 	log.Fatalf("res.Wait() failed: %+v", err)
// // }

// // log.Printf("res.Wait() returned: %+v", resEng)
