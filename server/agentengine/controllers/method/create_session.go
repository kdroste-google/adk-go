package method

import (
	"context"

	"google.golang.org/adk/server/agentengine/controllers"
	"google.golang.org/adk/server/agentengine/internal/models"
	"google.golang.org/adk/session"
)

type createSessionHandler struct {
	sessionservice session.Service
}

type createSessionData struct {
	UserID string `json:"userId"`
	LastUpdateTime float64 `json:"lastUpdateTime"`
	AppName string `json:"appName"`
	Id string `json:"id"`
	State map[string]any `json:"state"`
	Events []any `json:"events"`
}

type createSessionResult struct {
	Data createSessionData `json:"data"`
}

// handle implements [controllers.MethodHandler].
func (c *createSessionHandler) handle(context.Context, *models.Query) (any, error) {
	//data: {"output": {"userId": "u_1234","lastUpdateTime": 1775651267.219728,"appName": "app","id": "5696152341405761536","state": {},"events": []}}
	result:= createSessionResult{
		Data: createSessionData{
			UserID: "u_1234",
			LastUpdateTime: 1775651267.219728,
			AppName: "app",
			Id: "5696152341405761536",
			State: map[string]any{},
			Events: []any{},
		},
	}
	return result, nil
}

// metadata implements [controllers.MethodHandler].
func (c *createSessionHandler) metadata() controllers.Metadata {
	return controllers.Metadata{
		Description: "Create a new session",
		Parameters: {
			Properties:  { { "user_id", { "string"}}},	
			Required []string {"user_id"},
			Type: "object" ,
		},
		Name: "create_session",
		APIMode: "",

	}
	
}

// name implements [controllers.MethodHandler].
func (c *createSessionHandler) name() string {
	return "create_session"
}

func NewCreateSessionHandler(sessionservice session.Service) *createSessionHandler {
	return &createSessionHandler{sessionservice: sessionservice}
}

var _ controllers.MethodHandler = (*createSessionHandler)(nil)
