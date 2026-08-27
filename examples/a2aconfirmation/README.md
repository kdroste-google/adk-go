# Local tool, remote confirmation over A2A

A paired example simulating a situation where a tool that requires
confirmation runs **locally on the client**, but the confirmation decision is
supplied **remotely by the server**, over the A2A (Agent-To-Agent) protocol.
The server answers on the user's behalf — it impersonates the user and provides
the confirmation.

There are two programs:

- [`server/`](./server) — an A2A server exposing an `approval_agent`. Given a
  proposed action it answers `APPROVE` or `DENY`, acting as the account owner.
- [`client/`](./client) — a local `refund_agent` with a local `issue_refund`
  tool created with `RequireConfirmation: true`. The tool executes on the
  client, but when it needs confirmation the client asks the server (through a
  remote agent) and relays the decision back as the confirmation.

## The scenario

```
user ─▶ refund_agent (client, local LLM)
          │  call issue_refund(...)  ── RequireConfirmation ──▶ run suspends
          ▼                                                       │
   adk_request_confirmation (local)  ◀───────────────────────────┘
          │
          │  client forwards the proposed action over A2A
          ▼
       approval_agent (server, LLM)  ──▶ "APPROVE"   (server impersonates user)
          │
          │  client relays decision as the confirmation response
          ▼
   FunctionResponse{confirmed: true}  ─▶ run resumes
          │
   issue_refund runs LOCALLY on the client ─▶ "refunded"
```

1. The user asks the local agent to issue a refund.
2. The local agent calls its local `issue_refund` tool, which requires
   confirmation, so the run suspends with an `adk_request_confirmation`
   function call.
3. Instead of asking a human, the client sends the proposed action to the
   remote `approval_agent` over A2A and reads its `APPROVE`/`DENY` decision.
4. The client feeds that decision back as the confirmation response (as if the
   user had answered), resuming the run.
5. The local `issue_refund` tool executes **on the client**.

The key point: the tool runs on the client, but the confirmation originates on
the server. The server stands in for the user.

## Running it

Both agents use Gemini, so set an API key (or configure Vertex AI):

```bash
export GOOGLE_API_KEY=...
```

Start the server in one terminal:

```bash
go run ./examples/a2aconfirmation/server
# [server] A2A approval agent listening on http://127.0.0.1:8080
```

Run the client in another terminal:

```bash
go run ./examples/a2aconfirmation/client -server http://127.0.0.1:8080
```

You will see the local tool request confirmation, the client ask the server,
the server respond `APPROVE`, and the refund run locally. The client terminal
logs `executing issue_refund LOCALLY ...` only after the remote approval — proof
that the tool ran on the client but was gated by the server's decision.

The default request is a $42 refund, which the server approves (its rule is to
approve refunds of $100 or less). Try a larger amount to see it denied:

```bash
go run ./examples/a2aconfirmation/client -prompt "Please refund $500 for order B-9."
```

### Flags

Server:

- `-addr` — host:port to serve on (default `127.0.0.1:8080`).

Client:

- `-server` — base URL of the A2A approval server (default `http://127.0.0.1:8080`).
- `-prompt` — the message to send (default asks for a $42 refund).
```
