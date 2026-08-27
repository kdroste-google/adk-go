# Two servers: local tool, remote confirmation

Two cooperating servers where a tool that requires confirmation runs on the
**first** server, but the confirmation decision is supplied by a **second**,
remote agent. The second server answers on the user's behalf — it impersonates
the user and delivers the confirmation.

This is the two-server variant of [`../a2aconfirmation`](../a2aconfirmation),
where the tool holder is a client program. Here the tool holder is itself a
server.

There are two programs:

- [`agentserver/`](./agentserver) — **Server 1.** An HTTP server hosting a
  `refund_agent` with an `issue_refund` tool created with
  `RequireConfirmation: true`. The tool executes on this server, but when it
  needs confirmation the server asks Server 2 (through a remote agent) and
  relays the decision back as the confirmation.
- [`approvalserver/`](./approvalserver) — **Server 2.** An A2A server exposing
  an `approval_agent`. Given a proposed action it answers `APPROVE` or `DENY`,
  acting as the account owner. This is the remote agent that delivers the
  confirmation.

## The scenario

```
curl ─▶ agentserver /run  (Server 1, local LLM)
           │  call issue_refund(...)  ── RequireConfirmation ──▶ run suspends
           ▼                                                       │
    adk_request_confirmation (local)  ◀───────────────────────────┘
           │
           │  Server 1 forwards the proposed action over A2A
           ▼
        approval_agent (Server 2, LLM)  ──▶ "APPROVE"   (impersonates user)
           │
           │  Server 1 relays decision as the confirmation response
           ▼
    FunctionResponse{confirmed: true}  ─▶ run resumes
           │
    issue_refund runs on Server 1 ─▶ "refunded"
```

1. A caller sends a prompt asking Server 1 to issue a refund.
2. Server 1's agent calls its `issue_refund` tool, which requires confirmation,
   so the run suspends with an `adk_request_confirmation` function call.
3. Instead of asking a human, Server 1 sends the proposed action to the remote
   `approval_agent` on Server 2 over A2A and reads its `APPROVE`/`DENY`
   decision.
4. Server 1 feeds that decision back as the confirmation response (as if the
   user had answered), resuming the run.
5. The `issue_refund` tool executes **on Server 1**.

The key point: the tool runs on Server 1, but the confirmation originates on
Server 2. Server 2 stands in for the user.

## Running it

Both agents use Gemini, so set an API key (or configure Vertex AI):

```bash
export GOOGLE_API_KEY=...
```

Start the approval server (Server 2) in one terminal:

```bash
go run ./examples/serverconfirmation/approvalserver
# [approvalserver] A2A approval agent listening on http://127.0.0.1:8081
```

Start the agent server (Server 1) in a second terminal:

```bash
go run ./examples/serverconfirmation/agentserver -approver http://127.0.0.1:8081
# [agentserver] refund agent listening on http://127.0.0.1:8080
```

Drive it with `curl` in a third terminal:

```bash
curl "http://127.0.0.1:8080/run"
```

You will see Server 1's tool request confirmation, Server 1 ask Server 2,
Server 2 respond `APPROVE`, and the refund run on Server 1. Server 1 logs
`executing issue_refund ...` only after the remote approval — proof that the
tool ran on Server 1 but was gated by Server 2's decision.

The default request is a $42 refund, which Server 2 approves (its rule is to
approve refunds of $100 or less). Try a larger amount to see it denied:

```bash
curl "http://127.0.0.1:8080/run?q=Please+refund+%24500+for+order+B-9."
```

### Flags

Agent server (Server 1):

- `-addr` — host:port to serve on (default `127.0.0.1:8080`).
- `-approver` — base URL of the A2A approval server (default `http://127.0.0.1:8081`).

Approval server (Server 2):

- `-addr` — host:port to serve on (default `127.0.0.1:8081`).

### Endpoint

Agent server (Server 1):

- `GET /run` — runs one refund conversation. Optional `q` query parameter sets
  the prompt (defaults to a $42 refund request). Returns a plain-text transcript
  of the flow.
