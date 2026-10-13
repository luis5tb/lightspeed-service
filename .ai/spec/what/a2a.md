# A2A (Agent2Agent) Server Surface

OLS can expose an Agent2Agent protocol surface for delegated OpenShift investigations. This is separate from the versioned REST API under `/v1`.

## Behavioral Rules

### Enablement

1. The A2A surface must be disabled by default.
2. Enablement is controlled by the top-level `a2a.enabled` setting, overridden by `A2A_ENABLED` when set.
3. When disabled, OLS must not mount the A2A agent card, JSON-RPC routes, or A2A authentication middleware.
4. When enabled but `A2A_CLUSTER_ID` or `A2A_RPC_URL` is missing, A2A RPC and agent-card construction must fail with HTTP 503 and detail `A2A is not configured`.

### Routes

5. `GET /.well-known/agent-card.json` is public discovery.
6. `POST /a2a` supports `SendMessage`, `SendStreamingMessage`, `GetTask`, and `SubscribeToTask`. `CancelTask` must report unsupported until the investigation pipeline has a cancellation hook.
7. Push notifications are not supported.

### Authentication and cluster scope

8. A2A must use the selected OLS authentication module: `k8s` authentication uses Kubernetes TokenReview/SubjectAccessReview and `oidc` uses the configured OIDC JWT validator.
9. A2A must reject `noop` and `noop-with-token` authentication modes, and `dev_config.disable_auth` must not weaken A2A authentication.
10. Every `POST /a2a` call must include `X-OLS-Cluster` equal to the cluster id configured for this OLS service.
11. The validated caller bearer token must be passed unchanged through OLS's existing request/MCP path.
12. Client-supplied MCP header overrides must be ignored.

### Query modes

13. The agent card must advertise skills `ask` and `troubleshooting` with matching skill ids.
14. The agent card must declare the optional extension URI `https://openshift.io/ols/a2a/extensions/query-mode/v1` with params `modes: [ask, troubleshooting]`, `defaultMode: ask`, and `metadataKey: ols_mode`.
15. If message metadata `ols_mode` is set to a supported mode, use it; otherwise default to `ask`.
16. Interactive tool approval is not supported over A2A; such tasks must fail closed.

### Conversation continuity and task isolation

17. A2A `contextId` must map to an owned OLS conversation SUID keyed by `(caller task_scope, contextId)`. The raw `contextId` must not be used as the conversation id.
18. Subsequent messages with the same `contextId` from the same caller must reuse the same OLS conversation while that context binding is retained. Idle bindings may be evicted when the process-local capacity is reached; an evicted context starts a new conversation. Active and queued turns must not lose their binding during eviction.
19. Callers must not be able to continue another caller's context by guessing a `contextId`.
20. Task lookup must be owner-scoped: a task created by one `(caller, cluster)` is reported as not found to any other caller.
21. Task state may be process-local only for single-replica deployments. Reject new tasks before execution when the in-flight admission limit is reached; existing tasks may continue. Retain the latest terminal results in a separate bounded window so active tasks cannot evict just-completed results. Old terminal tasks may be evicted and then report not found. Multi-replica or restart-safe task storage is required before HA A2A is claimed.

### Audit

22. Each A2A investigation must emit a greppable `a2a_audit` log line without tokens, plus OLS `AuditContext` lifecycle events from the shared query pipeline.

## Query-mode extension contract

| Field | Value |
|-------|-------|
| URI | `https://openshift.io/ols/a2a/extensions/query-mode/v1` |
| required | false |
| params.modes | `ask`, `troubleshooting` |
| params.defaultMode | `ask` |
| params.metadataKey | `ols_mode` |

Clients may activate the extension via the `A2A-Extensions` header. OLS still honors `ols_mode` when present without the header.
