# A2A (Agent2Agent) Server Surface

OLS can expose an Agent2Agent protocol surface for ACME-style delegated OpenShift investigations. This is separate from the versioned REST API under `/v1`.

## Behavioral Rules

### Enablement

1. The A2A surface must be disabled by default.
2. Enablement is controlled by the top-level olsconfig `a2a.enabled` flag, overridden by the `A2A_ENABLED` environment variable when set (`true`/`false`).
3. When disabled, OLS must not mount the A2A agent card, JSON-RPC routes, or A2A auth middleware.
4. When enabled but required Keycloak/SPIFFE env configuration is missing, A2A RPC and agent-card construction must fail with HTTP 503 and detail `A2A is not configured` (never an unhandled 500).

### Routes

5. `GET /.well-known/agent-card.json` is public (unauthenticated) discovery.
6. `POST /a2a` is the JSON-RPC 2.0 endpoint. Supported methods: `SendMessage`, `SendStreamingMessage`, `GetTask`, `SubscribeToTask`. `CancelTask` is accepted but must report unsupported until the investigation pipeline has a real cancel hook.
7. Push notifications are not supported.

### Authentication (distinct from `/v1`)

8. A2A must not use Kubernetes TokenReview/SubjectAccessReview.
9. Every `POST /a2a` call must validate a Keycloak bearer JWT (signature via JWKS, issuer, audience `lightspeed-a2a` by default, expiry, and `azp` allow-list) and require `X-OLS-Cluster` to equal the configured cluster id.
10. For each investigation, OLS must mint MCP credentials via SPIFFE JWT-SVID + RFC 8693 token exchange. Only the exchanged token (Token B) may reach MCP; the caller's token (Token A) must never be forwarded as an MCP credential.
11. Client-supplied MCP header overrides must be ignored.
12. `dev_config.disable_auth` must not weaken A2A authentication.
13. OIDC discovery and JWKS documents must be cached with a TTL (same order as `A2A_JWKS_CACHE_SECONDS`); discovery must not be cached forever.

### Query modes

14. The agent card must advertise skills `ask` and `troubleshooting` with matching skill ids.
15. The agent card must declare the optional extension URI `https://openshift.io/ols/a2a/extensions/query-mode/v1` with params `modes: [ask, troubleshooting]`, `defaultMode: ask`, and `metadataKey: ols_mode`.
16. Mode selection: if message metadata `ols_mode` is set to a supported mode, use it; otherwise default to `ask`.
17. Interactive tool approval is not supported over A2A; such tasks must fail closed.

### Conversation continuity

18. A2A `contextId` must map to an owned OLS conversation SUID keyed by `(caller task_scope, contextId)`. The raw `contextId` must not be used as the conversation id.
19. Subsequent messages with the same `contextId` from the same caller must reuse the same OLS conversation so history continues.
20. Callers must not be able to continue another caller's context by guessing a `contextId`.

### Task store / HA

21. Task state may be process-local (`InMemoryTaskStore`) only for single-replica deployments. Multi-replica or restart-safe task storage is required before HA A2A is claimed.
22. Task lookup must be owner-scoped: a task created by one `(caller, cluster)` is reported as not found to any other caller.

### Audit

23. Each A2A investigation must emit a greppable `a2a_audit` log line without tokens, plus OLS `AuditContext` lifecycle events from the shared query pipeline.

## Extension contract (query-mode/v1)

| Field | Value |
|-------|-------|
| URI | `https://openshift.io/ols/a2a/extensions/query-mode/v1` |
| required | false |
| params.modes | `ask`, `troubleshooting` |
| params.defaultMode | `ask` |
| params.metadataKey | `ols_mode` |

Clients may activate the extension via the `A2A-Extensions` header. OLS still honors `ols_mode` when present without the header.
