# Configuration Model Guide

Read this when adding new configuration fields, new config classes, or modifying `ols/app/models/config.py`.

## Key Facts

- All config classes use Pydantic `BaseModel`
- `models[].parameters.temperature` is optional, finite, and non-negative. When absent, OLS
  does not set a sampling temperature. `temperature_supported` is no longer accepted.
- Config is loaded from YAML via `ProviderConfig(data_dict)` and similar constructors — not from keyword args directly
- Validation errors must raise `checks.InvalidConfigurationError`, not Pydantic's built-in errors
- `extra="forbid"` is used on provider-specific configs — unknown fields cause a hard error
- Credentials are never stored in the YAML directly; they are read from files via `checks.read_secret()`

## Config Class Patterns

### Standard field with validation

```python
class MyConfig(BaseModel):
    name: str
    optional_field: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise checks.InvalidConfigurationError("name must not be empty")
        return value
```

### Cross-field validation

```python
@model_validator(mode="after")
def validate_consistency(self) -> Self:
    if self.field_a and not self.field_b:
        raise checks.InvalidConfigurationError("field_b is required when field_a is set")
    return self
```

### Before-mode validator (for input transformation)

```python
@model_validator(mode="before")
@classmethod
def transform_inputs(cls, data: Any) -> Any:
    data["credentials"] = checks.read_secret(
        data, constants.CREDENTIALS_PATH_SELECTOR, constants.API_TOKEN_FILENAME
    )
    return data
```

## Reading Secrets

Credentials come from files, never from the YAML values directly:

```python
# Read API token from a file path specified in config
credentials = checks.read_secret(
    data,                               # the raw config dict
    constants.CREDENTIALS_PATH_SELECTOR,  # key = "credentials_path"
    constants.API_TOKEN_FILENAME,         # filename = "apitoken"
)
```

If `credentials_path` points to a directory, it reads `<directory>/apitoken`. If it points directly to a file, it reads that file. See `ols/utils/checks.py` for the full implementation.

## Adding a New Provider-Specific Config

1. Subclass `ProviderSpecificConfig` (which provides `url`, `token`, `api_key`):

```python
class MyProviderConfig(ProviderSpecificConfig, extra="forbid"):
    credentials_path: str
    my_specific_field: Optional[str] = None
```

2. Add an optional field to `ProviderConfig`:

```python
class ProviderConfig(BaseModel):
    ...
    my_provider_config: Optional[MyProviderConfig] = None
```

3. Add a `case` branch in `ProviderConfig.set_provider_specific_configuration()`:

```python
case constants.PROVIDER_MYPROVIDER:
    my_config = data.get("my_provider_config")
    self.check_provider_config(my_config)
    self.read_api_key(my_config)
    self.my_provider_config = MyProviderConfig(**my_config)
```

4. The config key in YAML follows the pattern `<provider_type>_config` (e.g. `openai_config`, `watsonx_config`).

## Config Validation Flow

Config classes have two validation phases:

1. **`__init__`** — Pydantic field validation + any custom `__init__` logic (reading secrets, setting derived fields)
2. **`validate_yaml()`** — Called explicitly after full config load; used for cross-object validation (e.g. TLS cert file existence checks)

If you add a new config class that needs file existence checks or cross-section validation, add a `validate_yaml()` method and call it from the appropriate parent config's `validate_yaml()`.

### OIDC authentication module

Set `ols_config.authentication_config.module` to `oidc` to validate bearer JWTs
against the configured OpenID Connect issuer. OIDC mode requires both
`oidc_issuer_url` and `oidc_audience`; the issuer's discovery document and JWKS
are cached for `oidc_jwks_cache_seconds` (default `300`). Tokens must use an
RS256 key published by the issuer and contain valid `iss`, `aud`, `exp`, `iat`,
and `sub` claims. The username comes from `oidc_username_claim` (default
`preferred_username`) and falls back to `sub`. OLS derives a stable UUID user
ID from the issuer and subject. To forward that token to an MCP server,
configure its `Authorization` header as `kubernetes`; the same placeholder
receives the exchanged token on A2A requests when token exchange is enabled.
The issuer and discovered JWKS URL must use HTTPS. OIDC always verifies system
TLS certificates and does not use the Kubernetes `skip_tls_verification`
setting. Configure the issuer as its exact OIDC identifier; its trailing
slash/path must match both discovery metadata and the
token's `iss` claim.

OIDC metrics access is denied unless both `oidc_metrics_access_claim` and
`oidc_metrics_access_value` are configured. The configured string must match
the claim's scalar string or one of its string-list values. This policy applies
only to the metrics permission scope and does not change ordinary REST access.

```yaml
ols_config:
  authentication_config:
    module: oidc
    oidc_issuer_url: "https://identity.example.com/realms/ols"
    oidc_audience: "openshift-lightspeed"
    oidc_username_claim: preferred_username
    oidc_jwks_cache_seconds: 300
    # Optional. Without both fields, OIDC callers cannot access Prometheus metrics.
    oidc_metrics_access_claim: groups
    oidc_metrics_access_value: metrics-readers
```

### Optional A2A server (`a2a`)

Top-level `a2a.enabled` (default `false`) gates mounting of the Agent2Agent
surface. A2A uses the selected `k8s` or `oidc` authentication module; development
authentication bypass and no-op modules are not accepted for A2A. `A2A_ENABLED`
overrides the YAML flag when set. Set `A2A_CLUSTER_ID` and `A2A_RPC_URL` for
cluster scoping and the public agent card. Token exchange is disabled by
default; set `a2a.token_exchange_enabled: true` to exchange the caller token for
an MCP-scoped token using the workload's SPIFFE JWT-SVID. This requires
an HTTPS `A2A_KEYCLOAK_ISSUER_URL` and the SPIFFE Workload API settings. The
issuer URL must match Keycloak discovery metadata exactly, including a trailing
slash if present. The `A2A_SPIFFE_TIMEOUT_SECONDS`,
`A2A_KEYCLOAK_TIMEOUT_SECONDS`, and `A2A_TOKEN_EXCHANGE_TIMEOUT_SECONDS`
values must be finite and in `(0, 300]`;
`A2A_KEYCLOAK_DISCOVERY_CACHE_SECONDS` must be in `[1, 86400]`. Invalid values
fail A2A configuration with HTTP 503. When exchange is disabled, OLS passes the
validated caller token directly to the MCP path. A2A task and context state is
process-local. At most 1,024 tasks may be in flight; excess new requests get
JSON-RPC error `-32603` with `retryable: true` metadata. The latest 1,024
terminal task results and 1,024 context bindings are retained separately.
Older terminal tasks may return not found, and an evicted idle context starts
a new OLS conversation. When Keycloak refuses an expired or revoked caller token
during exchange, the task pauses in `TASK_STATE_AUTH_REQUIRED`; the caller
renews by messaging the same task with a fresh token, and OLS re-runs the
original request. `A2A_AUTH_REQUIRED_TIMEOUT_SECONDS` (default `900`) bounds the
wait before the task fails and frees its slot; an invalid value is logged and
the default is used. See `.ai/spec/what/a2a.md`.

```yaml
a2a:
  enabled: true
  token_exchange_enabled: false
```

### Optional Solr hybrid RAG (`ols_config.solr_hybrid`)

Omit the key to leave Solr hybrid RAG off (no client, no tool). When present, values map to `SolrHybridSettings` and the feature is active. `solr_http_base` must be a valid `http` or `https` URL with a host (checked in `validate_yaml()`).

```yaml
ols_config:
  solr_hybrid:
    solr_http_base: "https://solr.example.com:8983"
    # max_results, hybrid_*, max_expansion_neighbors: optional overrides
```

Product documentation is served exclusively via Solr (OKP). Local FAISS indexes
under ``reference_content`` are for BYOK (user-supplied) content only.

### Hybrid search and chunk expansion

OLS sends a hybrid query to OKP Solr's ``/hybrid-search`` endpoint. The query
performs keyword retrieval first, then reranks candidates by KNN vector
similarity. See the
[OKP RAG Chunk Retrieval Strategy](https://docs.google.com/document/d/1W7G3Tbz5peMAh8cnGcpKvayQAciAyDXzXY_5KXdkY_0/edit?tab=t.0)
for the full specification.

Solr returns raw matched chunks. OLS implements chunk expansion client-side
(all logic in ``SolrHybridSearch`` in ``ols/src/rag_index/solr_support.py``):

1. **Deduplicate by parent** — keep only the first (highest-scored) chunk per
   ``parent_id``. Chunks without a ``parent_id`` are kept as-is.
2. **Cap results** — slice the deduped list to ``max_results``.
3. **Compute per-chunk budget** — divide the tool's ``tools_token_budget``
   (set by the execution framework via ``tool.metadata``) equally across the
   deduped chunks. When the budget is 0, skip expansion entirely and return
   the matched chunk as-is.
4. **Fetch family** — for each matched chunk, query Solr ``/select`` for all
   chunks sharing the same ``parent_id`` AND ``heading_id`` (ordered by
   ``chunk_index``). Orphan chunks (missing ``heading_id``) skip this step.
5. **Expand around match** — starting from the matched chunk, alternate
   between previous and next siblings until one of these limits is hit:
   - per-chunk token budget exhausted (tracked via ``num_tokens``)
   - ``max_expansion_neighbors`` reached on each side (default 2, configurable
     0–10 in ``SolrHybridSettings``; 0 disables expansion)
6. **Concatenate** — assemble the expanded chunks in ``chunk_index`` order,
   strip HTML, and return as the tool result (one content block per deduped
   match).

Relevant Solr fields for expansion:

| Field | Purpose |
|---|---|
| ``parent_id`` | Groups chunks by source document |
| ``chunk_index`` | Sequential ordering for neighbor expansion |
| ``heading_id`` | Family grouping (chunks under the same heading) |
| ``num_tokens`` | Token count per chunk for budget tracking |

Relevant config fields on ``SolrHybridSettings``:

| Field | Default | Purpose |
|---|---|---|
| ``max_results`` | 5 | Max deduped chunks returned |
| ``max_expansion_neighbors`` | 2 | Max siblings per side during expansion (0 disables) |

### OCP version resolution at startup

When ``solr_hybrid`` is configured, ``SolrHybridSearch.__init__`` resolves the
OCP product version for ``chunk_filter_query``:

1. Read the ``OCP_CLUSTER_VERSION`` environment variable (set by the operator).
   If not set, raise ``InvalidConfigurationError`` and stop.
2. Query Solr for available versions of ``openshift_container_platform``
   (facet on ``product_version`` with ``fq=product:openshift_container_platform``).
3. Clamp the requested version to the nearest available:
   - env version < lowest available → use lowest available
   - env version > highest available → use highest available
   - otherwise → use the closest available version ≤ requested
4. Build ``chunk_filter_query``:
   ``is_chunk:true AND product:openshift_container_platform AND product_version:<resolved>``


## Important Constants

Config-related constants live in `ols/constants.py`:

| Constant | Value | Used for |
|---|---|---|
| `CREDENTIALS_PATH_SELECTOR` | `"credentials_path"` | Key for secret file path in config dicts |
| `API_TOKEN_FILENAME` | `"apitoken"` | Default filename inside credentials directory |
| `DEFAULT_CONTEXT_WINDOW_SIZE` | (int) | Default if not set in model config |
| `DEFAULT_MAX_TOKENS_FOR_RESPONSE` | (int) | Default token limit |
| `SUPPORTED_PROVIDER_TYPES` | frozenset | Validated against in `ProviderConfig.set_provider_type()` |
| `SUPPORTED_AUTHENTICATION_MODULES` | frozenset | Validated in `AuthenticationConfig` |
| `tool_round_cap_fraction` | (YAML: `ols_config.tool_round_cap_fraction`) | Global per-round MCP tool budget cap; bounds `TOOL_ROUND_CAP_FRACTION_MIN` / `MAX` in `ols/constants.py` |
