# Authentication Configuration

This document describes OLS authentication configuration, with focus on OIDC (OpenID Connect) authentication and multi-issuer support.

## Authentication Modules

OLS supports several authentication modules:
- `k8s` - Kubernetes ServiceAccount token validation
- `oidc` - OpenID Connect JWT bearer token validation
- `noop` - No authentication (development only)
- `noop-with-token` - Pass-through authentication (development only)

## OIDC Authentication

OIDC authentication validates JWT bearer tokens from one or more trusted OIDC identity providers.

### Single Issuer Configuration (Legacy)

For a single trusted OIDC issuer:

```yaml
ols_config:
  authentication_config:
    module: oidc
    oidc_issuer_url: "https://keycloak.example.com/realms/ols"
    oidc_audience: "openshift-lightspeed"
    oidc_username_claim: "preferred_username"
    oidc_jwks_cache_seconds: 300
```

**Fields:**
- `oidc_issuer_url` (required) - OIDC issuer URL (must use HTTPS)
- `oidc_audience` (required) - Expected audience claim (`aud`) in tokens
- `oidc_username_claim` (optional, default: `preferred_username`) - JWT claim to use as username
- `oidc_jwks_cache_seconds` (optional, default: 300) - How long to cache JWKS keys

### Multi-Issuer Configuration

For environments that need to accept tokens from multiple trusted OIDC issuers (e.g., Red Hat SSO production, Red Hat SSO stage, and local Keycloak):

```yaml
ols_config:
  authentication_config:
    module: oidc
    oidc_issuers:
      - issuer_url: "https://sso.redhat.com/auth/realms/redhat"
        audience: "openshift-lightspeed"
        username_claim: "preferred_username"
        jwks_cache_seconds: 300
      - issuer_url: "https://sso.stage.redhat.com/auth/realms/redhat"
        audience: "openshift-lightspeed"
        username_claim: "email"
        jwks_cache_seconds: 300
      - issuer_url: "https://keycloak.example.com/realms/ols"
        audience: "openshift-lightspeed"
        username_claim: "preferred_username"
        jwks_cache_seconds: 300
```

**Per-Issuer Fields:**
- `issuer_url` (required) - OIDC issuer URL (must use HTTPS)
- `audience` (required) - Expected audience claim for this issuer
- `username_claim` (optional, default: `preferred_username`) - JWT claim to use as username for tokens from this issuer
- `jwks_cache_seconds` (optional, default: 300) - JWKS cache duration for this issuer

**Validation Rules:**
- You must configure either `oidc_issuer_url` (single issuer) or `oidc_issuers` (multi-issuer), but not both
- `oidc_issuers` must contain at least one issuer configuration
- Issuer URLs must be unique within the list
- All issuer URLs must use HTTPS

## Multi-Issuer Use Case: Red Hat SSO + Keycloak Token Exchange

A common deployment pattern is:

1. **Orchestrator** sends a token from Red Hat SSO (or Red Hat SSO stage)
2. **OLS** validates the Red Hat SSO token at the OLS layer using multi-issuer OIDC
3. **OLS** exchanges the Red Hat SSO token for a Keycloak token using RFC 8693 token exchange
4. **Keycloak token** is used to access MCP and OCP API

This pattern is necessary when:
- OpenShift clusters are not under Red Hat control (cannot add OIDC Authorization pointing to Red Hat SSO)
- Red Hat SSO does not support token exchange
- An on-prem Keycloak is required as an intermediate token exchange service

### Configuration for Token Exchange Flow

**OLS OIDC Configuration** (validates tokens from both Red Hat SSO and Keycloak):
```yaml
ols_config:
  authentication_config:
    module: oidc
    oidc_issuers:
      - issuer_url: "https://sso.redhat.com/auth/realms/redhat"
        audience: "openshift-lightspeed"
        username_claim: "preferred_username"
      - issuer_url: "https://sso.stage.redhat.com/auth/realms/redhat"
        audience: "openshift-lightspeed"
        username_claim: "preferred_username"
      - issuer_url: "https://keycloak.example.com/realms/ols"
        audience: "openshift-lightspeed"
        username_claim: "preferred_username"
```

**A2A Token Exchange** (environment variables for token exchange):
```bash
export A2A_CLUSTER_ID="my-cluster-id"
export A2A_RPC_URL="https://mcp.example.com/rpc"
export A2A_KEYCLOAK_ISSUER_URL="https://keycloak.example.com/realms/ols"
export A2A_KEYCLOAK_CA_BUNDLE="/path/to/keycloak-ca.crt"
export A2A_EXCHANGE_AUDIENCE="openshift-mcp"
export A2A_SPIFFE_JWT_AUDIENCE="https://keycloak.example.com/realms/ols"
```

**Keycloak Setup** (for token exchange):
1. Configure Identity Provider (IdP) brokering for Red Hat SSO
2. Map user attributes (sub, username, etc.)
3. Enable token exchange for the realm
4. Configure client permissions for SPIFFE-based token exchange
5. Set up audience and requested token type

## Metrics Access Control

You can optionally require a specific JWT claim to access the `/metrics` endpoint:

```yaml
ols_config:
  authentication_config:
    module: oidc
    oidc_issuer_url: "https://keycloak.example.com/realms/ols"
    oidc_audience: "openshift-lightspeed"
    oidc_metrics_access_claim: "groups"
    oidc_metrics_access_value: "metrics-readers"
```

Both `oidc_metrics_access_claim` and `oidc_metrics_access_value` must be configured together. The claim value can be:
- A string (exact match required)
- A list (value must be present in the list)

## User Identity

OLS generates a stable user ID for each authenticated user:

```python
user_id = uuid5(NAMESPACE_URL, f"{issuer}:{subject}")
```

Where:
- `issuer` is the OIDC issuer URL from the token's `iss` claim
- `subject` is the user's subject from the token's `sub` claim

This ensures:
- Same user from same issuer gets same user_id
- Same user from different issuers gets different user_ids (multi-tenant isolation)

## Security Considerations

- All OIDC issuer URLs must use HTTPS
- TLS certificate verification is always enabled for OIDC (independent of `skip_tls_verification`)
- JWKS keys are cached for the configured duration, then refreshed
- Unknown signing keys trigger a throttled JWKS refresh
- Token validation includes signature, issuer, audience, lifetime, and required claims
- The `iss` claim in the token must exactly match a configured issuer URL

## Testing

See `tests/unit/auth/test_oidc_auth.py` and `tests/unit/auth/test_oidc_multi_issuer_auth.py` for comprehensive test coverage of OIDC authentication, including:
- Single and multi-issuer validation
- Issuer routing
- Unknown issuer rejection
- Username claim extraction
- Backward compatibility
