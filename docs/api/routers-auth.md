# Auth

There is no local auth router any more. `/signup`, `/login`, `/refresh`, `/verify` and
`/forgot-password` are removed; the backend no longer stores or checks passwords.

## API call order

1. Sign in at the identity provider (Keycloak locally, Cognito per AWS environment), not at
   this API. The provider issues the access token.
2. Use that token as `Authorization: Bearer <access_token>` on every protected route.
3. Refresh through the provider's own token endpoint when it expires; there is no backend
   `/refresh` to call.

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Programmatic access without an identity-provider session

For scripts, CI and other machine callers, create an `eve_` API key from an authenticated
session (or from another key) and use it as the bearer credential instead of a provider
access token. See [API keys](api-keys.md) for the create/list/revoke contract and a curl
walkthrough.

```bash
curl -s "$BASE_URL/users/me" -H "Authorization: Bearer $EVE_API_KEY"
```

## Why

The identity-provider migration is documented in
[`adr/identity-provider-architecture.md`](../adr/identity-provider-architecture.md). In short:
the app only verifies issuer, audience and JWT signature; which product issues the token
(Keycloak locally, Cognito in the cloud) is not the backend's concern.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
