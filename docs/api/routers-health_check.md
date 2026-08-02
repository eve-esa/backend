# Health Check API

Health routes provide service liveness information.

## API call order

`GET /health` can be called any time (no auth/token prerequisite).

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Health check

`GET /health`

::: routers.health_check.health_check
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/health",
    timeout=10,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns a status payload for liveness checks, plus the identity of the running build:

```json
{
  "status": "healthy",
  "version": "v0.1.0",
  "commit": "1a2b3c4",
  "environment": "prod"
}
```

### Notes

- Commonly used by load balancers and uptime monitors.
- Does not require authentication.
- `status` is the only field the ALB health check and the deploy verification read, and its
  value never changes.
- `commit` is baked into the image at build time, `version` is injected at deploy time. Either
  reads `unknown` when the corresponding variable is not set, for example in local development.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
