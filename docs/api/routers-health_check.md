# Health Check API

Health routes provide service liveness information.

## API call order

`GET /health` can be called any time (no auth/token prerequisite).

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

Returns a simple status payload for liveness checks.

### Notes

- Commonly used by load balancers and uptime monitors.
- Does not require authentication.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
