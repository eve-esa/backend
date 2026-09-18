# User API

User routes manage the authenticated user's profile, usage metadata, and self-service API
keys.

## API call order

1. Complete auth flow (`/signup`, `/verify`, `/login`) or hold an `eve_` API key.
2. Use `access_token` (or the API key) in the `Authorization` header.
3. Call user routes (`/users/me`, `/users/me/token-usage`, `/users` patch, `/users/api-keys`).

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Get current user profile

`GET /users/me`

::: routers.user.me
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/users/me",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns the authenticated user's profile document.

### Notes

- Requires `Authorization: Bearer <access_token>`.

## Get token usage

`GET /users/me/token-usage`

::: routers.user.get_my_token_usage
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/users/me/token-usage",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns token budget and usage information for the current user.

### Notes

- Useful for client-side quota indicators.

## Update current user

`PATCH /users`

::: routers.user.update_user
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.patch(
    f"{BASE_URL}/users",
    json={"first_name": "Astro", "last_name": "User"},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Updates mutable profile fields for the authenticated user.

### Notes

- Route updates only the current authenticated user.

## API keys

Create, list and revoke self-service `eve_` API keys. Full concepts, error table and a
curl walkthrough live in [API keys](api-keys.md); this page only links the endpoints.

- `POST /users/api-keys`: create a key. Body optional, every field optional.
- `GET /users/api-keys[?include_revoked=true]`: list this user's keys.
- `DELETE /users/api-keys/{id}`: revoke a key, cascading to its children.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
