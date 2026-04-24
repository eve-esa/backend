# User API

User routes manage the authenticated user's profile and usage metadata.

## API call order

1. Complete auth flow (`/signup`, `/verify`, `/login`).
2. Use `access_token` in `Authorization` header.
3. Call user routes (`/users/me`, `/users/me/token-usage`, `/users` patch).

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

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
