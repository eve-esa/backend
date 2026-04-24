# Auth API

Authentication is the first step for all protected routes.

## API call order

1. `POST /signup`
2. `POST /verify`
3. `POST /login` to receive tokens
4. Use `access_token` in protected routes
5. `POST /refresh` when access token expires

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Signup

`POST /signup`

::: routers.auth.signup
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/signup",
    json={
        "email": "astro.user@example.com",
        "password": "StrongPassword123!",
        "first_name": "Astro",
        "last_name": "User",
    },
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Creates an inactive user account and triggers an activation email.

### Notes

- Verification is required before login.
- Email must be unique.

## Verify account

`POST /verify`

::: routers.auth.verify
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/verify",
    json={
        "email": "astro.user@example.com",
        "activation_code": "123456",
    },
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Activates the account using the code sent by email.

### Notes

- Call this before `POST /login`.
- Use `POST /resend-activation` if the code is missing/expired.

## Login

`POST /login`

::: routers.auth.login
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/login",
    json={
        "email": "astro.user@example.com",
        "password": "StrongPassword123!",
    },
    timeout=30,
)
resp.raise_for_status()
tokens = resp.json()
ACCESS_TOKEN = tokens["access_token"]
REFRESH_TOKEN = tokens["refresh_token"]
print(tokens)
```

### Explanation

Returns JWT tokens used by all protected APIs.

### Notes

- Add `Authorization: Bearer <access_token>` to protected requests.
- Keep `refresh_token` for access token renewal.

## Refresh access token

`POST /refresh`

::: routers.auth.refresh
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/refresh",
    json={"refresh_token": REFRESH_TOKEN},
    timeout=30,
)
resp.raise_for_status()
ACCESS_TOKEN = resp.json()["access_token"]
print(resp.json())
```

### Explanation

Issues a new access token without re-entering credentials.

### Notes

- Keep using the same header format with the new token.
- Use when protected endpoints return auth-expired errors.

## Resend activation email

`POST /resend-activation`

::: routers.auth.resend_activation
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/resend-activation",
    json={"email": "astro.user@example.com"},
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Sends a new activation code for unverified accounts.

### Notes

- Use only before successful verification.

## Token usage helper

```python
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
```

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
