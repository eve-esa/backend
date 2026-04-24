# Forgot Password API

Forgot-password routes reset credentials without requiring login.

## API call order

1. `POST /forgot-password/code` with the user email.
2. User receives code/reset link by email.
3. `POST /forgot-password/confirm` with code and new password fields.
4. Login again through `POST /login` with the new password.

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Send reset code

`POST /forgot-password/code`

::: routers.forgot_password.send_forgot_password_code
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/forgot-password/code",
    json={"email": "astro.user@example.com"},
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Generates a one-time reset code and emails the reset link.

### Notes

- No authentication token is required.
- Fails if the email does not exist.

## Confirm new password

`POST /forgot-password/confirm`

::: routers.forgot_password.confirm_new_password
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/forgot-password/confirm",
    json={
        "code": "ABC123",
        "new_password": "StrongPassword123!",
        "confirm_password": "StrongPassword123!",
    },
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Validates code and expiry, then updates the user's password.

### Notes

- `new_password` and `confirm_password` must match.
- Code must be valid and not expired.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
