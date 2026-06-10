# Backoffice user creation

The backoffice app calls `POST /admin/users` to provision users with immediate activation (`is_active=true`).

## Configure the backend

### 1. Set `ADMIN_API_KEY`

On each EC2 instance, add a strong random secret to the environment file used by `docker-compose.staging.yml` or `docker-compose.prod.yml`:

```bash
# Generate a key (example)
openssl rand -hex 32
```

```env
ADMIN_API_KEY=<your-staging-or-prod-secret>
```

Use **different** values for staging and production.

### 2. Redeploy

Push to `staging` or `main` (per your GitHub Actions workflows), or restart the backend container after updating the env file:

```bash
docker compose -f docker-compose.staging.yml up -d --force-recreate backend
```

### 3. Smoke test

```bash
curl -X POST "$STAGING_URL/admin/users" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Api-Key: $ADMIN_API_KEY" \
  -d '{"email":"test@example.com"}'
```

Expected: HTTP `201` with `is_active: true` and a generated password.

### 4. CORS

Add the backoffice origin to `CORS_ALLOWED_ORIGINS` (comma-separated):

```env
# Local dev (backoffice runs on port 5174)
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174

# Deployed backoffice
CORS_ALLOWED_ORIGINS=https://backoffice.yourdomain.com
```

Redeploy after updating CORS.

## API reference

**`POST /admin/users`**

Headers:

- `Content-Type: application/json`
- `X-Admin-Api-Key: <ADMIN_API_KEY>`

Body:

```json
{
  "email": "user@example.com",
  "password": "optional",
  "first_name": "optional",
  "last_name": "optional",
  "rate_limit_group": "eve_free"
}
```

Responses:

| Status | Meaning |
|--------|---------|
| `201` | User created |
| `400` | Duplicate email or validation error |
| `403` | Missing or invalid API key |
| `503` | `ADMIN_API_KEY` not set on server |

## Backoffice app

The UI lives in a separate repository (`eve-backoffice`). See that repo's README for local setup and static deployment.
