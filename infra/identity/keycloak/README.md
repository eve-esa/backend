# Local identity provider

Keycloak stands in for Cognito on a developer machine. The application cannot
tell them apart: it knows an issuer, an audience and a JWT.

## The realm

`realm/eve-realm.json` is imported by `start-dev --import-realm` on first boot.
It ships the `eve` realm, the public `eve-frontend` client (PKCE S256, no
secret), and three users with `emailVerified: true`.

Three settings in it are load-bearing and easy to lose in a reformat:

- **the audience protocol mapper**. Without it a Keycloak access token's only
  audience is `account`, and the backend rejects every token as
  `aud` mismatch. This is the mapper that puts `eve-frontend` in `aud`.
- **`accessTokenLifespan: 3600`**. The Keycloak default is five minutes, which
  is shorter than a long agentic run: the token expires mid-answer.
- **`sslRequired: "none"`**. The backend fetches discovery from inside the
  compose network, so the request arrives from a container address rather than
  localhost. The `external` default would refuse it over plain http.

## Editing it

`--import-realm` **skips** import when the realm already exists, so an edit to
this file does nothing on a restart. Recreate the container:

```sh
docker compose rm -sf keycloak && docker compose up -d keycloak
```

`docker compose down -v` also works, at the price of the rest of your data.

## Why the issuer says localhost while the backend talks to keycloak:8080

There is one `iss` value and both sides must agree on it. The browser can only
reach `http://localhost:8080`, the backend can only reach `http://keycloak:8080`.
`KC_HOSTNAME` pins the issuer to the browser-visible address and
`KC_HOSTNAME_BACKCHANNEL_DYNAMIC=true` lets the backend fetch discovery and JWKS
on the address it can actually reach, with the document still declaring the
`localhost` issuer the backend expects.
