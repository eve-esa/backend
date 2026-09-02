# Identity provider architecture

Status: accepted, 2026-09-01.

## Context

EVE authenticated humans itself. It stored passwords as unsalted SHA-256,
signed its own HS256 tokens with a shared secret, and ran its own signup,
email-verification and password-reset flows on top of python-jose. Two findings
from the Auth hardening review pointed at that code, and both of them were about
a job the application should not have had in the first place.

At the same time the deployment story needed an answer that is not
environment-specific: local development, dev, staging and production must all
authenticate the same way, and none of them should need a second implementation.

## Decision

Human authentication moves to OIDC, and the application keeps exactly one
verification path.

**Provider-neutral by construction.** The application knows an issuer, an
expected audience, a JWT, and its own stable internal user id. It does not know
which product answers at that issuer. Keycloak runs in local Docker Compose; one
Cognito user pool per environment is provisioned in Terraform. No code branches
on which one it is talking to.

**One path, not a dual mode.** A bearer token starting with `eve_` is an API
key; everything else is a provider access token. There is no configuration flag
that keeps the old tokens working, because a migration switch that can be left
half-flipped is a way to ship both code paths forever.

**The findings are retired by construction, not patched.** There is no password
hashing left to salt and no python-jose left to upgrade. The code that carried
the weakness is gone.

**AgentCore and MCP machine auth are untouched.** They authenticate services,
not people, and folding them into this would have coupled two problems that have
different lifetimes.

## The issuer and subject contract

The join key between a provider account and an EVE user is the pair
`(issuer, subject)`, stored in its own `external_identities` collection.

Not the email. Addresses get recycled, corrected and reassigned; a provider's
`sub` does not. Keying on email would mean that whoever ends up holding an old
address inherits the conversations of whoever held it before.

Its own collection, rather than a field on `users`, for two reasons. One user may
legitimately hold several provider accounts over time, and the uniqueness of the
pair has to be a real unique index. DocumentDB supports a compound unique index
on a top-level pair; it does not support one on array elements.

That index is doing more than lookup. The identity row is written **before** the
user row, so when two workers race the same first sign-in the duplicate key
picks the winner and the loser re-reads and adopts the winner's `user_id`.
Production runs two gunicorn workers; without this, a first sign-in in two tabs
is two accounts.

### Linking an existing account

A first sign-in adopts a pre-existing EVE account only when both hold:

1. the provider reports the email as verified, and
2. the matched account has no external identity of its own yet.

The second condition is the one that blocks recycled-address takeover. Without
it, an account already bound to one provider subject could be silently re-bound
to another. It refuses instead, loudly, and the request answers 403.

`email_verified` fails closed. Cognito returns the string `"true"`, Keycloak
returns a boolean, and both are accepted; anything else is read as unverified.
Guessing at an unfamiliar shape is how the rule above gets bypassed by accident.

## Audience: one rule that fits both

Cognito access tokens carry `client_id` and `token_use: "access"`, no `aud` and
no email. Keycloak access tokens carry `aud` only when the realm ships an
explicit audience protocol mapper (the default audience is `account`), plus
`azp`, and no `client_id`.

So the check is: `AUTH_CLIENT_ID` must appear in `aud` when the token has one,
and otherwise must equal the `client_id` claim. The local realm ships the
audience mapper. A Keycloak token whose only audience is `account` is refused,
which has a test of its own, because that is exactly what the tokens look like
if somebody edits the realm and drops the mapper.

## Caching, and how to correct it

Three in-process caches: the discovery document and the JWKS
(`AUTH_JWKS_CACHE_TTL_SECONDS`, default one hour) and the identity resolution
(60 seconds, bounded).

There is no invalidation API and there should not be one: a cache-poke endpoint
is another authenticated surface, and one that reaches across a fleet is a
distributed-systems problem bought for a rare event. Correcting a cached
resolution fleet-wide is a TTL lapse or a rolling restart. Sixty seconds is
chosen so that "wait it out" is a real answer rather than a shrug.

An unknown `kid` triggers one JWKS refetch behind a cooldown, so a legitimate
provider key rotation does not lock everybody out until the TTL lapses, and a
stream of forged kids does not turn that recovery into an amplifier.

## Failure modes are distinguished

A rejected token raises `PermissionError` and answers 401. An unreachable or
nonsensical provider raises `IdentityProviderUnavailable` and answers 503.
Collapsing the two would send every signed-in user into a sign-in that cannot
complete during a provider outage.

The JOSE header is parsed and vetted locally before any network call, so a
malformed token, a symmetric algorithm, or a missing `kid` cannot make this
process issue an outbound request.

The discovery document is trusted only after its own `issuer` field matches
`AUTH_ISSUER`. That check is what makes the `jwks_uri` it carries safe to fetch:
without it, a redirected or poisoned discovery response chooses the signing keys.

## sessionStorage, on the frontend

Tokens live in `sessionStorage` (the `oidc-client-ts` default) rather than
`localStorage`. The tradeoff is deliberate: a narrower XSS blast radius, at the
cost of not sharing a session across tabs directly. Persistence comes back
through the IdP session cookie via silent sign-in, which is the mechanism that
should own it anyway.

## What is deliberately temporary

Two things in this change are built to be deleted, and both say so in their own
source:

- `src/routers/migration.py`, the endpoint the Cognito Migrate-user Lambda calls
  during the production cutover so existing users sign in with the password they
  already have. It holds the last copy of the legacy hash function.
- `password_hash`, `is_active` and `activation_code` on the `User` model.
  Nothing writes them. They stay because `MongoModel.save()` is a full
  `replace_one` of `model_dump()`, so removing a field erases it from stored
  documents on the next save, which happens on every message. Deleting them now
  would destroy the hashes the migration needs and make rollback one-way.

Both go in the cleanup PR once the migration window closes, together with a
one-off `$unset` sweep.

`users.email` has a non-unique index for the same kind of reason: building a
unique index on a collection that already holds duplicates fails, and production
startup is the wrong place to discover that. `src/commands/report_duplicate_emails.py`
reports what is actually there; the index tightens when it comes back clean.

## Future: FLOCi

FLOCi (federated login across collaborating institutions) is the reason this is
issuer-and-subject rather than a Cognito integration. Adding a federated IdP
later is a new `(issuer, subject)` pair against the same user, an extra row in a
collection that already allows several per user. No schema change, no second
verification path.
