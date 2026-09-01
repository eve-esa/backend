import hashlib
import secrets


def generate_api_key() -> tuple[str, str]:
    """Generate a new SHA-256 hash API key.

    Returns ``(raw_token, key_hash)`` where ``raw_token`` is shown to the user
    exactly once and ``key_hash`` is the SHA-256 digest stored in the DB.

    The only credential this application still mints. Human sign-in belongs to
    the identity provider; an ``eve_`` key is a machine credential with no
    provider account behind it, so it stays here.
    """
    raw = "eve_" + secrets.token_hex(32)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, key_hash
