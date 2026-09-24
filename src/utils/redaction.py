"""Secret and PII redaction for anything that leaves the process as text.

Single implementation used by the Mongo error log, the frontend ``/log-error``
endpoint and the MCP interceptor previews. Everything here is pure and never
raises on odd input: a redactor that crashes would drop the log it protects.

Two entry points:

- :func:`redact_secrets` scrubs one string (header values, query strings,
  ``key=value`` assignments, JWTs, EVE and RunPod API keys, URL credentials,
  email addresses).
- :func:`redact_value` walks dicts, lists, tuples and exceptions, blanks values
  stored under secret looking keys and runs :func:`redact_secrets` on every
  string it meets.
"""

import re
from enum import Enum
from typing import Any, Optional

REDACTED = "[REDACTED]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"

# Dict keys whose value is a secret whatever it looks like. Matched on the last
# snake_case segment group so ``access_token``, ``x-api-key``, ``clientSecret``
# and ``set-cookie`` hit, while ``max_tokens`` and ``token_count`` do not.
_SECRET_KEY_RE = re.compile(
    r"(?i)^(?:.*_)?(?:authorization|proxy_authorization|password|passwd|pwd|"
    r"secret|token|api_?key|jwt|credentials?|cookie|passphrase|"
    r"(?:secret|access|secret_access|private|signing|encryption)_key)$"
)

# Scheme plus credential. Bearer keeps the old broad match; Basic needs a base64
# looking value with an uppercase letter, digit or padding, so prose such as
# "basic setup" is left alone.
_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+[a-z0-9._~+/=\-]+")
_BASIC_RE = re.compile(
    r"(?i)\b(basic)\s+(?-i:(?=[A-Za-z0-9+/]*[A-Z0-9+/=])[A-Za-z0-9+/]{8,}={0,2})"
)
_JWT_RE = re.compile(
    r"eyJ[a-zA-Z0-9_\-]+=*\.[a-zA-Z0-9_\-]+=*\.[a-zA-Z0-9_\-+=/.]*"
)
# EVE API keys are "eve_" + 64 hex chars, RunPod keys "rpa_" + alphanumerics.
# The length floor keeps identifiers like "eve_free" or "eve_jsc" readable.
_PREFIXED_KEY_RE = re.compile(r"(?<![A-Za-z0-9])(?:eve|rpa)_[A-Za-z0-9]{16,}")
# user:password@ in any URL (https, mongodb, redis, ...).
# Anchored on the start of a scheme character run, not on \b: a word boundary
# inside a long run such as "a-b.a-b." restarts the scheme scan at every
# hyphen or dot, which is quadratic on big inputs.
_URL_USERINFO_RE = re.compile(r"(?i)(?<![a-z0-9+.\-])([a-z][a-z0-9+.\-]*://)[^/\s:@]*:[^/\s@]+@")
# Query string parameters named like a credential. The name is kept so a log
# still says which parameter was there.
_QUERY_PARAM_RE = re.compile(
    r"(?i)([?&;][^=&#\s?;]*?(?:key|token|secret|password|passwd|pwd|signature|"
    r"credential|jwt|code|session)[^=&#\s?;]*=)[^&#\s;]*"
)
# Free text assignments: api_key=..., "password": "...", token: ... .
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([a-z0-9_\-]*(?:api[_-]?key|apikey|token|secret|password|passwd)"
    r"[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;}#]+"
)
# The lookbehind anchors a match at the start of a run of local part
# characters. Without it the engine retries from every position inside a long
# run with no "@" (a base64 blob, a long generated output), which is quadratic:
# about 400 seconds for one megabyte of letters.
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
)


def redact_secrets(value: str) -> str:
    """Return ``value`` with credentials and email addresses replaced.

    Each match becomes a fixed placeholder, :data:`REDACTED` for secrets and
    :data:`REDACTED_EMAIL` for addresses; the scheme or parameter name in
    front of a secret is kept. Non string input is returned unchanged.
    """
    if not value or not isinstance(value, str):
        return value
    out = _URL_USERINFO_RE.sub(lambda m: f"{m.group(1)}{REDACTED}@", value)
    out = _JWT_RE.sub(REDACTED, out)
    out = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
    out = _BASIC_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
    out = _PREFIXED_KEY_RE.sub(REDACTED, out)
    out = _QUERY_PARAM_RE.sub(lambda m: f"{m.group(1)}{REDACTED}", out)
    out = _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{REDACTED}", out)
    out = _EMAIL_RE.sub(REDACTED_EMAIL, out)
    return out


def is_secret_key(key: Any) -> bool:
    """True when a mapping key names a credential (``api_key``, ``accessToken``)."""
    if key is None:
        return False
    text = str(key)
    # camelCase to snake_case, then hyphens to underscores.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text).replace("-", "_")
    return bool(_SECRET_KEY_RE.match(text))


def _clip(text: str, max_len: Optional[int]) -> str:
    return text if max_len is None else text[:max_len]


def redact_value(
    value: Any,
    *,
    key: Optional[str] = None,
    max_len: Optional[int] = None,
    _depth: int = 0,
) -> Any:
    """Return a JSON friendly, redacted copy of ``value``.

    Dicts, lists, tuples and exceptions are walked recursively; values under a
    secret looking key become :data:`REDACTED`; strings go through
    :func:`redact_secrets` and are clipped to ``max_len`` when given; other
    objects are stringified first. Exceptions become
    ``{"type", "message", "args"}``. Recursion stops at depth 20.
    """
    if key is not None and is_secret_key(key):
        return REDACTED
    if _depth > 20:
        return REDACTED
    nxt = _depth + 1
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _clip(redact_secrets(str(value)), max_len),
            "args": [
                redact_value(arg, max_len=max_len, _depth=nxt)
                for arg in getattr(value, "args", ()) or ()
            ],
        }
    if isinstance(value, dict):
        return {
            k: redact_value(v, key=str(k), max_len=max_len, _depth=nxt)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item, max_len=max_len, _depth=nxt) for item in value]
    if isinstance(value, str):
        return _clip(redact_secrets(value), max_len)
    if isinstance(value, Enum):
        return _clip(redact_secrets(str(value.value)), max_len)
    if isinstance(value, (bool, int, float, type(None))):
        return value
    return _clip(redact_secrets(str(value)), max_len)
