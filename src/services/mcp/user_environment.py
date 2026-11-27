"""Turn a stored ``environment.yaml`` into the accounts the UI shows.

``servers/<name>/environment.yaml`` in eve-esa/mcp-tool-registry names each
upstream account: the label, who issues it, which tools need it, and the
header or variable the server reads. The file holds no secret values. The
backoffice writes the parsed file onto ``mcp_servers.user_environment``.

This module is the read side. ``public_accounts`` is what ``GET /mcp-servers``
returns. When the file names headers and the row has no ``user_credentials``
spec, ``credential_spec`` builds the spec the save route and the proxy already
use, so those headers are checked, stored and sent. A file that only names
environment variables is shown and is not turned into a spec: the proxy injects
headers, not process environment.

A malformed file is ignored with one warning. It must not drop the toolkit
from the list.

Reference:
    https://github.com/eve-esa/mcp-tool-registry/blob/main/CONTRIBUTING.md
    (section "Create environment.yaml")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional
from urllib.parse import urlparse

from src import config
from src.database.models.mcp_server import (
    MCPServer,
    UserCredentialField,
    UserCredentialSpec,
    is_reserved_credential_header,
)
from src.schemas.mcp_server import (
    MCPServerAccountCredentialPublic,
    MCPServerAccountPublic,
)

logger = logging.getLogger(__name__)

_MAX_DEPTH = 3
_MAX_SERVICES = 8
_MAX_CREDENTIALS = 8
_MAX_TOOLS = 64
_MAX_HEADER_FIELDS = 5

_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_HEADER = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TOOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

_FILE_KEYS = frozenset({"services"})
_SERVICE_KEYS = frozenset(
    {"id", "label", "provider", "url", "tools", "credentials", "services"}
)
_CREDENTIAL_KEYS = frozenset({"header", "variable", "label", "help", "url", "secret"})

# Server names already warned about, so a broken file logs once per process.
_warned: set[str] = set()


class _Invalid(Exception):
    """The stored file breaks a rule. ``code`` never contains operator text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _Credential:
    name: str
    label: str
    help_text: Optional[str]
    url: Optional[str]
    secret: bool
    via: Literal["header", "variable"]
    header: Optional[str]


@dataclass(frozen=True)
class _Service:
    id: str
    label: str
    provider: Optional[str]
    url: Optional[str]
    tools: tuple[str, ...]
    credentials: tuple[_Credential, ...]
    services: tuple["_Service", ...]


def _warn(name: str, code: str) -> None:
    key = f"{name}:{code}"
    if key in _warned:
        return
    _warned.add(key)
    logger.warning("mcp.user_environment.invalid server=%s error=%s", name, code)


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Invalid(code)
    return value


def _exact_keys(value: Mapping[str, Any], allowed: frozenset[str]) -> None:
    if not set(value).issubset(allowed):
        raise _Invalid("unknown")


def _text(value: Any, *, limit: int, code: str) -> str:
    if not isinstance(value, str):
        raise _Invalid(code)
    text = value.strip()
    if not text or len(text) > limit:
        raise _Invalid(code)
    return text


def _optional_text(value: Any, *, limit: int, code: str) -> Optional[str]:
    if value is None:
        return None
    return _text(value, limit=limit, code=code)


def _https(value: Any) -> Optional[str]:
    """An https URL, or None when the key is absent. Anything else rejects the file."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _Invalid("url")
    text = value.strip()
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise _Invalid("url")
    return text


def _field_name(source: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    if not slug[:1].isalpha():
        slug = f"f_{slug}".strip("_")
    slug = slug[:32].rstrip("_")
    if not _FIELD_NAME.fullmatch(slug):
        raise _Invalid("name")
    return slug


def _parse_credential(
    raw: Any, names: set[str], headers: set[str]
) -> _Credential:
    body = _mapping(raw, "credential")
    _exact_keys(body, _CREDENTIAL_KEYS)
    header = body.get("header")
    variable = body.get("variable")
    if (header is None) == (variable is None):
        raise _Invalid("transport")
    if header is not None:
        if not isinstance(header, str) or not _HEADER.fullmatch(header):
            raise _Invalid("header")
        if is_reserved_credential_header(header):
            raise _Invalid("reserved_header")
        lowered = header.lower()
        if lowered in headers:
            raise _Invalid("duplicate")
        headers.add(lowered)
        source = header
        via: Literal["header", "variable"] = "header"
    else:
        if not isinstance(variable, str) or not _VARIABLE.fullmatch(variable):
            raise _Invalid("variable")
        source = variable
        via = "variable"
    name = _field_name(source)
    if name in names:
        raise _Invalid("duplicate")
    names.add(name)
    secret = body.get("secret", True)
    if not isinstance(secret, bool):
        raise _Invalid("secret")
    return _Credential(
        name=name,
        label=_text(body.get("label"), limit=64, code="label"),
        help_text=_optional_text(body.get("help"), limit=200, code="help"),
        url=_https(body.get("url")),
        secret=secret,
        via=via,
        header=header if via == "header" else None,
    )


def _parse_service(
    raw: Any, *, depth: int, ids: set[str], names: set[str], headers: set[str]
) -> _Service:
    if depth > _MAX_DEPTH:
        raise _Invalid("depth")
    body = _mapping(raw, "service")
    _exact_keys(body, _SERVICE_KEYS)
    service_id = body.get("id")
    if not isinstance(service_id, str) or not _ID.fullmatch(service_id):
        raise _Invalid("id")
    if service_id in ids:
        raise _Invalid("duplicate")
    ids.add(service_id)
    tools_raw = body.get("tools", [])
    if not isinstance(tools_raw, list) or len(tools_raw) > _MAX_TOOLS:
        raise _Invalid("tool")
    tools: list[str] = []
    for tool in tools_raw:
        if not isinstance(tool, str) or not _TOOL.fullmatch(tool):
            raise _Invalid("tool")
        tools.append(tool)
    credentials_raw = body.get("credentials", [])
    services_raw = body.get("services", [])
    if not isinstance(credentials_raw, list) or len(credentials_raw) > _MAX_CREDENTIALS:
        raise _Invalid("credential")
    if not isinstance(services_raw, list) or len(services_raw) > _MAX_SERVICES:
        raise _Invalid("service")
    if not credentials_raw and not services_raw:
        raise _Invalid("empty")
    return _Service(
        id=service_id,
        label=_text(body.get("label"), limit=64, code="label"),
        provider=_optional_text(body.get("provider"), limit=120, code="provider"),
        url=_https(body.get("url")),
        tools=tuple(tools),
        credentials=tuple(
            _parse_credential(item, names, headers) for item in credentials_raw
        ),
        services=tuple(
            _parse_service(
                item, depth=depth + 1, ids=ids, names=names, headers=headers
            )
            for item in services_raw
        ),
    )


def _parse_file(raw: Any) -> tuple[_Service, ...]:
    body = _mapping(raw, "shape")
    _exact_keys(body, _FILE_KEYS)
    services = body.get("services")
    if not isinstance(services, list) or not services or len(services) > _MAX_SERVICES:
        raise _Invalid("service")
    ids: set[str] = set()
    names: set[str] = set()
    headers: set[str] = set()
    return tuple(
        _parse_service(item, depth=1, ids=ids, names=names, headers=headers)
        for item in services
    )


def load(raw: Any, server_name: str) -> Optional[tuple[_Service, ...]]:
    """The accounts in ``raw``, or None when ``raw`` is absent or unusable."""
    if raw is None:
        return None
    try:
        return _parse_file(raw)
    except _Invalid as exc:
        _warn(str(server_name), exc.code)
        return None


def _to_public(service: _Service) -> MCPServerAccountPublic:
    return MCPServerAccountPublic(
        id=service.id,
        label=service.label,
        provider=service.provider,
        url=service.url,
        tools=list(service.tools),
        credentials=[
            MCPServerAccountCredentialPublic(
                name=item.name,
                label=item.label,
                secret=item.secret,
                help_text=item.help_text,
                url=item.url,
                via=item.via,
            )
            for item in service.credentials
        ],
        services=[_to_public(child) for child in service.services],
    )


def _owner_raw_name(server: MCPServer | Mapping[str, Any]) -> tuple[Any, Any, str]:
    if isinstance(server, MCPServer):
        return server.user_id, server.user_environment, server.name
    return server.get("user_id"), server.get("user_environment"), str(server.get("name") or "")


def public_accounts(
    server: MCPServer | Mapping[str, Any],
) -> Optional[list[MCPServerAccountPublic]]:
    """Accounts to show for ``server``, or None.

    None when the feature is off, the row belongs to a user, the row has no
    file, or the file is malformed.
    """
    if not config.FEATURE_TOOLKIT_CREDENTIALS:
        return None
    owner, raw, name = _owner_raw_name(server)
    if owner or raw is None:
        return None
    loaded = load(raw, name)
    if not loaded:
        return None
    return [_to_public(service) for service in loaded]


def _header_fields(
    services: tuple[_Service, ...],
) -> tuple[list[UserCredentialField], Optional[str]]:
    fields: list[UserCredentialField] = []
    help_url: Optional[str] = None

    def walk(nodes: tuple[_Service, ...]) -> None:
        nonlocal help_url
        for service in nodes:
            for item in service.credentials:
                if item.header is None:
                    continue
                if help_url is None:
                    help_url = item.url or service.url
                fields.append(
                    UserCredentialField(
                        name=item.name,
                        label=item.label,
                        header=item.header,
                        secret=item.secret,
                        help_text=item.help_text,
                    )
                )
            walk(service.services)

    walk(services)
    return fields, help_url


def credential_spec(
    raw: Any, server_name: str
) -> Optional[UserCredentialSpec]:
    """A header spec derived from the file, or None.

    None when the file is absent, malformed, names no headers, or names more
    headers than a spec can hold. Variable-only files stay display-only.
    """
    loaded = load(raw, server_name)
    if not loaded:
        return None
    fields, help_url = _header_fields(loaded)
    if not fields:
        return None
    if len(fields) > _MAX_HEADER_FIELDS:
        _warn(str(server_name), "too_many_headers")
        return None
    return UserCredentialSpec(fields=fields, required=False, help_url=help_url, check=None)
