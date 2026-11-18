"""Mail templates from a published pack, with the bundled copy as the fallback.

The copy of every mail can change without a backend release: an operator
publishes a pack (``manifest.yaml`` plus the template files) to the bucket named
by ``MAIL_TEMPLATES_BUCKET``, and every worker picks it up on its next refresh.
The contract a pack must meet is written in ``docs/mail-templates.md``.

Three promises, each pinned by ``tests/test_mail_templates.py``:

* A blank bucket changes nothing. No S3 client is built, no task runs, and every
  mail renders from ``src/templates/mail/`` exactly as before.
* A pack is validated whole when it loads and rejected whole when anything is
  wrong with it. A rejected pack, or a failed refresh, keeps the last good pack.
* A send never waits for S3. The refresh runs in the background; a render reads
  one immutable ``Pack`` reference taken at its start, so a swap in the middle of
  a render cannot mix two packs. Any error while rendering a pack mail falls back
  to the bundled copy for that mail, and the mail still leaves.

Log lines carry the mail name, the pack version and the error class, never an
address or a rendered value.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from jinja2 import (
    BaseLoader,
    ChoiceLoader,
    DictLoader,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    meta,
    select_autoescape,
)
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src import config

logger = logging.getLogger(__name__)

# A pack declares the contract it was written for. A new mail or a new variable
# is additive and keeps the number; removing or renaming a variable bumps it.
MAIL_PACK_CONTRACT = 1
MANIFEST_KEY = "manifest.yaml"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 512 * 1024

# Shipped inside the image: the Dockerfile copies src/, templates included.
BUNDLED_DIR = Path(__file__).resolve().parent.parent / "templates" / "mail"

# The variables every mail receives. A file that no mail lists directly (the
# shared frame, a partial) may use only these.
BASE_VARIABLES = frozenset({"email", "app_url", "logo_url"})

SOURCE_PACK = "pack"
SOURCE_BUNDLED = "bundled"
NO_VERSION = "none"

FIRST_REFRESH_WAIT_S = 5.0
FAILURE_LOG_INTERVAL_S = 3600.0

_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class MailSpec:
    """What the backend supplies for one mail: the bundled files and the variables."""

    name: str
    subject: str
    html: str
    text: str
    variables: frozenset[str]


class RenderedMail(NamedTuple):
    subject: str
    html: str
    text: str
    source: str
    version: str


@dataclass(frozen=True)
class PackMail:
    subject: str
    html: str
    text: str


@dataclass(frozen=True)
class Pack:
    """One validated pack. Never mutated: a new pack is a new object."""

    version: str
    etag: str
    env: SandboxedEnvironment
    mails: Mapping[str, PackMail]


class PackRejected(Exception):
    """The pack breaks the contract. ``reason`` is a fixed word, ``detail`` names the part."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class _ManifestMail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    html: str
    text: str

    @field_validator("subject")
    @classmethod
    def _one_line(cls, value: str) -> str:
        # A subject is a header: a line break would let a pack add headers.
        if "\r" in value or "\n" in value:
            raise ValueError("subject must be one line")
        return value.strip()


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: int
    version: str
    files: dict[str, str]
    mails: dict[str, _ManifestMail]

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        if not _SEMVER.match(value):
            raise ValueError("version must be a semantic version")
        return value

    @field_validator("files")
    @classmethod
    def _files(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if not _FILE_NAME.match(name) or ".." in name:
                raise ValueError(f"file name not allowed: {name}")
            if not _SHA256.match(digest):
                raise ValueError(f"sha256 must be 64 lowercase hex characters: {name}")
        return value


def _environment(loader: BaseLoader) -> SandboxedEnvironment:
    # Autoescape by extension: an address with an ampersand must be escaped in
    # the HTML part and stay literal in the text part. StrictUndefined turns a
    # typo into an error the caller falls back on, never a blank in a mail.
    return SandboxedEnvironment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=select_autoescape(
            enabled_extensions=("html", "xml"), default_for_string=True
        ),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _parse_manifest(data: bytes) -> _Manifest:
    if len(data) > MAX_MANIFEST_BYTES:
        raise PackRejected("size", MANIFEST_KEY)
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PackRejected("yaml", type(exc).__name__) from None
    if not isinstance(raw, dict):
        raise PackRejected("schema", "manifest is not a mapping")
    # Checked before the schema, so a pack written for another contract says so.
    if raw.get("contract") != MAIL_PACK_CONTRACT:
        raise PackRejected("contract", str(raw.get("contract")))
    try:
        manifest = _Manifest.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise PackRejected("schema", location) from None
    for name, mail in manifest.mails.items():
        for file_name in (mail.html, mail.text):
            if file_name not in manifest.files:
                raise PackRejected("schema", f"mails.{name} names an unlisted file")
    return manifest


def load_pack(
    manifest_data: bytes,
    read_file: Callable[[str], bytes],
    specs: Mapping[str, MailSpec],
    bundled_dir: Path = BUNDLED_DIR,
    *,
    etag: str = "",
) -> Pack:
    """Validate a pack and build its environment, or raise ``PackRejected``.

    ``read_file`` returns the bytes of one listed file (S3 or a local directory).
    A ``KeyError`` or ``FileNotFoundError`` from it is a missing file. Mails this
    backend does not know are syntax checked and then ignored, so a pack can run
    ahead of an older backend.
    """
    manifest = _parse_manifest(manifest_data)

    sources: dict[str, str] = {}
    for name in sorted(manifest.files):
        try:
            data = read_file(name)
        except (KeyError, FileNotFoundError):
            raise PackRejected("file", f"{name} missing") from None
        if len(data) > MAX_FILE_BYTES:
            raise PackRejected("size", name)
        if hashlib.sha256(data).hexdigest() != manifest.files[name]:
            raise PackRejected("file", f"{name} sha256 mismatch")
        try:
            sources[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            raise PackRejected("file", f"{name} not utf-8") from None

    # The bundled directory comes second only as a safety net: validation below
    # proves every reference resolves inside the pack, so it is never reached.
    env = _environment(
        ChoiceLoader([DictLoader(sources), FileSystemLoader(str(bundled_dir))])
    )

    # Which variables each file may use: the intersection over the known mails
    # that list it; base variables for a file no mail lists (frame, partial);
    # syntax only for a file listed by unknown mails alone.
    known = {name: mail for name, mail in manifest.mails.items() if name in specs}
    listed_by_known: dict[str, list[frozenset[str]]] = {}
    listed_by_unknown: set[str] = set()
    for name, mail in manifest.mails.items():
        for file_name in (mail.html, mail.text):
            if name in specs:
                listed_by_known.setdefault(file_name, []).append(specs[name].variables)
            else:
                listed_by_unknown.add(file_name)
    allowed: dict[str, Optional[frozenset[str]]] = {}
    for name in sources:
        if name in listed_by_known:
            allowed[name] = frozenset.intersection(*listed_by_known[name])
        elif name in listed_by_unknown:
            allowed[name] = None
        else:
            allowed[name] = BASE_VARIABLES

    for name, source in sources.items():
        try:
            ast = env.parse(source, name=name)
            for target in meta.find_referenced_templates(ast):
                if target is None or target not in sources:
                    raise PackRejected("template", f"{name} references a template outside the pack")
            used = meta.find_undeclared_variables(ast)
            env.get_template(name)
        except PackRejected:
            raise
        except TemplateError as exc:
            raise PackRejected("template", f"{name} {type(exc).__name__}") from None
        permitted = allowed[name]
        if permitted is not None and not used <= permitted:
            unknown = ",".join(sorted(used - permitted))
            raise PackRejected("template", f"{name} uses undeclared variables {unknown}")

    mails = {
        name: PackMail(subject=mail.subject, html=mail.html, text=mail.text)
        for name, mail in known.items()
    }
    return Pack(version=manifest.version, etag=etag, env=env, mails=mails)


def _s3_client():
    return boto3.client(
        "s3",
        config=Config(
            connect_timeout=2,
            read_timeout=3,
            retries={"mode": "standard", "max_attempts": 2},
        ),
    )


def _error_reason(exc: BaseException) -> str:
    if isinstance(exc, ClientError):
        return str(exc.response.get("Error", {}).get("Code") or "ClientError")
    return type(exc).__name__


def _not_modified(exc: ClientError) -> bool:
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = exc.response.get("Error", {}).get("Code")
    return status == 304 or code in ("304", "NotModified")


class MailTemplateStore:
    """The current pack, its refresh task and the bundled environment."""

    def __init__(
        self,
        specs: Mapping[str, MailSpec],
        bundled_dir: Path = BUNDLED_DIR,
        *,
        bucket: Optional[str] = None,
        refresh_s: Optional[int] = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ):
        self.specs = specs
        self.bundled_dir = bundled_dir
        self._bundled = _environment(FileSystemLoader(str(bundled_dir)))
        self._bucket = bucket
        self._refresh_s = refresh_s
        self._client_factory = client_factory
        self._client = None
        self._pack: Optional[Pack] = None
        # The ETag of the last manifest read, good or rejected: a rejected pack
        # is reported once, not on every refresh. A failed read leaves it alone.
        self._etag: Optional[str] = None
        self._last_failure_log: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def bucket(self) -> str:
        value = self._bucket if self._bucket is not None else config.MAIL_TEMPLATES_BUCKET
        return (value or "").strip()

    @property
    def refresh_s(self) -> int:
        value = self._refresh_s if self._refresh_s is not None else config.MAIL_TEMPLATES_REFRESH_S
        return max(int(value), 1)

    @property
    def pack_version(self) -> str:
        pack = self._pack
        return pack.version if pack else NO_VERSION

    def carries(self, name: str) -> bool:
        """Whether the current pack has its own copy of ``name``."""
        pack = self._pack
        return pack is not None and name in pack.mails

    # ------------------------------------------------------------ render

    def render(
        self, name: str, context: Mapping[str, Any], *, subject: Optional[str] = None
    ) -> RenderedMail:
        """Render one mail from the current pack, else from the bundled copy.

        ``subject`` overrides both the pack and the bundled subject. Never calls
        S3 and never raises for a pack problem.
        """
        spec = self.specs[name]
        if set(context) != spec.variables:
            # A backend bug, not a pack problem: the contract says what we supply.
            raise ValueError(f"context for {name} does not match its variables")

        pack = self._pack  # one reference for the whole render
        reason: Optional[str] = None
        if pack is not None and name in pack.mails:
            mail = pack.mails[name]
            try:
                html = pack.env.get_template(mail.html).render(**context)
                text = pack.env.get_template(mail.text).render(**context)
                return RenderedMail(
                    (subject or mail.subject).strip(), html, text, SOURCE_PACK, pack.version
                )
            except Exception as exc:  # the sandbox can raise more than TemplateError
                reason = type(exc).__name__
        elif self.bucket:
            reason = "absent" if pack is not None else "no_pack"

        if reason is not None:
            logger.warning(
                "mail_template_fallback name=%s reason=%s pack_version=%s",
                name,
                reason,
                pack.version if pack else NO_VERSION,
            )
        html = self._bundled.get_template(spec.html).render(**context)
        text = self._bundled.get_template(spec.text).render(**context)
        return RenderedMail(
            (subject or spec.subject).strip(), html, text, SOURCE_BUNDLED, NO_VERSION
        )

    # ------------------------------------------------------------ refresh

    def _get_client(self):
        if self._client is None:
            factory = self._client_factory or _s3_client
            self._client = factory()
        return self._client

    def _read(self, client, key: str, **extra) -> tuple[bytes, str]:
        response = client.get_object(Bucket=self.bucket, Key=key, **extra)
        body = response["Body"]
        try:
            data = body.read(MAX_FILE_BYTES + 1 if key != MANIFEST_KEY else MAX_MANIFEST_BYTES + 1)
        finally:
            body.close()
        return data, str(response.get("ETag") or "")

    def _refresh_sync(self) -> bool:
        """One refresh. Blocking: run it in a thread. Returns whether a pack is installed."""
        bucket = self.bucket
        if not bucket:
            return False
        try:
            client = self._get_client()
            extra = {"IfNoneMatch": self._etag} if self._etag else {}
            try:
                manifest, etag = self._read(client, MANIFEST_KEY, **extra)
            except ClientError as exc:
                if _not_modified(exc):
                    self._last_failure_log = None
                    return self._pack is not None
                raise

            def read_file(name: str) -> bytes:
                try:
                    return self._read(client, name)[0]
                except ClientError as exc:
                    if _error_reason(exc) == "NoSuchKey":
                        raise KeyError(name) from None
                    raise

            try:
                pack = load_pack(manifest, read_file, self.specs, self.bundled_dir, etag=etag)
            except PackRejected as exc:
                self._etag = etag
                self._last_failure_log = None
                logger.warning(
                    "mail_templates_rejected reason=%s detail=%s etag=%s",
                    exc.reason,
                    exc.detail,
                    etag,
                )
                return self._pack is not None
        except Exception as exc:
            self._log_failure(_error_reason(exc))
            return self._pack is not None

        self._pack = pack
        self._etag = etag
        self._last_failure_log = None
        logger.info("mail_templates_loaded version=%s etag=%s", pack.version, etag)
        return True

    def _log_failure(self, reason: str) -> None:
        now = time.monotonic()
        last = self._last_failure_log
        if last is not None and now - last < FAILURE_LOG_INTERVAL_S:
            return
        self._last_failure_log = now
        logger.warning("mail_templates_refresh_failed reason=%s", reason)

    async def refresh(self) -> bool:
        """One refresh off the event loop. Never raises for S3 or pack problems."""
        if not self.bucket:
            return False
        return await asyncio.to_thread(self._refresh_sync)

    async def _loop(self, first: asyncio.Task) -> None:
        try:
            await first
        except Exception:
            logger.exception("mail_templates_refresh_failed reason=unexpected")
        while True:
            await asyncio.sleep(self.refresh_s)
            try:
                await self.refresh()
            except Exception:
                logger.exception("mail_templates_refresh_failed reason=unexpected")

    async def start(self, first_wait_s: float = FIRST_REFRESH_WAIT_S) -> None:
        """Start the refresh task; wait for the first refresh at most ``first_wait_s``.

        A blank bucket starts nothing. A slow or failing bucket never holds the
        lifespan past the cap: mail renders from the bundled copy until a pack loads.
        """
        if not self.bucket or self._task is not None:
            return
        first = asyncio.create_task(self.refresh())
        try:
            await asyncio.wait_for(asyncio.shield(first), first_wait_s)
        except asyncio.TimeoutError:
            self._log_failure("timeout")
        except Exception as exc:
            self._log_failure(type(exc).__name__)
        self._task = asyncio.create_task(self._loop(first))

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
