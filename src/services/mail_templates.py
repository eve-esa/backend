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
  Validation includes a node allowlist and a trial render of every mail with a
  size cap, run in the refresh thread: a pack cannot stall the event loop.
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
import unicodedata
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from jinja2 import (
    BaseLoader,
    DictLoader,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    meta,
    nodes,
)
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src import config

logger = logging.getLogger(__name__)

# A pack declares the contract it was written for. A new mail or a new variable
# is additive and keeps the number; removing or renaming a variable bumps it.
MAIL_PACK_CONTRACT = 1
MANIFEST_KEY = "manifest.yaml"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 512 * 1024
# Each rendered part (html, text) of a pack mail. Above it the mail falls back.
MAX_RENDERED_BYTES = 256 * 1024

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

# Used with fullmatch: ``$`` would accept a trailing newline.
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_MAIL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEMVER = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")
_LOG_SAFE = re.compile(r"[A-Za-z0-9_.-]{1,100}")

# Control, format, surrogate, private use and unassigned characters, and the
# Unicode line and paragraph separators: none belongs in a header, and some
# break the message build or the relay (NUL, vertical tab, U+2028, NEL).
_SUBJECT_BANNED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})

# What a pack template may contain. The mails need names, blocks, extends and
# include, if, one level of ``for`` over a variable, and a few filters. Anything
# else (arithmetic, calls, assignments, macros, imports, autoescape switches) is
# where a template can burn CPU or memory, so it is rejected at load.
_ALLOWED_NODES = (
    nodes.Template,
    nodes.Output,
    nodes.TemplateData,
    nodes.Const,
    nodes.Name,
    nodes.Extends,
    nodes.Block,
    nodes.Include,
    nodes.If,
    nodes.For,
    nodes.CondExpr,
    nodes.Filter,
    nodes.FilterBlock,
    nodes.Test,
    nodes.Compare,
    nodes.Operand,
    nodes.And,
    nodes.Or,
    nodes.Not,
    nodes.Concat,
    nodes.Getattr,
    nodes.Getitem,
    nodes.Slice,
    nodes.Tuple,
    nodes.List,
    nodes.Keyword,
)
# Bounded filters only: none of them grows its input by an argument.
_ALLOWED_FILTERS = frozenset({
    "capitalize", "count", "d", "default", "e", "escape", "first", "last",
    "length", "lower", "string", "striptags", "title", "trim", "upper", "urlencode",
})
# Inside a loop body: no second loop and nothing that pulls in another template.
_FORBIDDEN_IN_LOOP = (nodes.For, nodes.Include, nodes.Extends, nodes.Block)


@dataclass(frozen=True)
class MailSpec:
    """What the backend supplies for one mail: the bundled files and the variables."""

    name: str
    subject: str
    html: str
    text: str
    variables: frozenset[str]
    # Contexts for the trial render at load; each must carry exactly ``variables``.
    samples: tuple[Mapping[str, Any], ...] = ()

    def trial_contexts(self) -> tuple[Mapping[str, Any], ...]:
        if self.samples:
            return self.samples
        return ({variable: f"trial-{variable}" for variable in sorted(self.variables)},)


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


class RenderTooLarge(Exception):
    """A rendered part is above ``MAX_RENDERED_BYTES``."""


def _render_parts(
    env_html: ImmutableSandboxedEnvironment,
    env_text: ImmutableSandboxedEnvironment,
    html_name: str,
    text_name: str,
    context: Mapping[str, Any],
) -> tuple[str, str]:
    """Render the html part with autoescape on and the text part with it off.

    Autoescape follows the role of the part, never the file name: a pack that
    names its html part ``x.htm`` still escapes the address.
    """
    html = env_html.get_template(html_name).render(**context)
    if len(html.encode("utf-8")) > MAX_RENDERED_BYTES:
        raise RenderTooLarge("html")
    text = env_text.get_template(text_name).render(**context)
    if len(text.encode("utf-8")) > MAX_RENDERED_BYTES:
        raise RenderTooLarge("text")
    return html, text


@dataclass(frozen=True)
class Pack:
    """One validated pack. Never mutated: a new pack is a new object."""

    version: str
    etag: str
    env_html: ImmutableSandboxedEnvironment
    env_text: ImmutableSandboxedEnvironment
    mails: Mapping[str, PackMail]

    def render_parts(self, name: str, context: Mapping[str, Any]) -> tuple[str, str]:
        mail = self.mails[name]
        return _render_parts(self.env_html, self.env_text, mail.html, mail.text, context)


class PackRejected(Exception):
    """The pack breaks the contract. ``reason`` is a fixed word, ``detail`` names the part.

    ``transient`` marks what a publish in progress can cause (a listed file
    missing or not yet replaced): the next refresh tries the same manifest again.
    Every other rejection is final for that manifest.
    """

    def __init__(self, reason: str, detail: str = "", *, transient: bool = False):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.transient = transient


def subject_problem(value: str) -> Optional[str]:
    """Why ``value`` cannot be a mail subject, or None when it can."""
    if not value.strip():
        return "empty"
    for char in value:
        if unicodedata.category(char) in _SUBJECT_BANNED_CATEGORIES:
            return f"character U+{ord(char):04X} not allowed"
    return None


def _subject_builds(value: str) -> bool:
    """Whether ``value`` passes the subject rules and survives a message build."""
    if subject_problem(value) is not None:
        return False
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = value
        message.as_string()
    except Exception:
        return False
    return True


class _ManifestMail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    html: str
    text: str

    @field_validator("subject")
    @classmethod
    def _header_safe(cls, value: str) -> str:
        # A subject is a header: a line break would let a pack add headers, and
        # a control character can break the message build or the relay.
        problem = subject_problem(value)
        if problem is not None:
            raise ValueError(f"subject: {problem}")
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
        if not _SEMVER.fullmatch(value):
            raise ValueError("version must be a semantic version")
        return value

    # Both run before the inner validation, so an error location never carries
    # a key that has not passed its pattern: keys reach the logs.
    @field_validator("files", mode="before")
    @classmethod
    def _files(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for name, digest in value.items():
                if not isinstance(name, str) or not _FILE_NAME.fullmatch(name) or ".." in name:
                    raise ValueError("file name not allowed")
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("mails", mode="before")
    @classmethod
    def _mail_names(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for name in value:
                if not isinstance(name, str) or not _MAIL_NAME.fullmatch(name):
                    raise ValueError("mail name not allowed")
        return value


def _environment(loader: BaseLoader, *, autoescape: bool) -> ImmutableSandboxedEnvironment:
    # One environment per part: an address with an ampersand must be escaped in
    # the HTML part and stay literal in the text part. StrictUndefined turns a
    # typo into an error the caller falls back on, never a blank in a mail.
    # Immutable: a template cannot append to a list or update a dict. No
    # globals: range, lipsum, cycler, joiner, namespace and dict are gone.
    env = ImmutableSandboxedEnvironment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=autoescape,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.clear()
    return env


def _log_safe(part: Any) -> str:
    text = str(part)
    return text if _LOG_SAFE.fullmatch(text) else "<invalid>"


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
        location = ".".join(_log_safe(part) for part in first["loc"])
        raise PackRejected("schema", location) from None
    for name, mail in manifest.mails.items():
        for file_name in (mail.html, mail.text):
            if file_name not in manifest.files:
                raise PackRejected("schema", f"mails.{name} names an unlisted file")
    return manifest


def _check_nodes(ast: nodes.Template, name: str, permitted: frozenset[str]) -> None:
    """Reject any node outside the allowlist, and loops that are not flat and bounded."""
    for node in ast.find_all(nodes.Node):
        if not isinstance(node, _ALLOWED_NODES):
            raise PackRejected("template", f"{name} uses {type(node).__name__}")
        if isinstance(node, nodes.Filter) and node.name not in _ALLOWED_FILTERS:
            raise PackRejected("template", f"{name} uses a filter outside the allowlist")
        if isinstance(node, nodes.For):
            # A loop walks one variable the backend supplies (a list or a
            # string), never a literal or an expression, and holds no loop
            # and no other template: its cost stays linear in its input.
            if (
                node.recursive
                or not isinstance(node.iter, nodes.Name)
                or node.iter.name not in permitted
            ):
                raise PackRejected("template", f"{name} loops over something other than a variable")
            for inner in node.body + node.else_:
                if isinstance(inner, _FORBIDDEN_IN_LOOP) or any(
                    True for _ in inner.find_all(_FORBIDDEN_IN_LOOP)
                ):
                    raise PackRejected("template", f"{name} nests a loop or a template in a loop")


def _check_no_cycle(references: Mapping[str, set[str]]) -> None:
    done: set[str] = set()
    for start in sorted(references):
        if start in done:
            continue
        # Iterative depth-first search: (file, iterator over its references).
        path: list[str] = [start]
        on_path = {start}
        stack = [iter(sorted(references[start]))]
        while stack:
            target = next(stack[-1], None)
            if target is None:
                stack.pop()
                done.add(path[-1])
                on_path.discard(path.pop())
                continue
            if target in on_path:
                raise PackRejected("template", f"{target} references itself")
            if target not in done:
                path.append(target)
                on_path.add(target)
                stack.append(iter(sorted(references[target])))


def load_pack(
    manifest_data: bytes,
    read_file: Callable[[str], bytes],
    specs: Mapping[str, MailSpec],
    *,
    etag: str = "",
) -> Pack:
    """Validate a pack, trial render every mail it carries, or raise ``PackRejected``.

    ``read_file`` returns the bytes of one listed file (S3 or a local directory).
    A ``KeyError`` or ``FileNotFoundError`` from it is a missing file. Mails this
    backend does not know are syntax checked and then ignored, so a pack can run
    ahead of an older backend; a file only they list but another file references
    is held to the base variables. Blocking and CPU bound: run it off the loop.
    """
    manifest = _parse_manifest(manifest_data)

    sources: dict[str, str] = {}
    for name in sorted(manifest.files):
        try:
            data = read_file(name)
        except (KeyError, FileNotFoundError):
            raise PackRejected("file", f"{name} missing", transient=True) from None
        # Size and hash: a publish in progress shows up as either, so retry.
        if len(data) > MAX_FILE_BYTES:
            raise PackRejected("size", name, transient=True)
        if hashlib.sha256(data).hexdigest() != manifest.files[name]:
            raise PackRejected("file", f"{name} sha256 mismatch", transient=True)
        try:
            sources[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            raise PackRejected("file", f"{name} not utf-8") from None

    loader = DictLoader(sources)
    env_html = _environment(loader, autoescape=True)
    env_text = _environment(loader, autoescape=False)

    asts: dict[str, nodes.Template] = {}
    references: dict[str, set[str]] = {}
    for name, source in sources.items():
        try:
            asts[name] = env_html.parse(source, name=name)
        except TemplateError as exc:
            raise PackRejected("template", f"{name} {type(exc).__name__}") from None
        targets = set()
        for target in meta.find_referenced_templates(asts[name]):
            if target is None or target not in sources:
                raise PackRejected("template", f"{name} references a template outside the pack")
            targets.add(target)
        references[name] = targets
    _check_no_cycle(references)
    referenced = set().union(*references.values()) if references else set()

    # Which variables each file may use: the intersection over the known mails
    # that list it; base variables for a file no known mail lists (frame,
    # partial). A file listed by unknown mails alone and referenced by nothing
    # is never rendered here: syntax only.
    known = {name: mail for name, mail in manifest.mails.items() if name in specs}
    listed_by_known: dict[str, list[frozenset[str]]] = {}
    for name, mail in known.items():
        for file_name in (mail.html, mail.text):
            listed_by_known.setdefault(file_name, []).append(specs[name].variables)
    listed_by_unknown = {
        file_name
        for name, mail in manifest.mails.items()
        if name not in specs
        for file_name in (mail.html, mail.text)
    }

    for name, ast in asts.items():
        if name in listed_by_known:
            permitted = frozenset.intersection(*listed_by_known[name])
        elif name in listed_by_unknown and name not in referenced:
            continue
        else:
            permitted = BASE_VARIABLES
        # Before any compile: the optimizer folds constants, so ``9**9**9``
        # would run inside get_template.
        _check_nodes(ast, name, permitted)
        used = meta.find_undeclared_variables(ast)
        if not used <= permitted:
            unknown = ",".join(sorted(used - permitted))
            raise PackRejected("template", f"{name} uses undeclared variables {unknown}")

    for name in sources:
        try:
            env_html.get_template(name)
            env_text.get_template(name)
        except TemplateError as exc:
            raise PackRejected("template", f"{name} {type(exc).__name__}") from None

    mails = {
        name: PackMail(subject=mail.subject, html=mail.html, text=mail.text)
        for name, mail in known.items()
    }
    pack = Pack(
        version=manifest.version, etag=etag, env_html=env_html, env_text=env_text, mails=mails
    )
    # The trial render: every mail this backend sends from the pack, with its
    # sample contexts, under the size cap. A mail that fails here would only
    # fall back at send time, so the whole pack is refused now.
    for name in sorted(mails):
        for context in specs[name].trial_contexts():
            try:
                pack.render_parts(name, context)
            except RenderTooLarge as exc:
                raise PackRejected("size", f"{name} {exc}") from None
            except Exception as exc:  # the sandbox can raise more than TemplateError
                raise PackRejected("render", f"{name} {type(exc).__name__}") from None
    return pack


def _s3_client():
    # A session of its own: the default boto3 session is not safe for client
    # creation across threads. Built once, on the event loop thread; botocore
    # clients are safe to share with the refresh thread after that.
    return boto3.session.Session().client(
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
    """The current pack, its refresh task and the bundled environments."""

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
        bundled_loader = FileSystemLoader(str(bundled_dir))
        self._bundled_html = _environment(bundled_loader, autoescape=True)
        self._bundled_text = _environment(bundled_loader, autoescape=False)
        self._bucket = bucket
        self._refresh_s = refresh_s
        self._client_factory = client_factory
        self._client = None
        self._pack: Optional[Pack] = None
        # The ETag of the last manifest read when it was installed or finally
        # rejected: a rejected pack is reported once, not on every refresh. A
        # failed read or a transient rejection leaves it alone, so it is retried.
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
            try:
                html, text = pack.render_parts(name, context)
                chosen = (subject or pack.mails[name].subject).strip()
                if _subject_builds(chosen):
                    return RenderedMail(chosen, html, text, SOURCE_PACK, pack.version)
                # A subject the message build would refuse: the bundled mail
                # leaves instead of no mail at all.
                reason = "subject"
            except RenderTooLarge:
                reason = "size"
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
        html = self._bundled_html.get_template(spec.html).render(**context)
        text = self._bundled_text.get_template(spec.text).render(**context)
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
                pack = load_pack(manifest, read_file, self.specs, etag=etag)
            except PackRejected as exc:
                if exc.transient:
                    # A publish in progress: same manifest again next time.
                    self._log_failure(f"{exc.reason} detail={exc.detail!r} etag={etag!r}", rejected=True)
                    return self._pack is not None
                self._etag = etag
                self._last_failure_log = None
                logger.warning(
                    "mail_templates_rejected reason=%s detail=%r etag=%r",
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
        logger.info("mail_templates_loaded version=%s etag=%r", pack.version, etag)
        return True

    def _log_failure(self, reason: str, *, rejected: bool = False) -> None:
        now = time.monotonic()
        last = self._last_failure_log
        if last is not None and now - last < FAILURE_LOG_INTERVAL_S:
            return
        self._last_failure_log = now
        event = "mail_templates_rejected" if rejected else "mail_templates_refresh_failed"
        logger.warning("%s reason=%s", event, reason)

    def _ensure_client(self) -> bool:
        """Build the S3 client on the calling (event loop) thread. False on failure."""
        try:
            self._get_client()
        except Exception as exc:
            self._log_failure(_error_reason(exc))
            return False
        return True

    async def refresh(self) -> bool:
        """One refresh off the event loop. Never raises for S3 or pack problems."""
        if not self.bucket:
            return False
        if not self._ensure_client():
            return self._pack is not None
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
