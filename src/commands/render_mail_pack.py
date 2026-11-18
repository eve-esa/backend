"""Render a local mail template pack with the production loader, or fail.

    python -m src.commands.render_mail_pack --pack ./mail-templates --out ./out
    python -m src.commands.render_mail_pack --pack ./mail-templates --write-hashes

Loads ``<pack>/manifest.yaml`` and every file it lists through the same
validation the backend applies to a published pack (``load_pack``), then renders
each variant of ``<pack>/fixtures.yaml`` (or ``--fixtures``) into
``<out>/<variant>.html`` and ``<out>/<variant>.txt`` and prints one subject line
per variant. Unlike a live render there is no fallback: a rejected pack, a
fixture whose context does not match the mail's variables, a mail the pack does
not carry or a render error exits non-zero, so a broken pack never reaches a
bucket. ``fixtures.yaml``::

    variants:
      - name: account-pending
        mail: account_pending
        context: {email: ..., app_url: ..., logo_url: ...}

The variant name is the output file stem, so a gallery picks its own names
(``04-on-hold`` writes ``04-on-hold.html`` and ``04-on-hold.txt``). ``--out``
must be a new or empty directory, so no file from an earlier run sits next to
the new variants.

``--write-hashes`` rewrites the value of each entry under ``files:`` in
``manifest.yaml`` with the sha256 of that file, keeping every other line,
comment and line ending as written, so nobody computes a hash by hand. A line
under ``files:`` it cannot read fails the run, and the result must parse as
YAML with every listed file at its new digest before the manifest is replaced
(in one step, through a temporary file). With ``--out`` as well it then
validates and renders as usual.

A symlink in the pack (manifest or listed file) fails the run: it could pull a
file from outside the pack into the hashes, the render and the upload.

The command needs no network and no database: it reads the pack directory and
writes the output directory.

Exit codes: 0 done, 1 the pack or the fixtures are wrong, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from src.services.account_notifications import MAIL_SPECS
from src.services.mail_templates import MANIFEST_KEY, PackRejected, load_pack

# Used with fullmatch: ``$`` would accept a trailing newline.
_VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")


class FixtureError(Exception):
    """The fixtures file does not describe renderable variants."""


def _read_fixtures(path: Path) -> list[dict]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FixtureError(f"cannot read {path.name}: {type(exc).__name__}") from None
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, list) or not variants:
        raise FixtureError("fixtures need a non-empty 'variants' list")
    seen: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise FixtureError("each variant is a mapping")
        name = variant.get("name")
        if not isinstance(name, str) or not _VARIANT_NAME.fullmatch(name) or name in seen:
            raise FixtureError(f"variant name missing, invalid or repeated: {name!r}")
        seen.add(name)
        mail = variant.get("mail")
        if mail not in MAIL_SPECS:
            raise FixtureError(f"{name}: unknown mail {mail!r}")
        context = variant.get("context")
        if not isinstance(context, dict) or set(context) != MAIL_SPECS[mail].variables:
            expected = ",".join(sorted(MAIL_SPECS[mail].variables))
            raise FixtureError(f"{name}: context keys must be exactly {expected}")
    return variants


_FILES_HEADER = re.compile(r"files:\s*(#.*)?")
_FILES_ENTRY = re.compile(r"(\s+)([A-Za-z0-9][A-Za-z0-9_.-]*)(\s*:\s*)([^\s#]*)(\s+#.*)?")
# One line with its own ending (LF, CRLF or none on the last line), so a CRLF
# manifest stays CRLF.
_LINE = re.compile(r"[^\n]*\n|[^\n]+$")


def _read_regular(path: Path) -> bytes:
    """The bytes of a file in the pack; a symlink could pull in a file from outside it."""
    if path.is_symlink():
        raise FixtureError(f"symlink not allowed in the pack: {path.name}")
    return path.read_bytes()


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` in one step: an interrupted run never leaves half a manifest."""
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_hashes(pack_dir: Path) -> int:
    """Set each ``files:`` entry of the manifest to its file's sha256. Returns entries changed.

    The manifest is edited as text, so comments and layout stay as written. Any
    line inside ``files:`` the edit cannot parse fails the run, and the result is
    parsed as YAML before it is written: every listed file must then carry the
    digest just computed, or nothing is written.
    """
    manifest_path = pack_dir / MANIFEST_KEY
    text = _read_regular(manifest_path).decode("utf-8")
    lines = _LINE.findall(text)
    in_files = False
    changed = 0
    digests: dict[str, str] = {}
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if _FILES_HEADER.fullmatch(body):
            in_files = True
            continue
        if not in_files:
            continue
        if not body.strip() or body.lstrip().startswith("#"):
            continue
        if not body[0].isspace():
            in_files = False  # the next top-level key ends the block
            continue
        match = _FILES_ENTRY.fullmatch(body)
        if not match:
            raise FixtureError(f"manifest.yaml line {index + 1}: cannot read this files entry")
        indent, name, separator, old, comment = match.groups()
        if ".." in name:
            raise FixtureError(f"file name not allowed: {name}")
        digest = hashlib.sha256(_read_regular(pack_dir / name)).hexdigest()
        digests[name] = digest
        if old != digest:
            changed += 1
            ending = line[len(body):]
            lines[index] = f"{indent}{name}{separator}{digest}{comment or ''}{ending}"
    if not digests:
        raise FixtureError("manifest.yaml has no entries under files:")
    new_text = "".join(lines)
    try:
        parsed = yaml.safe_load(new_text)
    except yaml.YAMLError as exc:
        raise FixtureError(f"manifest.yaml does not parse after the edit: {type(exc).__name__}") from None
    files = parsed.get("files") if isinstance(parsed, dict) else None
    if files != digests:
        raise FixtureError("manifest.yaml files do not match the computed hashes after the edit")
    if new_text != text:
        _write_atomically(manifest_path, new_text.encode("utf-8"))
    return changed


def render_pack(pack_dir: Path, out_dir: Path, fixtures_path: Path) -> list[tuple[str, str]]:
    """Render every variant; returns ``(variant, subject)`` pairs. Raises on any problem."""
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        # Never delete files here: a stale variant in a reused directory would
        # look like part of the gallery, so the directory must start empty.
        raise FixtureError("--out must be a new or empty directory")
    manifest = _read_regular(pack_dir / MANIFEST_KEY)

    def read_file(name: str) -> bytes:
        return _read_regular(pack_dir / name)

    pack = load_pack(manifest, read_file, MAIL_SPECS)
    variants = _read_fixtures(fixtures_path)

    rendered: list[tuple[str, str, str, str]] = []
    for variant in variants:
        mail = pack.mails.get(variant["mail"])
        if mail is None:
            raise FixtureError(f"{variant['name']}: the pack does not carry {variant['mail']}")
        try:
            html, text = pack.render_parts(variant["mail"], variant["context"])
        except Exception as exc:
            raise FixtureError(f"{variant['name']}: render failed, {type(exc).__name__}") from None
        rendered.append((variant["name"], mail.subject, html, text))

    # Written only once every variant rendered: a failed run leaves no half gallery.
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, _, html, text in rendered:
        (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    return [(name, subject) for name, subject, _, _ in rendered]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.commands.render_mail_pack",
        description="Validate a mail template pack and render its fixtures.",
    )
    parser.add_argument("--pack", required=True, help="Directory holding manifest.yaml and the templates.")
    parser.add_argument("--out", help="Directory for the rendered variants; must be new or empty.")
    parser.add_argument("--fixtures", help="Fixtures file; <pack>/fixtures.yaml otherwise.")
    parser.add_argument(
        "--write-hashes",
        action="store_true",
        help="Rewrite the sha256 of every file listed in manifest.yaml from the files.",
    )
    args = parser.parse_args(argv)
    if not args.out and not args.write_hashes:
        parser.error("give --out, --write-hashes or both")

    pack_dir = Path(args.pack)
    fixtures = Path(args.fixtures) if args.fixtures else pack_dir / "fixtures.yaml"
    subjects: list[tuple[str, str]] = []
    try:
        if args.write_hashes:
            changed = write_hashes(pack_dir)
            print(f"manifest.yaml: {changed} hash(es) updated")
        if args.out:
            subjects = render_pack(pack_dir, Path(args.out), fixtures)
    except PackRejected as exc:
        print(f"error: pack rejected, {exc.reason}: {exc.detail}", file=sys.stderr)
        return 1
    except FixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {type(exc).__name__}: {exc.filename}", file=sys.stderr)
        return 1
    for name, subject in subjects:
        print(f"{name}: {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
