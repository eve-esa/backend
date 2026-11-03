"""Build the skills folder a turn hands to the deep graph.

Mongo skills are the rows already resolved onto the request. Packaged skills
(the agents checkout) always load, and a packaged skill of the same name wins.
The result is one directory of ``<name>/...`` trees. A digest names the
directory, so a turn reuses a tree it has already built.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(tempfile.gettempdir()) / "eve-skills"
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_NAME_MAX = 64
_DESCRIPTION_MAX = 1024
_COMPATIBILITY_MAX = 500


def skill_checksum(files: Sequence[Any]) -> str:
    """sha256 over the sorted ``(path, content)`` pairs.

    These are the bytes the sandbox receives, so the checksum does not depend
    on an encoding.
    """
    digest = hashlib.sha256()
    pairs = sorted((item.path, item.content) for item in files)
    for path, content in pairs:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def packaged_skills_dir() -> Optional[Path]:
    """Skills folder shipped with the deep graph.

    ``None`` when that graph cannot be imported. The path is returned even
    when the folder is missing, so an empty request still points at the same
    place ``skills_dir()`` does.
    """
    try:
        from agents.graphs.deep.graph import skills_dir
    except ImportError:
        return None
    return Path(skills_dir())


def validate_skill(skill: Any) -> Optional[str]:
    """Return an error string, or ``None`` when *skill* follows the spec.

    A failure here drops that one skill. It does not drop the others.
    https://agentskills.io/specification
    """
    try:
        return _validate_skill(skill)
    except Exception as exc:
        return str(exc) or type(exc).__name__


def _validate_skill(skill: Any) -> Optional[str]:
    name = skill.name
    if not isinstance(name, str) or not _valid_name(name):
        return f"name {name!r} does not match the Agent Skills format"
    skill_md = _file_named(skill, "SKILL.md")
    if skill_md is None:
        return "SKILL.md is missing"
    frontmatter = _frontmatter(skill_md)
    if frontmatter is None:
        return "SKILL.md has no YAML frontmatter"
    if frontmatter.get("name") != name:
        return "frontmatter name does not match the skill name"
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        return "description is empty"
    if len(description) > _DESCRIPTION_MAX:
        return f"description is longer than {_DESCRIPTION_MAX} characters"
    if description != skill.description:
        return "frontmatter description does not match the skill description"
    compatibility = frontmatter.get("compatibility", None)
    if "compatibility" in frontmatter:
        if not isinstance(compatibility, str) or not (
            1 <= len(compatibility) <= _COMPATIBILITY_MAX
        ):
            return f"compatibility must be 1..{_COMPATIBILITY_MAX} characters"
    if "metadata" in frontmatter:
        metadata = frontmatter["metadata"]
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            return "metadata must be a map of strings to strings"
    seen: set[str] = set()
    for item in skill.files:
        error = _validate_path(item.path)
        if error is not None:
            return error
        if item.path in seen:
            return f"duplicate path {item.path!r}"
        seen.add(item.path)
    return None


def _valid_name(name: str) -> bool:
    return 1 <= len(name) <= _NAME_MAX and _NAME_RE.fullmatch(name) is not None


def _file_named(skill: Any, path: str) -> Optional[bytes]:
    for item in skill.files:
        if item.path == path:
            return item.content
    return None


def _frontmatter(content: bytes) -> Optional[dict]:
    text = content.decode("utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None
    loaded = yaml.safe_load(match.group(1))
    if not isinstance(loaded, dict):
        return None
    return loaded


def _validate_path(path: str) -> Optional[str]:
    if not path or path.endswith("/") or path.endswith("\\"):
        return f"path {path!r} is not a file"
    if "\\" in path or path.startswith("/") or ":" in path:
        return f"path {path!r} escapes the skill folder"
    parts = Path(path).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return f"path {path!r} escapes the skill folder"
    if Path(path).is_absolute():
        return f"path {path!r} escapes the skill folder"
    return None


def build_skills_root(
    skills: Sequence[Any], packaged_root: Optional[Path]
) -> Path:
    """Write *skills* and copy *packaged_root* into a cached directory.

    A packaged skill directory of the same name replaces the Mongo skill.
    Invalid Mongo skills are skipped. The directory name is a digest of the
    bytes written, and an existing digest is reused.
    """
    packaged = packaged_root if packaged_root is not None and packaged_root.is_dir() else None
    kept = _select_skills(skills, packaged)
    digest = _digest(kept, packaged)
    root = _CACHE_DIR / digest
    if root.is_dir():
        return root
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=_CACHE_DIR))
    try:
        for skill in kept:
            _write_skill(tmp / skill.name, skill)
        if packaged is not None:
            _copy_contents(packaged, tmp)
        os.rename(tmp, root)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        if root.is_dir():
            return root
        raise
    return root


def resolve_skills_root(skill_configs: Optional[Sequence[Any]]) -> Optional[Path]:
    """Folder to mount or upload for this turn.

    An empty *skill_configs* returns the packaged folder unchanged. A build
    failure falls back to that folder so the turn still runs.
    """
    packaged = packaged_skills_dir()
    if not skill_configs:
        return packaged
    try:
        return build_skills_root(list(skill_configs), packaged)
    except Exception:
        logger.exception("Failed to build the skills tree; using packaged skills")
        return packaged


def _select_skills(skills: Sequence[Any], packaged: Optional[Path]) -> list[Any]:
    packaged_names = set()
    if packaged is not None:
        packaged_names = {path.name for path in packaged.iterdir() if path.is_dir()}
    kept: list[Any] = []
    for skill in skills:
        name = getattr(skill, "name", None)
        if name in packaged_names:
            logger.warning(
                "Mongo skill %s matches a packaged skill; the packaged skill wins",
                name,
            )
            continue
        error = validate_skill(skill)
        if error is not None:
            logger.warning("Skipping skill %s: %s", name, error)
            continue
        kept.append(skill)
    return kept


def _digest(skills: Sequence[Any], packaged: Optional[Path]) -> str:
    digest = hashlib.sha256()
    for skill in sorted(skills, key=lambda item: item.name):
        digest.update(skill.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(skill_checksum(skill.files).encode("ascii"))
        digest.update(b"\0")
    if packaged is not None:
        files = sorted(path for path in packaged.rglob("*") if path.is_file())
        for path in files:
            digest.update(path.relative_to(packaged).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _write_skill(folder: Path, skill: Any) -> None:
    root = folder.resolve()
    for item in skill.files:
        dest = (folder / item.path).resolve()
        if root not in dest.parents:
            raise ValueError(f"path {item.path!r} escapes the skill folder")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(item.content)


def _copy_contents(source: Path, dest: Path) -> None:
    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)
