"""Upsert the global ``roll-dice`` skill into the local ``skills`` collection.

    python -m src.commands.seed_dice_skill [--disable]

The files live in the document. Re-running replaces them in place.
"""

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from src.config import configure_logging
from src.database.models.skill import Skill, SkillFile
from src.database.mongo import async_mongo_manager
from src.services.agents.core.skills import skill_checksum, validate_skill

configure_logging()
logger = logging.getLogger(__name__)

SKILL_NAME = "roll-dice"
DICE_DESCRIPTION = (
    "Roll dice from notation such as d20 or 3d6+2 and report each roll and the total. "
    "Use when the user asks to roll a die, roll dice, a d20, or NdM notation."
)

SKILL_MD = f"""---
name: {SKILL_NAME}
description: {DICE_DESCRIPTION}
---

# Roll dice

1. Read the request as dice notation `NdM`, with an optional modifier `+K` or `-K`. `N` is the number of dice and `M` is the number of faces. A bare `d20` means `1d20`.
2. Run this skill's script `scripts/roll.py` with that notation. In the sandbox the command is `python3 /skills/roll-dice/scripts/roll.py <notation>`.
3. The script prints JSON with `notation`, `rolls`, `modifier`, and `total`. Report each roll and the total from that JSON.
4. Never invent a number. If the execute tool is missing, say the roll is not possible.
"""

ROLL_PY = '''\
"""Roll NdM(+/-K) and print JSON. Standard library only."""

import json
import re
import secrets
import sys

_NOTATION = re.compile(r"^(\\d*)d(\\d+)([+-]\\d+)?$", re.IGNORECASE)
_MAX_DICE = 100
_MIN_FACES = 2
_MAX_FACES = 1000


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: roll.py NdM(+/-K)", file=sys.stderr)
        return 2
    notation = argv[1].strip()
    match = _NOTATION.fullmatch(notation)
    if match is None:
        print(f"invalid notation: {notation}", file=sys.stderr)
        return 2
    count = int(match.group(1) or "1")
    faces = int(match.group(2))
    modifier = int(match.group(3) or "0")
    if count < 1 or count > _MAX_DICE:
        print(f"dice count must be 1..{_MAX_DICE}", file=sys.stderr)
        return 2
    if faces < _MIN_FACES or faces > _MAX_FACES:
        print(f"faces must be {_MIN_FACES}..{_MAX_FACES}", file=sys.stderr)
        return 2
    rolls = [secrets.randbelow(faces) + 1 for _ in range(count)]
    total = sum(rolls) + modifier
    json.dump(
        {
            "notation": notation,
            "rolls": rolls,
            "modifier": modifier,
            "total": total,
        },
        sys.stdout,
    )
    sys.stdout.write("\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''


def dice_files() -> list[SkillFile]:
    return [
        SkillFile(path="SKILL.md", content=SKILL_MD.encode("utf-8")),
        SkillFile(path="scripts/roll.py", content=ROLL_PY.encode("utf-8")),
    ]


def dice_skill(*, enabled: bool = True) -> Skill:
    """The global roll-dice skill, not yet saved."""
    files = dice_files()
    skill = Skill(
        user_id=None,
        name=SKILL_NAME,
        description=DICE_DESCRIPTION,
        enabled=enabled,
        files=files,
        checksum=skill_checksum(files),
    )
    error = validate_skill(skill)
    if error is not None:
        raise ValueError(f"roll-dice skill is invalid: {error}")
    return skill


async def seed_dice_skill(enabled: bool = True) -> None:
    """Upsert the global roll-dice skill.

    Reuses an already-open connection (the test suite's database) instead of
    reconnecting to the default URI.
    """
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    skill = dice_skill(enabled=enabled)
    existing = await Skill.find_one({"name": SKILL_NAME, "user_id": None})
    if existing:
        existing.enabled = skill.enabled
        existing.description = skill.description
        existing.files = skill.files
        existing.checksum = skill.checksum
        existing.deleted_at = None
        existing.updated_at = datetime.now(timezone.utc)
        await existing.save()
        print(f"Updated skill '{SKILL_NAME}' (enabled={enabled})")
        logger.info("Skill '%s' updated with enabled=%s", SKILL_NAME, enabled)
        return

    created = await Skill.create(
        user_id=None,
        name=skill.name,
        description=skill.description,
        enabled=skill.enabled,
        files=skill.files,
        checksum=skill.checksum,
    )
    print(f"Registered skill '{SKILL_NAME}' (id={created.id}, enabled={enabled})")
    logger.info("Skill '%s' created with id=%s", SKILL_NAME, created.id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert the global roll-dice skill.")
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Store the skill as disabled (enabled=False). Default is enabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(seed_dice_skill(enabled=not args.disable))
