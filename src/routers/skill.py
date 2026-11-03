from typing import Any, Dict

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException

from src.database.models.skill import Skill
from src.database.models.user import User
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_current_user
from src.schemas.common import Pagination
from src.schemas.skill import SkillFilePublic, SkillPublic

router = APIRouter()


def _to_public(skill: Skill) -> SkillPublic:
    return SkillPublic(
        id=skill.id,
        timestamp=skill.timestamp,
        name=skill.name,
        description=skill.description,
        enabled=skill.enabled,
        requires_mcp_servers=list(skill.requires_mcp_servers),
        files=[
            SkillFilePublic(path=item.path, size=len(item.content))
            for item in sorted(skill.files, key=lambda item: item.path)
        ],
        created_at=skill.created_at,
        updated_at=skill.updated_at,
        deleted_at=skill.deleted_at,
    )


def _visible_filter(user_id: str) -> Dict[str, Any]:
    """Rows a user may see: global ones and their own, never soft-deleted.

    The same ownership clause as the chat path. ``enabled`` is not filtered:
    the client shows disabled rows greyed out.
    """
    return {
        "deleted_at": None,
        "$or": [{"user_id": user_id}, {"user_id": None}],
    }


def _object_id_or_404(skill_id: str) -> ObjectId:
    try:
        return ObjectId(skill_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=404, detail="Skill not found")


@router.get("/skills", response_model=PaginatedResponse[SkillPublic])
async def list_skills(
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
):
    """List the skills the current user can request: global ones and their own."""
    result = await Skill.find_all_with_pagination(
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
        filter_dict=_visible_filter(requesting_user.id),
    )
    return PaginatedResponse[SkillPublic](
        data=[_to_public(skill) for skill in result.data],
        meta=result.meta,
    )


@router.get("/skills/{skill_id}", response_model=SkillPublic)
async def get_skill(
    skill_id: str,
    requesting_user: User = Depends(get_current_user),
):
    """Get one visible skill. Another user's skill answers 404."""
    skill = await Skill.find_one(
        {"_id": _object_id_or_404(skill_id), **_visible_filter(requesting_user.id)}
    )
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _to_public(skill)
