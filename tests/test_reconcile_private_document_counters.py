"""Tests for src.commands.reconcile_private_document_counters."""

from bson import ObjectId

import src.commands.reconcile_private_document_counters as reconcile
from src.database.models.document import Document
from src.database.models.user import User
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


async def _set_counter(user: User, value) -> None:
    users = User.get_collection()
    if value is None:
        await users.update_one(
            {"_id": ObjectId(user.id)}, {"$unset": {"private_document_count": ""}}
        )
    else:
        await users.update_one(
            {"_id": ObjectId(user.id)}, {"$set": {"private_document_count": value}}
        )


async def _counter(user: User):
    row = await User.get_collection().find_one({"_id": ObjectId(user.id)})
    return row.get("private_document_count")


async def _documents(user: User, count: int) -> None:
    for i in range(count):
        await Document.create(user_id=user.id, collection_id="c", name=f"d{i}")


async def _cleanup(*users: User) -> None:
    for user in users:
        await Document.delete_many({"user_id": user.id})
    await cleanup_models(list(users))


async def test_dry_run_reports_and_writes_nothing(capsys):
    user, _ = await create_test_user_and_token()
    try:
        await _documents(user, 2)
        await _set_counter(user, 7)

        remaining = await reconcile.reconcile_private_document_counters(apply=False)

        out = capsys.readouterr().out
        assert f"{user.id}\t7\t2" in out
        assert user.email not in out
        assert "Dry run: nothing written" in out
        assert remaining >= 1
        assert await _counter(user) == 7
    finally:
        await _cleanup(user)


async def test_apply_sets_the_counter_to_the_document_count(capsys):
    user, _ = await create_test_user_and_token()
    try:
        await _documents(user, 3)
        await _set_counter(user, 9)

        remaining = await reconcile.reconcile_private_document_counters(apply=True)

        assert remaining == 0
        assert await _counter(user) == 3
        assert await Document.count_documents({"user_id": user.id}) == 3

        capsys.readouterr()
        await reconcile.reconcile_private_document_counters(apply=False)
        assert user.id not in capsys.readouterr().out
    finally:
        await _cleanup(user)


async def test_correct_and_unset_counters_are_left_alone(capsys):
    right, _ = await create_test_user_and_token()
    unset, _ = await create_test_user_and_token()
    try:
        await _documents(right, 1)
        await _set_counter(right, 1)
        await _documents(unset, 2)
        await _set_counter(unset, None)

        await reconcile.reconcile_private_document_counters(apply=True)

        out = capsys.readouterr().out
        assert right.id not in out
        assert unset.id not in out
        assert await _counter(right) == 1
        assert await _counter(unset) is None
    finally:
        await _cleanup(right, unset)


async def test_a_counter_that_moves_during_the_run_is_skipped(capsys, monkeypatch):
    user, _ = await create_test_user_and_token()
    try:
        await _documents(user, 1)
        await _set_counter(user, 5)
        real_find = reconcile.find_mismatches

        async def find_then_upload(*args, **kwargs):
            found = await real_find(*args, **kwargs)
            # An upload reserves a slot between the read and the write.
            await _set_counter(user, 6)
            return found

        monkeypatch.setattr(reconcile, "find_mismatches", find_then_upload)

        remaining = await reconcile.reconcile_private_document_counters(apply=True)

        assert remaining == 1
        assert await _counter(user) == 6
        assert f"run again: {user.id}" in capsys.readouterr().out
    finally:
        await _cleanup(user)


async def test_documents_of_deleted_users_are_reported_not_counted(capsys):
    ghost_id = str(ObjectId())
    try:
        await Document.create(user_id=ghost_id, collection_id="c", name="orphan")

        await reconcile.reconcile_private_document_counters(apply=False)

        out = capsys.readouterr().out
        assert "belong to users that no longer exist" in out
        assert ghost_id not in out
    finally:
        await Document.delete_many({"user_id": ghost_id})
