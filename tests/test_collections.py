import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.collection import Collection


def _stub_vector_methods(monkeypatch):
    """Patch VectorStoreManager methods used by the collection router to no-op."""

    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.create_collection",
        lambda self, name: None,
    )
    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.delete_collection",
        lambda self, name: None,
    )


@pytest.mark.asyncio
async def test_collection_crud(async_client, monkeypatch):
    """End-to-end create → list → update → delete flow."""

    _stub_vector_methods(monkeypatch)

    user, token = await create_test_user_and_token()

    try:
        payload = {"name": "My Collection"}
        create_resp = await async_client.post(
            "/collections",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert create_resp.status_code == 200
        created = create_resp.json()
        coll_id = created["id"]

        list_resp = await async_client.get(
            "/collections", headers={"Authorization": f"Bearer {token}"}
        )
        assert list_resp.status_code == 200
        assert list_resp.json()["data"][0]["id"] == coll_id

        patch_payload = {"name": "Updated Collection"}
        patch_resp = await async_client.patch(
            f"/collections/{coll_id}",
            json=patch_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["name"] == patch_payload["name"]

        del_resp = await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert del_resp.status_code == 200
        assert await Collection.find_by_id(coll_id) is None

    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_collection_not_owner(async_client, monkeypatch):
    _stub_vector_methods(monkeypatch)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Owner Coll"},
                headers={"Authorization": f"Bearer {owner_token}"},
            )
        ).json()["id"]

        resp = await async_client.patch(
            f"/collections/{coll_id}",
            json={"name": "Hacked"},
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403

        await async_client.delete(
            f"/collections/{coll_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
    finally:
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_delete_collection_not_owner(async_client, monkeypatch):
    _stub_vector_methods(monkeypatch)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Owner Coll 2"},
                headers={"Authorization": f"Bearer {owner_token}"},
            )
        ).json()["id"]

        resp = await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {intr_token}"}
        )
        assert resp.status_code == 403

        await async_client.delete(
            f"/collections/{coll_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
    finally:
        await cleanup_models([owner, intruder])
