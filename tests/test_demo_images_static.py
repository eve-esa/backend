import yaml
import pytest

from src import config


@pytest.mark.asyncio
async def test_demo_image_is_served(async_client):
    resp = await async_client.get("/demo-images/sentinel2-satellite.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert len(resp.content) > 1000


@pytest.mark.asyncio
async def test_unknown_demo_image_404s(async_client):
    resp = await async_client.get("/demo-images/nope.jpg")
    assert resp.status_code == 404


@pytest.mark.no_db
def test_catalog_urls_are_relative_and_served_files_exist():
    """Every catalog URL must be a /demo-images path backed by a real file,
    so the catalog works unchanged on any environment (no absolute hosts)."""
    with open(config.IMAGE_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    images = data["images"]
    assert images, "catalog must not be empty"
    import os

    static_dir = os.path.join(
        os.path.dirname(config.IMAGE_CATALOG_PATH), "demo_images"
    )
    for entry in images:
        url = entry["url"]
        assert url.startswith("/demo-images/"), url
        assert os.path.isfile(
            os.path.join(static_dir, url.removeprefix("/demo-images/"))
        ), f"missing file for {url}"
