"""End-to-end artifact-ingestion smoke test — no LLM involved.

Exercises the full pipeline against a REAL MCP server (tests/e2e/dummy_mcp_server)
over the network, then verifies serving through the REAL, already-running
backend over real HTTP. Deterministic: no model calls anywhere.

Prerequisites (bring up once, outside pytest):
    docker compose up -d mongo redis minio minio-init dummy-mcp backend

Invocation (run INSIDE the already-running backend container, so
``http://localhost:8000`` reaches that same server; ``dummy-mcp`` resolves via
compose service DNS either way):
    docker compose exec -T backend pytest tests/e2e/test_artifact_e2e.py -v -m e2e

Marked ``e2e`` and self-skipping when the dummy server / backend aren't
reachable, so ``docker compose run --rm backend pytest -q`` (the default unit
suite) still collects this module but skips it rather than failing.

Database note: the default test fixtures in tests/conftest.py pin every test
to a dedicated ``eve_backend_test`` database, isolated from whatever the real
app is using. That isolation is exactly wrong here — this test's Mongo writes
(seeded users, persisted Artifacts) need to land in the SAME database the
live backend process queries, or the real-HTTP assertions in part 4 would
404/403 against data the server can't see. The ``_use_real_database``
fixture below reconnects to the real URI (``get_mongodb_uri()``) after the
autouse test-DB fixture runs, overriding it for this module only.
"""

import asyncio
import json
import os
import socket
import uuid
from urllib.parse import urlparse

import httpx
import pytest
import pytest_asyncio

from src.database.models.artifact import Artifact
from src.database.mongo import async_mongo_manager
from src.services.agents.core.interceptors import ErrorLoggingInterceptor
from src.services.agents.graphs_bundle import graphs_base_module
from src.services.mcp.artifact_context import (
    reset_artifact_context,
    set_artifact_context,
)
from src.services.mcp.artifact_ingestion import ArtifactInterceptor
from src.utils.helpers import get_mongodb_uri
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _mcp_adapters_available = True
except Exception:
    _mcp_adapters_available = False

pytestmark = pytest.mark.e2e

LatencyInterceptor = graphs_base_module().LatencyInterceptor

DUMMY_MCP_HOST = os.environ.get("E2E_DUMMY_MCP_HOST", "dummy-mcp")
DUMMY_MCP_PORT = int(os.environ.get("E2E_DUMMY_MCP_PORT", "8000"))
DUMMY_MCP_URL = f"http://{DUMMY_MCP_HOST}:{DUMMY_MCP_PORT}/mcp"

BACKEND_BASE_URL = os.environ.get("E2E_BACKEND_BASE_URL", "http://localhost:8000")

E2E_PASSWORD = "e2eArtifactPass123"


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_e2e_services():
    """Skip the whole module if the dummy MCP server or backend aren't reachable."""
    if not _mcp_adapters_available:
        pytest.skip("langchain_mcp_adapters not available")
    if not _tcp_reachable(DUMMY_MCP_HOST, DUMMY_MCP_PORT):
        pytest.skip(
            f"dummy-mcp not reachable at {DUMMY_MCP_HOST}:{DUMMY_MCP_PORT} "
            "— bring up with `docker compose up -d dummy-mcp`"
        )
    parsed = urlparse(BACKEND_BASE_URL)
    if not _tcp_reachable(parsed.hostname, parsed.port or 80):
        pytest.skip(
            f"backend not reachable at {BACKEND_BASE_URL} "
            "— bring up with `docker compose up -d backend`, or run this test "
            "via `docker compose exec backend pytest ...` from inside it"
        )


@pytest_asyncio.fixture(autouse=True)
async def _use_real_database():
    """Point this module's Mongo connection at the real app database.

    Runs after tests/conftest.py's autouse `_db_connection` fixture (which pins
    every test to the isolated `eve_backend_test` DB), reconnecting to the same
    database the live, already-running backend process queries.
    """
    await async_mongo_manager.connect(get_mongodb_uri())
    yield


@pytest_asyncio.fixture
async def mcp_client():
    """Build the same interceptor stack runner.py uses, against the dummy server.

    Returns (tools, client): the LangChain tool wrappers don't keep a strong
    reference to the underlying MultiServerMCPClient (confirmed directly: a
    client with no other referrer is garbage-collected as soon as the
    building function returns), and ArtifactInterceptor.bind_client() only
    holds a weakref to it — exactly the same class of bug the T3 test suite
    caught in its own fake-client fixtures. The caller must hold `client`
    alive for as long as it calls tools that touch ResourceLink resolution.
    """
    interceptor = ArtifactInterceptor()
    client = MultiServerMCPClient(
        {"dummy": {"transport": "streamable_http", "url": DUMMY_MCP_URL, "headers": {}}},
        tool_name_prefix=True,
        tool_interceptors=[
            LatencyInterceptor(),
            ErrorLoggingInterceptor(),
            interceptor,
        ],
    )
    interceptor.bind_client(client)
    tools = {t.name: t for t in await client.get_tools(server_name="dummy")}
    return tools, client


def _parse_stub(text: str) -> dict:
    lines = text.split("\n")
    assert len(lines) == 2, f"expected a 2-line stub, got: {text!r}"
    return json.loads(lines[1])


async def _login(client: httpx.AsyncClient, email: str, password: str) -> str:
    resp = await client.post("/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"login failed: {resp.status_code} {resp.text}"
    return resp.json()["access_token"]


async def _activate(user) -> None:
    """Mirror the --test flag in src/commands/create_user.py: skip email verification."""
    user.is_active = True
    user.activation_code = None
    await user.save()


@pytest.mark.asyncio
async def test_artifact_capture_and_serving_e2e(mcp_client):
    tools, _client = mcp_client  # keep `_client` alive: see mcp_client's docstring
    owner, _owner_test_token = await create_test_user_and_token(
        email=f"e2e-owner-{uuid.uuid4().hex[:8]}@example.com", password=E2E_PASSWORD
    )
    intruder, _intruder_test_token = await create_test_user_and_token(
        email=f"e2e-intruder-{uuid.uuid4().hex[:8]}@example.com", password=E2E_PASSWORD
    )
    await _activate(owner)
    await _activate(intruder)
    conversation_id = f"e2e-conv-{uuid.uuid4().hex[:8]}"

    try:
        # ── Call all three tools through the real interceptor stack ──────────
        ctx, token = set_artifact_context(
            user_id=owner.id, conversation_id=conversation_id
        )
        try:
            image_result = await tools["dummy_get_sample_image"].ainvoke(
                {"color": "green"}
            )
            report_result = await tools["dummy_get_sample_report"].ainvoke({})
            text_result = await tools["dummy_get_text_summary"].ainvoke({})
        finally:
            reset_artifact_context(token)

        # (3) text tool → no artifact created, content passed through untouched.
        assert "This is a plain text summary" in str(text_result)

        # Exactly two artifacts were persisted: the image and the CSV.
        assert len(ctx.collected_artifact_ids) == 2
        artifacts = {
            a.id: a
            for a in [
                await Artifact.find_by_id(aid) for aid in ctx.collected_artifact_ids
            ]
        }
        assert all(a is not None for a in artifacts.values())

        image_artifact = next(
            a for a in artifacts.values() if a.content_type == "image/png"
        )
        csv_artifact = next(
            a for a in artifacts.values() if a.content_type == "text/csv"
        )

        # (1) image tool → Artifact doc with source.type=mcp_tool.
        assert image_artifact.source.type == "mcp_tool"
        assert image_artifact.source.mcp_server == "dummy"
        assert image_artifact.source.tool_name == "get_sample_image"
        assert image_artifact.user_id == owner.id
        assert image_artifact.conversation_id == conversation_id
        assert image_artifact.key.startswith(f"users/{owner.id}/artifacts/")

        # (2) resource-link tool → CSV persisted via resources/read.
        assert csv_artifact.source.type == "mcp_tool"
        assert csv_artifact.source.tool_name == "get_sample_report"
        assert csv_artifact.filename == "sample-report.csv"

        # Stub text is well-formed: markdown line + parseable JSON matching the artifact.
        image_stub = _parse_stub(str(image_result[0]["text"]))
        assert image_stub["artifact_id"] == image_artifact.id
        assert image_stub["url"] == f"/artifacts/{image_artifact.id}"
        csv_stub = _parse_stub(str(report_result[0]["text"]))
        assert csv_stub["artifact_id"] == csv_artifact.id

        # Object really present in MinIO under users/{uid}/artifacts/ (independent
        # verification via boto3, not our own storage_service abstraction).
        import boto3
        from botocore.config import Config as BotoConfig

        from src.config import (
            S3_ACCESS_KEY_ID,
            S3_BUCKET_NAME,
            S3_ENDPOINT_URL,
            S3_REGION,
            S3_SECRET_ACCESS_KEY,
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL or None,
            aws_access_key_id=S3_ACCESS_KEY_ID or None,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY or None,
            region_name=S3_REGION or None,
            config=BotoConfig(signature_version="s3v4"),
        )
        listing = s3.list_objects_v2(
            Bucket=S3_BUCKET_NAME, Prefix=f"users/{owner.id}/artifacts/"
        )
        s3_keys = {obj["Key"] for obj in listing.get("Contents", [])}
        assert image_artifact.key in s3_keys
        assert csv_artifact.key in s3_keys
        image_object = s3.get_object(Bucket=S3_BUCKET_NAME, Key=image_artifact.key)
        image_bytes_in_s3 = image_object["Body"].read()
        csv_object = s3.get_object(Bucket=S3_BUCKET_NAME, Key=csv_artifact.key)
        csv_bytes_in_s3 = csv_object["Body"].read()

        # ── (4) Real HTTP against the running backend ────────────────────────
        async with httpx.AsyncClient(base_url=BACKEND_BASE_URL, timeout=10.0) as client:
            owner_token = await _login(client, owner.email, E2E_PASSWORD)
            owner_headers = {"Authorization": f"Bearer {owner_token}"}

            listed = await client.get("/artifacts", headers=owner_headers)
            assert listed.status_code == 200
            listed_ids = {item["id"] for item in listed.json()["data"]}
            assert image_artifact.id in listed_ids
            assert csv_artifact.id in listed_ids

            # Image: byte-identical, inline disposition, image content-type.
            img_resp = await client.get(
                f"/artifacts/{image_artifact.id}", headers=owner_headers
            )
            assert img_resp.status_code == 200
            assert img_resp.content == image_bytes_in_s3
            assert img_resp.headers["content-type"] == "image/png"
            assert img_resp.headers["content-disposition"].startswith("inline")

            # CSV: byte-identical, attachment disposition (text/csv isn't in the
            # inline allowlist), correct content-type.
            csv_resp = await client.get(
                f"/artifacts/{csv_artifact.id}", headers=owner_headers
            )
            assert csv_resp.status_code == 200
            assert csv_resp.content == csv_bytes_in_s3
            # Starlette appends "; charset=utf-8" to text/* media types automatically.
            assert csv_resp.headers["content-type"].startswith("text/csv")
            assert csv_resp.headers["content-disposition"].startswith("attachment")

            # Cross-user: the intruder gets 403 on the owner's artifact.
            intruder_token = await _login(client, intruder.email, E2E_PASSWORD)
            intruder_headers = {"Authorization": f"Bearer {intruder_token}"}
            forbidden = await client.get(
                f"/artifacts/{image_artifact.id}", headers=intruder_headers
            )
            assert forbidden.status_code == 403

    finally:
        await Artifact.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])
