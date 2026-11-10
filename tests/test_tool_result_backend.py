"""S3 offload of large tool results, and the routes that point at it."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from deepagents.backends import StateBackend

from src.config import TOOL_RESULT_MAX_TOKENS
from src.services.agents.core.sandbox import (
    _rewrite_skills_command,
    backend_for_compile,
    large_tool_result_permissions,
)
from src.services.agents.core.tool_results import (
    GREP_MAX_BYTES,
    GREP_SCAN_LIMIT,
    RESTORE_MAX_BYTES,
    S3ToolResultBackend,
    restore_tool_results,
)
from src.services.storage import storage_service

pytestmark = pytest.mark.no_db

_USER = "user-1"
_CONVERSATION = "conv-1"


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        return None


class FakeS3:
    """In-memory stand-in for the boto3 client the storage service holds."""

    def __init__(self) -> None:
        self.objects: dict = {}
        self.gets = 0
        self._clock = 0

    def put_object(self, **kwargs):
        self._clock += 1
        body = kwargs["Body"]
        if isinstance(body, str):
            body = body.encode()
        self.objects[kwargs["Key"]] = {
            "Body": body,
            "ContentType": kwargs.get("ContentType"),
            "Tagging": kwargs.get("Tagging"),
            "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=self._clock),
        }

    def head_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        return {}

    def put_object_tagging(self, **kwargs):
        item = self.objects[kwargs["Key"]]
        tags = (kwargs.get("Tagging") or {}).get("TagSet") or []
        item["Tagging"] = "&".join(f"{tag['Key']}={tag['Value']}" for tag in tags)

    def get_object(self, **kwargs):
        self.gets += 1
        item = self.objects.get(kwargs["Key"])
        if item is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        return {"Body": _Body(item["Body"])}

    def list_objects_v2(self, **kwargs):
        prefix = kwargs.get("Prefix") or ""
        contents = []
        for key, item in self.objects.items():
            if not key.startswith(prefix):
                continue
            contents.append(
                {
                    "Key": key,
                    "Size": len(item["Body"]),
                    "LastModified": item["LastModified"],
                }
            )
        return {"Contents": contents, "IsTruncated": False}


class _Sandbox:
    def __init__(self, error: Exception | None = None) -> None:
        self.uploads: list = []
        self._error = error

    def upload_files(self, files):
        if self._error is not None:
            raise self._error
        self.uploads.append(list(files))
        return [SimpleNamespace(path=path, error=None) for path, _content in files]


@pytest.fixture
def fake_s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(storage_service, "_s3", fake)
    return fake


def _backend(sandbox=None) -> S3ToolResultBackend:
    return S3ToolResultBackend(_USER, _CONVERSATION, sandbox)


def _key(name: str) -> str:
    return (
        f"users/{_USER}/conversations/{_CONVERSATION}/large_tool_results/{name}"
    )


def test_write_then_paged_read_returns_the_requested_lines(fake_s3):
    backend = _backend()
    text = "\n".join(f"line-{index}" for index in range(6))

    written = backend.write("/call-1", text)
    page = backend.read("/call-1", offset=2, limit=2)

    assert written.error is None
    assert written.path == "/call-1"
    assert page.error is None
    assert page.file_data["content"] == "line-2\nline-3\n"
    assert page.start_line == 3
    assert page.end_line == 4
    assert fake_s3.gets == 0


async def test_paging_one_file_costs_one_get(fake_s3):
    text = "\n".join(f"line-{index}" for index in range(4))
    fake_s3.objects[_key("call-1")] = {
        "Body": text.encode(),
        "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    backend = _backend()

    first = await backend.aread("/call-1", offset=0, limit=1)
    second = await backend.aread("/call-1", offset=1, limit=1)

    assert first.file_data["content"] == "line-0\n"
    assert second.file_data["content"] == "line-1\n"
    assert second.start_line == 2
    assert fake_s3.gets == 1


def test_a_missing_file_uses_the_state_backend_error(fake_s3):
    missing = _backend().read("/missing")

    assert missing.error == "File '/large_tool_results/missing' not found"


def test_grep_reports_hits_and_stops_at_the_scan_cap(fake_s3, monkeypatch):
    backend = _backend()
    backend.write("/old", "hit\n")
    backend.write("/new", "hit\nmiss\n")

    found = backend.grep("hit")
    assert {(match["path"], match["line"]) for match in found.matches} == {
        ("/old", 1),
        ("/new", 1),
    }
    assert found.truncated is False

    monkeypatch.setattr(
        "src.services.agents.core.tool_results.GREP_SCAN_LIMIT", 1
    )
    capped = backend.grep("hit")

    assert capped.truncated is True
    assert [(match["path"], match["line"]) for match in capped.matches] == [
        ("/new", 1)
    ]
    assert GREP_SCAN_LIMIT == 50


def test_grep_stops_before_a_get_past_the_byte_cap(fake_s3, monkeypatch):
    monkeypatch.setattr(
        "src.services.agents.core.tool_results.GREP_MAX_BYTES", 4
    )
    older = datetime(2026, 1, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 1, 2, tzinfo=timezone.utc)
    fake_s3.objects[_key("old")] = {"Body": b"hit\n", "LastModified": older}
    fake_s3.objects[_key("new")] = {"Body": b"hit\n", "LastModified": newer}

    found = _backend().grep("hit")

    assert found.truncated is True
    assert [(match["path"], match["line"]) for match in found.matches] == [
        ("/new", 1)
    ]
    assert fake_s3.gets == 1
    assert GREP_MAX_BYTES == 32 * 1024 * 1024


def test_edit_and_delete_are_refused(fake_s3):
    backend = _backend()
    backend.write("/call-1", "hello")

    edited = backend.edit("/call-1", "hello", "goodbye")
    deleted = backend.delete("/call-1")

    assert edited.error == "Error: offloaded tool results are read-only"
    assert deleted.error == "Error: offloaded tool results are read-only"
    assert fake_s3.objects[_key("call-1")]["Body"] == b"hello"


async def test_async_edit_and_delete_are_refused(fake_s3):
    backend = _backend()

    edited = await backend.aedit("/call-1", "hello", "goodbye")
    deleted = await backend.adelete("/call-1")

    assert edited.error == "Error: offloaded tool results are read-only"
    assert deleted.error == "Error: offloaded tool results are read-only"


def test_the_object_key_has_no_extension_and_carries_the_tag(fake_s3):
    written = _backend().write("/call-1", "hello")

    stored = fake_s3.objects[_key("call-1")]
    assert written.path == "/call-1"
    assert stored["Body"] == b"hello"
    assert stored["Tagging"] == "kind=tool-result"
    assert "." not in _key("call-1").rsplit("/", 1)[-1]


def test_a_write_mirrors_into_the_sandbox(fake_s3):
    sandbox = _Sandbox()

    written = _backend(sandbox).write("/call-1", "hello")

    assert written.error is None
    assert sandbox.uploads == [[("/large_tool_results/call-1", b"hello")]]


def test_a_sandbox_upload_failure_still_stores_the_object(fake_s3):
    written = _backend(_Sandbox(error=RuntimeError("disk full"))).write(
        "/call-1", "hello"
    )

    assert written.error is None
    assert written.path == "/call-1"
    assert fake_s3.objects[_key("call-1")]["Body"] == b"hello"


def test_write_does_not_replace_an_existing_object(fake_s3):
    sandbox = _Sandbox()
    backend = _backend(sandbox)

    first = backend.write("/call-1", "hello")
    blocked = backend.write("/call-1", "goodbye")
    fresh = backend.write("/call-2", "world")

    assert first.error is None
    assert blocked.error == "Error: tool result '/large_tool_results/call-1' already exists"
    assert fresh.error is None
    assert fake_s3.objects[_key("call-1")]["Body"] == b"hello"
    assert fake_s3.objects[_key("call-2")]["Body"] == b"world"
    assert sandbox.uploads == [
        [("/large_tool_results/call-1", b"hello")],
        [("/large_tool_results/call-2", b"world")],
    ]


async def test_awrite_does_not_replace_an_existing_object(fake_s3):
    backend = _backend(_Sandbox())

    assert (await backend.awrite("/call-1", "hello")).error is None
    blocked = await backend.awrite("/call-1", "goodbye")

    assert blocked.error == "Error: tool result '/large_tool_results/call-1' already exists"
    assert fake_s3.objects[_key("call-1")]["Body"] == b"hello"
    assert backend.sandbox_backend.uploads == [
        [("/large_tool_results/call-1", b"hello")]
    ]


def test_a_rejected_tag_still_stores_the_object(fake_s3):
    def boom(**_kwargs):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no tags"}},
            "PutObjectTagging",
        )

    fake_s3.put_object_tagging = boom

    written = _backend().write("/call-1", "hello")

    assert written.error is None
    assert fake_s3.objects[_key("call-1")]["Body"] == b"hello"


def test_an_s3_failure_returns_an_error_and_does_not_mirror(fake_s3):
    def boom(**_kwargs):
        raise RuntimeError("s3 down")

    fake_s3.put_object = boom
    sandbox = _Sandbox()

    written = _backend(sandbox).write("/call-1", "hello")

    assert written.error
    assert "s3 down" in written.error
    assert sandbox.uploads == []


async def test_restore_uploads_newest_objects_until_the_byte_cap(
    fake_s3, monkeypatch
):
    backend = _backend()
    backend.write("/old", "aaaa")
    backend.write("/new", "bbbb")
    monkeypatch.setattr(
        "src.services.agents.core.tool_results.RESTORE_MAX_BYTES", 4
    )
    sandbox = _Sandbox()

    await restore_tool_results(sandbox, _USER, _CONVERSATION)

    assert sandbox.uploads == [[("/large_tool_results/new", b"bbbb")]]
    assert RESTORE_MAX_BYTES == 32 * 1024 * 1024


def test_listing_sorts_by_time_across_pages(fake_s3):
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 6, 1, tzinfo=timezone.utc)
    prefix = _key("").rsplit("/", 1)[0] + "/"
    pages = [
        {
            "Contents": [
                {"Key": _key("a-old"), "Size": 1, "LastModified": old}
            ],
            "IsTruncated": True,
            "NextContinuationToken": "next",
        },
        {
            "Contents": [
                {"Key": _key("z-new"), "Size": 1, "LastModified": new}
            ],
            "IsTruncated": False,
        },
    ]

    def list_objects_v2(**kwargs):
        if kwargs.get("ContinuationToken"):
            return pages[1]
        return pages[0]

    fake_s3.list_objects_v2 = list_objects_v2

    records, truncated = storage_service.list_key_records(prefix, 1, scan_limit=10)

    assert truncated is True
    assert [record["key"] for record in records] == [_key("z-new")]


async def test_restore_logs_a_truncated_listing(fake_s3, monkeypatch, caplog):
    backend = _backend()
    backend.write("/old", "aaaa")
    backend.write("/new", "bbbb")
    monkeypatch.setattr(
        "src.services.agents.core.tool_results.RESTORE_MAX_OBJECTS", 1
    )

    with caplog.at_level("INFO"):
        await restore_tool_results(_Sandbox(), _USER, _CONVERSATION)

    assert "tool_results.restore_truncated" in caplog.text


def test_large_tool_results_route_to_s3_and_other_paths_do_not(fake_s3):
    sandbox = object()
    tool_results = _backend()
    composite = backend_for_compile(sandbox, tool_results=tool_results)

    routed, path = composite._get_backend_and_key("/large_tool_results/x")
    other, other_path = composite._get_backend_and_key("/tmp/a")
    history, _history_path = composite._get_backend_and_key(
        "/conversation_history/old"
    )

    assert routed is tool_results
    assert path == "/x"
    missing = composite.read("/large_tool_results/missing")
    assert missing.error == "File '/large_tool_results/missing' not found"
    assert other is sandbox
    assert other_path == "/tmp/a"
    assert isinstance(history, StateBackend)


def test_shell_rewrite_points_large_tool_results_at_the_session_cwd():
    command = "cat /large_tool_results/call-1 /skills/eo/SKILL.md"

    assert _rewrite_skills_command(command, "/opt/sandbox") == (
        "cat /opt/sandbox/large_tool_results/call-1 "
        "/opt/sandbox/skills/eo/SKILL.md"
    )
    assert (
        _rewrite_skills_command(
            "cat /opt/sandbox/large_tool_results/call-1", "/opt/sandbox"
        )
        == "cat /opt/sandbox/large_tool_results/call-1"
    )
    assert (
        _rewrite_skills_command("cat /large_tool_results/call-1", "/")
        == "cat /large_tool_results/call-1"
    )
    assert (
        _rewrite_skills_command("cat /large_tool_results/skills_1", "/opt/sandbox")
        == "cat /opt/sandbox/large_tool_results/skills_1"
    )


class _CompileAgent:
    """``compile`` declares ``backend`` and ``skills``, not the token limit."""

    name = "deep"

    def __init__(self) -> None:
        self.kwargs = None

    def compile(
        self,
        llm,
        tools,
        checkpointer,
        history=None,
        summary=None,
        fallback_llm=None,
        llm_run_timeout=None,
        llm_idle_timeout=None,
        on_policy=None,
        backend=None,
        permissions=None,
        skills=None,
    ):
        self.kwargs = {
            "backend": backend,
            "permissions": permissions,
            "skills": skills,
        }
        return "graph"


class _LimitAgent(_CompileAgent):
    def compile(
        self,
        llm,
        tools,
        checkpointer,
        history=None,
        summary=None,
        fallback_llm=None,
        llm_run_timeout=None,
        llm_idle_timeout=None,
        on_policy=None,
        backend=None,
        permissions=None,
        skills=None,
        tool_result_max_tokens=None,
    ):
        super().compile(
            llm,
            tools,
            checkpointer,
            history=history,
            summary=summary,
            fallback_llm=fallback_llm,
            llm_run_timeout=llm_run_timeout,
            llm_idle_timeout=llm_idle_timeout,
            on_policy=on_policy,
            backend=backend,
            permissions=permissions,
            skills=skills,
        )
        self.kwargs["tool_result_max_tokens"] = tool_result_max_tokens
        return "graph"


def _compile(tmp_path, *, conversation_id, agent):
    from src.services.agents.core.runner import _build_react_graph

    _build_react_graph(
        None,
        [],
        None,
        agent=agent,
        llm=object(),
        fallback_llm=object(),
        on_policy=lambda **_kwargs: None,
        skills_root=tmp_path,
        user_id=_USER,
        conversation_id=conversation_id,
    )
    return agent.kwargs


def test_no_sandbox_routes_large_results_to_s3_and_keeps_skills(tmp_path):
    kwargs = _compile(tmp_path, conversation_id=_CONVERSATION, agent=_CompileAgent())
    backend = kwargs["backend"]

    routed, path = backend._get_backend_and_key("/large_tool_results/x")
    skills, skills_path = backend._get_backend_and_key("/skills/eo/SKILL.md")
    other, _other_path = backend._get_backend_and_key("/tmp/a")

    assert isinstance(routed, S3ToolResultBackend)
    assert path == "/x"
    assert not isinstance(skills, S3ToolResultBackend)
    assert skills_path == "/eo/SKILL.md"
    assert isinstance(other, StateBackend)
    assert kwargs["permissions"]
    assert _denies_large_writes(kwargs["permissions"])
    assert any(
        rule.mode == "deny" and "/skills/**" in rule.paths
        for rule in kwargs["permissions"]
    )
    assert kwargs["skills"] == ["/skills/"]
    assert "tool_result_max_tokens" not in kwargs


def _denies_large_writes(permissions) -> bool:
    return any(
        rule.mode == "deny"
        and "write" in rule.operations
        and "/large_tool_results/**" in rule.paths
        for rule in permissions or []
    )


def test_no_conversation_keeps_large_results_on_the_default_backend(tmp_path):
    kwargs = _compile(tmp_path, conversation_id=None, agent=_CompileAgent())
    backend = kwargs["backend"]
    routed, path = backend._get_backend_and_key("/large_tool_results/x")

    assert "/large_tool_results/" not in backend.routes
    assert routed is backend.default
    assert isinstance(routed, StateBackend)
    assert path == "/large_tool_results/x"
    assert not _denies_large_writes(kwargs["permissions"])


def test_sandbox_without_conversation_leaves_the_path_on_the_sandbox(tmp_path):
    from src.services.agents.core.runner import _build_react_graph

    agent = _CompileAgent()
    sandbox = object()
    _build_react_graph(
        None,
        [],
        None,
        agent=agent,
        llm=object(),
        fallback_llm=object(),
        on_policy=lambda **_kwargs: None,
        sandbox_backend=sandbox,
        skills_root=tmp_path,
        user_id=_USER,
        conversation_id=None,
    )
    backend = agent.kwargs["backend"]
    routed, path = backend._get_backend_and_key("/large_tool_results/call-1")

    assert "/large_tool_results/" not in backend.routes
    assert routed is sandbox
    assert path == "/large_tool_results/call-1"
    assert agent.kwargs["permissions"] is None


def test_sandbox_composite_denies_writes_on_the_offload_route(tmp_path):
    from deepagents.middleware.filesystem import _all_paths_scoped_to_routes

    from src.services.agents.core.runner import _build_react_graph

    agent = _CompileAgent()
    sandbox = object()
    _build_react_graph(
        None,
        [],
        None,
        agent=agent,
        llm=object(),
        fallback_llm=object(),
        on_policy=lambda **_kwargs: None,
        sandbox_backend=sandbox,
        skills_root=tmp_path,
        user_id=_USER,
        conversation_id=_CONVERSATION,
    )
    backend = agent.kwargs["backend"]
    permissions = agent.kwargs["permissions"]

    assert "/large_tool_results/" in backend.routes
    assert permissions == large_tool_result_permissions()
    assert _all_paths_scoped_to_routes(permissions, backend)
    assert _denies_large_writes(permissions)


def test_the_token_limit_is_passed_only_when_compile_declares_it(tmp_path):
    undeclared = _compile(tmp_path, conversation_id=None, agent=_CompileAgent())
    declared = _compile(tmp_path, conversation_id=None, agent=_LimitAgent())

    assert "tool_result_max_tokens" not in undeclared
    assert declared["tool_result_max_tokens"] == TOOL_RESULT_MAX_TOKENS
