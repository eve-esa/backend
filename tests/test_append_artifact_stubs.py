"""Unit tests for runner.rewrite_answer_file_links.

The model often wraps an artifact id in a host or an old sandbox prefix.
The runner rewrites that markdown in place. It does not append artifacts
the answer never mentions.
"""

import logging

import pytest

from src.services.agents.core.runner import (
    _ARTIFACT_MARKDOWN_INSTRUCTION,
    rewrite_answer_file_links,
)

pytestmark = pytest.mark.no_db


@pytest.mark.asyncio
async def test_an_unreferenced_artifact_is_not_appended():
    original = "Here is an invented URL: https://storage.googleapis.com/x.png"
    answer = await rewrite_answer_file_links(original)

    assert answer == original
    assert "/artifacts/" not in answer


@pytest.mark.asyncio
async def test_an_already_loadable_link_is_left_as_written(caplog):
    artifact_id = "6af73630e81caeb4f8d8bffb"
    original = (
        f'Already embedded: ![sample.png](/artifacts/{artifact_id} "MCP: dummy/tool")\n'
        f"![sample.png](/artifacts/{artifact_id})"
    )
    answer = await rewrite_answer_file_links(original)

    assert answer == original
    assert "artifact.markdown_repaired" not in caplog.text


@pytest.mark.asyncio
async def test_empty_answer_is_untouched():
    assert await rewrite_answer_file_links("") == ""
    assert await rewrite_answer_file_links("text") == "text"


@pytest.mark.no_db
def test_the_instruction_says_to_embed_the_tool_path():
    assert "/artifacts/{id}/{filename}" in _ARTIFACT_MARKDOWN_INSTRUCTION
    # Naming a host, even as a "never", gave the model one to copy (U7 on
    # dev, 2026-11-18).
    assert "googleapis" not in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "Never put a scheme or a host in front of a path." in (
        _ARTIFACT_MARKDOWN_INSTRUCTION
    )
    assert "workspace/" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "/skills/" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "/large_tool_results/" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "There is no display tool. Do not call display." in (
        _ARTIFACT_MARKDOWN_INSTRUCTION
    )


@pytest.mark.asyncio
async def test_a_sandbox_image_path_becomes_the_url_the_chat_loads():
    artifact_id = "6af73630e81caeb4f8d8bffb"

    answer = await rewrite_answer_file_links(
        "### Severity Map\n"
        f"![Severity Map](/tool_artifacts/{artifact_id}/severity_map.png)\n"
    )

    assert answer == (
        "### Severity Map\n" f"![Severity Map](/artifacts/{artifact_id})\n"
    )


@pytest.mark.asyncio
async def test_a_sandbox_file_link_is_rewritten():
    artifact_id = "6af73630e81caeb4f8d8bffd"

    answer = await rewrite_answer_file_links(
        f"[metrics.json](/tool_artifacts/{artifact_id}/metrics.json)"
    )

    assert answer == f"[metrics.json](/artifacts/{artifact_id})"


@pytest.mark.asyncio
async def test_a_google_host_around_the_artifact_id_is_rewritten(caplog):
    caplog.set_level(logging.INFO)
    artifact_id = "6afb1ab73daca14aba4f887f"

    answer = await rewrite_answer_file_links(
        "![Severity Map]"
        f"(https://storage.googleapis.com/phi-lab-toolkits-public-010/artifacts/{artifact_id}/severity_map.png)\n"
        "*Figure 1: Severity map of the burned area.*\n"
    )

    assert answer == (
        f"![Severity Map](/artifacts/{artifact_id})\n"
        "*Figure 1: Severity map of the burned area.*\n"
    )
    assert "artifact.markdown_repaired" in caplog.text
    assert "storage.googleapis.com" in caplog.text
    assert f"to=/artifacts/{artifact_id}" in caplog.text
    assert not _warnings(caplog, "markdown_repaired")


def _warnings(caplog, needle):
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and needle in record.getMessage()
    ]


# A tool image returned in this turn, then linked by its bare filename.
# Found live on dev (dummy_image, 2026-11-18): the model wrote
# ``![get_sample_image-1.png](get_sample_image-1.png)`` and the browser asked
# ``/api/get_sample_image-1.png``.

_TURN_ARTIFACT_ID = "6afdf46fa8ffdf3d917139e5"


class _Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data

    def close(self):
        return None


class _File:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error


class _WorkspaceSandbox:
    """Answers downloads from ``workspace/``; anything else is file_not_found."""

    def __init__(self, files):
        self.files = dict(files)

    def download_files(self, paths):
        found = []
        for path in paths:
            key = path[len("workspace/") :] if path.startswith("workspace/") else None
            if key in self.files:
                found.append(_File(self.files[key]))
            else:
                found.append(_File(error="file_not_found"))
        return found


class _Store:
    def __init__(self):
        self.objects: dict = {}

    async def put_object(self, key, body, content_type, tagging=None):
        self.objects[key] = body

    async def get_object(self, key):
        from src.services.storage import ObjectNotFoundError

        if key not in self.objects:
            raise ObjectNotFoundError(key)
        return {"Body": _Body(self.objects[key])}


@pytest.fixture
def turn_context(monkeypatch):
    from src.services.agents.core import workspace
    from src.services.mcp.artifact_context import (
        reset_artifact_context,
        set_artifact_context,
    )

    store = _Store()
    monkeypatch.setattr(workspace, "storage_service", store)
    ctx, token = set_artifact_context("u1", "c1")
    try:
        ctx.collected_artifact_ids.append(_TURN_ARTIFACT_ID)
        ctx.collected_artifact_files["get_sample_image-1.png"] = _TURN_ARTIFACT_ID
        ctx.collected_artifact_types[_TURN_ARTIFACT_ID] = "image/png"
        ctx.sandbox_backend = _WorkspaceSandbox(
            {"ndvi.png": b"\x89PNG\r\n\x1a\nndvi"}
        )
        yield ctx, store
    finally:
        reset_artifact_context(token)


@pytest.mark.asyncio
async def test_a_bare_filename_of_a_tool_image_becomes_its_artifact_url(
    turn_context, caplog
):
    caplog.set_level(logging.INFO)
    answer = await rewrite_answer_file_links(
        "Here is the sample blue image:\n"
        "![get_sample_image-1.png](get_sample_image-1.png)\n"
    )

    assert answer == (
        "Here is the sample blue image:\n"
        f"![get_sample_image-1.png](/artifacts/{_TURN_ARTIFACT_ID})\n"
    )
    assert f"to=/artifacts/{_TURN_ARTIFACT_ID}" in caplog.text
    assert "workspace.markdown_unreadable" not in caplog.text
    assert not _warnings(caplog, "markdown_repaired")


@pytest.mark.asyncio
async def test_an_artifacts_path_without_the_id_becomes_its_artifact_url(
    turn_context,
):
    answer = await rewrite_answer_file_links(
        "![img](/artifacts/get_sample_image-1.png) [img](./get_sample_image-1.png)"
    )

    assert answer == (
        # A plain link to a tool image is drawn as an image.
        f"![img](/artifacts/{_TURN_ARTIFACT_ID}) ![img](/artifacts/{_TURN_ARTIFACT_ID})"
    )


@pytest.mark.asyncio
async def test_a_workspace_file_link_still_points_at_the_workspace(
    turn_context, caplog
):
    caplog.set_level(logging.INFO)
    answer = await rewrite_answer_file_links("![ndvi](ndvi.png)")

    assert answer == "![ndvi](/conversations/c1/workspace/ndvi.png)"
    # A repair is expected work: INFO, never a warning in the release gate.
    assert "workspace.markdown_repaired" in caplog.text
    assert not _warnings(caplog, "markdown_repaired")


@pytest.mark.asyncio
async def test_the_real_agentcore_workspace_path_points_at_the_workspace(turn_context):
    # Tracebacks and print output show the model the real AgentCore path.
    answer = await rewrite_answer_file_links(
        "![x](/opt/amazon/genesis1p-tools/var/workspace/ndvi.png)"
    )

    assert answer == "![x](/conversations/c1/workspace/ndvi.png)"


@pytest.mark.asyncio
async def test_an_explicit_workspace_path_is_not_taken_for_the_tool_image(
    turn_context,
):
    _, store = turn_context
    store.objects["users/u1/conversations/c1/workspace/get_sample_image-1.png"] = (
        b"\x89PNG\r\n\x1a\nmine"
    )

    answer = await rewrite_answer_file_links(
        "![mine](/workspace/get_sample_image-1.png)"
    )

    assert answer == (
        "![mine](/conversations/c1/workspace/get_sample_image-1.png)"
    )


@pytest.mark.asyncio
async def test_an_unknown_bare_filename_stays_and_warns(turn_context, caplog):
    original = "![other](other.png)"

    answer = await rewrite_answer_file_links(original)

    assert answer == original
    warnings = [
        record
        for record in caplog.records
        if record.levelname == "WARNING"
        and "workspace.markdown_unreadable" in record.getMessage()
    ]
    assert warnings


@pytest.mark.asyncio
async def test_an_external_url_with_the_same_filename_is_left_alone(turn_context):
    original = "![x](https://example.org/img/get_sample_image-1.png)"

    assert await rewrite_answer_file_links(original) == original


@pytest.mark.asyncio
async def test_a_tool_file_of_this_turn_wins_over_a_workspace_file_of_the_same_name(
    turn_context,
):
    """Pinned: the bare name a tool returned in this turn is the exact match
    and costs no sandbox read; the workspace file stays reachable through
    ``/workspace/{name}``."""
    ctx, store = turn_context
    ctx.sandbox_backend.files["get_sample_image-1.png"] = b"\x89PNG\r\n\x1a\nmine"
    store.objects["users/u1/conversations/c1/workspace/get_sample_image-1.png"] = (
        b"\x89PNG\r\n\x1a\nmine"
    )

    answer = await rewrite_answer_file_links(
        "![tool](get_sample_image-1.png) ![mine](/workspace/get_sample_image-1.png)"
    )

    assert answer == (
        f"![tool](/artifacts/{_TURN_ARTIFACT_ID}) "
        "![mine](/conversations/c1/workspace/get_sample_image-1.png)"
    )


@pytest.mark.asyncio
async def test_links_inside_code_are_left_as_written(turn_context):
    host_link = (
        "![m](https://storage.googleapis.com/b/artifacts/"
        f"{_TURN_ARTIFACT_ID}/m.png)"
    )
    original = (
        "Write `![a](get_sample_image-1.png)` like this:\n"
        "```markdown\n"
        f"![b](ndvi.png) {host_link}\n"
        "```\n"
        "~~~\n![c](get_sample_image-1.png)\n~~~\n"
        "``![d](ndvi.png)`` and here it is: ![e](get_sample_image-1.png) "
        "and ![f](ndvi.png)"
    )

    answer = await rewrite_answer_file_links(original)

    assert answer == original.replace(
        "![e](get_sample_image-1.png)", f"![e](/artifacts/{_TURN_ARTIFACT_ID})"
    ).replace("and ![f](ndvi.png)", "and ![f](/conversations/c1/workspace/ndvi.png)")


@pytest.mark.asyncio
async def test_citation_offsets_survive_the_three_rewrite_passes(turn_context):
    from src.services.sources.citations import (
        CitationRewriter,
        SourceRegistry,
        marker_spans,
    )

    registry = SourceRegistry()
    registry.register({"id": "src_a", "document_key": "doi:a", "title": "A"})
    rewriter = CitationRewriter(registry)
    streamed = (
        "Map [S1]. "
        "![m](https://storage.googleapis.com/b/artifacts/"
        f"{_TURN_ARTIFACT_ID}/m.png) fact [S1]. "
        "`![x](ndvi.png)` [S1] "
        "![s](get_sample_image-1.png) again [S1]. "
        "![n](ndvi.png) last [S1]."
    )
    text = rewriter.feed(streamed) + rewriter.flush()
    assert text.count("[1]") == 5

    passes: list = []
    answer = await rewrite_answer_file_links(text, passes)

    assert len(passes) == 3 and all(passes)
    assert rewriter.remap(answer, passes) == 0
    assert marker_spans(answer, rewriter.citations()) == ["[1]"] * 5


# U7 on dev, 2026-11-18 20:45 UTC (dummy_image): the model wrapped the id of
# the image it had just received in an invented bucket URL.
_U7_ARTIFACT_ID = "6afe0e94f584dfa22d6ed15d"
_U7_ANSWER = (
    "![get_sample_image-1.png](https://storage.googleapis.com/"
    f"phi-lab-ai-platform-artifacts/{_U7_ARTIFACT_ID}/get_sample_image-1.png)"
)


@pytest.fixture
def u7_turn(turn_context):
    ctx, store = turn_context
    ctx.collected_artifact_ids.append(_U7_ARTIFACT_ID)
    ctx.collected_artifact_files["get_sample_image-1.png"] = _U7_ARTIFACT_ID
    ctx.collected_artifact_types[_U7_ARTIFACT_ID] = "image/png"
    return ctx, store


@pytest.mark.asyncio
async def test_an_invented_host_around_an_artifact_of_this_turn_is_rewritten(
    u7_turn, caplog
):
    caplog.set_level(logging.INFO)

    answer = await rewrite_answer_file_links(_U7_ANSWER)

    assert answer == f"![get_sample_image-1.png](/artifacts/{_U7_ARTIFACT_ID})"
    assert f"reason=turn_id to=/artifacts/{_U7_ARTIFACT_ID}" in caplog.text
    assert not _warnings(caplog, "markdown")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        f"http://example.org/{_U7_ARTIFACT_ID}.png",
        f"https://cdn.example.org/a/b/{_U7_ARTIFACT_ID}",
        f"//host/x/{_U7_ARTIFACT_ID}/y.png?sig=1",
        f"gs://bucket/{_U7_ARTIFACT_ID}/get_sample_image-1.png",
    ],
)
async def test_any_scheme_and_host_around_a_turn_artifact_id_is_rewritten(
    u7_turn, target
):
    answer = await rewrite_answer_file_links(f"see [file]({target}) here")

    # The artifact is an image, so the plain link is drawn as one.
    assert answer == f"see ![file](/artifacts/{_U7_ARTIFACT_ID}) here"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original",
    [
        # An id the turn never captured.
        _U7_ANSWER.replace(_U7_ARTIFACT_ID, "6afe0e94f584dfa22d6ed15e"),
        # The id inside a longer hex run is not that id.
        f"![x](https://example.org/{_U7_ARTIFACT_ID}ff/a.png)",
        # Quoted as code.
        f"`{_U7_ANSWER}`",
    ],
)
async def test_a_link_without_a_captured_id_is_left_alone(u7_turn, original):
    assert await rewrite_answer_file_links(original) == original


@pytest.mark.asyncio
async def test_the_invented_host_stays_when_the_turn_captured_nothing():
    assert await rewrite_answer_file_links(_U7_ANSWER) == _U7_ANSWER


@pytest.mark.no_db
def test_the_instruction_forbids_guessing_an_image_the_model_cannot_view():
    # U8 on dev, 2026-11-18: the read notice said the image was not attached and
    # the model still described it from the file name. Dev, 2026-11-19
    # (conversation 6afeb0196ad76994fa4768aa): with a tool on, the model never
    # read the upload and described it from its name, so the rule no longer
    # waits for a read notice.
    assert (
        "You see an image only when it reaches you as image content."
        in _ARTIFACT_MARKDOWN_INSTRUCTION
    )
    assert (
        "Never describe an image you have not seen and never infer its content "
        "from its file name, its path or the user's words: say you cannot see it "
        "with the current model." in _ARTIFACT_MARKDOWN_INSTRUCTION
    )
    assert "reading it returns a notice" not in _ARTIFACT_MARKDOWN_INSTRUCTION


@pytest.mark.no_db
def test_the_instruction_shows_made_images_without_reading_them():
    # Dev, 2026-11-19: after #393 the model read ndvi.png, got the cannot-view
    # notice and answered with a plain link and "I cannot display the image".
    assert "Showing an image needs no vision." in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert (
        "To show an image you made or a tool returned, do not read it: write "
        "`![ndvi.png](ndvi.png)`" in _ARTIFACT_MARKDOWN_INSTRUCTION
    )
    # Scoped: a blanket "never say you cannot display a file" next to the U8
    # rule may lead the model to show a user's upload back and caption it.
    assert (
        "Never say you cannot display an image you made or a tool returned."
        in _ARTIFACT_MARKDOWN_INSTRUCTION
    )
    assert "Never say you cannot display a file." not in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "you can still show the image" not in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "\u2014" not in _ARTIFACT_MARKDOWN_INSTRUCTION


@pytest.mark.no_db
def test_the_u8_sentences_are_kept_word_for_word():
    # Updated on purpose on 2026-11-19: the rule is unconditional now.
    assert (
        "You see an image only when it reaches you as image content. A file "
        "name, a path, a listing, a tool result or the user's words are not the "
        "image. Never describe an image you have not seen and never infer its "
        "content from its file name, its path or the user's words: say you "
        "cannot see it with the current model." in _ARTIFACT_MARKDOWN_INSTRUCTION
    )


# Dev, 2026-11-19 (conversations 6afea68dfaf70660d4dad8e1 and two more): the
# chart was saved, but a plain link draws nothing.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    ["/conversations/c1/workspace/ndvi.png", "ndvi.png", "/workspace/ndvi.png"],
)
async def test_a_plain_link_to_a_workspace_image_becomes_an_image(
    turn_context, target
):
    answer = await rewrite_answer_file_links(
        f"See [Download the NDVI plot]({target}) [S1]."
    )

    assert answer == (
        "See ![Download the NDVI plot](/conversations/c1/workspace/ndvi.png) [S1]."
    )


@pytest.mark.asyncio
async def test_a_workspace_image_without_an_extension_is_typed_by_its_bytes(
    turn_context,
):
    ctx, _ = turn_context
    ctx.sandbox_backend.files["plot"] = b"\x89PNG\r\n\x1a\nplot"

    answer = await rewrite_answer_file_links("[plot](plot)")

    assert answer == "![plot](/conversations/c1/workspace/plot)"


@pytest.mark.asyncio
async def test_a_plain_link_to_a_workspace_csv_stays_a_link(turn_context):
    ctx, _ = turn_context
    ctx.sandbox_backend.files["ndvi.csv"] = b"nir,ndvi\n0.1,0.0\n"

    answer = await rewrite_answer_file_links("[ndvi.csv](ndvi.csv)")

    assert answer == "[ndvi.csv](/conversations/c1/workspace/ndvi.csv)"


@pytest.mark.asyncio
async def test_an_external_image_link_is_untouched(turn_context):
    original = "[chart](https://example.org/chart.png) and [c](https://x.org/a/b.jpg)"

    assert await rewrite_answer_file_links(original) == original


@pytest.mark.asyncio
async def test_a_plain_link_to_an_artifact_image_becomes_an_image(turn_context):
    # Only a stored type decides: an id this turn did not store keeps the
    # label the model wrote, whatever name it typed after the id.
    other = "6afe0e94f584dfa22d6ed15f"
    answer = await rewrite_answer_file_links(
        f"[map](/artifacts/{other}/map.png) [data](/artifacts/{other}/data.csv) "
        f"[tool](get_sample_image-1.png) [kept](/artifacts/{other}) "
        f"[file](/tool_artifacts/{_TURN_ARTIFACT_ID}/file)"
    )

    assert answer == (
        f"[map](/artifacts/{other}) [data](/artifacts/{other}) "
        f"![tool](/artifacts/{_TURN_ARTIFACT_ID}) [kept](/artifacts/{other}) "
        f"![file](/artifacts/{_TURN_ARTIFACT_ID})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "content_type"),
    [
        # Served as an octet-stream download: the chat cannot draw it.
        ("chart.svg", "image/svg+xml"),
        # A .png name over bytes the sniffer could not classify.
        ("fake.png", "application/octet-stream"),
    ],
)
async def test_a_turn_artifact_that_is_no_raster_image_stays_a_link(
    turn_context, name, content_type
):
    ctx, _ = turn_context
    other = "6afe0e94f584dfa22d6ed160"
    ctx.collected_artifact_ids.append(other)
    ctx.collected_artifact_files[name] = other
    ctx.collected_artifact_types[other] = content_type

    answer = await rewrite_answer_file_links(
        f"[a]({name}) [b](/artifacts/{other}/x.png)"
    )

    assert answer == f"[a](/artifacts/{other}) [b](/artifacts/{other})"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("chart.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'),
        ("fake.png", b"not an image at all"),
    ],
)
async def test_a_workspace_file_that_is_no_raster_image_stays_a_link(
    turn_context, name, data
):
    ctx, _ = turn_context
    ctx.sandbox_backend.files[name] = data

    answer = await rewrite_answer_file_links(f"[c]({name})")

    assert answer == f"[c](/conversations/c1/workspace/{name})"


@pytest.mark.no_db
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.png", True),
        ("a.JPG", True),
        ("a.webp", True),
        ("a.svg", False),
        ("a.csv", False),
        ("noext", False),
    ],
)
def test_without_a_stored_type_the_extension_decides(name, expected):
    from src.services.agents.core.workspace import is_image_file

    assert is_image_file(name) is expected


@pytest.mark.no_db
def test_a_stored_type_wins_over_the_extension():
    from src.services.agents.core.workspace import is_image_file

    assert is_image_file("a.png", "application/octet-stream") is False
    assert is_image_file("a.svg", "image/svg+xml") is False
    assert is_image_file("plot", "image/png") is True
    assert is_image_file("a.png", "") is False


@pytest.mark.asyncio
async def test_citation_offsets_survive_a_link_turned_into_an_image(turn_context):
    from src.services.sources.citations import (
        CitationRewriter,
        SourceRegistry,
        marker_spans,
    )

    registry = SourceRegistry()
    registry.register({"id": "src_a", "document_key": "doi:a", "title": "A"})
    rewriter = CitationRewriter(registry)
    streamed = (
        "Plot [S1]. [n](ndvi.png) fact [S1]. "
        f"[m](/artifacts/{_TURN_ARTIFACT_ID}/m.png) again [S1]. "
        "[s](get_sample_image-1.png) last [S1]."
    )
    text = rewriter.feed(streamed) + rewriter.flush()

    passes: list = []
    answer = await rewrite_answer_file_links(text, passes)

    assert answer.count("![") == 3
    assert rewriter.remap(answer, passes) == 0
    assert marker_spans(answer, rewriter.citations()) == ["[1]"] * 4


# Dev, 2026-11-19 (conversation 6afea7a9ac79eb8afdbaf7fe, dummy_image only): the
# tool returned the stub below, the answer announced the image and left the
# markdown out, so nothing rendered.
_DUMMY_IMAGE_ID = "6afea7ccac79eb8afdbaf804"
_DUMMY_IMAGE_TOOL_RESULT = (
    f'![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID} '
    '"MCP: dummy_image/get_sample_image")\n'
    '{"width": 512, "height": 512, "color": "blue"}'
)
_DUMMY_IMAGE_ANSWER = "Here is the sample blue image with a size of 512x512 pixels:"
_ID_B, _ID_C, _ID_D, _ID_E, _ID_F = (
    f"6afea7ccac79eb8afdbaf8{n:02x}" for n in range(5, 10)
)


def _record_turn_image(
    ctx, artifact_id, name, content_type="image/png", shown=True
):
    ctx.collected_artifact_ids.append(artifact_id)
    ctx.collected_artifact_files[name] = artifact_id
    ctx.collected_artifact_types[artifact_id] = content_type
    if shown:
        ctx.shown_artifact_ids.append(artifact_id)


@pytest.fixture
def dummy_image_turn(monkeypatch):
    from src.services.agents.core import workspace
    from src.services.mcp.artifact_context import (
        reset_artifact_context,
        set_artifact_context,
    )

    monkeypatch.setattr(workspace, "storage_service", _Store())
    ctx, token = set_artifact_context("u1", "c1")
    try:
        # What the interceptor records for the stub above; the interceptor
        # side is pinned in test_artifact_ingestion.py.
        _record_turn_image(ctx, _DUMMY_IMAGE_ID, "get_sample_image-1.png")
        yield ctx
    finally:
        reset_artifact_context(token)


@pytest.mark.asyncio
async def test_a_tool_image_the_answer_leaves_out_is_appended(dummy_image_turn):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    answer = append_unreferenced_turn_images(
        await rewrite_answer_file_links(_DUMMY_IMAGE_ANSWER)
    )

    assert answer == (
        f"{_DUMMY_IMAGE_ANSWER}\n\n"
        f"![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "written",
    [
        "Here: ![img](get_sample_image-1.png)",
        f"Here: [img](https://cdn.example.org/{_DUMMY_IMAGE_ID}.png)",
        f"Here: ![img](/artifacts/{_DUMMY_IMAGE_ID})",
    ],
)
async def test_a_tool_image_the_answer_links_is_not_appended_again(
    dummy_image_turn, written
):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    answer = append_unreferenced_turn_images(await rewrite_answer_file_links(written))

    assert answer.count(f"/artifacts/{_DUMMY_IMAGE_ID}") == 1


@pytest.mark.no_db
def test_only_raster_images_are_appended(dummy_image_turn):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    _record_turn_image(dummy_image_turn, _ID_B, "table.csv", "text/csv")
    _record_turn_image(dummy_image_turn, _ID_C, "chart.svg", "image/svg+xml")
    # A .png name the sniffer stored opaque.
    _record_turn_image(dummy_image_turn, _ID_D, "x.png", "application/octet-stream")

    answer = append_unreferenced_turn_images("Done.")

    assert answer == (
        f"Done.\n\n![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID})"
    )


@pytest.mark.no_db
@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_answer_gets_the_tool_image(dummy_image_turn, blank):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    assert append_unreferenced_turn_images(blank) == (
        f"![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID})"
    )


@pytest.mark.no_db
def test_an_image_folded_into_the_tool_json_is_not_appended(dummy_image_turn):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    dummy_image_turn.shown_artifact_ids.clear()

    assert append_unreferenced_turn_images("Done.") == "Done."


@pytest.mark.asyncio
async def test_tool_inputs_stay_out_when_the_answer_shows_its_own_image(
    dummy_image_turn,
):
    # Tiles fetched as input, then a chart drawn from them and embedded: the
    # model chose what to show.
    from src.services.agents.core.runner import append_unreferenced_turn_images

    dummy_image_turn.sandbox_backend = _WorkspaceSandbox(
        {"chart.png": b"\x89PNG\r\n\x1a\nchart"}
    )
    _record_turn_image(dummy_image_turn, _ID_B, "tile-2.png")

    answer = append_unreferenced_turn_images(
        await rewrite_answer_file_links("NDVI chart: ![chart](chart.png)")
    )

    assert answer == "NDVI chart: ![chart](/conversations/c1/workspace/chart.png)"


@pytest.mark.no_db
def test_at_most_four_images_are_appended(dummy_image_turn):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    for n, artifact_id in enumerate((_ID_B, _ID_C, _ID_D, _ID_E, _ID_F), start=2):
        _record_turn_image(dummy_image_turn, artifact_id, f"tile-{n}.png")

    answer = append_unreferenced_turn_images("Done.")

    assert answer.count("![") == 4
    assert _DUMMY_IMAGE_ID in answer and _ID_D in answer
    assert _ID_E not in answer and _ID_F not in answer


@pytest.mark.no_db
def test_an_id_quoted_in_code_does_not_count_as_shown(dummy_image_turn):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    written = f"Artifact `{_DUMMY_IMAGE_ID}` is ready."

    assert append_unreferenced_turn_images(written) == (
        f"{written}\n\n![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID})"
    )


@pytest.mark.no_db
def test_an_open_code_block_is_closed_before_the_image(dummy_image_turn):
    from src.services.agents.core.runner import append_unreferenced_turn_images

    written = "Code:\n```python\nprint(1)\n"

    assert append_unreferenced_turn_images(written) == (
        "Code:\n```python\nprint(1)\n```\n\n"
        f"![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID})"
    )
    closed = "Code:\n```\nx\n```"
    assert append_unreferenced_turn_images(closed) == (
        f"{closed}\n\n![get_sample_image-1.png](/artifacts/{_DUMMY_IMAGE_ID})"
    )


@pytest.mark.no_db
def test_nothing_is_appended_without_a_turn():
    from src.services.agents.core.runner import append_unreferenced_turn_images

    assert append_unreferenced_turn_images(_DUMMY_IMAGE_ANSWER) == _DUMMY_IMAGE_ANSWER
