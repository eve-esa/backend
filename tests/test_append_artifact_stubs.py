"""Unit tests for runner.append_missing_artifact_stubs.

The models routinely paraphrase artifact stubs into invented URLs, so the
runner deterministically re-attaches any collected artifact whose serving URL
never made it into the final answer. Artifact lookups are monkeypatched: the
helper imports the model lazily inside the function, so patching the class
attribute is enough — no database needed.
"""

from types import SimpleNamespace

import pytest

from src.services.agents.core.runner import append_missing_artifact_stubs


def _fake_artifact(artifact_id: str, content_type: str, filename: str):
    return SimpleNamespace(
        id=artifact_id,
        content_type=content_type,
        filename=filename,
        source=SimpleNamespace(mcp_server="dummy", tool_name="get_sample_image"),
    )


@pytest.fixture
def patched_artifacts(monkeypatch):
    store = {}

    async def fake_find_by_id(artifact_id):
        return store.get(artifact_id)

    from src.database.models.artifact import Artifact

    monkeypatch.setattr(Artifact, "find_by_id", staticmethod(fake_find_by_id))
    return store


@pytest.mark.asyncio
async def test_appends_image_and_link_stubs_for_unreferenced_artifacts(
    patched_artifacts,
):
    patched_artifacts["img1"] = _fake_artifact("img1", "image/png", "sample.png")
    patched_artifacts["csv1"] = _fake_artifact("csv1", "text/csv", "report.csv")

    answer = await append_missing_artifact_stubs(
        "Here is an invented URL: https://storage.googleapis.com/x.png",
        ["img1", "csv1"],
    )

    assert '![sample.png](/artifacts/img1 "MCP: dummy/get_sample_image")' in answer
    assert '[report.csv](/artifacts/csv1 "MCP: dummy/get_sample_image")' in answer


@pytest.mark.asyncio
async def test_referenced_artifacts_are_not_duplicated(patched_artifacts):
    patched_artifacts["img1"] = _fake_artifact("img1", "image/png", "sample.png")

    original = 'Already embedded: ![sample.png](/artifacts/img1 "MCP: dummy/tool")'
    answer = await append_missing_artifact_stubs(original, ["img1"])

    assert answer == original


@pytest.mark.asyncio
async def test_empty_answer_or_no_artifacts_is_untouched(patched_artifacts):
    assert await append_missing_artifact_stubs("", ["img1"]) == ""
    assert await append_missing_artifact_stubs("text", None) == "text"
    assert await append_missing_artifact_stubs("text", []) == "text"


@pytest.mark.asyncio
async def test_lookup_errors_fail_open(monkeypatch):
    from src.database.models.artifact import Artifact

    async def boom(artifact_id):
        raise RuntimeError("db down")

    monkeypatch.setattr(Artifact, "find_by_id", staticmethod(boom))

    answer = await append_missing_artifact_stubs("text", ["img1"])
    assert answer == "text"


@pytest.mark.asyncio
async def test_filenames_and_names_are_escaped_in_appended_stubs(patched_artifacts):
    artifact = _fake_artifact("img1", "image/png", "x](https://phish.example)\n\n# hi")
    artifact.source = SimpleNamespace(mcp_server='ev"il', tool_name="t")
    patched_artifacts["img1"] = artifact

    answer = await append_missing_artifact_stubs("Answer.", ["img1"])

    stub = answer.split("\n\n", 1)[1].strip()
    assert "\n" not in stub
    assert "](https://phish" not in stub
    assert stub.startswith("![x\\]\\(https://phish.example\\) # hi](/artifacts/img1 ")
    assert stub.endswith('"MCP: evil/t")')


@pytest.mark.asyncio
async def test_a_non_image_type_gets_a_link_stub(patched_artifacts):
    patched_artifacts["bin1"] = _fake_artifact("bin1", "image/svg+xml", "chart.svg")

    answer = await append_missing_artifact_stubs("Answer.", ["bin1"])

    assert "![" not in answer
    assert '[chart.svg](/artifacts/bin1 "MCP: dummy/get_sample_image")' in answer
