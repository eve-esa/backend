"""Unit tests for runner.rewrite_answer_file_links.

The model often wraps an artifact id in a host or an old sandbox prefix.
The runner rewrites that markdown in place. It does not append artifacts
the answer never mentions.
"""

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
    assert "storage.googleapis.com" in _ARTIFACT_MARKDOWN_INSTRUCTION
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
