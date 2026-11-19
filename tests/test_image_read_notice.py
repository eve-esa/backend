"""The notice a text-only model reads in place of an image it cannot see.

Dev, 2026-11-19 (conversations 6afeaf346ad76994fa4768a0 and two more, after
#396): the chart rendered, but the answer still said "I cannot display images
directly in this interface". The model read ``ndvi.png`` after saving it, got
deepagents' "was not attached because this model does not support image
content" and concluded it could not show the image. The notice now says what
is true and what to do: the model cannot see the image, must not describe it,
and can still show it with markdown the answer link rewrite understands.
"""

import asyncio
import json
import logging
import threading
from http.server import ThreadingHTTPServer

import pytest
from langchain_core.messages import HumanMessage

from src.core import llm_manager as llm_manager_module
from src.core.llm_manager import LLMManager, platform_model_profile
from src.services.agents.core import runner
from src.services.agents.core.image_notice import (
    image_read_notice,
    install_image_read_notice,
)
from tests.test_image_read_model_profile import (
    _png,
    _read_through_deep_agent,
)

pytestmark = pytest.mark.no_db

TEXT_ONLY = platform_model_profile("20 - EVE-Instruct")


def _png_of(size: int) -> bytes:
    """A valid PNG padded after IEND to exactly ``size`` bytes."""
    data = _png()
    return data + b"\x00" * (size - len(data))


@pytest.fixture(autouse=True)
def _installed():
    install_image_read_notice()


def test_a_workspace_png_read_by_a_text_only_model_says_how_to_show_it(tmp_path):
    read = _read_through_deep_agent(
        tmp_path, "/workspace/ndvi.png", TEXT_ONLY, data=_png_of(41 * 1024)
    )

    assert [b["type"] for b in read.content_blocks] == ["text"]
    assert read.text == (
        "[read_file: /workspace/ndvi.png] This is an image file (ndvi.png, 41 KB). "
        "This model cannot see image content, so do not describe it or guess "
        "what it shows. If the user asks what it shows, tell them you cannot see "
        "images with the current model. You can still show it to the user: "
        "write ![ndvi.png](ndvi.png) in your answer."
    )
    assert "was not attached" not in read.text
    assert "does not support" not in read.text


def test_a_nested_workspace_image_links_relative_to_the_workspace(tmp_path):
    read = _read_through_deep_agent(tmp_path, "/workspace/charts/ndvi.png", TEXT_ONLY)

    assert "write ![ndvi.png](charts/ndvi.png) in your answer." in read.text


def test_an_uploaded_image_says_do_not_describe_and_keeps_its_artifact_path(tmp_path):
    # U8: the user asks to describe an upload. The notice must keep the model
    # from inventing content; the link is the path the artifact rewrite reads.
    path = "/artifacts/6afdf5edf1a1fe86ea6ac150/light-home.png"
    read = _read_through_deep_agent(tmp_path, path, TEXT_ONLY)

    assert "This model cannot see image content, so do not describe it" in read.text
    assert "tell them you cannot see images with the current model." in read.text
    assert f"write ![light-home.png]({path}) in your answer." in read.text


def test_an_image_the_answer_cannot_link_gets_no_show_sentence(tmp_path):
    read = _read_through_deep_agent(tmp_path, "/large_tool_results/plot.png", TEXT_ONLY)

    assert "This is an image file (plot.png" in read.text
    assert "do not describe it" in read.text
    assert "You can still show it" not in read.text
    assert "![" not in read.text


def test_a_vision_model_still_gets_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_manager_module, "IMAGE_INPUT_MODEL_NAMES", "pixtral")

    read = _read_through_deep_agent(
        tmp_path, "/workspace/ndvi.png", platform_model_profile("pixtral")
    )

    assert [b["type"] for b in read.content_blocks] == ["image"]
    assert read.content_blocks[0]["mime_type"] == "image/png"


def test_other_unsupported_blocks_keep_the_deepagents_placeholder():
    from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware

    block = {"type": "image", "base64": "AAAA", "mime_type": "image/png"}
    # An image in a human message, not a read_file result.
    message = HumanMessage(content_blocks=[block])

    text = UnsupportedContentMiddleware()._replace(block, message)["text"]

    assert text.startswith("[read_file: the requested file was not attached")


def test_install_is_idempotent():
    from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware

    install_image_read_notice()
    first = UnsupportedContentMiddleware._replace
    install_image_read_notice()

    assert UnsupportedContentMiddleware._replace is first


@pytest.mark.parametrize(
    "size, shown",
    [(0, "0 bytes"), (512, "512 bytes"), (1024, "1 KB"), (41 * 1024 + 300, "41 KB"),
     (3 * 1024 * 1024, "3.0 MB")],
)
def test_the_size_reads_like_a_person_wrote_it(size, shown):
    assert f"(x.png, {shown})" in image_read_notice("/workspace/x.png", size)


def test_a_name_with_spaces_or_brackets_still_makes_a_valid_link():
    notice = image_read_notice("/workspace/my [chart] (v2).png", 10)

    assert "![my chart v2.png](my%20%5Bchart%5D%20%28v2%29.png)" in notice


def test_the_notice_logs_nothing(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    _read_through_deep_agent(tmp_path, "/workspace/secret-plan.png", TEXT_ONLY)

    assert "secret-plan" not in caplog.text


def test_the_runner_graph_sends_the_notice_to_a_text_only_model(tmp_path, monkeypatch):
    """Through the runner's own graph build: no explicit install in this path."""
    deep = pytest.importorskip("agents.graphs.deep.graph")
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware

    from tests.test_agentic_image_read import _OpenAICompatible

    import tests.test_agentic_image_read as wire

    # Undo the autouse install, so only the runner can put the notice in place.
    import src.services.agents.core.image_notice as notice_module

    monkeypatch.setattr(
        UnsupportedContentMiddleware, "_replace", notice_module._DEEPAGENTS_REPLACE
    )
    monkeypatch.setattr(wire, "IMAGE", "/workspace/ndvi.png")
    _OpenAICompatible.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatible)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setattr(llm_manager_module, "IMAGE_INPUT_MODEL_NAMES", "")
    monkeypatch.setattr(llm_manager_module, "EVE_JSC_API_KEY", "k")
    monkeypatch.setattr(llm_manager_module, "FALLBACK_MODEL_API_KEY", "k")
    manager = LLMManager()
    manager._eve_jsc_base_url = url
    manager._eve_jsc_model_name = "20 - EVE-Instruct"
    manager._fallback_base_url = url
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "ndvi.png").write_bytes(_png())

    graph = runner._build_react_graph(
        "eve_jsc",
        [],
        None,
        agent=deep.DeepAgent(),
        llm=manager._get_eve_jsc_llm(),
        fallback_llm=manager._get_fallback_llm(),
        sandbox_backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
    )

    async def run() -> None:
        query = {"role": "user", "content": "Plot NDVI and show it."}
        async for _ in graph.astream({"messages": [query]}, stream_mode="updates"):
            pass

    try:
        asyncio.run(run())
    finally:
        server.shutdown()

    assert len(_OpenAICompatible.bodies) == 2
    sent = json.dumps(_OpenAICompatible.bodies[1]["messages"])
    assert "This model cannot see image content, so do not describe it" in sent
    assert "![ndvi.png](ndvi.png)" in sent
    assert "was not attached" not in sent
