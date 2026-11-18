"""An image read by the deep agent reaches the model only when it can see images.

deepagents' ``read_file`` returns an image as an image content block, and its
``UnsupportedContentMiddleware`` replaces that block with a text notice when the
model's ``profile`` says it takes no images. The platform clients had no profile,
so the block went to EVE-Instruct, Blablador answered 400, and the retry sent a
generic "Unsupported content" line that the model answered with an invented
description. These tests pin the profile on the platform clients and run a real
deep agent read through a fake model to see what the model receives.
"""

import struct
import zlib
from typing import Any, List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.core import llm_manager as llm_manager_module
from src.core.llm_manager import LLMManager, platform_model_profile

pytestmark = pytest.mark.no_db

IMAGE_PATH = "/artifacts/6afdf5edf1a1fe86ea6ac150/light-home.png"
TEXT_PATH = "/artifacts/6afdf5edf1a1fe86ea6ac151/notes.txt"


def _png(width: int = 4, height: int = 3) -> bytes:
    """A valid PNG with the given size, built without an imaging library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class _ReadingModel(BaseChatModel):
    """Calls ``read_file`` once, then answers. Records what each call received."""

    path: str
    seen: List[List[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-reader"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ReadingModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if len(self.seen) == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_read", "name": "read_file", "args": {"file_path": self.path}}
                ],
            )
        else:
            message = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _read_through_deep_agent(tmp_path, path: str, profile: Any) -> ToolMessage:
    """Run a deep agent whose model reads ``path``; return the read as the model saw it."""
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend

    (tmp_path / path.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".png"):
        (tmp_path / path.lstrip("/")).write_bytes(_png())
    else:
        (tmp_path / path.lstrip("/")).write_text("line one\nline two\n")

    model = _ReadingModel(path=path, seen=[], profile=profile)
    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
    )
    agent.invoke({"messages": [{"role": "user", "content": "Describe the file."}]})
    assert len(model.seen) == 2
    reads = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert len(reads) == 1
    return reads[0]


def _block_types(message: ToolMessage) -> List[str]:
    return [block["type"] for block in message.content_blocks]


def test_an_image_read_by_a_text_only_model_names_the_image_it_cannot_see(tmp_path):
    read = _read_through_deep_agent(
        tmp_path, IMAGE_PATH, platform_model_profile("20 - EVE-Instruct")
    )

    assert "image" not in _block_types(read)
    assert read.text.startswith("[read_file: ")
    assert IMAGE_PATH in read.text
    assert "does not support image content (image/png)" in read.text


def test_a_text_read_by_a_text_only_model_is_unchanged(tmp_path):
    read = _read_through_deep_agent(
        tmp_path, TEXT_PATH, platform_model_profile("20 - EVE-Instruct")
    )

    assert _block_types(read) == ["text"]
    assert "line one" in read.text and "line two" in read.text
    assert "does not support" not in read.text


def test_an_image_read_by_a_listed_vision_model_keeps_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(
        llm_manager_module, "IMAGE_INPUT_MODEL_NAMES", "mistral-small-latest"
    )

    read = _read_through_deep_agent(
        tmp_path, IMAGE_PATH, platform_model_profile("mistral-small-latest")
    )

    assert _block_types(read) == ["image"]
    assert read.content_blocks[0]["mime_type"] == "image/png"


def test_platform_models_are_text_only_unless_listed(monkeypatch):
    monkeypatch.setattr(
        llm_manager_module, "IMAGE_INPUT_MODEL_NAMES", " mistral-small-latest , x "
    )

    assert platform_model_profile("20 - EVE-Instruct") == {
        "image_inputs": False,
        "image_tool_message": False,
    }
    assert platform_model_profile("mistral-small-latest") == {
        "image_inputs": True,
        "image_tool_message": True,
    }
    assert platform_model_profile(None)["image_inputs"] is False


def test_the_platform_clients_carry_the_profile(monkeypatch):
    monkeypatch.setattr(llm_manager_module, "IMAGE_INPUT_MODEL_NAMES", "")
    monkeypatch.setattr(llm_manager_module, "EVE_JSC_API_KEY", "k")
    monkeypatch.setattr(llm_manager_module, "FALLBACK_MODEL_API_KEY", "k")
    manager = LLMManager()
    manager._eve_jsc_base_url = "http://127.0.0.1:1/v1"
    manager._eve_jsc_model_name = "20 - EVE-Instruct"
    manager._fallback_base_url = "http://127.0.0.1:1/v1"
    manager._fallback_model_name = "mistral-small-latest"

    for client in (manager._get_eve_jsc_llm(), manager._get_fallback_llm()):
        assert client.profile["image_inputs"] is False
        assert client.profile["image_tool_message"] is False


def test_a_custom_model_keeps_its_own_profile():
    client = LLMManager().build_custom_client(
        base_url="http://127.0.0.1:1/v1", model_name="gpt-4o", api_key="sk-test"
    )

    assert not client.profile or client.profile.get("image_inputs") is not False
