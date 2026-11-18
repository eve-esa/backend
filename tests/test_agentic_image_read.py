"""An image read by the deep graph never reaches the model as base64 text.

On dev (2026-11-18, agents ce1a4b7) the deep graph rewrote the ``read_file``
result into ``[TOOL_RESULTS]`` text, so the PNG reached EVE-Instruct as base64
characters and the answer described a room that is not in the picture. Agents
#21 keeps tool results structured. The test runs the real JSC client from
LLMManager, the runner's graph build and the installed deep graph against a
local OpenAI-compatible server, and reads what was sent.
"""

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from src.core import llm_manager as llm_manager_module
from src.core.llm_manager import LLMManager
from src.services.agents.core import runner
from tests.test_image_read_model_profile import _png

pytestmark = pytest.mark.no_db

IMAGE = "/artifacts/a1/light-home.png"
PNG_B64 = base64.b64encode(_png()).decode()


class _OpenAICompatible(BaseHTTPRequestHandler):
    """First call asks to read the image, the next one answers. Keeps every body."""

    bodies: List[Dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.bodies.append(body)
        if len(self.bodies) == 1:
            call = {
                "id": "abcdefghi",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"file_path": IMAGE})},
            }
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
        else:
            message = {"role": "assistant", "content": "I cannot see images."}
        data = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_the_agentic_jsc_client_sends_no_image_bytes(tmp_path, monkeypatch):
    deep = pytest.importorskip("agents.graphs.deep.graph")
    from deepagents.backends.filesystem import FilesystemBackend

    _OpenAICompatible.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatible)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setattr(llm_manager_module, "IMAGE_INPUT_MODEL_NAMES", "")
    monkeypatch.setattr(llm_manager_module, "EVE_JSC_API_KEY", "k")
    monkeypatch.setattr(llm_manager_module, "FALLBACK_MODEL_API_KEY", "k")
    # Fresh clients: the guard wraps the instance it is given.
    manager = LLMManager()
    manager._eve_jsc_base_url = url
    manager._eve_jsc_model_name = "20 - EVE-Instruct"
    manager._fallback_base_url = url
    (tmp_path / "artifacts" / "a1").mkdir(parents=True)
    (tmp_path / "artifacts" / "a1" / "light-home.png").write_bytes(_png())

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
        query = {"role": "user", "content": f"Describe the attached picture.\n\n{IMAGE}"}
        async for _ in graph.astream({"messages": [query]}, stream_mode="updates"):
            pass

    try:
        asyncio.run(run())
    finally:
        server.shutdown()

    assert len(_OpenAICompatible.bodies) == 2
    sent = json.dumps(_OpenAICompatible.bodies[1]["messages"])
    assert PNG_B64[:40] not in sent
    assert "image_url" not in sent
    assert "cannot see images" in sent or "does not support image content" in sent
