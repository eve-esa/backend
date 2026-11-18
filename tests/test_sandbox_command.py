"""A sandbox shell command sees the same paths as the file tools, byte for byte.

The command text is never edited. ``workspace/x`` resolves through a
``workspace -> .`` self-link inside the workspace, ``/workspace/x`` through an
optional link where the process may write it.
"""

import asyncio
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.agents.core import sandbox
from src.services.agents.core.sandbox import (
    _bind_skills_shell,
    prepare_workspace_layout,
    workspace_shell_command,
)
from src.services.agents.core.sandbox_inspect import _find_command
from src.services.agents.core.workspace import _FIND, _command, _list_files

pytestmark = pytest.mark.no_db


def _sh(command: str, cwd: Path) -> SimpleNamespace:
    done = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return SimpleNamespace(output=done.stdout + done.stderr, exit_code=done.returncode)


class _LocalAgentCore:
    """AgentCore-like backend: a real /bin/sh in a session root that is not ``/``.

    Like ``AgentCoreSandbox``, the cwd is probed lazily through
    ``self.execute("pwd")`` and cached in ``_cwd``; file paths are stored under
    it and the shell starts there.
    """

    def __init__(self, root: Path, *, failing_probes: int = 0):
        self.root = root
        self._cwd = None
        self.failing_probes = failing_probes
        self.commands: list[str] = []

    def _get_cwd(self) -> str:
        if self._cwd is None:
            if self.failing_probes:
                self.failing_probes -= 1
                raise RuntimeError("Failed to detect sandbox working directory")
            result = self.execute("pwd")
            self._cwd = result.output.strip().rstrip("/")
        return self._cwd

    def execute(self, command, *, timeout=None):
        self.commands.append(command)
        return _sh(command, self.root)

    def write(self, file_path, content):
        cwd = self._get_cwd()
        real = file_path if file_path.startswith(cwd + "/") else f"{cwd}/{file_path.lstrip('/')}"
        Path(real).parent.mkdir(parents=True, exist_ok=True)
        Path(real).write_text(content)
        return SimpleNamespace(path=real, error=None)


class _RecordingOpenSandbox:
    """OpenSandbox-like backend: no session cwd, file paths are absolute."""

    def __init__(self):
        self.commands: list[str] = []

    def execute(self, command, *, timeout=None):
        self.commands.append(command)
        return SimpleNamespace(output="", exit_code=0)


@pytest.fixture
def alias(tmp_path, monkeypatch):
    """Point the ``/workspace`` link at a scratch path, never the real ``/``."""
    target = tmp_path / "virtual" / "workspace"
    target.parent.mkdir()
    monkeypatch.setattr(sandbox, "_VIRTUAL_WORKSPACE", str(target), raising=False)
    return target


@pytest.fixture
def agentcore(tmp_path, alias):
    root = tmp_path / "session"
    root.mkdir()
    inner = _LocalAgentCore(root)
    return inner, prepare_workspace_layout(_bind_skills_shell(inner))


@pytest.mark.parametrize(
    "written, run",
    [
        ("workspace/count.py", "python3 workspace/count.py"),
        ("workspace/count.py", "python3 count.py"),
        ("workspace/count.py", "python3 ./workspace/count.py"),
        ("count.py", "python3 workspace/count.py"),
        ("count.py", "cd workspace && python3 count.py"),
        ("/workspace/count.py", "python3 count.py"),
        ("/workspace/count.py", "cd workspace/workspace && python3 count.py"),
    ],
)
def test_a_written_file_runs_by_the_path_the_model_names(agentcore, written, run):
    inner, backend = agentcore

    backend.write(written, "print(6 * 7)\n")
    result = backend.execute(run)

    assert (inner.root / "workspace" / "count.py").is_file()
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "42"


def test_the_virtual_absolute_path_resolves_where_the_link_is_allowed(agentcore, alias):
    inner, backend = agentcore

    backend.write("count.py", "print(6 * 7)\n")
    result = backend.execute(f"python3 {alias}/count.py")

    assert alias.is_symlink()
    assert result.output.strip() == "42"


def test_commands_still_run_when_the_virtual_link_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sandbox, "_VIRTUAL_WORKSPACE", "/proc/eve-no-such-dir/workspace", raising=False
    )
    inner = _LocalAgentCore(tmp_path)
    backend = prepare_workspace_layout(_bind_skills_shell(inner))

    backend.write("count.py", "print(6 * 7)\n")
    result = backend.execute("python3 workspace/count.py")

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "42"


def test_a_script_reaches_workspace_paths_it_names_itself(agentcore):
    _, backend = agentcore
    backend.write("data.txt", "seven\n")
    backend.write("s.py", "print(open('workspace/data.txt').read().strip())\n")

    result = backend.execute("python3 s.py")

    assert result.exit_code == 0, result.output
    assert result.output == "seven\n"


@pytest.mark.parametrize(
    "command",
    [
        "grep -c '^workspace/a' data.txt",
        "sed -n 's#workspace/a#HIT#p' data.txt",
        "python3 -c \"import re; print(re.findall(r'workspace/(\\w+)', open('data.txt').read()))\"",
        "echo 'Saved to workspace/out.png' /workspace/x @workspace/foo",
        "cat <<'EOF'\nworkspace/a /workspace/b ./workspace/c\nEOF",
        "X=7; printf '%s\\n' \"a'b\" 'c\"d' '$X' \"$X\" \"it's \\\"quoted\\\"\"",
        "python3 -c \"d = [1, 2]; print(f'Dice: {d[0]}, {d[1]}')\"",
    ],
)
def test_the_command_text_reaches_the_shell_unchanged(agentcore, tmp_path, command):
    inner, backend = agentcore
    data = "workspace/a\nworkspace/b\n/workspace/c\n"
    backend.write("data.txt", data)
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "data.txt").write_text(data)

    result = backend.execute(command)

    assert result.exit_code == 0, result.output
    assert result.output == _sh(command, plain).output


def test_a_quoted_heredoc_body_is_written_byte_for_byte(agentcore):
    inner, backend = agentcore
    body = "path = 'workspace/a'\nroot = \"/workspace\"\ncd workspace\n"

    backend.execute(f"cat > out.txt <<'EOF'\n{body}EOF")

    assert (inner.root / "workspace" / "out.txt").read_text() == body


def test_a_command_is_its_own_script(agentcore):
    _, backend = agentcore

    assert backend.execute("# nothing to run").exit_code == 0
    multi = backend.execute("# roll\necho one; false || echo two")
    assert multi.output == "one\ntwo\n"


def test_skills_paths_are_still_rewritten(agentcore):
    inner, backend = agentcore
    (inner.root / "skills" / "roll").mkdir(parents=True)
    (inner.root / "skills" / "roll" / "roll.py").write_text("print('rolled')\n")

    result = backend.execute("python3 /skills/roll/roll.py")

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "rolled"


def test_the_self_link_comes_back_after_the_model_deletes_it(agentcore):
    inner, backend = agentcore
    backend.write("count.py", "print(6 * 7)\n")

    backend.execute("rm workspace")
    result = backend.execute("python3 workspace/count.py")

    assert result.output.strip() == "42"


def test_listings_skip_the_self_link(agentcore):
    inner, backend = agentcore
    backend.write("a.txt", "a")
    backend.write("sub/b.txt", "b")
    link = inner.root / "workspace" / "workspace"
    assert link.is_symlink()

    rows, truncated = asyncio.run(_list_files(backend))
    from_inside, _ = _command(SimpleNamespace(execute=lambda c: _sh(c, link.parent)), _FIND)
    inspect, ok = _command(backend, _find_command("", 50))

    assert [row[0] for row in rows] == ["a.txt", "sub/b.txt"]
    assert not truncated
    assert sorted(line.split("\t")[0] for line in from_inside.splitlines()) == [
        "a.txt",
        "sub/b.txt",
    ]
    assert ok
    assert sorted(line.split("\t")[0] for line in inspect.splitlines()) == [
        "workspace/a.txt",
        "workspace/sub/b.txt",
    ]


def test_a_failed_cwd_probe_never_writes_into_workspace_workspace(tmp_path, alias):
    inner = _LocalAgentCore(tmp_path, failing_probes=1)
    backend = prepare_workspace_layout(_bind_skills_shell(inner))

    backend.write("count.py", "print(6 * 7)\n")
    result = backend.execute("python3 count.py")

    assert inner._cwd == str(tmp_path)
    assert (tmp_path / "workspace" / "count.py").is_file()
    assert (tmp_path / "workspace" / "workspace").is_symlink()
    assert result.output.strip() == "42"


def test_the_wrapper_runs_from_the_workspace_with_the_command_as_one_argument(tmp_path):
    workspace = tmp_path / "with space" / "workspace"
    command = "pwd; printf '%s|' \"$0\" \"$@\"; echo; cat <<'EOF'\n$HOME `x`\nEOF"

    result = _sh(workspace_shell_command(command, str(workspace)), tmp_path)

    lines = result.output.splitlines()
    assert result.exit_code == 0, result.output
    assert lines[0] == str(workspace)
    assert lines[1] == "sh|"
    assert lines[2] == "$HOME `x`"
    assert (workspace / "workspace").is_symlink()


def test_opensandbox_runs_in_the_absolute_workspace_with_no_virtual_link():
    inner = _RecordingOpenSandbox()
    backend = prepare_workspace_layout(inner)
    command = "python3 workspace/count.py && echo 'done'"

    backend.execute(command)

    setup = inner.commands[0]
    argv = shlex.split(inner.commands[-1])
    assert "ln -s . /workspace/workspace" in setup
    assert "ln -s /workspace" not in setup
    assert argv[-4:] == ["exec", "sh", "-c", command]
    assert "cd /workspace &&" in inner.commands[-1]


def test_internal_listing_commands_run_from_the_root():
    inner = _RecordingOpenSandbox()
    backend = prepare_workspace_layout(inner)
    listing = _find_command("workspace/sub/a.txt", 10)

    _command(backend, listing)

    assert inner.commands[-1] == f"cd / && {listing}"


def test_agentcore_internal_commands_stay_raw(agentcore):
    inner, backend = agentcore

    _command(backend, "ps -eo pid=,args=")

    assert inner.commands[-1] == "ps -eo pid=,args="
