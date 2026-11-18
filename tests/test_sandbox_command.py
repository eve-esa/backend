"""A sandbox shell command runs from the workspace with its text unchanged."""

import logging
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.agents.core.runner import _ARTIFACT_MARKDOWN_INSTRUCTION
from src.services.agents.core.sandbox import (
    SandboxError,
    _bind_skills_shell,
    prepare_workspace_layout,
    workspace_shell_command,
)
from src.services.agents.core.sandbox_inspect import _find_command
from src.services.agents.core.workspace import _command

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

    def __init__(self, root: Path, *, failing_probes: int = 0, pwd_broken: bool = False):
        self.root = root
        self._cwd = None
        self.failing_probes = failing_probes
        self.pwd_broken = pwd_broken
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
        if self.pwd_broken and command == "pwd":
            return SimpleNamespace(output="", exit_code=1)
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
def agentcore(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    inner = _LocalAgentCore(root)
    return inner, prepare_workspace_layout(_bind_skills_shell(inner))


@pytest.mark.parametrize(
    "written",
    ["count.py", "workspace/count.py", "./workspace/count.py", "/workspace/count.py"],
)
def test_a_written_file_runs_as_its_name_from_the_workspace(agentcore, written):
    inner, backend = agentcore

    backend.write(written, "print(6 * 7)\n")
    result = backend.execute("python3 count.py")

    assert (inner.root / "workspace" / "count.py").is_file()
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "42"


def test_a_written_file_runs_by_its_real_absolute_path(agentcore):
    inner, backend = agentcore

    written = backend.write("count.py", "print(6 * 7)\n").path
    result = backend.execute(f"cd /tmp && python3 {written}")

    assert result.output.strip() == "42"


@pytest.mark.parametrize("path", ["workspace/count.py", "/workspace/count.py"])
def test_workspace_prefixed_paths_fail_loudly_in_the_shell(agentcore, path):
    _, backend = agentcore
    backend.write("count.py", "print(6 * 7)\n")

    result = backend.execute(f"python3 {path}")

    assert result.exit_code != 0
    assert "workspace/count.py" in result.output


def test_the_instruction_says_how_to_run_a_written_file():
    assert "The shell starts there." in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "`python3 x`" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "`workspace/x` or `/workspace/x` as `x`" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "never run a file through" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "Those paths do not exist there." in _ARTIFACT_MARKDOWN_INSTRUCTION


@pytest.mark.parametrize(
    "command",
    [
        "grep -c '^workspace/a' data.txt",
        "sed -n 's#workspace/a#HIT#p' data.txt",
        "python3 -c \"import re; print(re.findall(r'workspace/(\\w+)', open('data.txt').read()))\"",
        "echo 'Saved to workspace/out.png' /workspace/x @workspace/foo",
        "cat <<'EOF'\nworkspace/a /workspace/b ./workspace/c $HOME `x`\nEOF",
        "X=7; printf '%s\\n' \"a'b\" 'c\"d' '$X' \"$X\" \"it's \\\"quoted\\\"\"",
        "python3 -c \"d = [1, 2]; print(f'Dice: {d[0]}, {d[1]}')\"",
    ],
)
def test_the_command_text_reaches_the_shell_unchanged(agentcore, tmp_path, command):
    _, backend = agentcore
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


def test_a_failed_cwd_probe_is_read_again_through_the_raw_shell(tmp_path):
    inner = _LocalAgentCore(tmp_path, failing_probes=1)
    backend = prepare_workspace_layout(_bind_skills_shell(inner))

    backend.write("count.py", "print(6 * 7)\n")
    result = backend.execute("python3 count.py")

    assert inner._cwd == str(tmp_path)
    assert (tmp_path / "workspace" / "count.py").is_file()
    assert not (tmp_path / "workspace" / "workspace").exists()
    assert result.output.strip() == "42"


def test_an_unknown_cwd_fails_writes_instead_of_nesting_them(tmp_path, caplog):
    inner = _LocalAgentCore(tmp_path, failing_probes=1, pwd_broken=True)
    with caplog.at_level(logging.ERROR):
        backend = prepare_workspace_layout(_bind_skills_shell(inner))

    with pytest.raises(SandboxError):
        backend.write("count.py", "print(6 * 7)\n")

    assert "workspace.cwd_unknown" in caplog.text
    assert not (tmp_path / "workspace" / "workspace").exists()


def test_the_wrapper_runs_from_the_workspace_with_the_command_as_one_argument(tmp_path):
    workspace = tmp_path / "with space" / "workspace"
    command = "pwd; printf '%s|' \"$0\" \"$@\"; echo; cat <<'EOF'\n$HOME `x`\nEOF"

    result = _sh(workspace_shell_command(command, str(workspace)), tmp_path)

    lines = result.output.splitlines()
    assert result.exit_code == 0, result.output
    assert lines == [str(workspace), "sh|", "$HOME `x`"]


def test_opensandbox_runs_in_the_absolute_workspace():
    inner = _RecordingOpenSandbox()
    backend = prepare_workspace_layout(inner)
    command = "python3 workspace/count.py && echo 'done'"

    backend.execute(command)

    argv = shlex.split(inner.commands[-1])
    assert argv[:6] == ["mkdir", "-p", "/workspace", "&&", "cd", "/workspace"]
    assert argv[-4:] == ["exec", "sh", "-c", command]


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
