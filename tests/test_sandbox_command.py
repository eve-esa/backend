"""A sandbox shell command sees the same paths as the file tools, and keeps its quotes."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.agents.core.sandbox import (
    _bind_skills_shell,
    prepare_workspace_layout,
)
from src.services.agents.core.sandbox_inspect import _find_command
from src.services.agents.core.workspace import _command


class _LocalAgentCore:
    """AgentCore-like backend: a real /bin/sh in a session root that is not ``/``.

    Like ``AgentCoreSandbox``, file paths are stored under the session cwd and
    the shell starts in that cwd.
    """

    def __init__(self, root: Path):
        self.root = str(root)
        self.commands: list[str] = []

    def _get_cwd(self) -> str:
        return self.root

    def execute(self, command, *, timeout=None):
        self.commands.append(command)
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return SimpleNamespace(output=done.stdout + done.stderr, exit_code=done.returncode)

    def write(self, file_path, content):
        real = (
            file_path
            if file_path.startswith(self.root + "/")
            else f"{self.root}/{file_path.lstrip('/')}"
        )
        Path(real).parent.mkdir(parents=True, exist_ok=True)
        Path(real).write_text(content)
        return SimpleNamespace(path=file_path, error=None)


class _RecordingOpenSandbox:
    """OpenSandbox-like backend: no session cwd, file paths are absolute."""

    def __init__(self):
        self.commands: list[str] = []

    def execute(self, command, *, timeout=None):
        self.commands.append(command)
        return SimpleNamespace(output="", exit_code=0)


def _agentcore(tmp_path: Path):
    inner = _LocalAgentCore(tmp_path)
    return inner, prepare_workspace_layout(_bind_skills_shell(inner))


@pytest.mark.no_db
@pytest.mark.parametrize(
    "written, run",
    [
        ("workspace/count.py", "python3 workspace/count.py"),
        ("workspace/count.py", "python3 count.py"),
        ("workspace/count.py", "python3 /workspace/count.py"),
        ("workspace/count.py", "python3 ./workspace/count.py"),
        ("count.py", "python3 workspace/count.py"),
        ("/workspace/count.py", "cd /tmp && python3 workspace/count.py"),
    ],
)
def test_a_written_file_runs_by_the_path_the_model_names(tmp_path, written, run):
    _, backend = _agentcore(tmp_path)

    backend.write(written, "print(6 * 7)\n")
    result = backend.execute(run)

    assert (tmp_path / "workspace" / "count.py").is_file()
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "42"


@pytest.mark.no_db
def test_single_quotes_inside_double_quotes_reach_python(tmp_path):
    _, backend = _agentcore(tmp_path)
    command = (
        'python3 -c "dice = [1, 2, 3]; total = sum(dice) + 2; '
        "print(f'Dice rolls: {dice[0]}, {dice[1]}, {dice[2]}'); "
        "print(f'Total: {total}')\""
    )

    result = backend.execute(command)

    assert result.exit_code == 0, result.output
    assert result.output == "Dice rolls: 1, 2, 3\nTotal: 8\n"


@pytest.mark.no_db
def test_quotes_and_dollar_signs_are_left_to_the_shell(tmp_path):
    _, backend = _agentcore(tmp_path)
    command = """X=7; printf '%s\\n' "a'b" 'c"d' '$X' "$X" "it's \\"quoted\\"" """

    result = backend.execute(command)

    assert result.exit_code == 0, result.output
    assert result.output == "a'b\nc\"d\n$X\n7\nit's \"quoted\"\n"


@pytest.mark.no_db
def test_a_command_is_its_own_script(tmp_path):
    _, backend = _agentcore(tmp_path)

    assert backend.execute("# nothing to run").exit_code == 0
    multi = backend.execute("# roll\necho one; false || echo two")
    assert multi.output == "one\ntwo\n"


@pytest.mark.no_db
def test_skills_paths_are_still_rewritten(tmp_path):
    inner, backend = _agentcore(tmp_path)
    (tmp_path / "skills" / "roll").mkdir(parents=True)
    (tmp_path / "skills" / "roll" / "roll.py").write_text("print('rolled')\n")

    result = backend.execute("python3 /skills/roll/roll.py")

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "rolled"


@pytest.mark.no_db
def test_opensandbox_runs_in_the_workspace_the_file_tools_use():
    inner = _RecordingOpenSandbox()
    backend = prepare_workspace_layout(inner)

    backend.execute("python3 workspace/count.py && echo 'done'")

    assert inner.commands[-1] == (
        "mkdir -p /workspace && cd /workspace && exec sh -c "
        "'python3 /workspace/count.py && echo '\"'\"'done'\"'\"''"
    )


@pytest.mark.no_db
def test_internal_listing_commands_skip_the_model_path_rewrite():
    inner = _RecordingOpenSandbox()
    backend = prepare_workspace_layout(inner)
    listing = _find_command("workspace/sub/a.txt", 10)

    _command(backend, listing)

    assert inner.commands[-1] == f"cd / && {listing}"
    assert "/workspace/sub" not in inner.commands[-1]


@pytest.mark.no_db
def test_agentcore_internal_commands_stay_raw(tmp_path):
    inner, backend = _agentcore(tmp_path)

    _command(backend, "ps -eo pid=,args=")

    assert inner.commands[-1] == "ps -eo pid=,args="
