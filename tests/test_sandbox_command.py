"""A sandbox shell command runs from the workspace.

Its text reaches the shell unchanged except for the absolute ``/workspace``
token, which names the real workspace where the session root is not ``/``.
"""

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
    translate_workspace_paths,
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
        self.edits: list[str] = []

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

    def edit(self, file_path, old_string, new_string, replace_all=False):
        self.edits.append(old_string)
        cwd = self._get_cwd()
        real = file_path if file_path.startswith(cwd + "/") else f"{cwd}/{file_path.lstrip('/')}"
        text = Path(real).read_text()
        count = text.count(old_string)
        # Same messages as deepagents' _map_edit_error.
        if not count:
            return SimpleNamespace(path=None, error=f"Error: String not found in file: '{old_string}'")
        if count > 1 and not replace_all:
            return SimpleNamespace(
                path=None,
                error=f"Error: String '{old_string}' appears {count} times in file.",
            )
        Path(real).write_text(text.replace(old_string, new_string, -1 if replace_all else 1))
        return SimpleNamespace(path=real, error=None, occurrences=count)

    def grep(self, pattern, path=None, glob=None):
        # Like BaseSandbox.grep: the pattern is plain text in the command and
        # the command goes through self.execute. Paths resolve like write.
        cwd = self._get_cwd()
        target = path or cwd
        if not target.startswith(cwd + "/"):
            target = f"{cwd}/{target.lstrip('/')}"
        result = self.execute(f"grep -rHnF {shlex.quote(pattern)} {shlex.quote(target)}")
        return SimpleNamespace(matches=result.output.splitlines(), error=None)


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


def test_a_relative_workspace_prefix_fails_loudly_in_the_shell(agentcore):
    _, backend = agentcore
    backend.write("count.py", "print(6 * 7)\n")

    result = backend.execute("python3 workspace/count.py")

    assert result.exit_code != 0
    assert "workspace/count.py" in result.output


def test_the_instruction_says_how_to_name_a_written_file():
    assert "The shell starts there." in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "`python3 x`" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "`x` or `/workspace/x`" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "never `workspace/x`" in _ARTIFACT_MARKDOWN_INSTRUCTION
    assert "Those paths do not exist there." not in _ARTIFACT_MARKDOWN_INSTRUCTION


# Dev, 2026-11-19 (AgentCore, deep graph): the model wrote /workspace/plot_ndvi.py,
# ran `python3 /workspace/plot_ndvi.py` and saved with
# plt.savefig('/workspace/ndvi.png'). AgentCore has no /workspace and cannot
# create one, so each failed with "No such file or directory".


def test_a_file_written_as_slash_workspace_runs_by_that_path(agentcore):
    _, backend = agentcore

    backend.write("/workspace/plot_ndvi.py", "print(6 * 7)\n")
    result = backend.execute("python3 /workspace/plot_ndvi.py")

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "42"


def test_slash_workspace_works_from_another_directory(agentcore):
    _, backend = agentcore
    backend.write("count.py", "print(6 * 7)\n")

    result = backend.execute("cd /tmp && python3 /workspace/count.py")

    assert result.output.strip() == "42"


def test_a_written_script_saves_to_slash_workspace(agentcore):
    inner, backend = agentcore
    code = "with open('/workspace/ndvi.png', 'w') as f:\n    f.write('png')\n"

    backend.write("/workspace/plot_ndvi.py", code)
    result = backend.execute("python3 plot_ndvi.py")

    assert result.exit_code == 0, result.output
    assert (inner.root / "workspace" / "ndvi.png").read_text() == "png"


def test_code_in_the_command_saves_to_slash_workspace(agentcore):
    inner, backend = agentcore
    command = (
        "python3 - <<'EOF'\n"
        "import os\n"
        "os.makedirs('/workspace/out', exist_ok=True)\n"
        "open(os.path.join('/workspace', 'out', 'a.txt'), 'w').write('ok')\n"
        "EOF"
    )

    result = backend.execute(command)

    assert result.exit_code == 0, result.output
    assert (inner.root / "workspace" / "out" / "a.txt").read_text() == "ok"


@pytest.mark.parametrize(
    "name", ["config.json", "params.yaml", "paths.txt", "notes.md", "Makefile", "run"]
)
def test_a_text_file_names_the_real_workspace_like_a_heredoc(agentcore, name):
    inner, backend = agentcore
    text = "out: /workspace/x\n"

    backend.write(name, text)
    backend.execute(f"cat > heredoc-{name} <<'EOF'\n{text}EOF")

    real = f"out: {inner.root}/workspace/x\n"
    assert (inner.root / "workspace" / name).read_text() == real
    assert (inner.root / "workspace" / f"heredoc-{name}").read_text() == real


@pytest.mark.parametrize("name", ["a.png", "b.PDF", "c.parquet", "d.zip"])
def test_a_binary_file_is_never_translated(agentcore, name):
    inner, backend = agentcore
    text = "/workspace/x"

    backend.write(name, text)

    assert (inner.root / "workspace" / name).read_text() == text


def test_an_edit_matches_the_path_the_model_wrote(agentcore):
    inner, backend = agentcore
    backend.write("plot.py", "out = '/workspace/a.png'\n")

    result = backend.edit("plot.py", "'/workspace/a.png'", "'/workspace/b.png'")

    assert result.error is None
    assert backend.execute("python3 -c \"exec(open('plot.py').read()); print(out)\"").output.strip() == (
        f"{inner.root}/workspace/b.png"
    )


def test_an_edit_still_matches_a_file_saved_with_the_virtual_path(agentcore):
    inner, backend = agentcore
    (inner.root / "workspace").mkdir(exist_ok=True)
    (inner.root / "workspace" / "old.py").write_text("out = '/workspace/a.png'\n")

    result = backend.edit("old.py", "'/workspace/a.png'", "'/workspace/b.png'")

    assert result.error is None
    assert (inner.root / "workspace" / "old.py").read_text() == (
        f"out = '{inner.root}/workspace/b.png'\n"
    )


def test_an_edit_tries_the_quoted_string_first(agentcore):
    # A restored legacy line and a later translated line: the model quoted the
    # legacy one, so that one changes.
    inner, backend = agentcore
    path = inner.root / "workspace" / "mixed.py"
    path.parent.mkdir(exist_ok=True)
    real = f"{inner.root}/workspace"
    path.write_text(f"a = '/workspace/a.png'\na = '{real}/a.png'\n")

    result = backend.edit("mixed.py", "a = '/workspace/a.png'", "a = '/workspace/c.png'")

    assert result.error is None
    assert path.read_text() == f"a = '{real}/c.png'\na = '{real}/a.png'\n"
    assert inner.edits == ["a = '/workspace/a.png'"]


def test_an_edit_retries_only_when_the_string_is_not_found(agentcore):
    inner, backend = agentcore
    path = inner.root / "workspace" / "twice.py"
    path.parent.mkdir(exist_ok=True)
    real = f"{inner.root}/workspace"
    text = f"a = '/workspace/a.png'\na = '/workspace/a.png'\nb = '{real}/a.png'\n"
    path.write_text(text)

    result = backend.edit("twice.py", "'/workspace/a.png'", "'/workspace/c.png'")

    assert "appears 2 times" in result.error
    assert path.read_text() == text
    assert inner.edits == ["'/workspace/a.png'"]


def test_the_grep_tool_searches_for_the_pattern_as_written(agentcore):
    inner, backend = agentcore
    (inner.root / "workspace").mkdir(exist_ok=True)
    (inner.root / "workspace" / "notes.md").write_text("see /workspace/x\n")

    result = backend.grep("/workspace/x", "notes.md")

    assert any("see /workspace/x" in line for line in result.matches), result.matches


def test_a_model_shell_command_names_the_real_workspace(agentcore):
    inner, backend = agentcore
    backend.write("count.py", "print(1)\n")

    result = backend.execute("ls /workspace/count.py")

    assert result.exit_code == 0, result.output
    assert f"{inner.root}/workspace/count.py" in result.output


def test_a_translation_is_logged_with_the_count_only(agentcore, caplog):
    _, backend = agentcore
    caplog.set_level(logging.INFO, logger="src.services.agents.core.sandbox")

    backend.execute("echo secret-text /workspace/a /workspace/b")

    lines = [r.getMessage() for r in caplog.records if "path_translated" in r.getMessage()]
    assert lines == ["sandbox.path_translated count=2 tool=execute"]


def test_an_escaped_sed_on_a_legacy_script_still_runs(agentcore):
    inner, backend = agentcore
    path = inner.root / "workspace" / "old.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text("out = '/workspace/a.png'\n")

    result = backend.execute("sed -i 's/\\/workspace\\//.\\//g' old.py")

    assert result.exit_code == 0, result.output
    assert path.read_text() == "out = './a.png'\n"


@pytest.mark.parametrize(
    "text",
    [
        "workspace/a",
        "./workspace/a",
        "~/workspace/a",
        "${HOME}/workspace/a",
        "$(pwd)/workspace/a",
        "/workspaces/a",
        "/workspace.py",
        "/data/workspace/a",
        "https://host/workspace/a",
        "@workspace/a",
        "\\/workspace\\/a",
        "s/\\/workspace\\//.\\//g",
        '"$HOME"/workspace',
        "`pwd`/workspace",
        '"$(pwd)"/workspace',
        '$ROOT"/workspace"',
        "root + '/workspace'",
        'root+"/workspace/a"',
        "$DIR'/workspace'",
    ],
)
def test_only_the_absolute_workspace_token_is_translated(text):
    assert translate_workspace_paths(text, "/opt/s/workspace") == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/workspace", "/opt/s/workspace"),
        ("/workspace/a.png", "/opt/s/workspace/a.png"),
        ("'/workspace/a'", "'/opt/s/workspace/a'"),
        ("--out=/workspace/a", "--out=/opt/s/workspace/a"),
        ("cp a /workspace/", "cp a /opt/s/workspace/"),
        ("PATH=/bin:/workspace/bin", "PATH=/bin:/opt/s/workspace/bin"),
        ("echo hi >/workspace/a", "echo hi >/opt/s/workspace/a"),
        ("open('/workspace/a')", "open('/opt/s/workspace/a')"),
        ("f'/workspace/{n}.png'", "f'/opt/s/workspace/{n}.png'"),
        ('rb"/workspace/a"', 'rb"/opt/s/workspace/a"'),
        ('{"out": "/workspace/a"}', '{"out": "/opt/s/workspace/a"}'),
        ('["/workspace/a", "/workspace/b"]', '["/opt/s/workspace/a", "/opt/s/workspace/b"]'),
        ("x=1\n/workspace/a", "x=1\n/opt/s/workspace/a"),
        ("cat '/workspace/a'", "cat '/opt/s/workspace/a'"),
        ("chmod +x /workspace/a.sh", "chmod +x /opt/s/workspace/a.sh"),
    ],
)
def test_the_absolute_workspace_token_points_at_the_real_directory(text, expected):
    assert translate_workspace_paths(text, "/opt/s/workspace") == expected


def test_the_translation_runs_once():
    once = translate_workspace_paths("python3 /workspace/a.py", "/opt/s/workspace")

    assert translate_workspace_paths(once, "/opt/s/workspace") == once


@pytest.mark.parametrize("workspace_dir", [None, "", "/workspace"])
def test_no_translation_where_slash_workspace_is_real_or_unknown(workspace_dir):
    text = "python3 /workspace/a.py"

    assert translate_workspace_paths(text, workspace_dir) == text


@pytest.mark.parametrize(
    "command",
    [
        "grep -c '^workspace/a' data.txt",
        "sed -n 's#workspace/a#HIT#p' data.txt",
        "python3 -c \"import re; print(re.findall(r'workspace/(\\w+)', open('data.txt').read()))\"",
        "echo 'Saved to workspace/out.png' @workspace/foo",
        "cat <<'EOF'\nworkspace/a ./workspace/c $HOME `x`\nEOF",
        "X=7; printf '%s\\n' \"a'b\" 'c\"d' '$X' \"$X\" \"it's \\\"quoted\\\"\"",
        "python3 -c \"d = [1, 2]; print(f'Dice: {d[0]}, {d[1]}')\"",
    ],
)
def test_a_command_without_the_workspace_token_reaches_the_shell_unchanged(agentcore, tmp_path, command):
    _, backend = agentcore
    data = "workspace/a\nworkspace/b\n/workspace/c\n"
    backend.write("data.txt", data)
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "data.txt").write_text(data)

    result = backend.execute(command)

    assert result.exit_code == 0, result.output
    assert result.output == _sh(command, plain).output


def test_a_quoted_heredoc_body_keeps_its_quotes_and_dollars(agentcore):
    inner, backend = agentcore
    body = "path = 'workspace/a'\nroot = \"$HOME\"\ncd workspace\n"

    backend.execute(f"cat > out.txt <<'EOF'\n{body}EOF")

    assert (inner.root / "workspace" / "out.txt").read_text() == body


def test_a_quoted_heredoc_body_names_the_real_workspace(agentcore):
    inner, backend = agentcore
    body = "root = \"/workspace\"\nout = '$HOME'\n"

    backend.execute(f"cat > out.txt <<'EOF'\n{body}EOF")

    assert (inner.root / "workspace" / "out.txt").read_text() == (
        f"root = \"{inner.root}/workspace\"\nout = '$HOME'\n"
    )


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


def test_opensandbox_commands_keep_the_absolute_workspace():
    inner = _RecordingOpenSandbox()
    backend = prepare_workspace_layout(inner)
    command = "python3 /workspace/count.py"

    backend.execute(command)

    assert shlex.split(inner.commands[-1])[-1] == command


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
