"""Windows, simulated from a POSIX box. No models, no Windows, no subprocesses.

Every platform-dependent decision in this repo goes through a function that
takes the platform as an argument or reads `sys.platform` at call time — which
is what makes them testable here at all. The three that would otherwise only be
discovered on a real Windows machine, each of them silent rather than loud:

* the install table (a `brew install` line typed at a PowerShell prompt does
  nothing but confuse),
* the cancel path (`os.killpg` raises `AttributeError` on Windows, so a job the
  UI says is stopping keeps rendering), and
* the parent watchdog (`getppid()` never changes there, so it polls forever and
  never fires — a watchdog that reports for duty and sleeps).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from dubbing import tools
from dubbing_app import install as install_mod
from dubbing_app import runner as runner_mod
from dubbing_app import server as server_mod

WINDOWS, MAC, LINUX = "win32", "darwin", "linux"


# ---------------------------------------------------------------------------
# the install table
# ---------------------------------------------------------------------------

def test_each_platform_installs_the_tools_its_own_way():
    assert tools.command("ffmpeg", MAC) == "brew install ffmpeg"
    assert tools.command("sox", MAC) == "brew install sox"
    win = tools.command("ffmpeg", WINDOWS)
    assert win.startswith("winget install --id Gyan.FFmpeg -e")
    assert "--accept-source-agreements" in win and "--accept-package-agreements" in win
    assert tools.command("sox", WINDOWS).startswith("winget install --id ChrisBagwell.SoX -e")
    assert tools.command("ffmpeg", LINUX) == "sudo apt-get install -y ffmpeg"


def test_an_unknown_platform_is_treated_as_a_posix_box_not_as_a_hole():
    assert tools.platform_key("freebsd13") == LINUX
    assert tools.command("ffmpeg", "freebsd13") == "sudo apt-get install -y ffmpeg"
    assert tools.platform_key("win32") == WINDOWS
    # Cygwin is a POSIX environment with a POSIX package manager, not a Windows
    # one — `winget install` is the wrong answer there and `sys.platform` says so.
    assert tools.platform_key("cygwin") == LINUX


def test_only_the_managers_that_run_unattended_get_a_button():
    """`sudo apt-get` asks for a password on a terminal the app does not have;
    a spinner in front of a hidden prompt is a hang, not an install."""
    assert set(tools.auto_installers(MAC)) == {"ffmpeg", "sox", "uv"}
    assert set(tools.auto_installers(WINDOWS)) == {"ffmpeg", "sox", "uv"}
    assert tools.auto_installers(LINUX) == {}
    assert tools.unattended("ffmpeg", WINDOWS) and not tools.unattended("ffmpeg", LINUX)


def test_linux_is_told_nothing_about_installing_uv_because_there_is_nothing_true():
    """The one recipe that is deliberately missing rather than merely absent.

    brew and winget both publish uv, so `brew install uv` and `winget install
    --id astral-sh.uv` are lines a user can actually type. No Debian or Ubuntu
    does, so a `sudo apt-get install -y uv` line here would be a sentence the
    Setup screen printed and no machine on earth could run. Silence is the
    honest row, and the button on Linux comes from the release archive instead,
    which needs no package manager at all.
    """
    assert tools.command("uv", MAC) == "brew install uv"
    assert tools.command("uv", WINDOWS).startswith("winget install --id astral-sh.uv -e")
    assert tools.command("uv", LINUX) is None
    assert tools.install_hint("uv", LINUX) == ""


def test_the_module_table_is_this_machines_row():
    assert install_mod.INSTALLERS == tools.auto_installers(sys.platform)


def test_a_tool_with_no_button_still_hands_over_its_command():
    """The refusal, and the check's detail line, are all a Linux user gets —
    so they have to carry the command `dubbing/tools.py` knows."""
    assert install_mod.manual_command("ffmpeg", recipes={}) == tools.command("ffmpeg")
    inst = install_mod.Installer(lambda id_: None, recipes={})
    message = inst._refusal("model.translate")
    assert tools.command("ffmpeg") in message and tools.command("sox") in message


def test_every_manager_the_table_names_has_a_sentence_for_being_missing():
    """A manager with no message falls back to a generic line that names it —
    fine, but every one we ship a recipe for deserves its own URL."""
    for platform in (MAC, WINDOWS, LINUX):
        for argv in tools.recipes(platform).values():
            assert argv[0] in install_mod.MANAGERS, argv[0]
            filled = install_mod.MANAGERS[argv[0]].format(
                tool="ffmpeg", command=" ".join(argv), manager=argv[0])
            assert "{" not in filled


def test_a_windows_without_winget_still_gets_ffmpeg_and_is_told_about_the_rest(monkeypatch):
    """No winget is not the end of the road for the one tool every stage needs:
    the same static build a brewless Mac gets. `sox` has no such fallback, so it
    is still the refusal that names where winget comes from — which is the whole
    difference between "we cannot help you" and "we cannot help you with this"."""
    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    monkeypatch.setattr(tools, "platform_key", lambda *a: WINDOWS)
    assert install_mod.static_route("ffmpeg") is True
    assert "static build" in (install_mod.route("ffmpeg") or "")

    inst = install_mod.Installer(lambda id_: None,
                                 recipes=tools.auto_installers(WINDOWS))
    with pytest.raises(Exception) as exc:
        inst.start("sox")
    assert "winget" in exc.value.message and "Microsoft Store" in exc.value.message


def test_a_missing_tool_row_names_this_platforms_command(monkeypatch):
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(setup_mod.shutil, "which", lambda exe, *a, **k: None)
    row = setup_mod.tool("ffmpeg", *setup_mod.TOOLS["ffmpeg"])
    assert row["ok"] is False and tools.command("ffmpeg") in row["detail"]


# ---------------------------------------------------------------------------
# cancelling a job
# ---------------------------------------------------------------------------

class FakeProc:
    """Enough of `Popen` for the cancel path: what was asked of it, in order."""

    def __init__(self, alive: bool = True, pid: int = 4321):
        self.pid = pid
        self.calls: list[str] = []
        self.signals: list[int] = []
        self._alive = alive

    def poll(self):
        return None if self._alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self.calls.append("send_signal")

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


def test_the_job_child_is_detached_the_way_each_platform_spells_it():
    assert runner_mod.spawn_kwargs(LINUX) == {"start_new_session": True}
    assert runner_mod.spawn_kwargs(MAC) == {"start_new_session": True}
    win = runner_mod.spawn_kwargs(WINDOWS)
    assert win == {"creationflags": runner_mod.CREATE_NEW_PROCESS_GROUP}
    assert "start_new_session" not in win      # meaningless there, and a TypeError risk


def test_a_soft_cancel_on_windows_is_a_ctrl_break_not_a_killpg(monkeypatch):
    monkeypatch.setattr(runner_mod.os, "killpg",
                        lambda *a: pytest.fail("killpg has no meaning on Windows"))
    proc = FakeProc()
    runner_mod.terminate_tree(proc, hard=False, platform=WINDOWS)
    assert proc.calls == ["send_signal"]


def test_a_hard_cancel_on_windows_takes_the_children_with_it(monkeypatch):
    """`TerminateProcess` does not touch grandchildren, and the pipeline's
    grandchildren are a yt-dlp download and an ffmpeg re-encode."""
    ran: list[list[str]] = []
    monkeypatch.setattr(runner_mod.subprocess, "run",
                        lambda argv, **kw: ran.append(argv))
    proc = FakeProc()
    runner_mod.terminate_tree(proc, hard=True, platform=WINDOWS)
    assert ran and ran[0][:3] == ["taskkill", "/F", "/T"]
    assert ran[0][-1] == str(proc.pid)
    assert proc.calls == ["kill"]              # still alive after taskkill: make sure


def test_a_windows_cancel_never_ends_in_a_still_running_child(monkeypatch):
    """Whatever fails — no group, no taskkill — the child itself is killed."""
    def boom(*a, **kw):
        raise OSError("taskkill is not on PATH")

    monkeypatch.setattr(runner_mod.subprocess, "run", boom)
    proc = FakeProc()
    runner_mod.terminate_tree(proc, hard=True, platform=WINDOWS)
    assert proc.calls == ["kill"]


def test_posix_cancel_is_unchanged(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(runner_mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(runner_mod.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    proc = FakeProc()
    runner_mod.terminate_tree(proc, hard=False, platform=LINUX)
    runner_mod.terminate_tree(proc, hard=True, platform=LINUX)
    import signal as signal_mod
    assert sent == [(proc.pid, signal_mod.SIGTERM), (proc.pid, signal_mod.SIGKILL)]
    assert proc.calls == []


def test_a_posix_group_that_is_already_gone_falls_back_to_the_process(monkeypatch):
    monkeypatch.setattr(runner_mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(runner_mod.os, "killpg",
                        lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    proc = FakeProc()
    runner_mod.terminate_tree(proc, hard=False, platform=LINUX)
    runner_mod.terminate_tree(proc, hard=True, platform=LINUX)
    assert proc.calls == ["terminate", "kill"]


# ---------------------------------------------------------------------------
# the server's watchdogs
# ---------------------------------------------------------------------------

def test_the_ppid_watchdog_declines_on_windows():
    """Windows does not reparent, so the pid it polls never changes. Better to
    run no watchdog than one that can only ever say 'still fine'."""
    fired: list[bool] = []
    assert server_mod.watchdog(on_orphan=lambda: fired.append(True),
                               interval=0.01, platform=WINDOWS) is None
    assert fired == []


def test_the_ppid_watchdog_still_runs_on_posix():
    thread = server_mod.watchdog(on_orphan=lambda: None, interval=5.0, platform=LINUX)
    # None only when this process is already an orphan, which the test runner is not.
    assert thread is None or thread.is_alive()


def test_the_stdin_watchdog_is_the_platform_independent_one():
    """The pipe does not lie anywhere, which is why Windows relies on it."""
    import io
    closed: list[bool] = []
    thread = server_mod.stdin_watchdog(stream=io.BytesIO(b""),
                                       on_close=lambda: closed.append(True))
    thread.join(5.0)
    assert closed == [True]


# ---------------------------------------------------------------------------
# text that is not ASCII, on a console that is not UTF-8
# ---------------------------------------------------------------------------

class FakeStream:
    def __init__(self, encoding: str):
        self.encoding = encoding
        self.reconfigured: list[str] = []

    def reconfigure(self, encoding: str, errors: str = "strict"):
        self.reconfigured.append(encoding)
        self.encoding = encoding


def test_a_windows_console_is_switched_to_utf8_and_a_posix_one_is_left_alone():
    cp1252, utf8 = FakeStream("cp1252"), FakeStream("utf-8")
    tools.utf8_stdio((cp1252, utf8))
    assert cp1252.reconfigured == ["utf-8"]
    assert utf8.reconfigured == []             # already right; do not touch it


def test_a_stream_that_cannot_be_reconfigured_is_not_an_error():
    class Stubborn(FakeStream):
        def reconfigure(self, encoding, errors="strict"):
            raise OSError("not seekable")

    tools.utf8_stdio((Stubborn("cp1252"), object()))


# ---------------------------------------------------------------------------
# the translation worker's pipe
# ---------------------------------------------------------------------------

def test_the_worker_reads_its_pipe_without_select_on_windows(monkeypatch):
    """`select` on Windows speaks only sockets, so a readiness poll on the pipe
    would fail on the very first line the worker prints — its ready line. The
    reader thread + queue design never touches `select` on any platform."""
    import queue

    from dubbing import translate

    handle = translate.WorkerHandle.__new__(translate.WorkerHandle)
    handle._stderr_tail = __import__("collections").deque()
    handle._lines = queue.Queue()

    class Proc:
        returncode = None
        stdout = iter(['{"ready": true}\n'])

        def poll(self):
            return None

    handle._proc = Proc()
    monkeypatch.setattr(translate.sys, "platform", WINDOWS)
    monkeypatch.setattr(translate, "select", None, raising=False)
    handle._pump_stdout()                    # what the reader thread would do
    assert handle._read_line(5.0) == '{"ready": true}\n'


def test_a_dead_windows_worker_is_an_error_not_a_hang():
    import queue

    from dubbing import translate

    handle = translate.WorkerHandle.__new__(translate.WorkerHandle)
    handle._stderr_tail = __import__("collections").deque(["out of memory\n"])
    handle._lines = queue.Queue()

    class Dead:
        returncode = 137
        stdout: list[str] = []

        def poll(self):
            return 137

        def wait(self, timeout=None):
            return 137

    handle._proc = Dead()
    handle._pump_stdout()                    # stdout is closed: queues the EOF mark
    with pytest.raises(RuntimeError) as exc:
        handle._read_line(1.0)
    assert "died" in str(exc.value) and "out of memory" in str(exc.value)


# ---------------------------------------------------------------------------
# spawn-safety: what a Windows multiprocessing child re-imports
# ---------------------------------------------------------------------------

def test_the_entry_points_do_not_run_themselves_on_import():
    """Windows starts multiprocessing children with `spawn`, which re-imports
    the entry module. An unguarded `main()` would start a second run inside
    every worker process torch or demucs ever creates."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("dubbing/__main__.py", "dubbing_app/server.py", "dubbing_app/worker.py",
                "translator/worker.py"):
        source = (root / rel).read_text(encoding="utf-8")
        body = [ln for ln in source.splitlines()
                if ln.startswith(("main(", "raise SystemExit(", "sys.exit("))]
        assert body == [], f"{rel} runs {body} at import time"
        assert 'if __name__ == "__main__":' in source, rel


def test_no_module_reaches_for_a_posix_only_import():
    """fcntl/termios/pty do not exist on Windows; an import of one at module
    scope makes the whole package unimportable there."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    banned = ("import fcntl", "import termios", "import pty", "import grp", "import pwd")
    for path in list((root / "dubbing").rglob("*.py")) + \
            list((root / "dubbing_app").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for name in banned:
            assert name not in source, f"{path.name}: {name}"


def test_the_pipeline_never_hardcodes_a_posix_process_call_outside_the_runner():
    """`os.killpg`, `os.setsid` and `preexec_fn` are POSIX-only. The one module
    that may name them is `dubbing_app/runner.py`, which branches on platform."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in list((root / "dubbing").rglob("*.py")) + \
            list((root / "dubbing_app").rglob("*.py")):
        if path.name == "runner.py":
            continue
        source = path.read_text(encoding="utf-8")
        for name in ("os.killpg", "os.setsid", "preexec_fn", "os.getpgid"):
            assert name not in source, f"{path.name}: {name}"


def test_external_tools_are_resolved_with_which_not_assumed():
    """`shutil.which` is what makes `ffmpeg` find `ffmpeg.exe`; a bare
    existence check against a path would not. The lookup lives in
    `tools.resolve_tool` now (env override, workspace tools/bin, then PATH),
    and `which` is still its floor."""
    from dubbing import audio

    inspect = __import__("inspect")
    assert "resolve_tool" in inspect.getsource(audio.require_tools)
    assert "shutil.which" in inspect.getsource(tools.resolve_tool)


def test_the_ffmpeg_refusal_names_this_platforms_installer(monkeypatch):
    from dubbing import audio

    monkeypatch.setattr(audio.tools, "resolve_tool", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        audio.require_tools()
    assert "ffmpeg not found on PATH" in str(exc.value)
    assert tools.command("ffmpeg") in str(exc.value)


def test_subprocess_has_the_windows_flag_spelled_out_off_windows():
    """The constant only exists on Windows, so the value is written down —
    if that ever drifts, a job child spawns without its own group."""
    assert runner_mod.CREATE_NEW_PROCESS_GROUP == getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


# ---------------------------------------------------------------------------
# uv discovery, which was Mac-only and therefore wrong on two platforms
# ---------------------------------------------------------------------------
# `setup.find_uv` is the one probe that does not go through `resolve_tool`, and
# it is the one that had drifted: two hardcoded Homebrew paths, and a last
# resort that spelled the binary `uv` with no suffix. On Windows uv installs to
# `%USERPROFILE%\.local\bin\uv.exe`, which that chain looked straight past, so
# a machine with uv installed exactly where its own installer puts it reported
# the tool missing. These tests are the only place that behaviour can be pinned
# from here, so they pin all three platforms rather than only the broken one.


def _isolate_uv_env(monkeypatch, home):
    """A machine with nothing on PATH, no override, no uv at any of the literal
    install paths, and `home` for a home.

    The literal paths are emptied rather than left alone because the machine
    running this suite very probably has `/opt/homebrew/bin/uv` on it, and a
    chain that answers from rung three never reaches the rung under test. What
    those lists contain is asserted separately, where it is the subject.
    """
    from dubbing_app import setup as setup_mod

    monkeypatch.delenv(setup_mod.UV_PATH_ENV, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("USERPROFILE", raising=False)
    for key in setup_mod.UV_FALLBACKS:
        monkeypatch.setitem(setup_mod.UV_FALLBACKS, key, ())
    return setup_mod


def test_windows_finds_the_uv_its_own_installer_wrote(tmp_path, monkeypatch):
    """The reported bug, in one assertion.

    `install-server.ps1` and uv's official installer both write
    `%USERPROFILE%\\.local\\bin\\uv.exe`, and neither of them puts that directory
    on a GUI process's PATH. The old chain probed two Homebrew paths that do not
    exist on Windows and then a suffixless `~/.local/bin/uv`, so the row said
    MISSING and REQUIRED on a machine whose uv was installed correctly.
    """
    setup_mod = _isolate_uv_env(monkeypatch, tmp_path / "home")
    local = tmp_path / "home" / ".local" / "bin"
    local.mkdir(parents=True)

    # The suffixless spelling is not what Windows has, and must not answer.
    (local / "uv").write_text("not the binary\n")
    assert setup_mod.find_uv(WINDOWS) is None

    (local / "uv.exe").write_text("MZ\n")
    assert setup_mod.find_uv(WINDOWS) == str(local / "uv.exe")
    # And the same tree on the other two, where the suffixless name is right.
    assert setup_mod.find_uv(MAC) == str(local / "uv")
    assert setup_mod.find_uv(LINUX) == str(local / "uv")


def test_a_cargo_install_is_found_on_every_platform(tmp_path, monkeypatch):
    """`cargo install uv` lands in `~/.cargo/bin`, which the shell's chain has
    checked all along and this one had never heard of."""
    setup_mod = _isolate_uv_env(monkeypatch, tmp_path / "home")
    cargo = tmp_path / "home" / ".cargo" / "bin"
    cargo.mkdir(parents=True)
    for platform_key, exe in ((MAC, "uv"), (LINUX, "uv"), (WINDOWS, "uv.exe")):
        (cargo / exe).write_text("x\n")
        assert setup_mod.find_uv(platform_key) == str(cargo / exe), platform_key


def test_each_platform_probes_its_own_absolute_paths_and_no_others():
    """A Mac path list on a Linux box finds nothing and says "missing"; the
    reverse would find a `/usr/bin/uv` that a Mac does not have. The lists are
    `workspace.rs`'s three, verbatim."""
    from dubbing_app import setup as setup_mod

    assert setup_mod.uv_fallbacks(MAC) == ("/opt/homebrew/bin/uv", "/usr/local/bin/uv")
    assert setup_mod.uv_fallbacks(LINUX) == ("/usr/local/bin/uv", "/usr/bin/uv",
                                             "/home/linuxbrew/.linuxbrew/bin/uv")
    # Empty on purpose rather than unfinished: winget and uv's installer both
    # land on PATH or in the per-user `.local\bin` the tail of the chain checks.
    assert setup_mod.uv_fallbacks(WINDOWS) == ()
    assert setup_mod.uv_exe(WINDOWS) == "uv.exe"
    assert setup_mod.uv_exe(MAC) == setup_mod.uv_exe(LINUX) == "uv"


def test_the_path_search_tries_the_exe_spelling_first(tmp_path, monkeypatch):
    """`shutil.which("uv")` appends PATHEXT and never tries the bare name;
    `shutil.which("uv.exe")` never tries anything else. The shell tries both,
    in that order, inside each directory, so this does too. Otherwise the two
    sides can disagree about which uv a machine has."""
    from dubbing_app import setup as setup_mod

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("PATH", f"{first}{__import__('os').pathsep}{second}")
    (first / "uv").write_text("shim\n")
    (first / "uv.exe").write_text("MZ\n")
    assert setup_mod.uv_on_path("uv.exe") == str(first / "uv.exe")
    # A Git-Bash-style extensionless shim is still worth finding when it is the
    # only thing there, which is why the bare name is tried at all.
    (first / "uv.exe").unlink()
    assert setup_mod.uv_on_path("uv.exe") == str(first / "uv")


def test_the_home_directory_is_read_the_way_the_shell_reads_it(monkeypatch):
    """`HOME`, then `USERPROFILE`, which is not what `Path.home()` does.

    Python's `expanduser` reads `USERPROFILE` on Windows and `HOME` everywhere
    else; `workspace.rs` reads `HOME` first on all three, because that is what a
    Git-Bash or MSYS shell sets. Asking the same two variables in the same order
    is what makes "the shell and the server look in the same place" a fact.
    """
    from dubbing_app import setup as setup_mod

    monkeypatch.setenv("HOME", "/h")
    monkeypatch.setenv("USERPROFILE", r"C:\Users\x")
    assert setup_mod.uv_home() == __import__("pathlib").Path("/h")
    monkeypatch.delenv("HOME")
    assert setup_mod.uv_home() == __import__("pathlib").Path(r"C:\Users\x")
    monkeypatch.delenv("USERPROFILE")
    # None rather than a guess: a home-relative path with no home is not a path,
    # and `Path.home()` would raise or invent one.
    assert setup_mod.uv_home() is None


def test_the_bundled_sidecar_reaches_the_server_as_the_override(tmp_path, monkeypatch):
    """Every desktop bundle ships uv as a Tauri `externalBin` sidecar beside the
    app binary, and `runner/process.rs` passes the resolved path down on the
    child's environment. This process cannot resolve it any other way, because
    its `sys.executable` is the venv's Python, so the override is not a
    courtesy here: it is how a desktop install is green at all."""
    from dubbing_app import setup as setup_mod

    sidecar = tmp_path / "MamboDubb.app" / "Contents" / "MacOS" / "uv"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("x\n")
    monkeypatch.setenv(setup_mod.UV_PATH_ENV, str(sidecar))
    for platform_key in (MAC, WINDOWS, LINUX):
        assert setup_mod.find_uv(platform_key) == str(sidecar)


def test_a_missing_uv_is_not_a_reason_a_machine_cannot_dub(tmp_path, monkeypatch):
    """The row is `optional`, and that is the file's own rule applied rather
    than a softening.

    Blocking means "the run fails without it". No run fails: this server is
    already up in its environment, and `runner.SubprocessRunner` spawns every
    job child with `sys.executable`. Graded blocking, it told a desktop user
    whose app had been launched *by* the bundled uv that something REQUIRED was
    MISSING, which is the exact dishonesty the three grades exist to remove.
    """
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(setup_mod, "find_uv", lambda *a, **k: None)
    before = setup_mod.report(tmp_path)
    row = next(c for c in before["checks"] if c["id"] == "uv")
    assert row["ok"] is False
    assert row["severity"] == setup_mod.OPTIONAL and row["required"] is False
    assert "stage" not in row
    # The verdict is the conjunction of the required rows, so uv cannot move it.
    assert before["ok"] == setup_mod.report(tmp_path)["ok"]
    # And the sentence says what is actually lost, without claiming a dub fails.
    assert "uv not found" in row["detail"]
    assert "uv sync" in row["detail"]


def test_the_uv_button_exists_on_every_platform_and_says_which_route(monkeypatch):
    """A REQUIRED row with no button was the complaint; an OPTIONAL row with no
    button would still be a row a user can do nothing about. Package manager
    where there is one to drive, the official release archive where there is
    not, which is every Linux and any Mac or Windows without brew or winget."""
    from dubbing_app import install as inst

    monkeypatch.setattr(inst.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(inst, "INSTALLERS", tools.auto_installers(MAC))
    assert inst.route("uv") == "via Homebrew"
    monkeypatch.setattr(inst, "INSTALLERS", tools.auto_installers(WINDOWS))
    assert inst.route("uv") == "via winget"

    # No manager at all: the release archive, named, with its checksum promise
    # and the directory it lands in, because `~/.local/bin` is on PATH by
    # default on neither Windows nor a bare Linux login shell.
    monkeypatch.setattr(inst.shutil, "which", lambda name: None)
    monkeypatch.setattr(inst, "INSTALLERS", tools.auto_installers(LINUX))
    assert inst.installable("uv") is True
    route = inst.route("uv")
    assert "astral-sh/uv" in route and "SHA-256" in route
    assert str(inst.uv_install_dir()) in route
    # And it is the release route, not a package manager, that would run.
    assert inst.uv_release_route("uv") is True
    assert inst.uv_release_route("ffmpeg") is False


def test_the_uv_button_downloads_verifies_and_only_then_writes(tmp_path, monkeypatch):
    """Astral publishes a `.sha256` beside every archive, so a truncated or
    corrupted transfer is caught before a binary is written rather than after it
    fails to run. That is the whole reason this route is the release archive and
    not `curl … | sh`: a piped script leaves nothing to check.

    No network: `download` is a module global exactly so a test replaces it.
    """
    import hashlib
    import io
    import tarfile

    from dubbing_app import install as inst

    payload = b"#!/bin/sh\necho uv\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    blob = buf.getvalue()

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.setattr(inst, "uv_triple", lambda: "x86_64-unknown-linux-gnu")

    def fake_download(url, timeout=180.0):
        assert url.startswith(inst.UV_RELEASES), url
        if url.endswith(".sha256"):
            return f"{hashlib.sha256(blob).hexdigest()}  uv.tar.gz\n".encode()
        return blob

    monkeypatch.setattr(inst, "download", fake_download)
    lines: list[str] = []
    target = inst.install_uv(lines.append)
    assert target == home / ".local" / "bin" / "uv"
    assert target.read_bytes() == payload
    assert any("checksum verified" in line for line in lines)
    # Nothing is left behind half-written next to it.
    assert not (target.parent / "uv.incoming").exists()

    # A checksum that does not match is a refusal, and it refuses *before* the
    # binary is written: the old one on disk is still the one that runs.
    target.write_bytes(b"the previous uv\n")
    monkeypatch.setattr(inst, "download",
                        lambda url, timeout=180.0: b"0" * 64 if url.endswith(".sha256")
                        else blob)
    with pytest.raises(RuntimeError, match="published checksum"):
        inst.install_uv(lines.append)
    assert target.read_bytes() == b"the previous uv\n"


def test_the_diarization_restore_line_is_for_the_shell_that_will_read_it(monkeypatch):
    """`cp -R` at a PowerShell prompt answers "a positional parameter cannot be
    found", which is the worst kind of instruction: it looks like help, it is
    confidently wrong, and the user cannot tell which of the two of you is
    mistaken. `git` is spelled the same everywhere, so only the copy branch
    splits."""
    from pathlib import Path

    from dubbing_app import install as inst

    monkeypatch.setattr(inst, "diarization_source",
                        lambda: ("copy", Path("/src/weights")))
    monkeypatch.setattr(inst, "diarization_target", lambda: Path("/dst/weights"))
    monkeypatch.setattr(inst.tools, "platform_key", lambda *a: WINDOWS)
    assert inst.diarization_command().startswith("robocopy ")
    monkeypatch.setattr(inst.tools, "platform_key", lambda *a: MAC)
    assert inst.diarization_command().startswith("cp -R ")


def test_the_caches_are_resolved_the_way_their_libraries_resolve_them(tmp_path, monkeypatch):
    """`XDG_CACHE_HOME` moves both the Hugging Face and the torch cache, on
    every platform, and neither row was asking. A Linux-only bug, and the quiet
    kind: a model genuinely on disk reported missing, with a Download button
    that would fetch it into the directory the check refused to look in."""
    from dubbing_app import setup as setup_mod

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert setup_mod.default_cache_home() == tmp_path / "xdg"
    assert setup_mod.hf_hub_cache() == tmp_path / "xdg" / "huggingface" / "hub"

    # A `.th` under the torch hub cache is a demucs 3.x install, and the row has
    # to see it wherever XDG_CACHE_HOME says it lives.
    hub = tmp_path / "xdg" / "torch" / "hub" / "checkpoints"
    hub.mkdir(parents=True)
    (hub / "htdemucs_ft.th").write_bytes(b"x" * 32)
    assert setup_mod.demucs_check()["ok"] is True

    # The explicit variables still win over it, exactly as in both libraries.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert setup_mod.hf_hub_cache() == tmp_path / "explicit"


def test_the_python_uv_chain_still_matches_the_rust_one_it_claims_to():
    """The parity claim, checked against the Rust source instead of asserted in
    a comment.

    `setup.find_uv` says it is kept in step with `find_uv()` in the shell's
    `workspace.rs`, and for a long time it was not: the Rust grew per-platform
    lists, a `.exe` spelling and a `~/.cargo/bin` rung while the Python kept two
    Homebrew paths. Nobody noticed because the drift is invisible from a Mac.
    Reading the constants out of the .rs file is the only check that stays true
    without a Rust toolchain here, and it fails the moment either side moves.
    """
    import re
    from pathlib import Path

    from dubbing_app import setup as setup_mod

    source = Path(__file__).resolve().parents[1] / (
        "app/desktop/src-tauri/src/workspace.rs")
    if not source.is_file():                 # a payload-only checkout
        pytest.skip("the desktop shell's source is not in this tree")
    rs = source.read_text(encoding="utf-8")

    def rust_fallbacks(cfg: str) -> tuple[str, ...]:
        block = re.search(
            r'#\[cfg\(target_os = "%s"\)\]\s*\nconst UV_FALLBACKS: &\[&str\] = &\[(.*?)\];'
            % cfg, rs, re.S)
        assert block, f"no UV_FALLBACKS for {cfg} in workspace.rs"
        return tuple(re.findall(r'"([^"]+)"', block.group(1)))

    assert rust_fallbacks("macos") == setup_mod.uv_fallbacks(MAC)
    assert rust_fallbacks("linux") == setup_mod.uv_fallbacks(LINUX)
    # Windows falls to the catch-all arm, which is empty on both sides.
    assert re.search(r'#\[cfg\(not\(any\(target_os = "macos", target_os = "linux"\)\)\)\]'
                     r'\s*\nconst UV_FALLBACKS: &\[&str\] = &\[\];', rs)
    assert setup_mod.uv_fallbacks(WINDOWS) == ()

    # The same env var, the same binary name, and the same home-relative tail in
    # the same order.
    assert re.search(r'pub const UV_PATH_ENV: &str = "%s"' % setup_mod.UV_PATH_ENV, rs)
    assert '"uv.exe"' in rs and setup_mod.uv_exe(WINDOWS) == "uv.exe"
    assert re.findall(r'home\.join\("(\.[a-z]+)"\)\.join\("bin"\)', rs) == [".local", ".cargo"]
    # And the home directory itself: HOME first, USERPROFILE second, on all three.
    assert 'env::var_os("HOME")' in rs and 'var_os("USERPROFILE")' in rs
