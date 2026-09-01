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

import os
import subprocess
import sys
from pathlib import Path

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


def test_the_path_search_honours_pathext_on_windows(tmp_path, monkeypatch):
    """Scoop, Chocolatey and most corporate wrappers install uv as a `uv.cmd`
    or `uv.bat` shim. A scan that tried `uv.exe` and `uv` alone walked past a
    working uv, reported the row not-found, and offered a button whose whole
    reason for existing is to avoid a second uv that disagrees with the first.
    `.exe` still wins inside a directory, and the bare name is still last.
    Keyed off the spelling `uv_exe()` chose, so every branch runs on a Mac."""
    import os

    from dubbing import tools

    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    assert tools.path_names("uv.exe") == ("uv.exe", "uv.com", "uv.bat", "uv.cmd", "uv")
    # No PATHEXT at all falls back to the documented default rather than to
    # nothing, so the shim is found on a machine that never set the variable.
    monkeypatch.delenv("PATHEXT")
    assert "uv.cmd" in tools.path_names("uv.exe")
    # The POSIX spelling is its own name and nothing else.
    assert tools.path_names("uv") == ("uv",)

    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("PATH", f"{first}{os.pathsep}{second}")
    (second / "uv.cmd").write_text("@echo off\n")
    assert tools.uv_on_path("uv.exe") == str(second / "uv.cmd")
    # A real .exe in an earlier directory beats the shim in a later one, and a
    # real .exe beside the shim beats it too: the order is the shell's.
    (first / "uv.exe").write_text("MZ\n")
    assert tools.uv_on_path("uv.exe") == str(first / "uv.exe")
    (first / "uv.exe").unlink()
    (second / "uv.exe").write_text("MZ\n")
    assert tools.uv_on_path("uv.exe") == str(second / "uv.exe")


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


def test_a_missing_uv_is_not_a_reason_a_MAC_cannot_dub(tmp_path, monkeypatch):
    """On a Mac the row is `optional`, and that is the file's own rule applied
    rather than a softening.

    Blocking means "the run fails without it". No run fails *here*: the MLX
    backend loads the translator in this process, this server is already up in
    its environment, and `runner.SubprocessRunner` spawns every job child with
    `sys.executable`. Graded blocking, it told a desktop user whose app had been
    launched *by* the bundled uv that something REQUIRED was MISSING, which is
    the exact dishonesty the three grades exist to remove.

    Off a Mac the answer is the other one: see the test below.
    """
    from dubbing import translate
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(translate, "shells_out_to_uv", lambda: False)
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

    # The bag's `.th` files under the torch hub cache are a demucs 3.x install,
    # and the row has to see them wherever XDG_CACHE_HOME says they live. Named
    # by their real signatures, because the row wants this bag rather than any
    # file ending in `.th`.
    from dubbing import stems

    hub = tmp_path / "xdg" / "torch" / "hub" / "checkpoints"
    hub.mkdir(parents=True)
    for sig in setup_mod.demucs_signatures(stems.MODEL) or ():
        (hub / f"{sig}-deadbeef.th").write_bytes(b"x" * 32)
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


# ---------------------------------------------------------------------------
# the stale PATH after a winget install, which made every Windows tool row end
# in "restart the app" for an install that had already worked
# ---------------------------------------------------------------------------

def test_a_winget_install_is_seen_without_restarting_the_app(tmp_path, monkeypatch):
    """Windows keeps PATH in the registry and hands each process a copy, so a
    tool winget installed a second ago is invisible to a server that started
    before it. `install-server.ps1` rebuilds PATH from the registry for exactly
    this reason; the button has to do the same, or every re-probe says the tool
    is missing and every row ends in a restart nobody should need.

    The registry read itself is Windows-only and unrunnable here; the merge is
    not, so `entries` is injectable and this is the merge.
    """
    gained = tmp_path / "WinGet" / "Links"
    gained.mkdir(parents=True)
    env = {"PATH": os.pathsep.join([str(tmp_path / "already")])}
    (tmp_path / "already").mkdir()

    install_mod.refresh_path(env, [str(gained)])
    assert env["PATH"].split(os.pathsep) == [str(tmp_path / "already"), str(gained)]

    # Idempotent, including across the trailing separator Windows writes.
    install_mod.refresh_path(env, [str(gained) + os.sep, str(gained)])
    assert env["PATH"].count(str(gained)) == 1

    # A directory that is not there is not worth putting on the PATH of every
    # child this process will ever spawn, which is `server.widen_path`'s rule.
    install_mod.refresh_path(env, [str(tmp_path / "ghost")])
    assert "ghost" not in env["PATH"]

    # And off Windows there is no registry to read, so this is a no-op rather
    # than a guess.
    if sys.platform != "win32":
        assert install_mod.registry_path_entries() == []


def test_the_manager_route_refreshes_path_before_it_re_probes(monkeypatch):
    """The refresh has to land between the child exiting and the check running,
    or the check is still looking at the PATH the process started with."""
    order = []
    monkeypatch.setattr(install_mod, "refresh_path",
                        lambda *a, **k: order.append("refresh"))
    inst = install_mod.Installer(
        lambda id_: (order.append("probe"),
                     {"id": id_, "ok": True, "label": id_, "detail": "there"})[1],
        recipes={"ffmpeg": ("/bin/sh", "-c", "true")})
    inst.start("ffmpeg")
    assert inst.wait(10.0)
    assert order == ["refresh", "probe"]

    # The routes that write where PATH is not consulted do not pay for it.
    order.clear()
    inst2 = install_mod.Installer(
        lambda id_: (order.append("probe"),
                     {"id": id_, "ok": True, "label": id_, "detail": "there"})[1])
    monkeypatch.setattr(install_mod, "restore_diarization", lambda log: None)
    monkeypatch.setattr(install_mod, "diarization_source", lambda: ("copy", Path(".")))
    inst2.start(install_mod.DIARIZATION_ID)
    assert inst2.wait(10.0)
    assert order == ["probe"]


# ---------------------------------------------------------------------------
# the GPU that was there all along
# ---------------------------------------------------------------------------
#
# A real Windows user with an NVIDIA card ran a dub in which stem separation
# took 57576 seconds, which is sixteen hours, because PyPI's `torch` wheel is a CUDA
# build on Linux and a CPU-only build on Windows, and nothing anywhere said so.
# The ASR was the only stage that noticed, and all it said was "cuda unusable
# (Library cublas64_12.dll is not found or cannot be loaded) falling back to
# cpu", which names a DLL nobody has ever installed on purpose.
#
# None of this can be run for real here: there is no Windows machine and no
# NVIDIA GPU. What is testable is every decision made *before* the DLL is
# loaded, and that is what these cover: which directories get registered, what
# the fallback message tells the user to do, that the whole thing is inert on a
# Mac, and that the detection cannot mistake a CPU-only torch for a CUDA one.

from dubbing import nvlibs                                            # noqa: E402
from dubbing_app import setup as setup_mod                            # noqa: E402


def _fake_torch(tmp_path: Path, cuda: str | None) -> Path:
    """A `torch/version.py` of the shape the real build script writes."""
    d = tmp_path / "torch"
    d.mkdir(parents=True, exist_ok=True)
    literal = f"'{cuda}'" if cuda else "None"
    (d / "version.py").write_text(
        "__version__ = '2.13.0'\ndebug = False\n"
        f"cuda: Optional[str] = {literal}\ngit_version = 'abc'\n", encoding="utf-8")
    return d


def test_a_cpu_only_torch_is_told_apart_from_a_cuda_one_without_importing_it(tmp_path):
    """The Setup screen asks this on every poll, so it may not import torch.

    The answer is a literal in a generated file, so reading the file is both
    cheaper and exactly as true as importing half a gigabyte to ask.
    """
    assert nvlibs.torch_cuda_build(_fake_torch(tmp_path / "cpu", None)) is None
    assert nvlibs.torch_cuda_build(_fake_torch(tmp_path / "gpu", "12.6")) == "12.6"
    # No torch at all reads as no CUDA, not as a crash.
    assert nvlibs.torch_cuda_build(tmp_path / "nothing") is None


def test_the_asr_fallback_names_the_fix_for_the_platform_it_is_on(monkeypatch):
    """"falling back to cpu" is a diagnosis. A user needs a prescription."""
    monkeypatch.setattr(sys, "platform", WINDOWS)
    win = nvlibs.cuda_hint()
    assert "--extra cuda" in win and "WINDOWS.md" in win

    # Linux gets a CUDA torch from PyPI, so the only thing that can be missing
    # there is the CUDA 12 cuBLAS that CTranslate2 wants and torch's CUDA 13
    # stack does not provide. Different fix, so a different sentence.
    monkeypatch.setattr(sys, "platform", LINUX)
    lin = nvlibs.cuda_hint()
    assert "nvidia-cublas-cu12" in lin and "--extra cuda" not in lin

    # And on a Mac there is no CUDA to repair. Silence rather than an
    # instruction that would send a Mac user shopping for a graphics card.
    monkeypatch.setattr(sys, "platform", MAC)
    assert nvlibs.cuda_hint() == ""


def test_the_dll_directories_are_the_wheel_ones_and_torchs_own(tmp_path, monkeypatch):
    """PATH is not searched for an extension module's dependent DLLs on 3.8+.

    So the directories have to be named. There are three sources and they are
    not interchangeable: `nvidia/*/bin` is where the cuBLAS and cuDNN wheels
    unpack, `nvidia/*/lib` is how a few of them spell the same thing, and
    `torch/lib` is where a CUDA torch wheel drops its own copies.
    """
    site = tmp_path / "site-packages"
    for rel in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cusparse/lib",
                "torch/lib"):
        (site / rel).mkdir(parents=True)
    fake_nvidia = type(sys)("nvidia")
    fake_nvidia.__path__ = [str(site / "nvidia")]
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    monkeypatch.setattr(nvlibs, "_torch_dir", lambda: site / "torch")

    found = {str(p) for p in nvlibs.windows_dll_dirs()}
    assert found == {str(site / rel) for rel in
                     ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cusparse/lib",
                      "torch/lib")}


def test_preload_is_a_no_op_off_linux_and_windows(monkeypatch):
    """It runs at import of `dubbing/__main__.py` on every platform, so on a Mac
    it has to do nothing at all rather than nothing much."""
    monkeypatch.setattr(sys, "platform", MAC)
    called = []
    monkeypatch.setattr(nvlibs, "windows_dll_dirs", lambda: called.append(1) or [])
    nvlibs.preload()                       # must not raise, must not look
    assert called == []


def test_the_gpu_warning_fires_only_on_the_machine_it_is_about(monkeypatch, capsys):
    """A card, a driver, and a torch that cannot use either. Nothing else."""
    monkeypatch.setattr(sys, "platform", WINDOWS)
    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: r"C:\Windows\nvidia-smi.exe")
    monkeypatch.setattr(nvlibs, "_torch_dir", lambda: Path("/torch"))
    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: None)
    assert nvlibs.warn_if_gpu_unused(once=False) is not None
    printed = capsys.readouterr().err
    assert "CPU-only" in printed and "--extra cuda" in printed

    # A CUDA torch on the same machine: nothing to say.
    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: "12.6")
    assert nvlibs.warn_if_gpu_unused(once=False) is None

    # No driver: no card, so a CPU-only torch is the correct install and
    # warning about it would be telling every CPU box it is broken.
    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: None)
    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: None)
    assert nvlibs.warn_if_gpu_unused(once=False) is None

    # And a Mac is never asked, driver or not, because MPS is the GPU there.
    monkeypatch.setattr(sys, "platform", MAC)
    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: "/usr/bin/nvidia-smi")
    assert nvlibs.warn_if_gpu_unused(once=False) is None


def test_the_setup_row_appears_only_where_there_is_a_card_to_talk_about(monkeypatch):
    """A permanently grey "not applicable" row is a line nobody needs twice."""
    monkeypatch.setattr(sys, "platform", MAC)
    assert setup_mod.gpu_check() is None

    monkeypatch.setattr(sys, "platform", WINDOWS)
    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: None)
    assert setup_mod.gpu_check() is None

    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: r"C:\nvidia-smi.exe")
    monkeypatch.setattr(setup_mod, "gpu_memory_bytes", lambda **_k: 12 * 1024**3)

    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: None)
    bad = setup_mod.gpu_check()
    assert bad["id"] == "gpu" and bad["ok"] is False
    # Degrades, not blocking: the run does finish. And not optional either,
    # which is the grade for something somebody chose.
    assert bad["severity"] == setup_mod.DEGRADES and bad["required"] is False
    assert "CPU-only" in bad["detail"] and "--extra cuda" in bad["detail"]

    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: "12.6")
    good = setup_mod.gpu_check()
    assert good["ok"] is True and "12.6" in good["detail"]


def test_the_installer_asks_for_the_cuda_extra_and_the_lock_can_answer():
    """The two halves of the fix have to agree, and they live in two files.

    `install-server.ps1` adds `--extra cuda` when `nvidia-smi` is there;
    `pyproject.toml` is where that extra has to exist, with the CUDA torch (for
    demucs and everything else torch drives) and the cuBLAS/cuDNN wheels (for
    CTranslate2, which bundles neither). A rename on either side is a Windows
    install that quietly resolves to the CPU wheels again.
    """
    root = Path(__file__).resolve().parents[1]
    ps1 = (root / "install-server.ps1").read_text(encoding="utf-8")
    assert "nvidia-smi" in ps1 and "'--extra', 'cuda'" in ps1
    # And it must not print the advice it used to print: cu124 has no build of
    # the torch this project locks, and `uv pip install` is undone by the next
    # `uv run`, which re-syncs the venv to the lockfile.
    assert "cu124" not in ps1
    printed = [ln for ln in ps1.splitlines() if ln.strip().startswith("Info ")]
    assert not [ln for ln in printed if "pip install" in ln]

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\ncuda = [" in pyproject
    for pkg in ("torch", "torchaudio", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        assert f'"{pkg}' in pyproject.split("\ncuda = [")[1].split("]")[0]
    # Every line of the extra is markered to Windows as well, so `--extra cuda`
    # on a Mac or a Linux box resolves to nothing rather than to a download.
    body = pyproject.split("\ncuda = [")[1].split("]")[0]
    assert body.count("sys_platform == 'win32'") == 4


# --- Setup's two "ready" lies, both found by review ---------------------------


def _hf_repo(cache, hub, *blobs):
    """A Hugging Face cache repo holding exactly `blobs`, by file name."""
    repo = cache / f"models--{hub.replace('/', '--')}"
    (repo / "blobs").mkdir(parents=True)
    for name in blobs:
        (repo / "blobs" / name).write_bytes(b"x" * 16)
    return repo


def test_a_download_still_running_is_not_a_model_in_the_cache(tmp_path, monkeypatch):
    """`huggingface_hub` opens `<etag>.incomplete` in blobs/ before the first
    byte lands, so "blobs/ has something in it" is true one percent into a 9.7 GB
    fetch. Counting that as present is how Setup called a machine ready for a run
    that then died at translate."""
    from dubbing_app import setup

    monkeypatch.setattr(setup, "hf_hub_cache", lambda: tmp_path)
    _hf_repo(tmp_path, "org/model", "abc123.incomplete")
    assert setup.hf_cache_repo("org/model") is None


def test_a_finished_blob_beside_a_partial_still_counts(tmp_path, monkeypatch):
    """The other side of the same line: a repo part-way through its second file
    has real bytes on disk, and `model_ready`'s size floor is what grades those.
    Refusing the repo outright would swap one wrong answer for another."""
    from dubbing_app import setup

    monkeypatch.setattr(setup, "hf_hub_cache", lambda: tmp_path)
    repo = _hf_repo(tmp_path, "org/model", "done1", "abc123.incomplete")
    assert setup.hf_cache_repo("org/model") == repo


def test_an_empty_models_dir_is_not_rescued_by_the_cache(tmp_path, monkeypatch):
    """Every loader is `str(local) if local.is_dir() else hub`, so an empty
    directory wins over the hub id and the library is handed nothing. A cached
    copy cannot save that load, so it must not turn the row green either."""
    from dubbing_app import setup

    monkeypatch.setattr(setup, "hf_hub_cache", lambda: tmp_path / "hf")
    (tmp_path / "hf").mkdir()
    _hf_repo(tmp_path / "hf", "org/model", "done1")
    empty = tmp_path / "models" / "model"
    empty.mkdir(parents=True)

    where, size, state = setup._model_location(
        empty, "org/model", True, 0)
    assert state == setup.MISSING
    assert where == empty


def test_a_missing_models_dir_is_still_rescued_by_the_cache(tmp_path, monkeypatch):
    """The case the fallback exists for is untouched: no local directory at all
    means the loader does reach the hub id, and the cache is where that lands."""
    from dubbing_app import setup

    monkeypatch.setattr(setup, "hf_hub_cache", lambda: tmp_path / "hf")
    (tmp_path / "hf").mkdir()
    repo = _hf_repo(tmp_path / "hf", "org/model", "done1")

    where, size, state = setup._model_location(
        tmp_path / "models" / "model", "org/model", True, 0)
    assert where == repo
    assert state != setup.MISSING


def test_the_installer_never_runs_uv_without_the_extras_it_synced():
    """A bare `uv run` is a re-sync without the extra, and the torch the lock
    names without `--extra cuda` is the CPU wheel.

    The first cut of the installer synced with the extras and then ran two
    bare `uv run`s: one to *check* that CUDA had taken, one to start the
    server. Either would have put the CPU torch back, and the check would then
    have reported the very failure it had just caused. So every `uv run` in
    the script either carries the extras it synced or says `--no-sync`.
    """
    root = Path(__file__).resolve().parents[1]
    ps1 = (root / "install-server.ps1").read_text(encoding="utf-8")
    runs = [ln for ln in ps1.splitlines()
            if "$Uv" in ln and " run " in ln and not ln.strip().startswith("#")]
    assert runs, "expected the installer to run uv at least once"
    for ln in runs:
        assert "@Extras" in ln or "--no-sync" in ln or "$Extras" in ln, ln
    # And the translator venv, which has its own torch, is synced by the
    # installer rather than left for the first translate stage to create.
    assert "--project (Join-Path $Dir 'translator')" in ps1


def test_the_stem_stage_names_a_device_without_loading_torch_to_ask(monkeypatch):
    """The stems process spawns a demucs child that loads torch itself. The
    parent naming the device by importing torch too would keep a second copy
    resident for the whole separation, on the machines with the least memory."""
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch_device imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)

    monkeypatch.setattr(sys, "platform", MAC)
    monkeypatch.setattr(nvlibs.platform, "machine", lambda: "arm64")
    assert nvlibs.torch_device() == "mps"
    monkeypatch.setattr(nvlibs.platform, "machine", lambda: "x86_64")
    assert nvlibs.torch_device() == "cpu"

    monkeypatch.setattr(sys, "platform", WINDOWS)
    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: r"C:\Windows\System32\nvidia-smi.exe")
    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: "12.6")
    assert nvlibs.torch_device() == "cuda"
    # A driver with a CPU-only torch is the sixteen-hour case, and it says cpu.
    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: None)
    assert nvlibs.torch_device() == "cpu"
    monkeypatch.setattr(nvlibs, "nvidia_smi", lambda: None)
    monkeypatch.setattr(nvlibs, "torch_cuda_build", lambda *_a: "12.6")
    assert nvlibs.torch_device() == "cpu"


def test_the_translator_venv_sends_windows_to_the_cuda_torch_without_a_flag():
    """The translator has its own torch, CPU-only from PyPI on Windows.

    The choice lives in its lockfile, as a platform marker on the index source,
    and not as an extra. An extra was tried first and uv unified Linux onto the
    CUDA 12 wheel too, off the CUDA 13 stack the root project pins. A marker
    forks where it says, and a bare `uv run --project translator` on Windows
    gets the card without any launch line having to remember a flag.
    """
    from dubbing import translate

    root = Path(__file__).resolve().parents[1]
    toml = (root / "translator" / "pyproject.toml").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cu126" in toml
    assert "marker = \"sys_platform == 'win32'\"" in toml
    assert "\ncuda = [" not in toml
    # The lock agrees: Windows resolves torch from the CUDA index, and nowhere
    # else does, which is the fork the marker promises.
    lock = (root / "translator" / "uv.lock").read_text(encoding="utf-8")
    edges = [ln for ln in lock.splitlines()
             if ln.strip().startswith('{ name = "torch"') and "download.pytorch.org" in ln]
    assert edges and all("sys_platform == 'win32'" in ln for ln in edges)
    assert 'name = "nvidia-cudnn-cu13"' in lock

    # And no launch line carries a cuda flag for it, on any platform.
    for plat in (WINDOWS, LINUX, MAC):
        cmd = translate._worker_cmd("transformers")
        assert "cuda" not in cmd


# ---------------------------------------------------------------------------
# the uv row off a Mac, and the uv the translate stage actually launches
# ---------------------------------------------------------------------------

def test_the_uv_row_is_blocking_and_names_translate_where_a_dub_needs_uv(tmp_path,
                                                                        monkeypatch):
    """The regression this file exists for, in its newest costume.

    `translate._worker_cmd` is `uv run --project translator …`, and on Windows
    and Linux `_backend()` is never "mlx", so that line *is* the translate
    stage. Graded optional, the screen showed a grey row, the user started a
    dub, fetch and stems and transcript all succeeded, and the run died at
    translate on a FileNotFoundError. A row that kills a stage is blocking, and
    a blocking row has to name the stage or "runs will fail" is true and
    useless.
    """
    from dubbing import translate
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(translate, "shells_out_to_uv", lambda: True)
    monkeypatch.setattr(setup_mod, "find_uv", lambda *a, **k: None)
    report = setup_mod.report(tmp_path)
    row = next(c for c in report["checks"] if c["id"] == "uv")
    assert row["severity"] == setup_mod.BLOCKING and row["required"] is True
    assert row["stage"] == "translate"
    assert "translate" in row["detail"]
    # And it drags the whole report red, which is the point of "required".
    assert report["ok"] is False
    # `probe`, which is what the install button re-reads, agrees with `report`.
    assert setup_mod.probe("uv")["severity"] == setup_mod.BLOCKING
    assert setup_mod.probe("uv")["stage"] == "translate"


def test_the_backend_and_not_the_platform_decides_whether_uv_is_blocking(monkeypatch):
    """A Mac with the transformers backend forced needs uv exactly as much as a
    Windows machine does, and a `sys.platform` test would have called it
    optional. So the question asked is the backend's."""
    from dubbing import translate

    monkeypatch.setenv(translate.BACKEND_ENV, "transformers")
    assert translate.shells_out_to_uv() is True
    monkeypatch.setenv(translate.BACKEND_ENV, "mlx")
    assert translate.shells_out_to_uv() is False


def test_the_translate_worker_launches_the_uv_this_machine_actually_has(tmp_path,
                                                                       monkeypatch):
    """A bare `"uv"` is resolved against PATH and nothing else, and PATH is the
    one rung of the five a packaged desktop install does not have: the bundle
    ships uv as a Tauri sidecar and passes its path down as `DUBSTUDIO_UV_PATH`.
    So the launch line has to go through the same lookup the Setup row and the
    Rust shell use, or a desktop user gets FileNotFoundError at translate with
    uv sitting right there."""
    from dubbing import translate

    sidecar = tmp_path / "MamboDubb.app" / "Contents" / "MacOS" / "uv.exe"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("MZ\n")
    monkeypatch.setenv(tools.UV_PATH_ENV, str(sidecar))
    cmd = translate._worker_cmd("transformers")
    assert cmd[0] == str(sidecar)
    assert cmd[1:3] == ["run", "--project"]


def test_a_translate_stage_with_no_uv_says_so_instead_of_raising_filenotfound(
        monkeypatch):
    """`FileNotFoundError: 'uv'` names a file. This names the stage, the reason
    and the fix, and it is the same sentence the Setup row carries."""
    from dubbing import translate

    monkeypatch.setattr(tools, "find_uv", lambda *a, **k: None)
    with pytest.raises(RuntimeError) as caught:
        translate._worker_cmd("transformers")
    said = str(caught.value)
    assert "translate" in said and "uv run" in said
    assert "docs.astral.sh/uv" in said


# ---------------------------------------------------------------------------
# PATHEXT: the shims a Windows PATH scan has to see
# ---------------------------------------------------------------------------

def test_the_uv_path_scan_honours_pathext_so_a_cmd_shim_is_not_invisible(
        tmp_path, monkeypatch):
    """Scoop, Chocolatey and most corporate wrappers install uv as a `uv.cmd` or
    `uv.bat`. A scan that tried exactly `uv.exe` and then `uv` walked past all of
    them, reported the row not-found, and offered a button that installs a
    *second* uv, which is the two-uvs-that-disagree outcome
    `install.uv_install_dir` exists to prevent. `resolve_tool` has honoured
    PATHEXT all along because it ends at `shutil.which`; this was the one lookup
    in the repo that did not."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    (tmp_path / "uv.cmd").write_text("@echo off\n")
    assert tools.uv_on_path("uv.exe") == str(tmp_path / "uv.cmd")

    # Order is still astral's build first, whatever else is beside it, because
    # `workspace.rs` tries `uv.exe` first and the two must not disagree about a
    # directory holding both.
    (tmp_path / "uv.exe").write_text("MZ\n")
    assert tools.uv_on_path("uv.exe") == str(tmp_path / "uv.exe")

    # And the machine that never set PATHEXT gets the documented default, so the
    # `.cmd` shim is found there too.
    monkeypatch.delenv("PATHEXT")
    (tmp_path / "uv.exe").unlink()
    assert tools.uv_on_path("uv.exe") == str(tmp_path / "uv.cmd")
    assert "uv.cmd" in tools.path_names("uv.exe")

    # Off Windows the name is the whole answer; nothing is appended to it.
    assert tools.path_names("uv") == ("uv",)


def test_the_uv_lookup_lives_where_both_callers_can_reach_it():
    """`dubbing.translate` needs it and `dubbing_app.setup` needs it, and the
    pipeline is not allowed to import the app. One implementation, one spelling
    of the env var, and `setup.find_uv` still means what it always meant."""
    from dubbing_app import setup as setup_mod

    assert setup_mod.find_uv is tools.find_uv
    assert setup_mod.uv_on_path is tools.uv_on_path
    assert setup_mod.UV_PATH_ENV == tools.UV_PATH_ENV == "DUBSTUDIO_UV_PATH"
    source = (Path(__file__).resolve().parents[1] / "dubbing" / "translate.py")
    text = source.read_text(encoding="utf-8")
    assert "import dubbing_app" not in text and "from dubbing_app" not in text


# ---------------------------------------------------------------------------
# the `cuda` extra actually gates the CUDA torch
# ---------------------------------------------------------------------------

def _root_base_torch_edges() -> list[str]:
    """The `torch` edges of the root project's *base* dependencies in the lock.

    The base list, not `[package.optional-dependencies]`, because the base list
    is where the bug was: an extra's own entries were always platform-markered
    and always will be, while the plain `torch>=2.4.0` every install pulls in is
    the one that came back saying cu126 with nothing to stop it.
    """
    lock = (Path(__file__).resolve().parents[1] / "uv.lock").read_text(encoding="utf-8")
    block = lock.split('name = "dubbingqwen"', 1)[1]
    base = block.split("[package.optional-dependencies]", 1)[0]
    return [ln.strip() for ln in base.splitlines()
            if ln.strip().startswith('{ name = "torch')]


def test_the_cuda_extra_gates_the_cuda_torch_instead_of_decorating_it():
    """An extra alone does not gate an index.

    torch is a plain dependency, so uv resolved it once per platform and, with
    only the cu126 directive to go on, wrote `2.13.0+cu126` into the lock for
    *every* Windows resolution. `uv export --frozen --extra app` emitted it with
    no cuda extra requested. Two things followed: a Windows laptop with no
    NVIDIA card was told by `install-server.ps1` it was getting CPU wheels and
    then downloaded gigabytes it could never load, and
    `nvlibs.torch_cuda_build()` reported 12.6 on every Windows machine, so
    `gpu_check()` and `warn_if_gpu_unused()` could never fire on the one
    platform they were written for.

    `[tool.uv] conflicts` is what makes uv fork the resolution, and a fork is
    what lets one lockfile hold two answers.
    """
    root = Path(__file__).resolve().parents[1]
    toml = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "conflicts = [" in toml
    assert '[{ extra = "cpu" }, { extra = "cuda" }]' in toml

    # The CPU-only Windows wheel PyPI publishes is in the lock, which is what
    # the machine with no card downloads. It was not there before this fix.
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert "torch-2.13.0-cp311-cp311-win_amd64.whl" in lock

    # Every cu126 edge of the base dependency list is gated on the extra *and*
    # on the platform. That conjunction is the fix: the extra alone let it reach
    # every Windows install, and the platform alone let it reach a Mac.
    base = _root_base_torch_edges()
    cuda_edges = [e for e in base if "download.pytorch.org" in e]
    assert cuda_edges, base
    for edge in cuda_edges:
        assert "extra-11-dubbingqwen-cuda" in edge, edge
        assert "sys_platform == 'win32'" in edge, edge
    # And the PyPI edge is what a resolution with no cuda extra reaches, on
    # Windows as everywhere else.
    plain = [e for e in base if "pypi.org" in e]
    assert plain and all("sys_platform != 'win32'" in e
                         or "extra != 'extra-11-dubbingqwen-cuda'" in e for e in plain), plain


@pytest.mark.skipif(tools.find_uv() is None, reason="needs uv to resolve the lock")
def test_uv_export_puts_the_cuda_torch_behind_the_extra_and_nowhere_else():
    """The claim above, asked of uv rather than of the file uv wrote.

    Offline: `--frozen` resolves from the lockfile and touches no index.
    """
    root = Path(__file__).resolve().parents[1]

    def torch_lines(*extras: str) -> list[str]:
        out = subprocess.run([tools.find_uv(), "export", "--frozen", *extras],
                             cwd=str(root), capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return [ln for ln in out.stdout.splitlines()
                if ln.startswith(("torch==", "torchaudio=="))]

    plain = torch_lines("--extra", "app")
    assert plain and not any("+cu126" in ln for ln in plain), plain

    withcuda = torch_lines("--extra", "app", "--extra", "cuda")
    cuda_lines = [ln for ln in withcuda if "+cu126" in ln]
    assert cuda_lines, withcuda
    # And only on Windows: a Mac or a Linux box asking for the extra still gets
    # the wheel it always had, which is the promise the extra's comment makes.
    assert all("sys_platform == 'win32'" in ln for ln in cuda_lines), cuda_lines
    assert any("sys_platform != 'win32'" in ln and "+cu126" not in ln
               for ln in withcuda), withcuda


# ---------------------------------------------------------------------------
# a failed install cannot be laundered into a success
# ---------------------------------------------------------------------------

class _Slot:
    """The two halves of `Installer` these tests drive, without the HTTP."""

    def __init__(self, route, probe_ok=True, probe=None):
        self.route = route
        self.probe_ok = probe_ok
        self.probe = probe


def _installer(monkeypatch, route, check):
    inst = install_mod.Installer.__new__(install_mod.Installer)
    inst._lock = __import__("threading").Lock()
    inst._tail = __import__("collections").deque(maxlen=200)
    inst._route = route
    inst._probe = lambda id_: check
    inst._running = True
    inst._ok = None
    inst._error = None
    inst._check = None
    inst._finished = None
    return inst


def test_a_green_reprobe_cannot_vouch_for_work_it_did_not_look_at():
    """`_run_static` copies ffmpeg and *then* ffprobe. When the second copy
    raised, the ffmpeg probe was green, the row went green, and no failure was
    recorded anywhere; the next dub died in `dubbing.audio` at the first
    ffprobe call. The override exists for one thing only: a package manager's
    exit code that lies. It may not speak for anything else."""
    green = {"id": "ffmpeg", "ok": True, "detail": "/usr/bin/ffmpeg"}

    inst = _installer(None, install_mod.STATIC, green)
    inst._finish("ffmpeg", False, "OSError: [Errno 28] No space left on device")
    assert inst._ok is False
    assert "No space left" in inst._error

    # The case it was written for still works: winget answers non-zero for
    # "already installed", and the machine has the tool.
    inst = _installer(None, install_mod.MANAGER, green)
    inst._finish("ffmpeg", False, "`winget install …` exited 1", exit_code_only=True)
    assert inst._ok is True and inst._error is None


def test_a_thirty_minute_timeout_kill_is_not_laundered_by_a_stale_green_row(
        monkeypatch):
    """A child killed at the timeout installed nothing. A row that is green in
    spite of it is green for some earlier reason with nothing to say about this
    attempt, and replacing "was still running after 30 minutes" with a success
    is how a user learns the app cannot be believed."""
    green = {"id": "ffmpeg", "ok": True, "detail": "/usr/bin/ffmpeg"}
    inst = _installer(None, install_mod.MANAGER, green)
    inst._finish("ffmpeg", False, "`brew install ffmpeg` was still running after "
                 "30 minutes and was stopped", exit_code_only=False)
    assert inst._ok is False
    assert "30 minutes" in inst._error


def test_the_ffmpeg_row_probes_ffprobe_too(monkeypatch):
    """There is no ffprobe row anywhere, and there should not be: every route
    that installs ffmpeg installs both, so a second row would be a second button
    running the first one's install. What was wrong was the claim: the row said
    "ffmpeg" and meant it, while `dubbing.audio` shells out to ffprobe for every
    input file."""
    from dubbing_app import setup as setup_mod

    assert setup_mod.COMPANIONS["ffmpeg"] == ("ffprobe",)
    real = tools.resolve_tool
    monkeypatch.setattr(tools, "resolve_tool",
                        lambda name: None if name == "ffprobe" else real(name))
    row = setup_mod.probe("ffmpeg")
    assert row["ok"] is False
    assert "ffprobe" in row["detail"]
    # And it does not claim ffmpeg is missing, because ffmpeg is right there.
    assert "ffmpeg not on PATH" not in row["detail"]


def test_a_message_that_cannot_be_composed_does_not_strand_the_install_slot():
    """`_disagreement` ran outside the try that protected the re-probe, and it
    was the last statement before the slot was released. Its RESTORE branch
    imports `dubbing.segments` and its UV_ROUTE branch calls `uv_install_dir`,
    which raises when neither HOME nor USERPROFILE is set. The worker thread
    died with `_running` still true and every install button answered 409 until
    the app was restarted."""
    red = {"id": "uv", "ok": False, "detail": "uv not found"}
    inst = _installer(None, install_mod.UV_ROUTE, red)
    inst._disagreement = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no HOME"))
    inst._finish("uv", True, None)
    assert inst._running is False               # the slot is free, which is the point
    assert inst._ok is False
    assert "no HOME" in inst._error


# ---------------------------------------------------------------------------
# the VRAM cache
# ---------------------------------------------------------------------------

def test_the_vram_cache_is_not_published_before_the_number_is_known(monkeypatch):
    """`_vram = 0` was written *before* a ten-second `nvidia-smi`, and
    `_vram is not None` was the "have I read this" flag, so for the whole of
    those ten seconds a concurrent poll was told this machine has no GPU as a
    cached fact. `low_vram_check` feeds that straight to
    `translate.choose_low_vram`, which reads 0 as "no CUDA device" and offers
    the 9.7 GB bf16 translator to a 12 GB card. `git_commit` keeps a separate
    read flag for exactly this reason."""
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(setup_mod, "_vram", 0)
    monkeypatch.setattr(setup_mod, "_vram_read", False)
    seen = []

    def slow_read():
        seen.append(setup_mod._vram_read)       # what a concurrent poll would see
        return 12 * 1024**3

    monkeypatch.setattr(setup_mod, "_read_gpu_memory_bytes", slow_read)
    assert setup_mod.gpu_memory_bytes() == 12 * 1024**3
    assert seen == [False], "the cache announced an answer it did not have yet"


def test_a_driver_that_will_not_answer_is_not_cached_as_zero_forever(monkeypatch):
    """`out.returncode` was never checked and empty stdout raised IndexError
    inside the try whose except cached 0, so one bad moment at startup made the
    whole process believe there was no card until it was restarted."""
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(setup_mod.shutil, "which", lambda *_a, **_k: "/usr/bin/nvidia-smi")

    class Out:
        def __init__(self, code, stdout):
            self.returncode, self.stdout = code, stdout

    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **k: Out(9, ""))
    assert setup_mod._read_gpu_memory_bytes() == 0
    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **k: Out(0, ""))
    assert setup_mod._read_gpu_memory_bytes() == 0
    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **k: Out(0, "not a number\n"))
    assert setup_mod._read_gpu_memory_bytes() == 0
    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **k: Out(0, "12288\n"))
    assert setup_mod._read_gpu_memory_bytes() == 12288 * 1024**2


# ---------------------------------------------------------------------------
# the registry PATH, in the order Windows composes it
# ---------------------------------------------------------------------------

def test_the_registry_path_is_read_system_first_then_user(monkeypatch):
    """Windows hands a new process the system PATH with the per-user PATH
    appended, so a per-user entry is a fallback and never a shadow. Read the
    other way round, a stale per-user ffmpeg shadowed a good system one, so the
    server and a fresh cmd.exe disagreed about what this machine has, which is
    the exact failure `refresh_path` exists to prevent.
    `install-server.ps1`'s `Update-PathFromRegistry` joins machine then user and
    always did."""
    import types

    fake = types.SimpleNamespace(HKEY_CURRENT_USER="HKCU", HKEY_LOCAL_MACHINE="HKLM")
    # Real drive letters, because that is what a real registry holds and it is
    # the shape the parser has to survive. It could not, until the split moved
    # from `os.pathsep` to `;`: this test runs on a Mac, where `os.pathsep` is
    # ":", so every `C:\...` entry came back cut in half at its drive letter and
    # the fixtures had to be written around the defect to stay green.
    system = r"C:\Program Files\ffmpeg\bin"
    user = r"C:\Users\eli\AppData\Local\old-ffmpeg"
    values = {("HKLM", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"):
              system, ("HKCU", "Environment"): user}

    class Key:
        def __init__(self, pair):
            self.pair = pair

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake.OpenKey = lambda root, key: Key((root, key))
    fake.QueryValueEx = lambda handle, name: (values[handle.pair], 1)
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(sys, "platform", WINDOWS)

    # System first, so a stale per-user copy is a fallback and never a shadow.
    assert install_mod.registry_path_entries() == [system, user]

    # And the shell agrees, in the same order, which is the half of this that
    # cannot be asserted from Python alone.
    ps1 = (Path(__file__).resolve().parents[1] / "install-server.ps1").read_text(
        encoding="utf-8")
    assert "@($machine, $user)" in ps1


# ---------------------------------------------------------------------------
# Windows on ARM
# ---------------------------------------------------------------------------

def test_the_windows_installer_refuses_arm64_instead_of_failing_minutes_in():
    """There is no `win_arm64` torch wheel anywhere: not on PyPI, not on
    download.pytorch.org, and so not in the lock. The installer used to map
    ARM64 to a triple and claim both architectures had wheels, then install git,
    ffmpeg, sox, uv and Node before dying inside `uv sync` on a resolution error
    naming a package the user never asked for."""
    root = Path(__file__).resolve().parents[1]
    ps1 = (root / "install-server.ps1").read_text(encoding="utf-8")
    arm = ps1[ps1.index("    'ARM64'"):ps1.index("    default {")]
    assert "Die" in arm
    assert "win_arm64" in arm
    assert "aarch64-pc-windows-msvc" not in arm
    # And the reason it refuses is true: the lock has one Windows wheel family.
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert "torch-2.13.0-cp311-cp311-win_amd64.whl" in lock
    assert not [ln for ln in lock.splitlines()
                if "win_arm64" in ln and "/torch-" in ln]


# ---------------------------------------------------------------------------
# nvlibs: the same libraries in the parent as in the child
# ---------------------------------------------------------------------------

def test_the_linux_preload_registers_cublas_as_well_as_cudnn(tmp_path, monkeypatch):
    """cuDNN 9 links cuBLAS, and `preload` filtered `_lib_dirs()` down to the
    cuDNN directory alone, so `libcudnn_cnn.so.9` resolved `libcublas.so.12`
    through ldconfig and got whatever system CUDA the machine happened to have,
    which is the wrong-copy-answers failure this module exists to prevent, one
    library over. `subprocess_env()` has always put every `nvidia/*/lib` on
    LD_LIBRARY_PATH, so a demucs child got the wheels' cuBLAS and the parent did
    not: two answers to one question inside one run."""
    from dubbing import nvlibs

    cublas = tmp_path / "cublas" / "lib"
    cudnn = tmp_path / "cudnn" / "lib"
    for d in (cublas, cudnn):
        d.mkdir(parents=True)
    (cublas / "libcublas.so.12").write_text("x")
    (cudnn / "libcudnn.so.9").write_text("x")
    (cudnn / "libcudnn_cnn.so.9").write_text("x")

    loaded: list[str] = []
    monkeypatch.setattr(sys, "platform", LINUX)
    monkeypatch.setattr(nvlibs, "_lib_dirs", lambda: [cublas, cudnn])
    monkeypatch.setattr(nvlibs.ctypes, "CDLL",
                        lambda path, mode=0: loaded.append(path))

    assert nvlibs.preload() == []
    assert str(cublas / "libcublas.so.12") in loaded
    # cuBLAS before cuDNN: RTLD_GLOBAL means the earlier load satisfies the
    # later one's undefined symbols, so the order is the whole point.
    assert loaded.index(str(cublas / "libcublas.so.12")) < \
        loaded.index(str(cudnn / "libcudnn.so.9"))
    # The versioned soname before the sub-libraries it would otherwise dlopen.
    assert loaded.index(str(cudnn / "libcudnn.so.9")) < \
        loaded.index(str(cudnn / "libcudnn_cnn.so.9"))
    # And LD_LIBRARY_PATH for the children names the same two directories, which
    # is what "the parent and the child agree" means here.
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    assert str(cublas) in nvlibs.subprocess_env()["LD_LIBRARY_PATH"]
    assert str(cudnn) in nvlibs.subprocess_env()["LD_LIBRARY_PATH"]


def test_a_directory_where_nothing_loaded_is_said_out_loud(tmp_path, monkeypatch,
                                                           capsys):
    """The CDLL failures were swallowed whole. One stale file among a dozen good
    ones is not worth a line; a directory where *nothing* loaded is a directory
    that will not be there when a convolution asks, and the error that arrives
    then says nothing about this."""
    from dubbing import nvlibs

    cudnn = tmp_path / "cudnn" / "lib"
    cudnn.mkdir(parents=True)
    (cudnn / "libcudnn.so.9").write_text("x")

    def refuse(path, mode=0):
        raise OSError("wrong ELF class")

    monkeypatch.setattr(sys, "platform", LINUX)
    monkeypatch.setattr(nvlibs, "_lib_dirs", lambda: [cudnn])
    monkeypatch.setattr(nvlibs.ctypes, "CDLL", refuse)
    failed = nvlibs.preload()
    assert failed == [str(cudnn / "libcudnn.so.9")]
    assert "cudnn" in capsys.readouterr().err


def test_the_job_child_registers_the_cuda_libraries_and_warns_about_a_cpu_torch():
    """`preload()` and `warn_if_gpu_unused()` were wired into
    `dubbing/__main__.py` alone, which the desktop app's job child never
    imports: `runner.SubprocessRunner` spawns `python -m dubbing_app.worker`. So
    on the desktop app the Windows DLL registration happened only as a side
    effect of `load_whisper` reaching its CUDA branch, four stages after Demucs
    had already run on the CPU, and the warning was never printed at all."""
    source = (Path(__file__).resolve().parents[1] / "dubbing_app" / "worker.py")
    text = source.read_text(encoding="utf-8")
    assert "nvlibs.preload()" in text
    assert "nvlibs.warn_if_gpu_unused()" in text
    # `preload` at module scope, because a spawned multiprocessing child
    # re-imports this module and then loads torch; the warning inside `main`,
    # because it is a sentence for a person and a person wants it once.
    body = text[text.index("def main("):]
    assert "nvlibs.warn_if_gpu_unused()" in body
    assert "nvlibs.preload()" not in body


def test_the_cli_warns_once_per_run_and_not_once_per_spawned_worker():
    """`warn_if_gpu_unused()` sat at module scope in `dubbing/__main__.py`, and
    spawn re-imports that module as `__mp_main__` for every worker torch and
    demucs create. The one line whose whole value is being first in the log
    arrived eight more times, in the middle of the stem separation it was
    warning about. `preload()` stays outside the guard, because registration
    genuinely has to happen in every process."""
    source = (Path(__file__).resolve().parents[1] / "dubbing" / "__main__.py")
    text = source.read_text(encoding="utf-8")
    guard = text.index('if __name__ == "__main__":')
    assert text.index("nvlibs.preload()") < guard
    assert text.index("nvlibs.warn_if_gpu_unused()") > guard


# ---------------------------------------------------------------------------
# one uv release table, not two
# ---------------------------------------------------------------------------

def test_the_uv_release_table_exists_once(tmp_path, monkeypatch):
    """`install.py` and `scripts/stage_desktop_payload.py` each carried a
    verbatim copy of the triple table, the download, the checksum and the
    archive unpacking, and they had already drifted about what an unknown triple
    means. `dubbing/uvrelease.py` is the one copy; the refusal is still each
    caller's own, which is the one thing they are allowed to differ about."""
    import stat as stat_mod

    from dubbing import uvrelease

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import stage_desktop_payload as staging

    assert staging.SUPPORTED_TRIPLES is uvrelease.SUPPORTED_TRIPLES
    assert install_mod.uv_triple() == uvrelease.host_triple()
    assert install_mod.UV_RELEASES == uvrelease.UV_RELEASES
    # The stage script no longer owns a copy of any of the four.
    text = (Path(__file__).resolve().parents[1] / "scripts"
            / "stage_desktop_payload.py").read_text(encoding="utf-8")
    assert "hashlib" not in text and "zipfile" not in text and "tarfile" not in text

    # The machines astral publishes nothing for get one answer and two
    # readings: None here, `SystemExit` there.
    assert uvrelease.host_triple("linux", "riscv64") is None
    monkeypatch.setattr(uvrelease, "host_triple", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        staging.host_triple()
    assert stat_mod.S_IXUSR                      # the import is used, and staging needs it


def test_the_shared_uv_fetch_verifies_the_checksum_before_it_writes(tmp_path):
    """One implementation of "download, check, unpack", so a corrupted transfer
    is caught in one place rather than in two that can drift."""
    import io
    import tarfile

    from dubbing import uvrelease

    payload = b"#!/bin/sh\necho uv\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    blob = buf.getvalue()
    digest = __import__("hashlib").sha256(blob).hexdigest()

    def fetch(url, timeout=180.0):
        return f"{digest}  uv.tar.gz\n".encode() if url.endswith(".sha256") else blob

    got = uvrelease.fetch_uv("x86_64-unknown-linux-gnu", fetch=fetch)
    assert got == payload

    def corrupt(url, timeout=180.0):
        return b"0" * 64 + b"  uv.tar.gz\n" if url.endswith(".sha256") else blob

    with pytest.raises(RuntimeError) as caught:
        uvrelease.fetch_uv("x86_64-unknown-linux-gnu", fetch=corrupt)
    assert "checksum" in str(caught.value)

    # And the write is through an `.incoming` sibling, so an interrupted
    # download leaves the old uv rather than a truncated one.
    target = tmp_path / "bin" / "uv"
    assert uvrelease.write_binary(payload, target) == target
    assert target.read_bytes() == payload
    assert not (tmp_path / "bin" / "uv.incoming").exists()
