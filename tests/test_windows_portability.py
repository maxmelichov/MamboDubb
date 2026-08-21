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
    assert set(tools.auto_installers(MAC)) == {"ffmpeg", "sox"}
    assert set(tools.auto_installers(WINDOWS)) == {"ffmpeg", "sox"}
    assert tools.auto_installers(LINUX) == {}
    assert tools.unattended("ffmpeg", WINDOWS) and not tools.unattended("ffmpeg", LINUX)


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
