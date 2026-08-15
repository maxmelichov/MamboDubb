//! Spawning the studio server and blocking on its one-line stdout handshake, after
//! MamboRambo's `RunnerProcess::spawn`.
//!
//! The difference is what gets spawned: not a bundled sidecar binary but
//! `uv run --project <workspace> python -m dubbing_app.server ...` out of the user's
//! checkout. The handshake, the stderr buffer and the kill-on-drop are the same.

use std::{
    collections::VecDeque,
    io::BufRead,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
};

use super::dto::{parse_ready_line, ReadySignal, ServerInfo};

/// How many stderr lines to keep. The server logs progress there, so the tail is what
/// matters when something goes wrong 40 lines into a model load.
const STDERR_RING_LINES: usize = 300;
const STDERR_LINE_CAP: usize = 2000;

pub struct RunnerState {
    pub process: Mutex<Option<RunnerProcess>>,
}

impl Default for RunnerState {
    fn default() -> Self {
        Self {
            process: Mutex::new(None),
        }
    }
}

/// A bounded tail of the child's stderr, shared with the pump thread.
#[derive(Default)]
pub struct StderrRing {
    lines: VecDeque<String>,
}

impl StderrRing {
    fn push(&mut self, line: &str) {
        let mut line = line.trim_end().to_string();
        line.truncate(STDERR_LINE_CAP);
        if self.lines.len() == STDERR_RING_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    fn text(&self) -> String {
        self.lines
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    }
}

pub struct RunnerProcess {
    port: u16,
    version: Option<String>,
    workspace: PathBuf,
    child: Child,
    stderr: Arc<Mutex<StderrRing>>,
}

impl RunnerProcess {
    pub fn spawn(uv: &Path, workspace: &Path) -> Result<Self, String> {
        let outputs = workspace.join("outputs");
        let mut cmd = Command::new(uv);
        cmd.args(server_args(workspace, &outputs))
            .current_dir(workspace)
            // Piped, not null: the server watches this pipe (`--exit-on-stdin-close`)
            // and EOF is how it learns the shell died even when the pid chain lies
            // (a surviving `uv` wrapper, a SIGKILL that skipped Drop). We never
            // write to it; holding the write end for the process's lifetime is the
            // whole mechanism.
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "failed to spawn the studio server with {} in {}: {err}",
                uv.display(),
                workspace.display()
            )
        })?;

        // Pump stderr from the very start: a first run is a multi-minute `uv sync`, and
        // its progress is the only sign of life while we block on the ready line.
        let stderr_ring = Arc::new(Mutex::new(StderrRing::default()));
        if let Some(stderr) = child.stderr.take() {
            let ring = stderr_ring.clone();
            std::thread::spawn(move || {
                let mut reader = std::io::BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if let Ok(mut ring) = ring.lock() {
                        ring.push(&line);
                    }
                    line.clear();
                }
            });
        }

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                kill_child(&mut child);
                return Err("failed to capture studio server stdout".to_string());
            }
        };
        let mut reader = std::io::BufReader::new(stdout);
        let mut line = String::new();

        if let Err(err) = reader.read_line(&mut line) {
            let stderr = tail(&stderr_ring);
            kill_child(&mut child);
            return Err(start_error(
                "failed to read ready signal",
                &err.to_string(),
                &stderr,
            ));
        }

        let signal: ReadySignal = match parse_ready_line(&line) {
            Ok(signal) => signal,
            Err(err) => {
                let stderr = tail(&stderr_ring);
                kill_child(&mut child);
                return Err(start_error("studio server did not start", &err, &stderr));
            }
        };

        // Drain the rest of stdout so a server that ever writes there cannot block on a
        // full pipe. Nothing but the handshake is supposed to arrive.
        std::thread::spawn(move || {
            let mut buf = String::new();
            while reader.read_line(&mut buf).unwrap_or(0) > 0 {
                buf.clear();
            }
        });

        Ok(Self {
            port: signal.port,
            version: signal.version,
            workspace: workspace.to_path_buf(),
            child,
            stderr: stderr_ring,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn info(&self) -> ServerInfo {
        ServerInfo {
            base_url: self.base_url(),
            port: self.port,
            version: self.version.clone(),
            workspace: self.workspace.display().to_string(),
        }
    }

    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn recent_stderr(&self) -> String {
        tail(&self.stderr)
    }

    pub fn kill(&mut self) {
        kill_child(&mut self.child);
    }
}

/// The child is ours; nothing outlives the shell. The server also watches its own
/// parent pid (docs/APP_ARCHITECTURE.md) belt and braces, same as MamboRambo.
impl Drop for RunnerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// The argv after `uv`, exactly as documented in the process contract.
pub fn server_args(workspace: &Path, outputs: &Path) -> Vec<String> {
    vec![
        "run".into(),
        "--project".into(),
        workspace.display().to_string(),
        "python".into(),
        "-m".into(),
        "dubbing_app.server".into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        "0".into(),
        "--outputs".into(),
        outputs.display().to_string(),
        "--exit-on-stdin-close".into(),
    ]
}

fn tail(ring: &Arc<Mutex<StderrRing>>) -> String {
    ring.lock().map(|ring| ring.text()).unwrap_or_default()
}

fn kill_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn start_error(context: &str, error: &str, stderr: &str) -> String {
    if stderr.trim().is_empty() {
        format!("{context}: {error}")
    } else {
        format!("{context}: {error}\n\nserver stderr:\n{}", stderr.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_argv_matches_the_process_contract() {
        let args = server_args(Path::new("/ws"), Path::new("/ws/outputs"));
        assert_eq!(
            args,
            vec![
                "run",
                "--project",
                "/ws",
                "python",
                "-m",
                "dubbing_app.server",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--outputs",
                "/ws/outputs",
                "--exit-on-stdin-close",
            ]
        );
    }

    #[test]
    fn the_ring_keeps_the_tail_not_the_head() {
        let mut ring = StderrRing::default();
        for i in 0..(STDERR_RING_LINES + 5) {
            ring.push(&format!("line {i}\n"));
        }
        let text = ring.text();
        assert!(!text.contains("line 0\n"), "the head is dropped");
        assert!(text.ends_with(&format!("line {}", STDERR_RING_LINES + 4)));
        assert_eq!(text.lines().count(), STDERR_RING_LINES);
    }

    #[test]
    fn a_single_absurd_line_cannot_grow_without_bound() {
        let mut ring = StderrRing::default();
        ring.push(&"x".repeat(STDERR_LINE_CAP * 4));
        assert_eq!(ring.text().len(), STDERR_LINE_CAP);
    }

    #[test]
    fn the_stderr_tail_is_attached_to_the_error() {
        let message = start_error("studio server did not start", "boom", "  Traceback...\n");
        assert!(message.contains("boom"));
        assert!(message.contains("Traceback..."));
    }

    #[test]
    fn a_silent_failure_still_reads_cleanly() {
        assert_eq!(start_error("ctx", "boom", "   "), "ctx: boom");
    }

    #[test]
    fn spawning_a_missing_uv_fails_without_hanging() {
        let result = RunnerProcess::spawn(
            Path::new("/nonexistent/uv"),
            Path::new("/nonexistent/workspace"),
        );
        let err = match result {
            Ok(_) => panic!("spawning a nonexistent uv should not succeed"),
            Err(err) => err,
        };
        assert!(err.contains("failed to spawn the studio server"), "{err}");
    }

    /// A stub standing in for `uv`, so the whole spawn path pipes, handshake, stderr
    /// pump, kill is exercised without a 10 GB venv.
    #[cfg(unix)]
    fn stub_uv(tag: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("dubstudio-uv-{tag}-{unique}.sh"));
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn a_ready_line_yields_a_live_process_on_the_announced_port() {
        let uv = stub_uv(
            "ready",
            "echo 'starting' 1>&2\n\
             echo '{\"status\":\"ready\",\"port\":54321,\"version\":\"0.1.0\"}'\n\
             sleep 30",
        );
        let mut process = RunnerProcess::spawn(&uv, Path::new(".")).unwrap();
        assert_eq!(process.base_url(), "http://127.0.0.1:54321");
        assert_eq!(process.info().version.as_deref(), Some("0.1.0"));
        assert!(process.is_alive());
        process.kill();
        assert!(!process.is_alive(), "kill reaps the child");
        let _ = std::fs::remove_file(&uv);
    }

    #[cfg(unix)]
    #[test]
    fn a_child_that_dies_before_announcing_reports_its_stderr() {
        let uv = stub_uv(
            "crash",
            "echo \"ModuleNotFoundError: No module named 'dubbing_app'\" 1>&2\nexit 1",
        );
        let err = match RunnerProcess::spawn(&uv, Path::new(".")) {
            Ok(_) => panic!("a crashing server should not look ready"),
            Err(err) => err,
        };
        assert!(err.contains("without a ready signal"), "{err}");
        assert!(
            err.contains("ModuleNotFoundError"),
            "the stderr tail is shown: {err}"
        );
        let _ = std::fs::remove_file(&uv);
    }
}
