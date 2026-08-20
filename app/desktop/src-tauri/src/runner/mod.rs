mod dto;
mod process;

use tauri::{Manager, State};

pub use dto::ServerInfo;
pub use process::RunnerState;

use crate::provision::resolve_workspace;
use crate::workspace::{find_uv, inspect_workspace};
use process::RunnerProcess;

/// Start the studio server out of the configured workspace, or return the one already
/// running. Idempotent: the webview calls it on mount without checking first.
#[tauri::command]
pub async fn start_server(app: tauri::AppHandle) -> Result<ServerInfo, String> {
    tauri::async_runtime::spawn_blocking(move || ensure_server(&app))
        .await
        .map_err(|err| format!("failed to join server start task: {err}"))?
}

#[tauri::command]
pub async fn stop_server(state: State<'_, RunnerState>) -> Result<(), String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(mut process) = guard.take() {
        process.kill();
    }
    Ok(())
}

/// Where the server is, or `None` if it is not running. Never starts one this is what
/// the UI polls to decide between the studio and the setup screen.
#[tauri::command]
pub async fn get_server_url(state: State<'_, RunnerState>) -> Result<Option<ServerInfo>, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(process) = guard.as_mut() {
        if process.is_alive() {
            return Ok(Some(process.info()));
        }
    }
    *guard = None;
    Ok(None)
}

/// The tail of the server's stderr, for the boot panel and the error panel.
///
/// Reads the boot ring, not the process table: the live process shares the same
/// ring, and the ring exists *before* the process does, so this answers all through
/// the minutes a first-run `start_server` spends blocking on `uv sync` which is
/// when the UI polls it hardest. It must therefore never wait on `process` (held
/// only for moments now, but why queue) and never on `start_gate` (held for the
/// whole start).
#[tauri::command]
pub async fn get_server_log(state: State<'_, RunnerState>) -> Result<String, String> {
    let ring = state
        .boot_log
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    Ok(ring.text())
}

pub fn stop_managed_server(app: &tauri::AppHandle) {
    let state = app.state::<RunnerState>();
    match state.process.lock() {
        Ok(mut guard) => {
            if let Some(mut process) = guard.take() {
                process.kill();
            }
        }
        Err(_) => eprintln!("runner state lock poisoned during app shutdown"),
    };
}

fn ensure_server(app: &tauri::AppHandle) -> Result<ServerInfo, String> {
    let state = app.state::<RunnerState>();
    // One starter at a time. The slow part below runs *outside* the `process` lock
    // so `get_server_log` keeps answering during a first-run sync; this gate is what
    // replaces the exclusion that lock used to provide.
    let _starting = state
        .start_gate
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    {
        let mut guard = state
            .process
            .lock()
            .map_err(|_| "runner state lock poisoned".to_string())?;
        if let Some(process) = guard.as_mut() {
            if process.is_alive() {
                return Ok(process.info());
            }
            let stderr = process.recent_stderr();
            if !stderr.is_empty() {
                eprintln!("studio server exited; recent stderr:\n{stderr}");
            }
            *guard = None;
        }
    }

    let result = start_fresh(app, &state);
    if let Err(err) = &result {
        // The webview's desktop seam maps a rejected command to a null base URL and
        // the reason to a console.warn nobody reads; the ring is the one channel the
        // boot panel actually shows, so the reason lands there too.
        if let Ok(mut ring) = state.boot_log.lock() {
            ring.push(&format!("[start error] {err}"));
        }
    }
    result
}

fn start_fresh(app: &tauri::AppHandle, state: &RunnerState) -> Result<ServerInfo, String> {
    // Provisions on a fresh install (`ensure_server` already runs on a blocking
    // thread, so the copy is fine here); a stored or default checkout passes through.
    let workspace = resolve_workspace(app)?;
    let uv = find_uv();
    let report = inspect_workspace(&workspace, uv.as_deref());
    if !report.ready {
        return Err(describe_not_ready(&report));
    }
    let uv = uv.expect("ready implies uv was found");

    // The shared ring, so the sync's progress is readable while spawn blocks.
    let process = RunnerProcess::spawn(&uv, &workspace, state.boot_log.clone())?;
    let info = process.info();
    *state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())? = Some(process);
    Ok(info)
}

fn describe_not_ready(report: &crate::workspace::WorkspaceReport) -> String {
    if !report.uv_found {
        return format!(
            "uv was not found. Install it (brew install uv) or set {} to its path.",
            crate::workspace::UV_PATH_ENV
        );
    }
    if !report.exists {
        return format!("workspace not found: {}", report.path);
    }
    format!(
        "{} does not look like a DubbingQwen checkout (needs pyproject.toml and dubbing_app/)",
        report.path
    )
}
