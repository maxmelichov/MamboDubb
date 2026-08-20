//! The workspace: a MamboDubb checkout on disk that the shell runs the studio
//! server out of.
//!
//! Unlike MamboRambo, the sidecar is not a bundled binary the pipeline is a ~10 GB
//! Python environment plus tens of gigabytes of model weights, so it cannot ship inside
//! a .dmg. The app instead remembers *where the checkout lives* and drives it with `uv`.
//! Everything here is either a pure predicate over a directory or a `PATH`-style lookup,
//! which is what makes it testable without a Tauri app handle.

use std::{
    env,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use tauri::Wry;
use tauri_plugin_store::{Store, StoreExt};

/// Store file and key the workspace path is persisted under.
pub const STORE_FILE: &str = "settings.json";
pub const WORKSPACE_KEY: &str = "workspace";

/// Overrides, mirroring MamboRambo's env-var-first discovery.
pub const UV_PATH_ENV: &str = "DUBSTUDIO_UV_PATH";
pub const WORKSPACE_ENV: &str = "DUBSTUDIO_WORKSPACE";

/// What `check_workspace` reports back to the setup screen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceReport {
    pub path: String,
    /// The directory itself is there.
    pub exists: bool,
    /// It looks like a MamboDubb checkout: `pyproject.toml` + `dubbing_app/`.
    pub has_project: bool,
    /// A `.venv` with an interpreter absent is only a slow first run, not an error.
    pub has_venv: bool,
    /// `uv` was found; without it there is nothing to run the server with.
    pub uv_found: bool,
    /// The resolved `uv`, so the setup screen can show what it would use.
    pub uv_path: Option<String>,
    /// Everything the shell strictly needs before `start_server` can work.
    pub ready: bool,
}

/// Inspect a candidate workspace directory. Pure but for the filesystem reads.
pub fn inspect_workspace(path: &Path, uv: Option<&Path>) -> WorkspaceReport {
    let exists = path.is_dir();
    let has_project = exists && is_project_dir(path);
    let has_venv = exists && venv_python(path).is_some();
    let uv_found = uv.is_some();
    WorkspaceReport {
        path: path.display().to_string(),
        exists,
        has_project,
        has_venv,
        uv_found,
        uv_path: uv.map(|p| p.display().to_string()),
        ready: has_project && uv_found,
    }
}

/// The marker files that make a directory a MamboDubb checkout rather than any
/// old folder the user pointed at.
pub fn is_project_dir(path: &Path) -> bool {
    path.join("pyproject.toml").is_file() && path.join("dubbing_app").is_dir()
}

/// The interpreter inside `.venv`, if the checkout has been synced.
pub fn venv_python(path: &Path) -> Option<PathBuf> {
    let candidates = if cfg!(target_os = "windows") {
        [path.join(".venv/Scripts/python.exe")]
    } else {
        [path.join(".venv/bin/python")]
    };
    candidates.into_iter().find(|candidate| candidate.is_file())
}

/// Find `uv`: explicit override, then the bundled sidecar, then Homebrew's usual
/// spots, then `PATH`.
///
/// The sidecar is the case that matters for a real install: the .dmg bundles `uv` via
/// Tauri `externalBin`, which lands next to the app binary (`Contents/MacOS/uv`), so a
/// clean Mac needs no Homebrew at all. The rest of the chain is for dev runs and for
/// users who prefer their own uv — and a bundled .app inherits almost nothing of the
/// user's shell `PATH` on macOS, so the literal Homebrew path matters more than it
/// looks: that is the case that works when a brew-only setup launches from Finder.
pub fn find_uv() -> Option<PathBuf> {
    if let Some(raw) = env::var_os(UV_PATH_ENV) {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Some(path) = sidecar_uv() {
        return Some(path);
    }
    for candidate in UV_FALLBACKS {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Some(path) = find_in_path("uv") {
        return Some(path);
    }
    home_dir().and_then(|home| {
        let path = home.join(".local/bin/uv");
        path.is_file().then_some(path)
    })
}

const UV_FALLBACKS: &[&str] = &["/opt/homebrew/bin/uv", "/usr/local/bin/uv"];

/// The `uv` Tauri bundles as an `externalBin` sidecar. Sidecars are placed next to the
/// app executable — `Contents/MacOS/` in the .app, `target/debug/` under `tauri dev` —
/// so resolving from `current_exe` covers both without needing an app handle (which is
/// what keeps `find_uv` callable from plain tests).
fn sidecar_uv() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let name = if cfg!(target_os = "windows") { "uv.exe" } else { "uv" };
    let candidate = exe.parent()?.join(name);
    candidate.is_file().then_some(candidate)
}

fn find_in_path(binary_name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .map(|dir| dir.join(binary_name))
        .find(|path| path.is_file())
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

/// Where the workspace defaults to before the user has ever chosen one.
pub fn default_workspace() -> PathBuf {
    if let Some(raw) = env::var_os(WORKSPACE_ENV) {
        return PathBuf::from(raw);
    }
    home_dir()
        .map(|home| default_workspace_in(&home))
        .unwrap_or_else(|| PathBuf::from("MamboDubb"))
}

/// The rename-aware half, split out so a test can hand it a fake home. The
/// repo was renamed DubbingQwen → MamboDubb, so a fresh install resolves to
/// `Documents/MamboDubb` — but a checkout cloned under the old name keeps
/// working: the pre-rename directory wins only when it is an actual checkout
/// and the new name is not. A bare `is_dir` check is not enough here: a stray
/// empty `MamboDubb` folder would shadow a real legacy checkout and leave
/// onboarding stuck on a directory with nothing in it.
fn default_workspace_in(home: &Path) -> PathBuf {
    let preferred = home.join("Documents/MamboDubb");
    if is_project_dir(&preferred) {
        return preferred;
    }
    let legacy = home.join("Documents/DubbingQwen");
    if is_project_dir(&legacy) {
        return legacy;
    }
    // Neither is a checkout: point at where the README tells the user to clone.
    preferred
}

fn store(app: &tauri::AppHandle) -> Result<std::sync::Arc<Store<Wry>>, String> {
    app.store(STORE_FILE)
        .map_err(|err| format!("failed to open settings store: {err}"))
}

/// The workspace the user explicitly chose, if any. `None` means nothing was ever
/// stored — which is what lets first-launch provisioning tell "fresh install" apart
/// from "user pointed us at a checkout".
pub fn stored_workspace_setting(app: &tauri::AppHandle) -> Result<Option<PathBuf>, String> {
    Ok(store(app)?
        .get(WORKSPACE_KEY)
        .and_then(|value| value.as_str().map(str::to_string))
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from))
}


pub fn store_workspace(app: &tauri::AppHandle, path: &Path) -> Result<(), String> {
    let store = store(app)?;
    store.set(
        WORKSPACE_KEY,
        serde_json::Value::String(path.display().to_string()),
    );
    store
        .save()
        .map_err(|err| format!("failed to save settings store: {err}"))
}

// --- commands ---------------------------------------------------------------

#[tauri::command]
pub async fn get_workspace(app: tauri::AppHandle) -> Result<WorkspaceReport, String> {
    // Off the async runtime: on a first launch this is where provisioning copies the
    // bundled source (~40 MB) into Application Support.
    tauri::async_runtime::spawn_blocking(move || {
        let path = crate::provision::resolve_workspace(&app)?;
        Ok(inspect_workspace(&path, find_uv().as_deref()))
    })
    .await
    .map_err(|err| format!("failed to join workspace task: {err}"))?
}

#[tauri::command]
pub async fn check_workspace(path: String) -> Result<WorkspaceReport, String> {
    Ok(inspect_workspace(
        Path::new(path.trim()),
        find_uv().as_deref(),
    ))
}

/// Persist a workspace. Refuses a directory that is not a checkout, so the stored
/// value is always one `start_server` could actually use.
#[tauri::command]
pub async fn set_workspace(app: tauri::AppHandle, path: String) -> Result<WorkspaceReport, String> {
    let path = PathBuf::from(path.trim());
    if !path.is_dir() {
        return Err(format!("no such directory: {}", path.display()));
    }
    if !is_project_dir(&path) {
        return Err(format!(
            "{} does not look like a MamboDubb checkout (needs pyproject.toml and dubbing_app/)",
            path.display()
        ));
    }
    store_workspace(&app, &path)?;
    Ok(inspect_workspace(&path, find_uv().as_deref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = env::temp_dir().join(format!("dubstudio-{tag}-{unique}"));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn make_checkout(root: &Path) {
        fs::write(root.join("pyproject.toml"), "[project]\nname='x'\n").unwrap();
        fs::create_dir_all(root.join("dubbing_app")).unwrap();
    }

    #[test]
    fn missing_directory_is_not_ready() {
        let report = inspect_workspace(Path::new("/nope/does/not/exist"), None);
        assert!(!report.exists);
        assert!(!report.has_project);
        assert!(!report.has_venv);
        assert!(!report.uv_found);
        assert!(!report.ready);
    }

    #[test]
    fn a_directory_without_the_markers_is_not_a_project() {
        let dir = TempDir::new("empty");
        let report = inspect_workspace(dir.path(), Some(Path::new("/opt/homebrew/bin/uv")));
        assert!(report.exists);
        assert!(!report.has_project, "an empty dir is not a checkout");
        assert!(!report.ready);
    }

    #[test]
    fn pyproject_alone_is_not_enough() {
        let dir = TempDir::new("pyproject-only");
        fs::write(dir.path().join("pyproject.toml"), "[project]\n").unwrap();
        assert!(!is_project_dir(dir.path()));
    }

    #[test]
    fn a_checkout_with_uv_is_ready_even_without_a_venv() {
        let dir = TempDir::new("checkout");
        make_checkout(dir.path());
        let report = inspect_workspace(dir.path(), Some(Path::new("/opt/homebrew/bin/uv")));
        assert!(report.has_project);
        assert!(!report.has_venv, "uv sync has not run yet");
        assert!(report.uv_found);
        assert!(report.ready, "uv can create the venv on first run");
        assert_eq!(report.uv_path.as_deref(), Some("/opt/homebrew/bin/uv"));
    }

    #[test]
    fn a_checkout_without_uv_is_not_ready() {
        let dir = TempDir::new("no-uv");
        make_checkout(dir.path());
        let report = inspect_workspace(dir.path(), None);
        assert!(report.has_project);
        assert!(!report.ready, "nothing can run the server");
    }

    #[test]
    fn a_synced_checkout_reports_its_venv() {
        let dir = TempDir::new("synced");
        make_checkout(dir.path());
        let bin = if cfg!(target_os = "windows") {
            dir.path().join(".venv/Scripts")
        } else {
            dir.path().join(".venv/bin")
        };
        fs::create_dir_all(&bin).unwrap();
        let python = if cfg!(target_os = "windows") {
            bin.join("python.exe")
        } else {
            bin.join("python")
        };
        fs::write(&python, "").unwrap();
        let report = inspect_workspace(dir.path(), None);
        assert!(report.has_venv);
        assert_eq!(venv_python(dir.path()), Some(python));
    }

    #[test]
    fn the_default_prefers_the_post_rename_name() {
        let home = TempDir::new("home-both");
        let preferred = home.path().join("Documents/MamboDubb");
        let legacy = home.path().join("Documents/DubbingQwen");
        fs::create_dir_all(&preferred).unwrap();
        fs::create_dir_all(&legacy).unwrap();
        make_checkout(&preferred);
        make_checkout(&legacy);
        assert_eq!(default_workspace_in(home.path()), preferred);
    }

    #[test]
    fn a_pre_rename_checkout_still_resolves() {
        let home = TempDir::new("home-legacy");
        let legacy = home.path().join("Documents/DubbingQwen");
        fs::create_dir_all(&legacy).unwrap();
        make_checkout(&legacy);
        assert_eq!(default_workspace_in(home.path()), legacy);
    }

    #[test]
    fn a_stray_empty_new_dir_does_not_shadow_a_legacy_checkout() {
        // The failure this guards against: an empty `MamboDubb` folder (a stray
        // Finder creation, an aborted clone) next to a real pre-rename checkout.
        // Preferring it on bare existence would block onboarding on nothing.
        let home = TempDir::new("home-shadow");
        let legacy = home.path().join("Documents/DubbingQwen");
        fs::create_dir_all(home.path().join("Documents/MamboDubb")).unwrap();
        fs::create_dir_all(&legacy).unwrap();
        make_checkout(&legacy);
        assert_eq!(default_workspace_in(home.path()), legacy);
    }

    #[test]
    fn with_neither_a_checkout_the_default_is_the_new_name() {
        // Two bare directories, neither a checkout: nothing to rescue, so point
        // at the post-rename name the README tells the user to clone into.
        let home = TempDir::new("home-bare");
        fs::create_dir_all(home.path().join("Documents/MamboDubb")).unwrap();
        fs::create_dir_all(home.path().join("Documents/DubbingQwen")).unwrap();
        assert_eq!(
            default_workspace_in(home.path()),
            home.path().join("Documents/MamboDubb")
        );
    }

    #[test]
    fn with_neither_directory_the_default_is_the_new_name() {
        // A first launch on a machine with no checkout at all: point at where
        // the README will tell the user to clone, not at the pre-rename path.
        let home = TempDir::new("home-empty");
        assert_eq!(
            default_workspace_in(home.path()),
            home.path().join("Documents/MamboDubb")
        );
    }

    #[test]
    fn the_env_override_wins_for_the_default_workspace() {
        // SAFETY: single-threaded assertion over process env; restored immediately.
        let previous = env::var_os(WORKSPACE_ENV);
        env::set_var(WORKSPACE_ENV, "/tmp/elsewhere");
        assert_eq!(default_workspace(), PathBuf::from("/tmp/elsewhere"));
        match previous {
            Some(value) => env::set_var(WORKSPACE_ENV, value),
            None => env::remove_var(WORKSPACE_ENV),
        }
    }
}
