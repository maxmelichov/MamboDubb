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

/// Find `uv`: explicit override, then the bundled sidecar, then the platform's usual
/// install spots, then `PATH`, then uv's own per-user install dir.
///
/// The sidecar is the case that matters for a real install: every bundle (.dmg, .msi /
/// .exe, .deb / .AppImage) ships `uv` via Tauri `externalBin`, which lands next to the
/// app binary, so a clean machine needs no package manager at all. The rest of the
/// chain is for dev runs and for users who prefer their own uv — and a GUI process
/// inherits almost nothing of the user's shell `PATH` (a Finder-launched .app, a
/// .desktop launcher), so the literal install paths matter more than they look.
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
    if let Some(path) = find_in_path(UV_EXE) {
        return Some(path);
    }
    // uv's own installer (`uv-installer.sh` / `uv-installer.ps1`) puts it here on all
    // three platforms, and `cargo install uv` in the second.
    home_dir().and_then(|home| {
        [
            home.join(".local").join("bin").join(UV_EXE),
            home.join(".cargo").join("bin").join(UV_EXE),
        ]
        .into_iter()
        .find(|path| path.is_file())
    })
}

/// The sidecar/binary file name, which is the only part of the uv lookup that is
/// spelled differently per platform.
pub const UV_EXE: &str = if cfg!(target_os = "windows") { "uv.exe" } else { "uv" };

/// Absolute paths worth probing before `PATH`, per platform. Windows has no
/// equivalent convention — winget and uv's installer both land on `PATH` or in the
/// per-user `.local\bin` the tail of `find_uv` checks — so the list is empty there.
#[cfg(target_os = "macos")]
const UV_FALLBACKS: &[&str] = &["/opt/homebrew/bin/uv", "/usr/local/bin/uv"];
#[cfg(target_os = "linux")]
const UV_FALLBACKS: &[&str] = &[
    "/usr/local/bin/uv",
    "/usr/bin/uv",
    "/home/linuxbrew/.linuxbrew/bin/uv",
];
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
const UV_FALLBACKS: &[&str] = &[];

/// The `uv` Tauri bundles as an `externalBin` sidecar. Sidecars are placed next to the
/// app executable — `Contents/MacOS/` in the .app, the install dir on Windows, `/usr/
/// lib/<product>/` for a .deb, `target/debug/` under `tauri dev` — so resolving from
/// `current_exe` covers all of them without needing an app handle (which is what keeps
/// `find_uv` callable from plain tests).
fn sidecar_uv() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let candidate = exe.parent()?.join(UV_EXE);
    candidate.is_file().then_some(candidate)
}

/// `which`, near enough. On Windows the name on `PATH` carries an extension, and the
/// bare name would never match — so try the `.exe` spelling first and the bare one
/// after (a Git-Bash-style shim without a suffix is still worth finding).
fn find_in_path(binary_name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    let bare = binary_name.trim_end_matches(".exe");
    for dir in env::split_paths(&path_var) {
        for name in [binary_name, bare] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// The user's home directory. `HOME` on macOS and Linux; Windows sets `USERPROFILE`
/// and only defines `HOME` inside MSYS/Git-Bash shells, which a bundled .exe launched
/// from the Start menu never inherits — so without the second variable every
/// home-relative lookup here silently returns `None` on Windows.
fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
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
    fn the_sidecar_name_carries_an_extension_only_on_windows() {
        // Tauri strips the target triple from `uv-<triple>[.exe]` and drops the
        // result next to the app binary, so this is the name the shell must look
        // for — and the one stage_desktop_payload.py must produce.
        assert_eq!(UV_EXE, if cfg!(target_os = "windows") { "uv.exe" } else { "uv" });
    }

    #[test]
    fn a_path_lookup_finds_a_binary_in_a_listed_directory() {
        let dir = TempDir::new("path-lookup");
        let name = if cfg!(target_os = "windows") { "fakeuv.exe" } else { "fakeuv" };
        let binary = dir.path().join(name);
        fs::write(&binary, "").unwrap();

        // SAFETY: single-threaded assertion over process env; restored immediately.
        let previous = env::var_os("PATH");
        env::set_var("PATH", dir.path());
        let found = find_in_path(if cfg!(target_os = "windows") { "fakeuv.exe" } else { "fakeuv" });
        // The Windows spelling is also reachable from the bare name, which is what
        // makes `find_uv`'s single `UV_EXE` call correct on all three platforms.
        let found_bare = find_in_path("fakeuv");
        match previous {
            Some(value) => env::set_var("PATH", value),
            None => env::remove_var("PATH"),
        }
        assert_eq!(found.as_deref(), Some(binary.as_path()));
        assert_eq!(found_bare.as_deref(), Some(binary.as_path()));
    }

    #[test]
    fn the_home_lookup_accepts_the_windows_variable() {
        // A bundled .exe gets USERPROFILE and no HOME; without the fallback every
        // home-relative path in this module would be silently unreachable there.
        // SAFETY: single-threaded assertion over process env; restored immediately.
        let home = env::var_os("HOME");
        let profile = env::var_os("USERPROFILE");
        env::remove_var("HOME");
        env::set_var("USERPROFILE", "/tmp/profile");
        let resolved = home_dir();
        env::remove_var("USERPROFILE");
        let empty_is_not_a_home = home_dir();
        match home {
            Some(value) => env::set_var("HOME", value),
            None => env::remove_var("HOME"),
        }
        if let Some(value) = profile {
            env::set_var("USERPROFILE", value);
        }
        assert_eq!(resolved, Some(PathBuf::from("/tmp/profile")));
        assert_eq!(empty_is_not_a_home, None, "no home variable at all is None");
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
