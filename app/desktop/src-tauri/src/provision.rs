//! First-launch provisioning: turning a bare .dmg install into a runnable workspace.
//!
//! The .app bundles the pipeline *source* (`Contents/Resources/workspace`, staged by
//! `scripts/stage_desktop_payload.py`) but a bundle is read-only and `uv sync` has to
//! write a `.venv` next to the pyproject. So on a machine with no stored workspace and
//! no checkout at the default path, the shell copies the payload to
//! `~/Library/Application Support/MamboDubb/workspace`, stores that as the workspace,
//! and the runner proceeds exactly as if the user had cloned it by hand.
//!
//! The copy is versioned: a `.mambodubb-payload` marker at the workspace root records
//! the app version that wrote it. On upgrade the source trees are replaced wholesale
//! (dir-by-dir, so deleted upstream files cannot linger and shadow), while `.venv`,
//! `.env`, `models/` and `outputs/` — everything the payload never contains but the
//! user's machine accumulates — are left alone.
//!
//! `set_workspace` stays the escape hatch: a stored path that is not the provisioned
//! root is the user's own checkout, and it is never touched.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use tauri::Manager;

use crate::workspace::{
    default_workspace, is_project_dir, store_workspace, stored_workspace_setting,
};

/// Marker file recording which app version last wrote the payload copy.
pub const PAYLOAD_MARKER: &str = ".mambodubb-payload";

/// Workspace entries a refresh must never touch: none of them exist in the payload,
/// all of them are expensive or private on the user's machine.
/// `tools/` is here for the same reason as the rest even though the payload has never
/// contained it: it is where `dubbing/tools.py` puts binaries the app installed for
/// this machine (the static ffmpeg a brewless Mac or a winget-less Windows box gets),
/// and re-downloading those on every app upgrade would be a silent 100 MB tax.
pub const PRESERVED: &[&str] =
    &[".venv", ".env", "models", "outputs", "tools", PAYLOAD_MARKER];

/// The workspace the runner should use, provisioning it first if this is a fresh
/// install. The precedence is deliberate:
///
/// 1. a stored workspace (the user chose it, or an earlier launch provisioned it) —
///    refreshed in place on app upgrade *only* when it is the provisioned copy;
/// 2. an existing checkout at the default path (the pre-bundling install story);
/// 3. the bundled payload, copied somewhere writable — the clean-Mac path;
/// 4. the default path anyway, when there is no payload to copy (a dev build run
///    before staging): the setup screen reports it not-ready, same as before.
pub fn resolve_workspace(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let version = app.package_info().version.to_string();

    if let Some(stored) = stored_workspace_setting(app)? {
        if provisioned_root(app).is_ok_and(|root| root == stored) {
            if let Ok(source) = payload_source(app) {
                refresh_if_stale(&source, &stored, &version)?;
            }
        }
        return Ok(stored);
    }

    let default = default_workspace();
    if is_project_dir(&default) {
        return Ok(default);
    }

    let source = match payload_source(app) {
        Ok(source) => source,
        Err(_) => return Ok(default),
    };
    let dest = provisioned_root(app).map_err(|err| err.to_string())?;
    if !is_project_dir(&dest) {
        provision(&source, &dest, &version)?;
    } else {
        // A previous install provisioned it but the store was lost (or wiped);
        // adopt it and let the refresh below bring it up to date.
        refresh_if_stale(&source, &dest, &version)?;
    }
    store_workspace(app, &dest)?;
    Ok(dest)
}

/// Where the writable copy lives. Not `app_data_dir` (which is the identifier-named
/// directory holding the settings store) but a product-named sibling, so the path the
/// user sees in the setup screen reads as "MamboDubb", not a reverse-DNS string.
///
/// `local_data_dir`, not `data_dir`, and the difference only exists on Windows:
/// `data_dir` is `%APPDATA%` (the *roaming* profile), and on a managed/domain machine
/// everything under it is copied to the server at logon. This directory grows a ~10 GB
/// `.venv` and tens of GB of model weights, so roaming it would be a disaster. macOS
/// (`~/Library/Application Support`) and Linux (`~/.local/share`) resolve both to the
/// same path, so existing installs there are unaffected.
fn provisioned_root(app: &tauri::AppHandle) -> Result<PathBuf, tauri::Error> {
    Ok(app.path().local_data_dir()?.join("MamboDubb").join("workspace"))
}

/// The bundled payload, per the `bundle.resources` map in tauri.conf.json.
fn payload_source(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .resource_dir()
        .map_err(|err| format!("no resource dir: {err}"))?
        .join("workspace");
    if dir.join("pyproject.toml").is_file() {
        Ok(dir)
    } else {
        Err(format!("no bundled workspace payload at {}", dir.display()))
    }
}

// --- the pure half: filesystem in, filesystem out, no app handle -------------------

/// First copy: payload → empty (or absent) destination, then stamp the marker.
pub fn provision(source: &Path, dest: &Path, version: &str) -> Result<(), String> {
    copy_tree(source, dest).map_err(|err| {
        format!(
            "failed to provision workspace {} from {}: {err}",
            dest.display(),
            source.display()
        )
    })?;
    write_marker(dest, version)
}

/// Replace the payload-owned entries of an existing copy when the marker says it was
/// written by a different app version. `PRESERVED` entries are never touched; each
/// payload entry is removed and re-copied whole.
pub fn refresh_if_stale(source: &Path, dest: &Path, version: &str) -> Result<(), String> {
    if payload_version(dest).as_deref() == Some(version) {
        return Ok(());
    }
    let entries = fs::read_dir(source)
        .map_err(|err| format!("failed to read payload {}: {err}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read payload entry: {err}"))?;
        let name = entry.file_name();
        if PRESERVED.iter().any(|keep| name == *keep) {
            continue;
        }
        let target = dest.join(&name);
        remove_entry(&target)
            .and_then(|_| copy_tree(&entry.path(), &target))
            .map_err(|err| {
                format!("failed to refresh {} in workspace: {err}", target.display())
            })?;
    }
    write_marker(dest, version)
}

/// The version recorded in the marker, `None` when absent or unreadable (both mean
/// "refresh": pre-marker copies predate this scheme and deserve one).
pub fn payload_version(root: &Path) -> Option<String> {
    let text = fs::read_to_string(root.join(PAYLOAD_MARKER)).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn write_marker(root: &Path, version: &str) -> Result<(), String> {
    fs::write(root.join(PAYLOAD_MARKER), format!("{version}\n"))
        .map_err(|err| format!("failed to write payload marker: {err}"))
}

fn remove_entry(path: &Path) -> io::Result<()> {
    match path.symlink_metadata() {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// Recursive copy that forces owner-write onto everything it writes: the source sits
/// inside a read-only .app, and `fs::copy` preserves the read-only mode bits, which
/// would leave `uv sync` unable to touch its own project.
fn copy_tree(source: &Path, dest: &Path) -> io::Result<()> {
    if source.is_dir() {
        fs::create_dir_all(dest)?;
        make_writable(dest)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            copy_tree(&entry.path(), &dest.join(entry.file_name()))?;
        }
    } else {
        fs::copy(source, dest)?;
        make_writable(dest)?;
    }
    Ok(())
}

#[cfg(unix)]
fn make_writable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(perms.mode() | 0o200);
    fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
fn make_writable(path: &Path) -> io::Result<()> {
    let mut perms = fs::metadata(path)?.permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    fs::set_permissions(path, perms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = env::temp_dir().join(format!("dubstudio-provision-{tag}-{unique}"));
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

    /// A miniature payload: the two project markers plus a nested source file.
    fn make_payload(root: &Path) {
        fs::write(root.join("pyproject.toml"), "[project]\nname='x'\n").unwrap();
        fs::create_dir_all(root.join("dubbing_app")).unwrap();
        fs::write(root.join("dubbing_app/server.py"), "# v1\n").unwrap();
    }

    #[test]
    fn provisioning_copies_the_tree_and_stamps_the_marker() {
        let source = TempDir::new("src");
        make_payload(source.path());
        let dest = TempDir::new("dst");
        let dest = dest.path().join("workspace");

        provision(source.path(), &dest, "0.1.3").unwrap();

        assert!(dest.join("pyproject.toml").is_file());
        assert!(dest.join("dubbing_app/server.py").is_file());
        assert_eq!(payload_version(&dest).as_deref(), Some("0.1.3"));
        // What we provisioned must pass the same predicate set_workspace applies.
        assert!(is_project_dir(&dest));
    }

    #[cfg(unix)]
    #[test]
    fn provisioned_files_are_writable_even_from_a_read_only_source() {
        use std::os::unix::fs::PermissionsExt;
        let source = TempDir::new("ro-src");
        make_payload(source.path());
        let file = source.path().join("pyproject.toml");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o444)).unwrap();

        let dest = TempDir::new("ro-dst");
        let dest = dest.path().join("workspace");
        provision(source.path(), &dest, "1").unwrap();

        let mode = fs::metadata(dest.join("pyproject.toml")).unwrap().permissions().mode();
        assert_ne!(mode & 0o200, 0, "uv sync needs to write here");
    }

    #[test]
    fn a_matching_marker_makes_refresh_a_no_op() {
        let source = TempDir::new("noop-src");
        make_payload(source.path());
        let dest = TempDir::new("noop-dst");
        let dest = dest.path().join("workspace");
        provision(source.path(), &dest, "0.2.0").unwrap();

        // The user edited a provisioned file; same version → not overwritten.
        fs::write(dest.join("dubbing_app/server.py"), "# local edit\n").unwrap();
        refresh_if_stale(source.path(), &dest, "0.2.0").unwrap();
        let text = fs::read_to_string(dest.join("dubbing_app/server.py")).unwrap();
        assert_eq!(text, "# local edit\n");
    }

    #[test]
    fn an_upgrade_replaces_source_but_preserves_user_state() {
        let source = TempDir::new("up-src");
        make_payload(source.path());
        let dest = TempDir::new("up-dst");
        let dest = dest.path().join("workspace");
        provision(source.path(), &dest, "0.1.0").unwrap();

        // Simulate what a machine accumulates between versions...
        fs::create_dir_all(dest.join(".venv/bin")).unwrap();
        fs::write(dest.join(".venv/bin/python"), "").unwrap();
        fs::write(dest.join(".env"), "HF_TOKEN=secret\n").unwrap();
        fs::create_dir_all(dest.join("models/gemma")).unwrap();
        fs::write(dest.join("models/gemma/weights.bin"), "81GB, honest").unwrap();
        fs::create_dir_all(dest.join("outputs")).unwrap();
        // ...a local edit that the upgrade is expected to roll over...
        fs::write(dest.join("dubbing_app/server.py"), "# stale local edit\n").unwrap();
        // ...and a file the new payload no longer ships.
        fs::write(dest.join("dubbing_app/removed_module.py"), "gone upstream\n").unwrap();

        // The new version ships a changed server.py.
        fs::write(source.path().join("dubbing_app/server.py"), "# v2\n").unwrap();
        refresh_if_stale(source.path(), &dest, "0.2.0").unwrap();

        assert_eq!(
            fs::read_to_string(dest.join("dubbing_app/server.py")).unwrap(),
            "# v2\n"
        );
        assert!(
            !dest.join("dubbing_app/removed_module.py").exists(),
            "dir-wholesale refresh must not leave deleted files to shadow imports"
        );
        assert!(dest.join(".venv/bin/python").is_file(), ".venv survives upgrades");
        assert_eq!(fs::read_to_string(dest.join(".env")).unwrap(), "HF_TOKEN=secret\n");
        assert!(dest.join("models/gemma/weights.bin").is_file(), "models survive");
        assert!(dest.join("outputs").is_dir());
        assert_eq!(payload_version(&dest).as_deref(), Some("0.2.0"));
    }

    #[test]
    fn a_missing_marker_counts_as_stale() {
        let source = TempDir::new("nomark-src");
        make_payload(source.path());
        let dest = TempDir::new("nomark-dst");
        let dest = dest.path().join("workspace");
        copy_tree(source.path(), &dest).unwrap();
        assert_eq!(payload_version(&dest), None);

        fs::write(source.path().join("dubbing_app/server.py"), "# v2\n").unwrap();
        refresh_if_stale(source.path(), &dest, "0.2.0").unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("dubbing_app/server.py")).unwrap(),
            "# v2\n"
        );
    }
}
