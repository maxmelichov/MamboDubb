//! First-launch provisioning: turning a bare .dmg install into a runnable workspace.
//!
//! The .app bundles the pipeline *source* (`Contents/Resources/workspace`, staged by
//! `scripts/stage_desktop_payload.py`) but a bundle is read-only and `uv sync` has to
//! write a `.venv` next to the pyproject. So on a machine with no stored workspace and
//! no checkout at the default path, the shell copies the payload to
//! `~/Library/Application Support/MamboDubb/workspace`, stores that as the workspace,
//! and the runner proceeds exactly as if the user had cloned it by hand.
//!
//! One set of model weights rides in that copy: the 31 MB CC-BY-4.0 diarization
//! pipeline under `third_party/pyannote-speaker-diarization-community-1`. It is
//! payload, not user state — which is the right side of the line, because it means a
//! fresh install can tell two speakers apart before it has touched the network, and a
//! deleted or half-copied one heals on the next upgrade instead of staying broken.
//!
//! The copy is stamped: a `.mambodubb-payload` marker at the workspace root records
//! the app version that wrote it *and a digest of the payload tree it was written
//! from*. On upgrade the source trees are replaced wholesale (dir-by-dir, so deleted
//! upstream files cannot linger and shadow), while `.venv`, `.env`, `models/` and
//! `outputs/` — everything the payload never contains but the user's machine
//! accumulates — are left alone.
//!
//! The digest is the load-bearing half, and it was learned the hard way (issue #15).
//! The marker used to hold the version alone, which quietly assumed that a version
//! string identifies a payload. It does not. Shipping a payload hotfix means
//! rebuilding the disk image under the tag that is already broken, because the fix is
//! to the *contents* and the release it repairs is the one people already have. Under
//! a version-only marker that rebuild is indistinguishable from the copy on disk, so
//! the refresh declines, and a user who redownloads the fixed image gets the identical
//! failure with nothing to explain it. That is exactly what happened to the 0.4.0
//! rebuild for #14: the .app carried the restored `qwen_tts/core/models/`, the
//! extracted workspace never received it, and every dub came out as untranslated
//! source audio.
//!
//! So the identity recorded is a content identity: every relative path and every byte
//! of the bundled payload, folded into one SHA-256 (`payload_digest`). Any difference
//! that could matter (a restored module, a changed file, a file dropped upstream)
//! moves the digest and earns a refresh, whether or not the version moved with it.
//! Hashing ~44 MB costs a fraction of a second and buys the guarantee that what is on
//! disk is what shipped.
//!
//! A marker written by an older build carries a version and no digest. It is treated
//! as stale exactly once: there is no way to know what tree produced it, the honest
//! answer is "refresh and find out", and the useful side effect is that every install
//! predating this scheme collects the #14 payload fix on its next launch. The rewritten
//! marker carries a digest, so the refresh does not repeat.
//!
//! `set_workspace` stays the escape hatch: a stored path that is not the provisioned
//! root is the user's own checkout, and it is never touched.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};
use tauri::Manager;

use crate::workspace::{
    default_workspace, is_project_dir, store_workspace, stored_workspace_setting,
};

/// Marker file recording which app version last wrote the payload copy, and the
/// digest of the payload tree it was written from: `"<version> <hex>\n"`.
pub const PAYLOAD_MARKER: &str = ".mambodubb-payload";

/// Workspace entries a refresh must never touch: none of them exist in the payload,
/// all of them are expensive or private on the user's machine.
/// `tools/` is here for the same reason as the rest even though the payload has never
/// contained it: it is where `dubbing/tools.py` puts binaries the app installed for
/// this machine (the static ffmpeg a brewless Mac or a winget-less Windows box gets),
/// and re-downloading those on every app upgrade would be a silent 100 MB tax.
pub const PRESERVED: &[&str] = &[
    ".venv",
    ".env",
    "models",
    "outputs",
    "tools",
    PAYLOAD_MARKER,
];

/// The workspace the runner should use, provisioning it first if this is a fresh
/// install. The precedence is deliberate:
///
/// 1. a stored workspace (the user chose it, or an earlier launch provisioned it),
///    but only while `stored_is_usable` still holds, and refreshed in place on app
///    upgrade *only* when it is the provisioned copy;
/// 2. an existing checkout at the default path (the pre-bundling install story);
/// 3. the bundled payload, copied somewhere writable — the clean-Mac path;
/// 4. the default path anyway, when there is no payload to copy (a dev build run
///    before staging): the setup screen reports it not-ready, same as before.
///
/// A stored path that no longer answers falls through to 2 to 4 instead of being
/// handed back, and whatever wins is stored in its place, so the store cannot stay
/// pointed at a hole.
pub fn resolve_workspace(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let version = app.package_info().version.to_string();

    let stored = stored_workspace_setting(app)?;
    if let Some(stored) = stored.as_ref().filter(|path| stored_is_usable(path)) {
        if provisioned_root(app).is_ok_and(|root| &root == stored) {
            if let Ok(source) = payload_source(app) {
                refresh_if_stale(&source, stored, &version)?;
            }
        }
        return Ok(stored.clone());
    }

    let resolved = resolve_without_store(app, &version)?;
    // Rewrite the setting whenever it disagrees with what we are about to run. This is
    // the whole point of the fall-through: leave the dead path in the store and the
    // next launch walks into the same wall.
    if stored.as_deref() != Some(resolved.as_path()) {
        store_workspace(app, &resolved)?;
    }
    Ok(resolved)
}

/// Whether a stored workspace setting still names something the runner can use.
///
/// It used to be enough that the setting existed. Then a workspace got moved (or the
/// user emptied `~/Library/Application Support` to reclaim the disk), and the app was
/// unrecoverable: `resolve_workspace` handed the dead path straight to the runner, `uv`
/// died on `failed to refresh <path>/uv.lock`, and because the setting was still there
/// no launch ever re-provisioned, with a complete payload sitting in the bundle the
/// whole time. Only deleting settings.json by hand got out of it.
///
/// The same predicate `set_workspace` applies, deliberately: a directory the setup
/// screen would refuse to accept as a workspace is not one we should silently keep
/// running against. A half-deleted workspace (the directory survives, `pyproject.toml`
/// does not) fails it too, which is what we want; re-provisioning replaces the source
/// tree and preserves `.venv`, `models` and the rest per `PRESERVED`.
fn stored_is_usable(stored: &Path) -> bool {
    is_project_dir(stored)
}

/// Steps 2 to 4 of `resolve_workspace`'s order: everything that does not depend on the
/// stored setting. Split out so the caller can store the result exactly once, whether
/// it got here from a fresh install or from a stored path that went missing.
fn resolve_without_store(app: &tauri::AppHandle, version: &str) -> Result<PathBuf, String> {
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
        provision(&source, &dest, version)?;
    } else {
        // A previous install provisioned it but the store was lost (or wiped);
        // adopt it and let the refresh below bring it up to date.
        refresh_if_stale(&source, &dest, version)?;
    }
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
    Ok(app
        .path()
        .local_data_dir()?
        .join("MamboDubb")
        .join("workspace"))
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
    let digest = payload_digest(source)?;
    copy_tree(source, dest).map_err(|err| {
        format!(
            "failed to provision workspace {} from {}: {err}",
            dest.display(),
            source.display()
        )
    })?;
    write_marker(dest, version, &digest)
}

/// Replace the payload-owned entries of an existing copy unless the marker proves the
/// copy already came from this exact payload: same app version *and* same content
/// digest. Either one moving is a refresh, and a marker with no digest at all (written
/// before this scheme) counts as stale. `PRESERVED` entries are never touched; each
/// payload entry is removed and re-copied whole.
pub fn refresh_if_stale(source: &Path, dest: &Path, version: &str) -> Result<(), String> {
    let digest = payload_digest(source)?;
    if payload_stamp(dest) == Some((version.to_string(), digest.clone())) {
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
            .map_err(|err| format!("failed to refresh {} in workspace: {err}", target.display()))?;
    }
    write_marker(dest, version, &digest)
}

/// The identity recorded in the marker: version *and* payload digest. `None` whenever
/// either is missing, which is the whole point of returning a pair rather than just the
/// version. An absent marker, an unreadable one, and a version-only one from an older
/// build all mean the same thing here: nothing on disk can claim this copy matches a
/// particular payload, so none of them gets to short-circuit a refresh.
pub fn payload_stamp(root: &Path) -> Option<(String, String)> {
    let text = fs::read_to_string(root.join(PAYLOAD_MARKER)).ok()?;
    let mut fields = text.split_whitespace();
    let version = fields.next()?.to_string();
    let digest = fields.next()?.to_string();
    Some((version, digest))
}

/// A content identity for a payload tree: every relative path and every byte of every
/// file it holds, in a fixed order, folded into one SHA-256.
///
/// Directory listings arrive in whatever order the filesystem feels like, so names are
/// sorted at every level; without that the same tree hashes differently on two machines
/// and every launch would "detect drift". Paths go into the hash beside the bytes, so
/// moving a file is a change even when no byte of content moved, and each entry carries
/// a type tag and its length so that no arrangement of files can be reshuffled into the
/// same byte stream as another.
///
/// Symlinks are followed, matching `copy_tree`: this must hash exactly what a copy
/// would write, or the marker would describe a tree that was never provisioned.
pub fn payload_digest(source: &Path) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hash_tree(&mut hasher, source, "")
        .map_err(|err| format!("failed to hash payload {}: {err}", source.display()))?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn hash_tree(hasher: &mut Sha256, path: &Path, rel: &str) -> io::Result<()> {
    if path.is_dir() {
        hasher.update(b"dir\0");
        hasher.update(rel.as_bytes());
        hasher.update(b"\0");
        let mut names: Vec<_> = fs::read_dir(path)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<io::Result<Vec<_>>>()?;
        names.sort();
        for name in names {
            let child = name.to_string_lossy();
            let child_rel = if rel.is_empty() {
                child.into_owned()
            } else {
                format!("{rel}/{child}")
            };
            hash_tree(hasher, &path.join(&name), &child_rel)?;
        }
    } else {
        let bytes = fs::read(path)?;
        hasher.update(b"file\0");
        hasher.update(rel.as_bytes());
        hasher.update(b"\0");
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(())
}

fn write_marker(root: &Path, version: &str, digest: &str) -> Result<(), String> {
    fs::write(root.join(PAYLOAD_MARKER), format!("{version} {digest}\n"))
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
        let (version, digest) = payload_stamp(&dest).expect("marker stamped");
        assert_eq!(version, "0.1.3");
        assert_eq!(digest, payload_digest(source.path()).unwrap());
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

        let mode = fs::metadata(dest.join("pyproject.toml"))
            .unwrap()
            .permissions()
            .mode();
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

    /// Issue #15: the payload hotfix case. Same version, different contents, and the
    /// old version-only marker declined to refresh, so the fix never landed.
    #[test]
    fn a_rebuilt_payload_under_the_same_version_is_stale() {
        let source = TempDir::new("rebuild-src");
        make_payload(source.path());
        let dest = TempDir::new("rebuild-dst");
        let dest = dest.path().join("workspace");
        provision(source.path(), &dest, "0.4.0").unwrap();

        // The .dmg is rebuilt under the same tag with the missing module restored.
        fs::create_dir_all(source.path().join("dubbing/core/models")).unwrap();
        fs::write(
            source.path().join("dubbing/core/models/__init__.py"),
            "# the fix\n",
        )
        .unwrap();

        refresh_if_stale(source.path(), &dest, "0.4.0").unwrap();

        assert!(
            dest.join("dubbing/core/models/__init__.py").is_file(),
            "a payload hotfix must reach an existing workspace without a version bump"
        );
    }

    /// A marker from a build that recorded the version alone: refresh once, then stamp
    /// the new format so it does not refresh forever.
    #[test]
    fn a_version_only_marker_is_stale_exactly_once() {
        let source = TempDir::new("legacy-src");
        make_payload(source.path());
        let dest = TempDir::new("legacy-dst");
        let dest = dest.path().join("workspace");
        copy_tree(source.path(), &dest).unwrap();
        fs::write(dest.join(PAYLOAD_MARKER), "0.4.0\n").unwrap();
        assert_eq!(payload_stamp(&dest), None, "no digest to compare against");

        fs::write(source.path().join("dubbing_app/server.py"), "# v2\n").unwrap();
        refresh_if_stale(source.path(), &dest, "0.4.0").unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("dubbing_app/server.py")).unwrap(),
            "# v2\n"
        );

        // Now stamped in full, a second launch is a no-op again.
        let (version, digest) = payload_stamp(&dest).expect("rewritten in the new format");
        assert_eq!(version, "0.4.0");
        assert_eq!(digest, payload_digest(source.path()).unwrap());
        fs::write(dest.join("dubbing_app/server.py"), "# local edit\n").unwrap();
        refresh_if_stale(source.path(), &dest, "0.4.0").unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("dubbing_app/server.py")).unwrap(),
            "# local edit\n"
        );
    }

    #[test]
    fn the_digest_is_stable_across_trees_and_moves_with_content() {
        let a = TempDir::new("dig-a");
        make_payload(a.path());
        let b = TempDir::new("dig-b");
        make_payload(b.path());
        assert_eq!(
            payload_digest(a.path()).unwrap(),
            payload_digest(b.path()).unwrap(),
            "identical trees in different places must hash the same"
        );

        let before = payload_digest(a.path()).unwrap();
        fs::write(a.path().join("dubbing_app/server.py"), "# v2\n").unwrap();
        assert_ne!(before, payload_digest(a.path()).unwrap(), "content change");

        // A rename moves no bytes, and must still register.
        let renamed = payload_digest(b.path()).unwrap();
        fs::rename(
            b.path().join("dubbing_app/server.py"),
            b.path().join("dubbing_app/other.py"),
        )
        .unwrap();
        assert_ne!(renamed, payload_digest(b.path()).unwrap(), "path change");
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
        fs::write(
            dest.join("dubbing_app/removed_module.py"),
            "gone upstream\n",
        )
        .unwrap();

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
        assert!(
            dest.join(".venv/bin/python").is_file(),
            ".venv survives upgrades"
        );
        assert_eq!(
            fs::read_to_string(dest.join(".env")).unwrap(),
            "HF_TOKEN=secret\n"
        );
        assert!(
            dest.join("models/gemma/weights.bin").is_file(),
            "models survive"
        );
        assert!(dest.join("outputs").is_dir());
        assert_eq!(
            payload_stamp(&dest),
            Some(("0.2.0".to_string(), payload_digest(source.path()).unwrap()))
        );
    }

    /// Issue #17, the dead end. A stored workspace whose directory is gone must stop
    /// being honoured, or `resolve_workspace` hands the runner a path `uv` cannot open
    /// and no launch ever re-provisions.
    #[test]
    fn a_stored_workspace_that_is_gone_is_not_usable() {
        let home = TempDir::new("gone");
        let stored = home.path().join("moved-away/workspace");
        assert!(!stored.exists());
        assert!(
            !stored_is_usable(&stored),
            "a path that is not there must fall through to provisioning"
        );
    }

    /// The half-deleted case: the directory survived, the project did not. Just as dead
    /// to `uv`, and `set_workspace` would refuse it too.
    #[test]
    fn a_stored_directory_that_is_not_a_project_is_not_usable() {
        let home = TempDir::new("gutted");
        let stored = home.path().join("workspace");
        fs::create_dir_all(&stored).unwrap();
        assert!(!stored_is_usable(&stored), "an empty directory is not one");

        // Everything back except the manifest: still not a project.
        fs::create_dir_all(stored.join("dubbing_app")).unwrap();
        assert!(
            !stored_is_usable(&stored),
            "dubbing_app without pyproject.toml is a torn workspace, not a workspace"
        );
    }

    /// And the case that must not regress: a real provisioned workspace is honoured, so
    /// the fix costs nobody a re-provision on a working install.
    #[test]
    fn a_provisioned_workspace_stays_usable() {
        let source = TempDir::new("keep-src");
        make_payload(source.path());
        let dest = TempDir::new("keep-dst");
        let dest = dest.path().join("workspace");
        provision(source.path(), &dest, "0.5.0").unwrap();

        assert!(stored_is_usable(&dest));
        // A plain checkout the user pointed at by hand qualifies the same way.
        assert!(stored_is_usable(source.path()));
    }

    #[test]
    fn a_missing_marker_counts_as_stale() {
        let source = TempDir::new("nomark-src");
        make_payload(source.path());
        let dest = TempDir::new("nomark-dst");
        let dest = dest.path().join("workspace");
        copy_tree(source.path(), &dest).unwrap();
        assert_eq!(payload_stamp(&dest), None);

        fs::write(source.path().join("dubbing_app/server.py"), "# v2\n").unwrap();
        refresh_if_stale(source.path(), &dest, "0.2.0").unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("dubbing_app/server.py")).unwrap(),
            "# v2\n"
        );
    }
}
