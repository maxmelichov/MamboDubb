//! The two filesystem capabilities the webview cannot have on its own: revealing a
//! path in the file manager, and getting a real absolute path out of a native open
//! dialog.
//!
//! The browser build can only ever hand the server a `File` blob; the pipeline takes a
//! path. This is the one thing the studio UI needs from the shell.

use std::{panic, path::PathBuf};

use tauri_plugin_dialog::DialogExt;

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "mkv", "webm", "m4v", "avi", "mpg", "mpeg", "ts", "wmv", "flv",
];
const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"];
// What dubbing/transcript.py can actually parse — srt, vtt, and YouTube's
// json3. No txt: without timestamps the pipeline has nothing to place.
const TRANSCRIPT_EXTENSIONS: &[&str] = &["srt", "vtt", "json3"];

/// Reveal a path in the platform's file manager: Finder, Explorer, or whatever
/// implements the freedesktop `FileManager1` D-Bus interface (Nautilus, Dolphin,
/// Thunar…). This goes through `tauri-plugin-opener` rather than the `showfile` crate
/// because the plugin is the one with a Linux story that survives a session without a
/// running `FileManager1` service: it falls back to opening the containing directory
/// with `xdg-open`, where `showfile` would leave the button doing nothing at all.
/// `catch_unwind` stays because these are FFI/D-Bus calls and a panic across the
/// command boundary would take the webview's promise with it.
#[tauri::command]
pub async fn reveal_path(path: PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("path does not exist: {}", path.display()));
    }
    tauri::async_runtime::spawn_blocking(move || {
        match panic::catch_unwind(|| tauri_plugin_opener::reveal_item_in_dir(&path)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(format!("failed to reveal path in file manager: {err}")),
            Err(_) => Err("failed to reveal path in file manager".to_string()),
        }
    })
    .await
    .map_err(|err| format!("failed to join reveal task: {err}"))?
}

/// Native open dialog for a source video. `Ok(None)` is a cancel, not an error.
#[tauri::command]
pub async fn pick_video_file(app: tauri::AppHandle) -> Result<Option<String>, String> {
    pick(app, "Video", VIDEO_EXTENSIONS, AUDIO_EXTENSIONS, false).await
}

/// Native open dialog for a transcript the user already has (import screen's
/// "A transcript I have"). `Ok(None)` is a cancel, not an error.
#[tauri::command]
pub async fn pick_transcript_file(app: tauri::AppHandle) -> Result<Option<String>, String> {
    pick(app, "Transcript", TRANSCRIPT_EXTENSIONS, &[], false).await
}

/// Native open dialog for the DubbingQwen checkout, for the setup screen.
#[tauri::command]
pub async fn pick_workspace_dir(app: tauri::AppHandle) -> Result<Option<String>, String> {
    pick(app, "", &[], &[], true).await
}

async fn pick(
    app: tauri::AppHandle,
    label: &'static str,
    extensions: &'static [&'static str],
    also: &'static [&'static str],
    directory: bool,
) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dialog = app.dialog().clone();
        let picked = if directory {
            dialog.file().blocking_pick_folder()
        } else {
            let mut file = dialog.file().add_filter(label, extensions);
            // A second filter only when there is a second family; an empty
            // "Audio" entry in the dropdown would filter to nothing.
            if !also.is_empty() {
                file = file.add_filter("Audio", also);
            }
            file.blocking_pick_file()
        };
        // A file dialog only ever yields an on-disk path on desktop; a URI would mean
        // a mobile content provider, which the pipeline could not open anyway.
        picked
            .map(|path| {
                path.into_path()
                    .map(|path| path.display().to_string())
                    .map_err(|err| format!("dialog returned a path we cannot use: {err}"))
            })
            .transpose()
    })
    .await
    .map_err(|err| format!("failed to join file dialog task: {err}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_video_filter_covers_what_ffmpeg_gets_handed() {
        for expected in ["mp4", "mov", "mkv", "webm"] {
            assert!(VIDEO_EXTENSIONS.contains(&expected), "missing {expected}");
        }
        assert!(
            VIDEO_EXTENSIONS.iter().all(|ext| !ext.starts_with('.')),
            "extensions are bare, the dialog adds the dot"
        );
    }

    #[tokio::test]
    async fn revealing_a_missing_path_is_an_error_not_a_panic() {
        let err = reveal_path(PathBuf::from("/nope/nothing/here"))
            .await
            .unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
    }
}
