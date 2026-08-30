//! The window control the webview cannot reach for itself.
//!
//! Everything the studio UI does with size and shape it does in CSS, and that is the
//! right default: the page owns the page. Fullscreen is the one exception, because the
//! thing the user is asking for is bigger than the page.
//!
//! The player asks the *element* to go fullscreen first, which is the correct request
//! on every platform and the one that brings the platform's own escape key and menu-bar
//! behaviour with it. This command is what it falls back to when a webview refuses, and
//! a webview does refuse: WebKit only honours the Fullscreen API when `fullScreenEnabled`
//! is set on the webview before it is built, which is what the `macos-private-api`
//! feature in `Cargo.toml` is now there to do. That feature is a private Apple key going
//! through wry, so it is exactly the kind of thing that stops working one release later;
//! when it does, the player fills the page and asks the window to come with it, and the
//! user still gets a picture that covers the screen.

/// Put the main window into fullscreen, or take it back out.
///
/// `Ok(())` means the window manager accepted the request, not that the transition has
/// finished: on macOS it is an animation, and asking `is_fullscreen()` on the next line
/// would report the state we just left. That is why nothing is returned. The caller's
/// icon is driven by the fill it applied itself, which is the half of this it can
/// actually observe, and an `Err` here only downgrades the sentence it shows.
#[tauri::command]
pub fn set_window_fullscreen(window: tauri::Window, value: bool) -> Result<(), String> {
    window
        .set_fullscreen(value)
        .map_err(|err| format!("failed to set window fullscreen: {err}"))
}
