mod files;
mod provision;
mod runner;
mod workspace;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .manage(runner::RunnerState::default())
        .invoke_handler(tauri::generate_handler![
            workspace::get_workspace,
            workspace::set_workspace,
            workspace::check_workspace,
            runner::start_server,
            runner::stop_server,
            runner::get_server_url,
            runner::get_server_log,
            files::reveal_path,
            files::pick_video_file,
            files::pick_transcript_file,
            files::pick_workspace_dir,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| {
        if matches!(
            event,
            tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
        ) {
            runner::stop_managed_server(app);
        }
    });
}
