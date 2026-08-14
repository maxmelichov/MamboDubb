use serde::{Deserialize, Serialize};

/// The one line the studio server prints on stdout before it serves:
/// `{"status":"ready","port":54321,"version":"0.1.0"}` (docs/APP_ARCHITECTURE.md).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ReadySignal {
    pub status: String,
    pub port: u16,
    #[serde(default)]
    pub version: Option<String>,
}

/// What `start_server` / `get_server_url` hand the webview.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ServerInfo {
    pub base_url: String,
    pub port: u16,
    pub version: Option<String>,
    pub workspace: String,
}

/// Parse the handshake line. Anything but a well-formed `"ready"` is an error with
/// the offending line quoted — a Python traceback on stdout would land here, and the
/// setup screen is the only place the user will see it.
pub fn parse_ready_line(line: &str) -> Result<ReadySignal, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err("studio server closed stdout without a ready signal".to_string());
    }
    let signal: ReadySignal = serde_json::from_str(trimmed)
        .map_err(|err| format!("failed to parse ready signal ({err}): {trimmed}"))?;
    if signal.status != "ready" {
        return Err(format!("unexpected studio server status: {}", signal.status));
    }
    if signal.port == 0 {
        return Err("studio server reported port 0".to_string());
    }
    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_line() {
        let signal =
            parse_ready_line("{\"status\":\"ready\",\"port\":54321,\"version\":\"0.1.0\"}\n")
                .unwrap();
        assert_eq!(signal.port, 54321);
        assert_eq!(signal.version.as_deref(), Some("0.1.0"));
    }

    #[test]
    fn version_is_optional() {
        let signal = parse_ready_line("{\"status\":\"ready\",\"port\":8756}").unwrap();
        assert_eq!(signal.port, 8756);
        assert_eq!(signal.version, None);
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let signal =
            parse_ready_line("{\"status\":\"ready\",\"port\":1,\"outputs\":\"/tmp/o\"}").unwrap();
        assert_eq!(signal.port, 1);
    }

    #[test]
    fn an_empty_line_is_a_dead_child() {
        let err = parse_ready_line("   \n").unwrap_err();
        assert!(err.contains("without a ready signal"), "{err}");
    }

    #[test]
    fn a_traceback_on_stdout_is_reported_verbatim() {
        let err = parse_ready_line("ModuleNotFoundError: No module named 'dubbing_app'").unwrap_err();
        assert!(err.contains("ModuleNotFoundError"), "{err}");
    }

    #[test]
    fn a_non_ready_status_is_refused() {
        let err = parse_ready_line("{\"status\":\"error\",\"port\":1}").unwrap_err();
        assert!(err.contains("unexpected studio server status"), "{err}");
    }

    #[test]
    fn port_zero_is_refused() {
        // 0 means "OS, pick one" on the way in; on the way out it means the server
        // announced before it bound, which no client could connect to.
        assert!(parse_ready_line("{\"status\":\"ready\",\"port\":0}").is_err());
    }
}
