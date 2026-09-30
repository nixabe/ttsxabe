//! Enforce a RAM budget before any model is opened.
//!
//! `launch` enters a systemd user scope; `verify` checks the kernel's cgroup
//! controls, never trusting the relaunch marker as proof. No RSS polling or
//! address-space limit: neither provides the requested physical-memory bound.

use crate::EngineError;
use std::path::{Component, Path, PathBuf};

const MARKER: &str = "LLMTIE_RAM_SCOPE_ENTERED";

/// Parse an exact positive byte count, optionally with a binary unit.
pub fn parse_bytes(value: &str) -> Result<u64, String> {
    let split = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split);
    let scale = match unit {
        "" | "B" => 1,
        "KiB" => 1u64 << 10,
        "MiB" => 1u64 << 20,
        "GiB" => 1u64 << 30,
        "TiB" => 1u64 << 40,
        _ => return Err("use bytes or an integer followed by KiB, MiB, GiB or TiB".into()),
    };
    number
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(scale))
        .filter(|&n| n > 0)
        .ok_or_else(|| "RAM limit must be a positive byte count that fits in u64".into())
}

fn refused(reason: impl std::fmt::Display) -> EngineError {
    EngineError::MemoryLimit(reason.to_string())
}

fn cgroup_dir(root: &Path, membership: &str) -> Result<PathBuf, EngineError> {
    let group = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| refused("Linux cgroup v2 is required"))?;
    let relative = group
        .strip_prefix('/')
        .ok_or_else(|| refused("invalid cgroup path"))?;
    if Path::new(relative)
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(refused("cgroup path is outside the visible hierarchy"));
    }
    Ok(root.join(relative))
}

fn check_controls(dir: &Path, bytes: u64) -> Result<(), EngineError> {
    let read = |name| std::fs::read_to_string(dir.join(name)).map_err(refused);
    let max = read("memory.max")?;
    let swap = read("memory.swap.max")?;
    if max.trim().parse::<u64>().ok().is_none_or(|cap| cap > bytes) || swap.trim() != "0" {
        return Err(refused(format!(
            "RAM cap is not active (memory.max={}, memory.swap.max={}); refusing to load models",
            max.trim(),
            swap.trim()
        )));
    }
    Ok(())
}

/// Check the actual kernel limit; library callers must enter a capped cgroup first.
pub fn verify(bytes: u64) -> Result<(), EngineError> {
    if bytes == 0 {
        return Err(refused("RAM limit must be positive"));
    }
    let membership = std::fs::read_to_string("/proc/self/cgroup").map_err(refused)?;
    check_controls(
        &cgroup_dir(Path::new("/sys/fs/cgroup"), &membership)?,
        bytes,
    )
}

/// Relaunch the CLI in a capped scope, preserving arguments, environment and IO.
/// Failure never falls back to an unrestricted run.
pub fn launch(limit: Option<u64>) -> Result<(), EngineError> {
    let Some(bytes) = limit else {
        return Ok(());
    };
    if verify(bytes).is_ok() {
        return Ok(());
    }
    if std::env::var_os(MARKER).is_some() {
        return verify(bytes);
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let executable = std::env::current_exe().map_err(refused)?;
        tracing::info!(
            bytes,
            "starting RAM-limited scope; requires systemd user memory delegation"
        );
        let error = std::process::Command::new("systemd-run")
            .args(["--user", "--scope", "--quiet"])
            .arg(format!("--property=MemoryMax={bytes}"))
            .arg("--property=MemorySwapMax=0")
            .arg("--")
            .arg(executable)
            .args(std::env::args_os().skip(1))
            .env(MARKER, "1")
            .exec();
        Err(refused(format!("cannot launch systemd-run: {error}")))
    }
    #[cfg(not(target_os = "linux"))]
    Err(refused("Linux cgroup v2 and systemd --user are required"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_accepts_exact_sizes_and_rejects_invalid_limits() {
        for (s, n) in [("32GiB", 32u64 << 30), ("512MiB", 512 << 20), ("123", 123)] {
            let args = crate::Args::try_parse_from(["llmtie-rs", "--max-system-ram", s]).unwrap();
            assert_eq!(args.max_system_ram, Some(n));
        }
        for s in ["0", "-1", "1.5GiB", "32GB", "18446744073709551615TiB"] {
            assert!(crate::Args::try_parse_from(["llmtie-rs", "--max-system-ram", s]).is_err());
        }
    }

    #[test]
    fn cgroup_paths_cannot_escape_mount() {
        let root = Path::new("/sys/fs/cgroup");
        assert_eq!(
            cgroup_dir(root, "0::/user.slice/run.scope\n").unwrap(),
            root.join("user.slice/run.scope")
        );
        assert_eq!(cgroup_dir(root, "0::/\n").unwrap(), root);
        for s in ["1:memory:/a", "0::/../a", "0::relative"] {
            assert!(cgroup_dir(root, s).is_err());
        }
    }

    #[test]
    fn refuses_unlimited_or_ineffective_controls() {
        let dir = std::env::temp_dir().join(format!("xabe-memory-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        assert!(check_controls(&dir, 1024).is_err());
        for (max, swap, valid) in [
            ("1024", "0", true),
            ("512", "0", true),
            ("max", "0", false),
            ("2048", "0", false),
            ("1024", "max", false),
            ("garbage", "0", false),
        ] {
            std::fs::write(dir.join("memory.max"), max).unwrap();
            std::fs::write(dir.join("memory.swap.max"), swap).unwrap();
            assert_eq!(check_controls(&dir, 1024).is_ok(), valid);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
