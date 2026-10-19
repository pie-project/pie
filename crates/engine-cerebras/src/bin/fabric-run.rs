//! Runs one compiled program on the fabric simulator in its own process.
//!
//! The SDK's simulator can be created once per process, so anything that
//! runs programs repeatedly (the kernel bench, the engine's tests) spawns this
//! binary per run. Usage:
//!
//! ```text
//! fabric-run <artifacts-dir> <spec.json>
//! ```
//!
//! `spec.json`:
//!
//! ```json
//! { "target": "wse3", "entry": "run", "args": [1, 2],
//!   "inputs":  [{ "name": "b1", "file": "b1.in" }],
//!   "outputs": [{ "name": "b2", "len": 12, "file": "b2.out" }] }
//! ```
//!
//! Buffer files are raw little-endian 32-bit words, PE after PE over the
//! spec's `"rect": [w, h]` (default one PE), each PE's share contiguous.

use std::path::Path;
use std::process::ExitCode;

use engine_cerebras::sdk::{DataType, Order, Platform, Rect, Runtime, Sdk, Simulator, Target};

fn words(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// One spec's inputs in, launch, outputs out, on a loaded runtime. A
/// `persistent` input (a weight) is uploaded once per loaded program:
/// `uploaded` remembers it across specs.
fn run_spec(
    rt: &mut Runtime,
    spec_path: &Path,
    uploaded: &mut std::collections::HashSet<String>,
) -> Result<(), String> {
    let spec_dir = std::fs::canonicalize(spec_path.parent().unwrap_or(Path::new(".")))
        .map_err(|e| e.to_string())?;
    let spec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(spec_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let entry = spec["entry"].as_str().unwrap_or("run");
    let launch_args: Vec<u32> = spec["args"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_default();
    let (w, h) = match spec["rect"].as_array() {
        Some(r) if r.len() == 2 => (
            r[0].as_u64().unwrap_or(1) as i32,
            r[1].as_u64().unwrap_or(1) as i32,
        ),
        _ => (1, 1),
    };
    let pe = Rect { x: 0, y: 0, w, h };
    let pes = (w * h).max(1) as usize;
    let persistent: Vec<&str> = spec["persistent"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    // An `init` entry runs before any copy: it points the kept slots'
    // symbols into the arena (a comptime pointer cast is not supported).
    if let Some(init) = spec["init"].as_str() {
        rt.launch(init, &[]).map_err(|e| e.to_string())?;
    }
    for input in spec["inputs"].as_array().into_iter().flatten() {
        let name = input["name"].as_str().ok_or("input without name")?;
        if persistent.contains(&name) {
            if uploaded.contains(name) {
                continue;
            }
            uploaded.insert(name.to_string());
        }
        let data = words(&spec_dir.join(input["file"].as_str().ok_or("input without file")?))?;
        rt.memcpy_h2d(
            name,
            &data,
            pe,
            data.len() / pes,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
    }
    rt.launch(entry, &launch_args).map_err(|e| e.to_string())?;
    for output in spec["outputs"].as_array().into_iter().flatten() {
        let name = output["name"].as_str().ok_or("output without name")?;
        let len = output["len"].as_u64().ok_or("output without len")? as usize;
        let mut data = vec![0u32; len];
        rt.memcpy_d2h(
            name,
            &mut data,
            pe,
            len / pes,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(
            spec_dir.join(output["file"].as_str().ok_or("output without file")?),
            bytes,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The parent process (from `/proc`; 1 once it has died and init has
/// adopted the server; 0 where `/proc` is not available).
fn parent_pid() -> u32 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            // `pid (comm) state ppid ...`: the command may hold spaces.
            let rest = stat.rsplit_once(')')?.1;
            rest.split_whitespace().nth(1)?.parse().ok()
        })
        .unwrap_or(0)
}

fn target_of(name: &str) -> Target {
    match name {
        "wse2" => Target::Wse2,
        _ => Target::Wse3,
    }
}

/// Loads the program once and runs it in place of its cwd: every line of
/// stdin names a spec, each answered with `ok` or `err: ...` on stdout;
/// EOF stops the runtime.
fn serve(artifacts: &str, target: &str) -> Result<(), String> {
    let artifacts = std::fs::canonicalize(artifacts).map_err(|e| format!("{artifacts}: {e}"))?;
    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let mut rt = Runtime::new(
        sdk,
        &artifacts,
        Platform::Simulator(Simulator::new(target_of(target))),
        "WARNING",
    )
    .map_err(|e| e.to_string())?;
    rt.load();
    rt.run();
    // A server outlives its parent only by accident (one was found hours
    // after its shell exited, spinning on four cores): a watcher polls the
    // parent and ends the process once it is gone, whatever holds stdin.
    let parent = parent_pid();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            if parent_pid() != parent {
                std::process::exit(0);
            }
        }
    });
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut uploaded = std::collections::HashSet::new();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let spec = line.trim();
        if spec.is_empty() {
            continue;
        }
        let answer = match run_spec(&mut rt, Path::new(spec), &mut uploaded) {
            Ok(()) => "ok".to_string(),
            Err(e) => format!("err: {}", e.replace('\n', " ")),
        };
        let _ = writeln!(out, "{answer}");
        let _ = out.flush();
    }
    rt.stop();
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or(
        "usage: fabric-run <artifacts-dir> <spec.json> | --serve <artifacts-dir> <target>",
    )?;
    if first == "--serve" {
        let artifacts = args
            .next()
            .ok_or("usage: fabric-run --serve <artifacts-dir> <target>")?;
        let target = args.next().unwrap_or_else(|| "wse3".to_string());
        // The SDK notes the cwd when its libraries load and writes its logs
        // there: the server's own directory beside the artifacts.
        let serve_dir = Path::new(&artifacts).join("serve");
        std::fs::create_dir_all(&serve_dir).map_err(|e| e.to_string())?;
        std::env::set_current_dir(&serve_dir)
            .map_err(|e| format!("cd {}: {e}", serve_dir.display()))?;
        return serve(&artifacts, &target);
    }
    let artifacts = first;
    let spec_path = args
        .next()
        .ok_or("usage: fabric-run <artifacts-dir> <spec.json>")?;
    let spec_dir = Path::new(&spec_path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let spec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&spec_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let target = target_of(spec["target"].as_str().unwrap_or("wse3"));

    // The SDK notes the cwd when its libraries load and the simulator writes
    // its logs and scratch directories there: move into the spec's directory
    // before opening it.
    let artifacts = std::fs::canonicalize(&artifacts).map_err(|e| format!("{artifacts}: {e}"))?;
    let spec_dir = std::fs::canonicalize(&spec_dir).map_err(|e| e.to_string())?;
    std::env::set_current_dir(&spec_dir).map_err(|e| format!("cd {}: {e}", spec_dir.display()))?;

    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let mut rt = Runtime::new(
        sdk,
        &artifacts,
        Platform::Simulator(Simulator::new(target)),
        "WARNING",
    )
    .map_err(|e| e.to_string())?;
    rt.load();
    rt.run();
    run_spec(
        &mut rt,
        Path::new(&spec_path),
        &mut std::collections::HashSet::new(),
    )?;
    rt.stop();
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fabric-run: {e}");
            ExitCode::FAILURE
        }
    }
}
