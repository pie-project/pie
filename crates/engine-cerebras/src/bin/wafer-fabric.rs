//! `wafer-fabric`: one stream program's SDK runtime in a process of its own, driven over its
//! stdin / stdout by the engine (`reading::Children`, `[cerebras.debug] stream_processes`): the
//! simulator holds one program a process, so readings each linked on their own run in a child
//! each (a co-resident program is the default: `stream::coresident`).
//!
//! ```text
//! wafer-fabric --sdk-lib <dir> [--cmaddr <ip:port>] [--threads N]
//! ```
//!
//! A request is a JSON line, a write's words after it (`n` little-endian u32); every reply is a
//! JSON line (`{"ok": ..}`), a run's rows after it (each back's `lens[i]` f32):
//! `{"op": "open", "link": ..}`, `{"op": "write", "sym", "x", "y", "w", "h", "len", "u16", "n"}`,
//! `{"op": "run", "dir": <the reading's program directory>}`, `{"op": "launch", "dir": ..}` (a
//! launch reading nothing back: a weight group's before the last).

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;

use engine_cerebras::reading::{Array, Fabric, Programs, Reading};
use engine_cerebras::sdk::Rect;
use engine_cpu::forward::StreamStep;
use serde_json::{Value, json};

type R<T> = Result<T, String>;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn num(v: &Value, k: &str) -> R<i64> {
    v[k].as_i64().ok_or_else(|| format!("`{k}`?"))
}

fn serve(
    fab: &mut Programs,
    readings: &mut HashMap<PathBuf, Reading>,
    req: &Value,
    words: &[u32],
) -> R<(Value, Vec<f32>)> {
    match req["op"].as_str() {
        Some("open") => {
            let link = PathBuf::from(req["link"].as_str().ok_or("`link`?")?);
            fab.switch(&link)?;
            Ok((json!({"ok": true}), Vec::new()))
        }
        Some("write") => {
            let a = Array {
                sym: req["sym"].as_str().ok_or("`sym`?")?.to_string(),
                rect: Rect {
                    x: num(req, "x")? as i32,
                    y: num(req, "y")? as i32,
                    w: num(req, "w")? as i32,
                    h: num(req, "h")? as i32,
                },
                len: num(req, "len")? as usize,
                u16: req["u16"].as_bool().unwrap_or(false),
                file: None,
                entry: None,
                statics: None,
                group: None,
                rowbcast: req["rowbcast"].as_bool().unwrap_or(false),
                kept: false,
            };
            fab.write(&a, words)?;
            Ok((json!({"ok": true}), Vec::new()))
        }
        Some("run") => {
            let dir = PathBuf::from(req["dir"].as_str().ok_or("`dir`?")?);
            if !readings.contains_key(&dir) {
                readings.insert(dir.clone(), Reading::open(&dir)?);
            }
            let step = StreamStep {
                lanes: Vec::new(),
                inputs: Vec::new(),
                tokens: Vec::new(),
            };
            let (got, cycles) = fab.run(&readings[&dir], &step)?;
            let lens: Vec<usize> = got.iter().map(Vec::len).collect();
            Ok((
                json!({"ok": true, "cycles": cycles, "lens": lens}),
                got.into_iter().flatten().collect(),
            ))
        }
        Some("launch") => {
            let dir = PathBuf::from(req["dir"].as_str().ok_or("`dir`?")?);
            if !readings.contains_key(&dir) {
                readings.insert(dir.clone(), Reading::open(&dir)?);
            }
            let cycles = fab.launch(&readings[&dir])?;
            Ok((
                json!({"ok": true, "cycles": cycles, "lens": []}),
                Vec::new(),
            ))
        }
        other => Err(format!("an op {other:?}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sdk_lib = PathBuf::from(arg(&args, "--sdk-lib").unwrap_or_else(|| "/cbcore/lib".into()));
    let platform = match arg(&args, "--cmaddr") {
        Some(cmaddr) => engine_cerebras::sdk::Platform::System { cmaddr },
        None => engine_cerebras::sdk::Platform::Simulator(engine_cerebras::sdk::Simulator::new(
            engine_cerebras::sdk::Target::Wse3,
            arg(&args, "--threads")
                .and_then(|t| t.parse().ok())
                .unwrap_or(8),
        )),
    };
    let mut fab = Programs {
        sdk_lib,
        platform,
        rt: None,
    };
    let mut readings = HashMap::new();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut out = std::io::stdout().lock();
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let reply = (|| -> R<(Value, Vec<f32>)> {
            let req: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            let n = req["n"].as_u64().unwrap_or(0) as usize;
            let mut bytes = vec![0u8; 4 * n];
            input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            let words: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            serve(&mut fab, &mut readings, &req, &words)
        })();
        let (head, data) = reply.unwrap_or_else(|e| (json!({"ok": false, "err": e}), Vec::new()));
        let _ = writeln!(out, "{head}");
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = out.write_all(&bytes);
        let _ = out.flush();
    }
}
