//! Co-resident readings (GENERIC.md §1, M1i): a load's readings (a diffusion poem's text encoder,
//! denoiser and VAE) as one wafer program, each compiled on its own (`stream::compile_reading`)
//! and laid side by side, so one program load serves them all (the simulator holds one program a
//! process; a system pays a load a switch).
//!
//! [`combine`] takes the readings' directories under one root: each keeps its kernels, its
//! printed regions and its `ops/` (cslc resolves a kernel's imports from its own directory), and
//! the root's `layout.csl` places reading `i` at its column offset, every PE's code the reading's
//! own file with a reading id (`RD`). Every kernel's `run` takes the id launched and starts only
//! where it is its reading's (the others unblock at once), so a reading's launch runs its PEs
//! alone. Each reading's `host.json` gains its `origin` (every array's PEs moved by it) and its
//! `rd`, so its host side is as it was.

use std::fmt::Write as _;
use std::path::Path;

type R<T> = Result<T, String>;

/// The launch entry every stream and grid kernel opens with (`emit::stream::COMMON`).
const RUN: &str = "fn run() void { timestamp.enable_tsc(); @activate(T_GO); }";
/// It, gated on the reading launched.
const GATED: &str = "fn run(rd: u16) void { if (rd == RD) { timestamp.enable_tsc(); @activate(T_GO); } else { sys_mod.unblock_cmd_stream(); } }";

/// A reading's place in the combined program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub name: String,
    pub x: u32,
    pub w: u32,
    pub h: u32,
}

fn io(p: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", p.display())
}

/// `n` in `line` after `pre`, and the rest.
fn num_after<'a>(line: &'a str, pre: &str) -> Option<(u32, &'a str)> {
    let at = line.find(pre)? + pre.len();
    let rest = &line[at..];
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    Some((rest[..end].parse().ok()?, &rest[end..]))
}

/// Combines the readings in `root/<name>` (each a compiled reading: `layout.csl`, `fabric.txt`,
/// `host.json`, its kernels) into one program at `root` (`layout.csl`, `fabric.txt`, the op
/// modules in `ops/` beside every kernel): reading `i` at the columns after reading `i - 1`'s,
/// its kernels gated on `RD = i`, its `host.json` moved there. Returns each reading's place.
#[allow(clippy::too_many_lines)]
pub fn combine(root: &Path, names: &[String]) -> R<Vec<Placed>> {
    let mut placed = Vec::new();
    let mut bf16: Option<String> = None;
    let mut x = 0u32;
    for n in names {
        let dir = root.join(n);
        let fab = std::fs::read_to_string(dir.join("fabric.txt")).map_err(io(&dir))?;
        let f: Vec<&str> = fab.split_whitespace().collect();
        let [w, h, b] = f[..] else {
            return Err(format!("{}: fabric.txt is not `w h bf16`", dir.display()));
        };
        let (w, h): (u32, u32) = (
            w.parse().map_err(|e| format!("{w}: {e}"))?,
            h.parse().map_err(|e| format!("{h}: {e}"))?,
        );
        match &bf16 {
            Some(b0) if b0 != b => {
                return Err(format!(
                    "{n}: a 16-bit type other than the other readings': one build's"
                ));
            }
            _ => bf16 = Some(b.to_string()),
        }
        placed.push(Placed {
            name: n.clone(),
            x,
            w,
            h,
        });
        x += w;
    }
    let (w, h) = (x, placed.iter().map(|p| p.h).max().unwrap_or(1));
    let mut names_out: std::collections::BTreeMap<String, String> = Default::default();
    let mut body = String::new();
    let mut covered = std::collections::BTreeSet::new();
    for (rd, p) in placed.iter().enumerate() {
        let dir = root.join(&p.name);
        // the reading's kernels, gated on its id
        for e in std::fs::read_dir(&dir).map_err(io(&dir))? {
            let path = e.map_err(io(&dir))?.path();
            if path.extension().is_none_or(|x| x != "csl") || path.ends_with("layout.csl") {
                continue;
            }
            let t = std::fs::read_to_string(&path).map_err(io(&path))?;
            if !t.contains(RUN) {
                continue;
            }
            let t = t.replacen(RUN, GATED, 1).replacen(
                "param memcpy_params;\n",
                "param memcpy_params;\nparam RD: u16;\n",
                1,
            );
            std::fs::write(&path, t).map_err(io(&path))?;
        }
        let lay = std::fs::read_to_string(dir.join("layout.csl")).map_err(io(&dir))?;
        for line in lay.lines() {
            let l = line.trim_start();
            if let Some(rest) = l.strip_prefix("@set_tile_code(") {
                let (px, rest) = num_after(rest, "").ok_or("a tile's x")?;
                let (py, rest) = num_after(rest, ", ").ok_or("a tile's y")?;
                let rest = rest
                    .strip_prefix(", \"")
                    .ok_or_else(|| format!("a tile's file: {line}"))?;
                let (file, rest) = rest.split_once('"').ok_or("a tile's file")?;
                let params = rest.replacen(
                    &format!("memcpy.get_params({px})"),
                    &format!("memcpy.get_params({}), .RD = {rd}", px + p.x),
                    1,
                );
                covered.insert((px + p.x, py));
                let _ = writeln!(
                    body,
                    "  @set_tile_code({}, {py}, \"{}/{file}\"{params}",
                    px + p.x,
                    p.name
                );
            } else if let Some(rest) = l.strip_prefix("@set_color_config(") {
                let (px, rest) = num_after(rest, "").ok_or("a route's x")?;
                let _ = writeln!(body, "  @set_color_config({}{rest}", px + p.x);
            } else if let Some(rest) = l.strip_prefix("@export_name(\"") {
                let (name, ty) = rest.split_once('"').ok_or("an export")?;
                if name == "run" {
                    continue;
                }
                match names_out.get(name) {
                    Some(t0) if t0 != ty => {
                        return Err(format!("`{name}` exported as {t0} and {ty}"));
                    }
                    _ => {
                        names_out.insert(name.to_string(), ty.to_string());
                    }
                }
            }
        }
        // its host side, moved to its place
        let hp = dir.join("host.json");
        let mut host: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&hp).map_err(io(&hp))?)
                .map_err(|e| format!("{}: {e}", hp.display()))?;
        for list in ["h2d", "d2h"] {
            if let Some(items) = host[list].as_array_mut() {
                for it in items {
                    let x0 = it["x"].as_u64().unwrap_or(0);
                    it["x"] = serde_json::json!(x0 + u64::from(p.x));
                }
            }
        }
        host["origin"] = serde_json::json!([p.x, 0]);
        host["rd"] = serde_json::json!(rd);
        std::fs::write(
            &hp,
            serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
        )
        .map_err(io(&hp))?;
    }
    // the op modules beside every kernel that imports them (cslc reads a file's imports from
    // its own directory)
    for d in std::iter::once(root.to_path_buf()).chain(placed.iter().map(|p| root.join(&p.name))) {
        let ops = d.join("ops");
        std::fs::create_dir_all(&ops).map_err(io(&ops))?;
        for (n, src) in crate::sources::OPS.iter().chain(crate::sources::STREAM_OPS) {
            std::fs::write(ops.join(n), src).map_err(io(&ops))?;
        }
    }
    // the PEs no reading takes: idle
    let filler = format!(
        "// An idle PE of a co-resident program (kernels-cerebras stream::coresident).\nparam memcpy_params;\nparam RD: u16;\n{}task go() void {{ sys_mod.unblock_cmd_stream(); }}\ncomptime {{\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n}}\n",
        crate::emit::stream::COMMON.replacen(RUN, GATED, 1)
    );
    std::fs::write(root.join("idle.csl"), filler).map_err(io(root))?;
    for y in 0..h {
        for xx in 0..w {
            if !covered.contains(&(xx, y)) {
                let _ = writeln!(
                    body,
                    "  @set_tile_code({xx}, {y}, \"idle.csl\", .{{ .memcpy_params = memcpy.get_params({xx}), .RD = 65535 }});"
                );
            }
        }
    }
    let mut t = format!(
        "// Co-resident readings {:?} (kernels-cerebras stream::coresident): {w} x {h} PEs.\nconst memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {w}, .height = {h} }});\nlayout {{\n  @set_rectangle({w}, {h});\n",
        names
    );
    t.push_str(&body);
    for (n, ty) in &names_out {
        let _ = writeln!(t, "  @export_name(\"{n}\"{ty}");
    }
    t.push_str("  @export_name(\"run\", fn(u16) void);\n}\n");
    std::fs::write(root.join("layout.csl"), t).map_err(io(root))?;
    std::fs::write(
        root.join("fabric.txt"),
        format!("{w} {h} {}\n", bf16.unwrap_or_else(|| "1".into())),
    )
    .map_err(io(root))?;
    Ok(placed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// flux2-mini's text encoder and VAE decode, each compiled on its own, combine into one
    /// program: the VAE's PEs after the text's columns, every kernel gated on its reading's id,
    /// the VAE's arrays moved by its origin.
    #[test]
    fn two_readings_combine_into_one_program() {
        let trace = poem_compiler::catalog::repository()
            .parse("flux2-mini-bf16-kv-bf16")
            .expect("flux2-mini parses")
            .trace(poem_ir::Platform::Cerebras);
        let dir = std::env::temp_dir().join(format!("coresident-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        struct Gone(std::path::PathBuf);
        impl Drop for Gone {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let tmp = Gone(dir);
        let tmp = &tmp;
        let names = vec!["text".to_string(), "vae.decode".to_string()];
        let mut sizes = Vec::new();
        for n in &names {
            let o = crate::stream::Options {
                reading: n.clone(),
                rows: 16,
                round16: true,
                upto: None,
                taps: Vec::new(),
            };
            sizes.push(
                crate::stream::compile_reading(&trace, &o, &tmp.0.as_path().join(n))
                    .unwrap_or_else(|e| panic!("{n}: {e}")),
            );
        }
        let before: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(tmp.0.as_path().join("vae.decode/host.json"))
                .expect("host.json"),
        )
        .expect("json");
        let placed = combine(tmp.0.as_path(), &names).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(placed[1].x, sizes[0].0);
        let lay = std::fs::read_to_string(tmp.0.as_path().join("layout.csl")).expect("layout");
        assert!(lay.contains("\"text/"));
        assert!(lay.contains(&format!("@set_tile_code({}, ", sizes[0].0)));
        assert!(lay.contains(".RD = 1"));
        assert!(lay.contains("@export_name(\"run\", fn(u16) void);"));
        let gx =
            std::fs::read_to_string(tmp.0.as_path().join("vae.decode/gx_0.csl")).expect("a kernel");
        assert!(gx.contains(GATED) && gx.contains("param RD: u16;"));
        let after: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(tmp.0.as_path().join("vae.decode/host.json"))
                .expect("host.json"),
        )
        .expect("json");
        assert_eq!(after["rd"], 1);
        assert_eq!(
            after["h2d"][0]["x"].as_u64().unwrap(),
            before["h2d"][0]["x"].as_u64().unwrap() + u64::from(sizes[0].0)
        );
        let fab = std::fs::read_to_string(tmp.0.as_path().join("fabric.txt")).expect("fabric");
        assert_eq!(
            fab.trim(),
            format!(
                "{} {} 1",
                sizes[0].0 + sizes[1].0,
                sizes[0].1.max(sizes[1].1)
            )
        );
    }
}
