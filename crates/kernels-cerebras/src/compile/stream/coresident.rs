//! Co-resident readings: a load's readings (e.g. a diffusion poem's text encoder, denoiser and
//! VAE), each compiled on its own and laid side by side as one wafer program, so one program load
//! serves them all. Every kernel's `run` takes the launched reading id (`RD`) and runs only on its
//! own reading's PEs; each reading's `host.json` gains its `origin` and `rd`.

use std::fmt::Write as _;
use std::path::Path;

type R<T> = Result<T, String>;

/// The launch entry every stream and grid kernel opens with (`emit::stream::COMMON`).
const RUN: &str = "fn run() void { timestamp.enable_tsc(); @activate(T_GO); }";
/// It, gated on the reading launched.
const GATED: &str = "fn run(rd: u16) void { if (rd == RD) { timestamp.enable_tsc(); @activate(T_GO); } else { sys_mod.unblock_cmd_stream(); } }";

/// A kernel's text with its `run` gated on the reading launched (`RD` a parameter); `None` for a
/// file with no launch entry (a module).
#[must_use]
pub fn gated(text: &str) -> Option<String> {
    text.contains(RUN).then(|| {
        text.replacen(RUN, GATED, 1).replacen(
            "param memcpy_params;\n",
            "param memcpy_params;\nparam RD: u16;\n",
            1,
        )
    })
}

/// A program's files (name, text) as the combined program links them ([`gated`]).
#[must_use]
pub fn gated_files(files: &[(String, String)]) -> Vec<(String, String)> {
    files
        .iter()
        .map(|(n, t)| (n.clone(), gated(t).unwrap_or_else(|| t.clone())))
        .collect()
}

/// A reading's place in the combined program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub name: String,
    pub x: u32,
    /// Its first row (0 where the readings fit side by side).
    pub y: u32,
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

/// Combines the compiled readings in `root/<name>` into one program at `root`: reading `i` at the
/// columns after reading `i - 1`'s, its kernels gated on `RD = i`, its `host.json` moved there.
/// Returns each reading's place.
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
            y: 0,
            w,
            h,
        });
        x += w;
    }
    // side by side where they fit the system's fabric, else packed in 2D ([`crate::compile::place::pack`])
    let (w, h) = place(&mut placed)?;
    let mut names_out: std::collections::BTreeMap<String, String> = Default::default();
    let mut body = String::new();
    let mut covered = std::collections::BTreeSet::new();
    for (rd, p) in placed.iter().enumerate() {
        let dir = root.join(&p.name);
        for e in std::fs::read_dir(&dir).map_err(io(&dir))? {
            let path = e.map_err(io(&dir))?.path();
            if path.extension().is_none_or(|x| x != "csl") || path.ends_with("layout.csl") {
                continue;
            }
            let t = std::fs::read_to_string(&path).map_err(io(&path))?;
            let Some(t) = gated(&t) else {
                continue;
            };
            std::fs::write(&path, t).map_err(io(&path))?;
        }
        let lay = std::fs::read_to_string(dir.join("layout.csl")).map_err(io(&dir))?;
        for line in lay.lines() {
            let l = line.trim_start();
            // a loop over the reading's columns (a tiled program's): its x range moved, its
            // tiles' files the reading's, gated on its id
            if let Some(rest) = l.strip_prefix("for (@range(u16, ") {
                let (x0, rest) = num_after(rest, "").ok_or("a loop's x")?;
                let (x1, rest) = num_after(rest, ", ").ok_or("a loop's x")?;
                let (sx, rest) = num_after(rest, ", ").ok_or("a loop's step")?;
                let mut rest = rest.to_string();
                if let Some(at) = rest.find("@set_tile_code(x, y, \"") {
                    let inner = &rest[at..];
                    let (y0, r2) = num_after(&rest, "@range(u16, ").ok_or("a loop's y")?;
                    let (y1, r2) = num_after(r2, ", ").ok_or("a loop's y")?;
                    let (sy, _) = num_after(r2, ", ").ok_or("a loop's step")?;
                    let mut x = x0;
                    while x < x1 {
                        let mut y = y0;
                        while y < y1 {
                            covered.insert((x + p.x, y + p.y));
                            y += sy;
                        }
                        x += sx;
                    }
                    let fixed = inner
                        .replacen(
                            "@set_tile_code(x, y, \"",
                            &format!("@set_tile_code(x, y, \"{}/", p.name),
                            1,
                        )
                        .replacen(
                            ".memcpy_params = mp",
                            &format!(".memcpy_params = mp, .RD = {rd}"),
                            1,
                        );
                    rest = format!("{}{fixed}", &rest[..at]);
                }
                let rest = shift_y(&rest, p.y)?;
                let _ = writeln!(
                    body,
                    "  for (@range(u16, {}, {}, {sx}{rest}",
                    x0 + p.x,
                    x1 + p.x
                );
                continue;
            }
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
                covered.insert((px + p.x, py + p.y));
                let _ = writeln!(
                    body,
                    "  @set_tile_code({}, {}, \"{}/{file}\"{params}",
                    px + p.x,
                    py + p.y,
                    p.name
                );
            } else if let Some(rest) = l.strip_prefix("@set_color_config(") {
                let (px, rest) = num_after(rest, "").ok_or("a route's x")?;
                let (py, rest) = num_after(rest, ", ").ok_or("a route's y")?;
                let _ = writeln!(body, "  @set_color_config({}, {}{rest}", px + p.x, py + p.y);
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
        let hp = dir.join("host.json");
        let mut host: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&hp).map_err(io(&hp))?)
                .map_err(|e| format!("{}: {e}", hp.display()))?;
        for list in ["h2d", "d2h", "argmax"] {
            if let Some(items) = host.get_mut(list).and_then(serde_json::Value::as_array_mut) {
                for it in items {
                    let x0 = it["x"].as_u64().unwrap_or(0);
                    it["x"] = serde_json::json!(x0 + u64::from(p.x));
                    if p.y > 0 {
                        let y0 = it["y"].as_u64().unwrap_or(0);
                        it["y"] = serde_json::json!(y0 + u64::from(p.y));
                    }
                }
            }
        }
        // a decode reading's state root: its regions' PEs the host addresses, moved alike
        if let Some(st) = host.get_mut("state").filter(|s| !s.is_null()) {
            let mut root: crate::compile::emit::state::StateRoot =
                serde_json::from_value(st.clone())
                    .map_err(|e| format!("{}: state: {e}", p.name))?;
            for r in &mut root.regions {
                for pe in r.pos.iter_mut().chain(&mut r.rows).chain(&mut r.tables) {
                    pe.x += p.x;
                    pe.y += p.y;
                }
            }
            *st = serde_json::to_value(root).map_err(|e| e.to_string())?;
        }
        host["origin"] = serde_json::json!([p.x, p.y]);
        host["rd"] = serde_json::json!(rd);
        std::fs::write(
            &hp,
            serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
        )
        .map_err(io(&hp))?;
    }
    // the op modules beside every kernel that imports them (cslc reads a file's imports from
    // its own directory)
    ops(root, names)?;
    // the PEs no reading takes: idle
    let filler = format!(
        "// An idle PE of a co-resident program (kernels-cerebras stream::coresident).\nparam memcpy_params;\nparam RD: u16;\n{}task go() void {{ sys_mod.unblock_cmd_stream(); }}\ncomptime {{\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n}}\n",
        crate::compile::emit::stream::COMMON.replacen(RUN, GATED, 1)
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

/// Whether readings of these sizes (w, h) are placed together on a system's fabric as
/// [`combine`] places them ([`crate::compile::place::pack_within`]); where not, by how much they miss.
pub fn fits(sizes: &[(u32, u32)]) -> Result<(), crate::compile::place::pack::Miss> {
    use crate::compile::place::{BlockKind, NetKind, Problem, pack};
    let b = crate::compile::floorplan::stream::Bounds::system();
    let mut x = 0;
    let rects = sizes
        .iter()
        .enumerate()
        .map(|(i, &(w, h))| {
            let r = crate::compile::floor::Rect::new(x, 0, w, h);
            x += w;
            (BlockKind::Reading(i), r)
        })
        .collect();
    let h0 = sizes.iter().map(|s| s.1).max().unwrap_or(1);
    // (a problem that will not form misses by the whole fabric)
    let whole = pack::Miss { dx: b.w, dy: b.h };
    let pr = Problem::of(
        x,
        h0,
        rects,
        &crate::compile::wiring::Wiring::default(),
        &|_| NetKind::Other,
    )
    .map_err(|_| whole)?;
    pack::pack_within(&pr, b.w, b.h).map(|_| ())
}

/// The readings' places within the system's fabric: as they are (side by side) where they fit,
/// else packed in 2D. Returns the program's size.
fn place(placed: &mut [Placed]) -> R<(u32, u32)> {
    let b = crate::compile::floorplan::stream::Bounds::system();
    place_within(placed, b.w, b.h)
}

/// [`place`] within a `w` x `h` fabric.
fn place_within(placed: &mut [Placed], w: u32, h: u32) -> R<(u32, u32)> {
    use crate::compile::place::{BlockKind, NetKind, Problem, pack};
    let rects = placed
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                BlockKind::Reading(i),
                crate::compile::floor::Rect::new(p.x, p.y, p.w, p.h),
            )
        })
        .collect();
    let (w0, h0) = (
        placed.iter().map(|p| p.x + p.w).max().unwrap_or(0),
        placed.iter().map(|p| p.y + p.h).max().unwrap_or(1),
    );
    let pr = Problem::of(
        w0,
        h0,
        rects,
        &crate::compile::wiring::Wiring::default(),
        &|_| NetKind::Other,
    )?;
    let s = pack::pack(&pr, w, h)?;
    for (p, at) in placed.iter_mut().zip(&s.at) {
        p.x = at.x;
        p.y = at.y;
    }
    Ok((s.w.max(1), s.h.max(1)))
}

/// A loop's rest past its x range, its y range (`@range(u16, y0, y1, ..`) moved by `dy`.
fn shift_y(rest: &str, dy: u32) -> R<String> {
    if dy == 0 {
        return Ok(rest.to_string());
    }
    let pre = "@range(u16, ";
    let at = rest.find(pre).ok_or("a loop's y")? + pre.len();
    let (y0, r) = num_after(&rest[at..], "").ok_or("a loop's y")?;
    let (y1, r) = num_after(r, ", ").ok_or("a loop's y")?;
    Ok(format!("{}{}, {}{r}", &rest[..at], y0 + dy, y1 + dy))
}

/// The op modules (`ops/`) in `root` and in each reading's directory `root/<name>`.
pub fn ops(root: &Path, names: &[String]) -> R<()> {
    for d in std::iter::once(root.to_path_buf()).chain(names.iter().map(|n| root.join(n))) {
        let ops = d.join("ops");
        std::fs::create_dir_all(&ops).map_err(io(&ops))?;
        for (n, src) in crate::sources::OPS
            .iter()
            .chain(crate::sources::stream_ops())
        {
            std::fs::write(ops.join(n), src).map_err(io(&ops))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decode reading (its state root's PEs the host addresses) combined after another
    /// reading: every region's `pos`, `rows` and `tables` PEs moved by its origin.
    #[test]
    fn a_decode_readings_state_moves_with_it() {
        let dir = std::env::temp_dir().join(format!("coresident-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        struct Gone(std::path::PathBuf);
        impl Drop for Gone {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let tmp = Gone(dir.clone());
        let mini = poem_compiler::catalog::repository()
            .parse("flux2-mini-bf16-kv-bf16")
            .expect("flux2-mini parses")
            .trace(poem_ir::Platform::Cerebras);
        let o = crate::compile::stream::Options {
            reading: "text".into(),
            rows: 16,
            round16: true,
            upto: None,
            taps: Vec::new(),
            anneal: Default::default(),
        };
        let (w0, _) =
            crate::compile::stream::compile_reading(&mini, &o, &dir.join("text")).unwrap();
        let q = poem_compiler::catalog::repository()
            .parse("qwen3-0.6b-bf16-kv-bf16-l1")
            .expect("qwen3 parses")
            .trace(poem_ir::Platform::Cerebras);
        crate::compile::stream::compile_decode(
            &q,
            64,
            1,
            true,
            None,
            &Default::default(),
            &dir.join("decode"),
        )
        .unwrap();
        let read = |p: &std::path::Path| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
        };
        let before = read(&dir.join("decode/host.json"));
        let names = vec!["text".to_string(), "decode".to_string()];
        let placed = combine(&dir, &names).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((placed[1].x, placed[1].y), (w0, 0));
        let after = read(&dir.join("decode/host.json"));
        let pes = |v: &serde_json::Value| -> Vec<(u64, u64)> {
            (v["state"]["regions"].as_array().into_iter().flatten())
                .flat_map(|r| ["pos", "rows", "tables"].map(|k| r[k].clone()))
                .flat_map(|l| l.as_array().cloned().unwrap_or_default())
                .map(|pe| (pe["x"].as_u64().unwrap(), pe["y"].as_u64().unwrap()))
                .collect()
        };
        let (b, a) = (pes(&before), pes(&after));
        assert!(!b.is_empty(), "the decode reading has state PEs");
        assert_eq!(
            a,
            b.iter()
                .map(|&(x, y)| (x + u64::from(w0), y))
                .collect::<Vec<_>>()
        );
        drop(tmp);
    }
}
