//! A plan built and linked: everything a plan builds ([`emit`]: the program's sources, the
//! artifact the engine runs, what the tools read), linked by `cslc` with every PE image's stack
//! room checked ([`link`]), under the key of everything that went into it ([`ProgramKey`]).
//!
//! A build directory holds, after [`emit`]:
//! - `floor.json`: units (rects, cx / cy, layers), relays, head block, stores, and the kind
//!   rectangles (`Floor::rects_by_kind`);
//! - `rcfg.npy` (u32, [h][w][words], zero-padded to the longest; idle PEs all zero), `rcfg_n.npy`
//!   (u32, [h][w]: each PE's word count) and `rcfg.txt` (`emit::rcfg::dump`);
//! - `kernels.json`: each role kind's source, owned symbols and params (levers applied);
//! - `layout/`: `layout.csl` and the per-kind sources (`emit::layout::write`);
//! - `host.json`, `bindings.json`, `program.json`: the [`Artifact`] (with the rcfg arrays);
//! - `plan.json`: the plan it built;
//!
//! and after [`link`]: `out/` (cslc's: the PE images, `out.json`) and `out/artifact/`, the
//! program the engine opens.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::artifact::Artifact;
use super::emit::params::{Levers, kernels};
use super::emit::{layout, rcfg};
use super::floor::{HeadBlock, Rect, StoreBlock, Unit};
use super::plan::Plan;
use super::program::Program;
use super::role::Kind;
use super::sources;
use super::toolchain::{Arch, Compile, Cslc, memcpy_fabric};

type R<T> = Result<T, String>;

/// What a build emits, by name (`wafer-compile build --emit`).
pub const EMITS: [&str; 8] = [
    "floor", "rcfg", "kernels", "layout", "host", "plan", "bindings", "program",
];

/// Bumped whenever what a build emits for the same inputs changes, so no cached program of an
/// older compiler is reused.
pub const FORMAT: u32 = 1;

/// The least stack room a PE image may leave: the stack grows down from `0xC000` past what the
/// linker counts.
pub const STACK_ROOM: u32 = 768;

/// The artifact's files, copied beside the linked program.
const ARTIFACT: [&str; 5] = [
    "program.json",
    "host.json",
    "bindings.json",
    "rcfg.npy",
    "rcfg_n.npy",
];

/// `floor.json`: the floor without its per-PE role grid (the kind rectangles stand for it).
#[derive(Serialize)]
struct FloorFile<'a> {
    w: u32,
    h: u32,
    units: &'a [Unit],
    relays: &'a [Rect],
    head: &'a HeadBlock,
    stores: &'a [StoreBlock],
    kinds: Vec<(Kind, Rect)>,
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> R<()> {
    let p = dir.join(name);
    std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))
}

fn json(dir: &Path, name: &str, v: &impl Serialize) -> R<()> {
    let s = serde_json::to_string_pretty(v).map_err(|e| format!("{name}: {e}"))?;
    write(dir, name, format!("{s}\n").as_bytes())
}

/// Writes what `prog` builds into `out` (the named [`EMITS`]), its kernels from `kernels_dir`
/// when given (else the embedded ones).
pub fn emit(
    prog: &Program,
    levers: &Levers,
    kernels_dir: Option<&Path>,
    out: &Path,
    emit: &[&str],
) -> R<()> {
    if let Some(e) = emit.iter().find(|e| !EMITS.contains(e)) {
        return Err(format!("emit {e}: of {}", EMITS.join(",")));
    }
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let want = |w: &str| emit.contains(&w);
    if want("floor") {
        let f = &prog.floor;
        json(
            out,
            "floor.json",
            &FloorFile {
                w: f.w,
                h: f.h,
                units: &f.units,
                relays: &f.relays,
                head: &f.head,
                stores: &f.stores,
                kinds: f.rects_by_kind(),
            },
        )?;
    }
    if ["rcfg", "host", "bindings", "program"]
        .iter()
        .any(|w| want(w))
    {
        let art = Artifact::of(prog)?;
        if want("rcfg") {
            art.write_rcfg(out)?;
            write(out, "rcfg.txt", rcfg::dump(prog)?.as_bytes())?;
        }
        if want("host") {
            art.write_host(out)?;
        }
        if want("bindings") {
            art.write_bindings(out)?;
        }
        if want("program") {
            art.write_header(out)?;
        }
    }
    if want("kernels") {
        json(out, "kernels.json", &kernels(prog, levers)?)?;
    }
    if want("layout") {
        layout::write(prog, levers, kernels_dir, &out.join("layout"))?;
    }
    if want("plan") {
        json(out, "plan.json", &prog.plan)?;
    }
    Ok(())
}

/// Links the build in `dir` ([`emit`]'s, every param a literal): cslc over its `layout/` at the
/// wafer program's dims (IEEE f16, one memcpy channel) into `dir/out`, every PE image's stack
/// room checked, the artifact copied to `dir/out/artifact`. Returns the least stack room.
pub fn link(dir: &Path, cslc: &Cslc, arch: Arch) -> R<u32> {
    let floor: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("floor.json")).map_err(|e| format!("{}: {e}", dir.display()))?,
    )
    .map_err(|e| format!("floor.json: {e}"))?;
    let dim = |k: &str| {
        floor[k]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("floor.json: no {k}"))
    };
    let (w, h) = (dim("w")?, dim("h")?);
    let (fabric_dims, fabric_offsets) = memcpy_fabric(w, h);
    let out = dir.join("out");
    if out.exists() {
        std::fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    cslc.compile(&Compile {
        arch,
        layout: dir.join("layout/layout.csl"),
        fabric_dims,
        fabric_offsets,
        params: Vec::new(),
        memcpy: true,
        channels: 1,
        import_paths: Vec::new(),
        out: out.clone(),
        bf16: false,
    })
    .map_err(|e| e.to_string())?;
    let room = cslc.stack_room(&out).map_err(|e| e.to_string())?;
    if room < STACK_ROOM {
        return Err(format!(
            "{}: a PE image leaves {room} B of stack, under {STACK_ROOM}",
            out.display()
        ));
    }
    let art = out.join("artifact");
    std::fs::create_dir_all(&art).map_err(|e| format!("{}: {e}", art.display()))?;
    for f in ARTIFACT {
        std::fs::copy(dir.join(f), art.join(f)).map_err(|e| format!("{f}: {e}"))?;
    }
    Ok(room)
}

/// The key of a compiled program: a digest of everything its bytes follow from (the plan, the
/// levers, every kernel source, this compiler's [`FORMAT`] and version, the toolchain and the
/// architecture), and those parts as a readable record (`key.json`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramKey {
    pub hex: String,
    pub parts: serde_json::Value,
}

impl ProgramKey {
    pub fn of(plan: &Plan, levers: &Levers, arch: Arch, toolchain: &Path) -> R<ProgramKey> {
        let mut k = blake3::Hasher::new();
        for (name, src) in sources::KERNELS {
            k.update(name.as_bytes())
                .update(&[0])
                .update(src.as_bytes())
                .update(&[0]);
        }
        let parts = serde_json::json!({
            "plan": plan,
            "levers": levers,
            "kernels": k.finalize().to_hex().to_string(),
            "format": FORMAT,
            "compiler": env!("CARGO_PKG_VERSION"),
            "arch": format!("{arch:?}"),
            "toolchain": toolchain.display().to_string(),
        });
        let text = serde_json::to_string(&parts).map_err(|e| e.to_string())?;
        let hex = blake3::hash(text.as_bytes()).to_hex()[..16].to_string();
        Ok(ProgramKey { hex, parts })
    }
}

/// The directory a program of `key` lives in under `root` (`p-<key>`).
pub fn program_dir(root: &Path, key: &ProgramKey) -> PathBuf {
    root.join(format!("p-{}", key.hex))
}
