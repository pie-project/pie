//! Compile every `// pie:instantiate NAME KEY=VALUE...` line under
//! `kernels/` with slangc, once a target the features ask for: SPIR-V
//! (`spirv`, engine-vulkan's) and WGSL (`wgsl`, engine-wgpu's, with
//! `PIE_WGSL` defined). An instantiation naming `PIE_TARGET=spirv` (or
//! `wgsl`) is built for that target alone. `PIE_TIER=<tier>` builds it at an
//! architecture tier (`subgroup`, `fp16`, `coopmat`; `baseline` without
//! one), defining `PIE_SUBGROUP`, `PIE_FP16` or `PIE_COOPMAT`: one name may
//! be instantiated once a tier, and a tier never shows in a name. The tables
//! land in OUT_DIR as `census.rs`, `spirv.rs` and `wgsl.rs`, keyed by
//! `name` at the baseline and `name.tier` above it; a target not asked for
//! is an empty table.
//!
//! A unit binding the state pool's segments (`PIE_POOL_SEGMENTS`, or
//! `pie_seg0` declared itself) is built once more for WGSL with
//! `PIE_POOL_SEG_COUNT=1`, keyed `<key>.s1`: the form an adapter binding a
//! stage too few storage buffers for four segments beside a kernel's own
//! takes (`kernels/prim/pool.slang`). It is in the WGSL table only, never
//! in the census; SPIR-V is built as before. `wgsl.rs` also states
//! `POOL_OWN_MAX`: the most storage buffers any four-segment pool form
//! declares for itself beside its four segments.
//!
//! The tree's rules, checked here: `prim/` holds the headers and no
//! instantiation; every other unit instantiates at least one entry, and
//! every entry is named for its path (`linear/moe/route.slang` instantiates
//! `linear_moe_route` or `linear_moe_route_<variant>`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The architecture tiers above the baseline, and the define each sets.
const TIERS: [(&str, &str); 3] = [
    ("subgroup", "PIE_SUBGROUP"),
    ("fp16", "PIE_FP16"),
    ("coopmat", "PIE_COOPMAT"),
];

struct Variant {
    name: String,
    file: PathBuf,
    defines: Vec<(String, String)>,
    /// The one target this instantiation is built for, when it names one.
    only: Option<String>,
    /// The tier it is built at; `None` is the baseline.
    tier: Option<String>,
    /// The one-segment form of a pool unit (WGSL only, keyed `<key>.s1`).
    one_segment: bool,
    /// Whether the unit binds the pool's segments.
    pool: bool,
}

impl Variant {
    /// The table key: the name, and the tier above the baseline.
    fn key(&self) -> String {
        let key = match &self.tier {
            Some(t) => format!("{}.{t}", self.name),
            None => self.name.clone(),
        };
        if self.one_segment {
            format!("{key}.s1")
        } else {
            key
        }
    }

    fn builds_for(&self, target: Target) -> bool {
        self.only.as_deref().is_none_or(|t| t == target.slang())
    }
}

#[derive(Clone, Copy)]
enum Target {
    Spirv,
    Wgsl,
}

impl Target {
    fn slang(self) -> &'static str {
        match self {
            Target::Spirv => "spirv",
            Target::Wgsl => "wgsl",
        }
    }

    fn ext(self) -> &'static str {
        match self {
            Target::Spirv => "spv",
            Target::Wgsl => "wgsl",
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=PIE_SLANGC");

    let root = manifest_dir();
    let kernels = root.join("kernels");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let variants = collect(&kernels);
    assert!(
        !variants.is_empty(),
        "no `// pie:instantiate` line under {}",
        kernels.display()
    );
    emit_census(&out, &variants);

    let slangc =
        std::env::var_os("PIE_SLANGC").map_or_else(|| PathBuf::from("slangc"), PathBuf::from);
    for (target, feature, table, ty) in [
        (Target::Spirv, "CARGO_FEATURE_SPIRV", "spirv.rs", "&[u8]"),
        (Target::Wgsl, "CARGO_FEATURE_WGSL", "wgsl.rs", "&str"),
    ] {
        let rows = if std::env::var_os(feature).is_some() {
            let dir = out.join(target.ext());
            std::fs::create_dir_all(&dir).expect("OUT_DIR is writable");
            compile_all(&slangc, &kernels, &dir, &variants, target);
            variants
                .iter()
                .filter(|(_, v)| v.builds_for(target))
                .map(|(key, v)| {
                    (
                        key.clone(),
                        v.file.to_string_lossy().replace('\\', "/"),
                        dir.join(format!("{key}.{}", target.ext())),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        emit_table(&out, table, ty, target, &rows);
        if let Target::Wgsl = target {
            let own_max = rows
                .iter()
                .filter(|(key, _, _)| variants.get(key).is_some_and(|v| v.pool && !v.one_segment))
                .map(|(_, _, path)| {
                    let text = std::fs::read_to_string(path).expect("slangc wrote it");
                    text.matches("var<storage").count().saturating_sub(4)
                })
                .max()
                .unwrap_or(0);
            let mut table = std::fs::OpenOptions::new()
                .append(true)
                .open(out.join("wgsl.rs"))
                .expect("emit_table wrote it");
            std::io::Write::write_all(
                &mut table,
                format!(
                    "/// The most storage buffers a four-segment pool form declares beside its four segments.\n\
                     pub const POOL_OWN_MAX: u32 = {own_max};\n"
                )
                .as_bytes(),
            )
            .expect("OUT_DIR is writable");
        }
    }
}

fn compile_all(
    slangc: &Path,
    kernels: &Path,
    dir: &Path,
    variants: &BTreeMap<String, Variant>,
    target: Target,
) {
    let jobs: Vec<(&String, &Variant)> = variants
        .iter()
        .filter(|(_, v)| v.builds_for(target))
        .collect();
    let width = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for _ in 0..width.min(jobs.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((name, variant)) = jobs.get(i) else {
                        return;
                    };
                    if let Err(why) = compile(slangc, kernels, dir, name, variant, target) {
                        failures.lock().expect("no panicking worker").push(why);
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().expect("no panicking worker");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

fn compile(
    slangc: &Path,
    kernels: &Path,
    dir: &Path,
    name: &str,
    variant: &Variant,
    target: Target,
) -> Result<(), String> {
    let src = kernels.join(&variant.file);
    let dst = dir.join(format!("{name}.{}", target.ext()));
    let mut cmd = Command::new(slangc);
    cmd.arg("-target")
        .arg(target.slang())
        .arg("-stage")
        .arg("compute")
        .arg("-entry")
        .arg("main")
        .arg("-I")
        .arg(kernels)
        .arg("-O2");
    if let Target::Spirv = target {
        // GLSL intrinsics too: the kernels that came from kernels-vulkan use
        // `subgroupShuffleXor` and `gl_LocalInvocationIndex`.
        cmd.arg("-emit-spirv-directly").arg("-allow-glsl");
    } else {
        cmd.arg("-DPIE_WGSL=1");
    }
    for (key, value) in &variant.defines {
        cmd.arg(format!("-D{key}={value}"));
    }
    if variant.one_segment {
        cmd.arg("-DPIE_POOL_SEG_COUNT=1");
    }
    if let Some(tier) = &variant.tier {
        let define = TIERS
            .iter()
            .find(|(t, _)| t == tier)
            .map(|(_, d)| d)
            .expect("collect checks the tier");
        cmd.arg(format!("-D{define}=1"));
    }
    cmd.arg("-o").arg(&dst).arg(&src);
    let output = cmd.output().map_err(|e| {
        format!(
            "cannot run `{}` for `{name}`: {e}. Put slangc on PATH or set PIE_SLANGC.",
            slangc.display()
        )
    })?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "slangc refused `{name}` ({}, {}): {}\n{}",
        variant.file.display(),
        target.slang(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn emit_census(out: &Path, variants: &BTreeMap<String, Variant>) {
    let mut text = String::from(
        "// Generated by build.rs: every instantiation by key, its name, its tier and its unit.\n\
         pub static CENSUS: &[(&str, &str, &str, &str)] = &[\n",
    );
    for (key, v) in variants.iter().filter(|(_, v)| !v.one_segment) {
        let file = v.file.to_string_lossy().replace('\\', "/");
        let tier = v.tier.as_deref().unwrap_or("baseline");
        text.push_str(&format!(
            "    ({key:?}, {:?}, {tier:?}, {file:?}),\n",
            v.name
        ));
    }
    text.push_str("];\n");
    std::fs::write(out.join("census.rs"), text).expect("OUT_DIR is writable");
}

fn emit_table(
    out: &Path,
    table: &str,
    ty: &str,
    target: Target,
    rows: &[(String, String, PathBuf)],
) {
    let include = match target {
        Target::Spirv => "include_bytes!",
        Target::Wgsl => "include_str!",
    };
    let konst = table.trim_end_matches(".rs").to_uppercase();
    let mut text = format!(
        "// Generated by build.rs: every compiled instantiation by key, its unit and its {}.\n\
         pub static {konst}: &[(&str, &str, {ty})] = &[\n",
        target.slang()
    );
    for (name, file, path) in rows {
        let path = path.to_string_lossy().replace('\\', "/");
        text.push_str(&format!("    ({name:?}, {file:?}, {include}({path:?})),\n"));
    }
    text.push_str("];\n");
    std::fs::write(out.join(table), text).expect("OUT_DIR is writable");
}

fn collect(kernels: &Path) -> BTreeMap<String, Variant> {
    let mut out = BTreeMap::new();
    let mut files = Vec::new();
    walk(kernels, &mut files);
    files.sort();
    // The headers that bind the pool's segments for the units including
    // them (`prim/experts.slang`); `prim/pool.slang` only defines the macro.
    let pool_headers: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(kernels)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(path).ok()?;
            (rel.starts_with("prim/")
                && rel != "prim/pool.slang"
                && text.contains("PIE_POOL_SEGMENTS("))
            .then_some(rel)
        })
        .collect();
    for path in files {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let rel = path
            .strip_prefix(kernels)
            .expect("walk yields paths under kernels/")
            .to_path_buf();
        let unit = rel.with_extension("").to_string_lossy().replace('\\', "/");
        let header = unit.starts_with("prim/");
        let pool = text.contains("PIE_POOL_SEGMENTS(")
            || text.contains("> pie_seg0;")
            || pool_headers
                .iter()
                .any(|header| text.contains(&format!("#include \"{header}\"")));
        let stem = unit.replace('/', "_");
        let mut entries = 0;
        for (lineno, line) in text.lines().enumerate() {
            let Some(rest) = line.trim().strip_prefix("// pie:instantiate ") else {
                continue;
            };
            let at = format!("{}:{}", rel.display(), lineno + 1);
            let mut words = rest.split_whitespace();
            let name = words
                .next()
                .unwrap_or_else(|| panic!("{at}: a `pie:instantiate` names its entrypoint first"))
                .to_string();
            assert!(
                !header,
                "{at}: `prim/` holds headers; `{name}` is instantiated in one"
            );
            assert!(
                name == stem || name.starts_with(&format!("{stem}_")),
                "{at}: `{name}` is not named for its path: `{stem}` or `{stem}_<variant>`"
            );
            entries += 1;
            let mut defines: Vec<(String, String)> = words
                .map(|w| {
                    let (k, v) = w
                        .split_once('=')
                        .unwrap_or_else(|| panic!("{at}: `{w}` is not a `KEY=VALUE` define"));
                    (k.to_string(), v.to_string())
                })
                .collect();
            let only = defines
                .iter()
                .position(|(k, _)| k == "PIE_TARGET")
                .map(|i| defines.remove(i).1);
            if let Some(t) = &only {
                assert!(
                    t == "spirv" || t == "wgsl",
                    "{at}: `PIE_TARGET={t}` names no target (spirv, wgsl)"
                );
            }
            let tier = defines
                .iter()
                .position(|(k, _)| k == "PIE_TIER")
                .map(|i| defines.remove(i).1);
            if let Some(t) = &tier {
                assert!(
                    TIERS.iter().any(|(name, _)| name == t),
                    "{at}: `PIE_TIER={t}` names no tier (subgroup, fp16, coopmat)"
                );
            }
            for (key, _) in &defines {
                assert!(
                    TIERS.iter().all(|(_, d)| d != key),
                    "{at}: `{key}` is a tier's define; say `PIE_TIER=` instead"
                );
            }
            if pool && only.as_deref() != Some("spirv") {
                let form = Variant {
                    name: name.clone(),
                    file: rel.clone(),
                    defines: defines.clone(),
                    only: Some("wgsl".to_string()),
                    tier: tier.clone(),
                    one_segment: true,
                    pool,
                };
                out.insert(form.key(), form);
            }
            let variant = Variant {
                name,
                file: rel.clone(),
                defines,
                only,
                tier,
                one_segment: false,
                pool,
            };
            if let Some(prior) = out.insert(variant.key(), variant) {
                panic!(
                    "`{}` is instantiated twice at one tier: {} and {}",
                    prior.name,
                    prior.file.display(),
                    rel.display()
                );
            }
        }
        assert!(
            header || entries > 0,
            "{}: a unit outside `prim/` instantiates at least one entry; a header \
             belongs in `prim/`",
            rel.display()
        );
    }
    out
}

fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            walk(&path, files);
        } else if path.extension().is_some_and(|e| e == "slang") {
            files.push(path);
        }
    }
}

/// This crate's directory, read when the build script runs: a path baked in
/// when the script was compiled would name another checkout's crate where
/// worktrees share a target directory.
fn manifest_dir() -> std::path::PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .expect("cargo sets CARGO_MANIFEST_DIR for a build script")
        .into()
}
