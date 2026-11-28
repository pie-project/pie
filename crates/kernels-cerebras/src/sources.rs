//! The wafer program's CSL kernels and the op modules they import, compiled into the crate from
//! `kernels/<family>`: a build needs no source tree, and their digest keys every compiled program
//! ([`crate::compile::build::ProgramKey`]). A build writes each by its file name.

/// `(file name, source)` of `kernels/<family>/<file>`.
macro_rules! csl {
    ($family:literal, $file:literal) => {
        (
            $file,
            include_str!(concat!("../kernels/", $family, "/", $file)),
        )
    };
}

/// Every role kernel source (a PE's program), by file name.
pub const KERNELS: &[(&str, &str)] = &[
    csl!("runtime", "cx.csl"),
    csl!("runtime", "cy.csl"),
    csl!("runtime", "head_ctl.csl"),
    csl!("runtime", "head_pe.csl"),
    csl!("runtime", "ple_gemma.csl"),
    csl!("runtime", "relay.csl"),
    csl!("runtime", "weight.csl"),
];

/// Every op module, by file name: the ops a role kernel imports with comptime params
/// (`@import_module("ops/<name>", ..)`). A build writes them all beside the kinds.
pub const OPS: &[(&str, &str)] = &[
    csl!("elemwise", "act.csl"),
    csl!("sampling", "argmax.csl"),
    csl!("attn", "att.csl"),
    csl!("prim", "cpe.csl"),
    csl!("attn", "cy_att.csl"),
    csl!("ssm", "cy_gdn.csl"),
    csl!("attn", "cy_kv.csl"),
    csl!("linear", "cy_moe.csl"),
    csl!("gather", "cy_ple.csl"),
    csl!("attn", "cy_swa.csl"),
    csl!("prim", "fabric.csl"),
    csl!("ssm", "gdn.csl"),
    csl!("attn", "kv.csl"),
    csl!("prim", "math.csl"),
    csl!("linear", "moe.csl"),
    csl!("linear", "mv.csl"),
    csl!("collective", "red.csl"),
    csl!("elemwise", "rope.csl"),
    csl!("runtime", "seats.csl"),
    csl!("collective", "seg.csl"),
    csl!("attn", "swa.csl"),
    csl!("prim", "wpe.csl"),
];

/// The stream regime's shared op modules: the chain reduction, the matvec over a row stream and
/// the rotation tables. No decode kind imports them, so a decode build neither writes nor keys
/// them. The kernels' own are their modules' ([`crate::kernel`]).
pub const STREAM_LIBS: &[(&str, &str)] = &[
    csl!("collective", "schain.csl"),
    csl!("linear", "smv.csl"),
    csl!("elemwise", "srope.csl"),
];

/// Every stream op module: each registered kernel's sources ([`crate::kernel::KERNELS`], in
/// order), then the shared ones ([`STREAM_LIBS`]).
#[must_use]
pub fn stream_ops() -> &'static [(&'static str, &'static str)] {
    static ALL: std::sync::LazyLock<Vec<(&'static str, &'static str)>> =
        std::sync::LazyLock::new(|| {
            (crate::kernel::KERNELS.iter())
                .flat_map(|k| k.sources().iter().copied())
                .chain(STREAM_LIBS.iter().copied())
                .collect()
        });
    &ALL
}

/// The source of the kernel file `name`.
pub fn get(name: &str) -> Option<&'static str> {
    KERNELS.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

/// The file `name` of a kernel tree laid out as `kernels/<family>/<name>` (`wafer-compile build
/// --kernels`), read from disk.
pub fn read(dir: &std::path::Path, name: &str) -> Result<String, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let p = (rd.filter_map(Result::ok))
        .map(|e| e.path().join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| format!("{}: no {name}", dir.display()))?;
    std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))
}

#[cfg(test)]
mod tests {
    /// The tables are the tree: a kernel or op added under `kernels/` and not here would not
    /// build. (The stream regimes' PE images are their emitters'.)
    #[test]
    fn every_kernel_and_op_is_embedded() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let mut files: Vec<String> = std::fs::read_dir(root)
            .unwrap()
            .flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".csl") && n != "gemm_pe.csl" && n != "tiled_pe.csl")
            .collect();
        files.sort();
        let mut names: Vec<&str> = (super::KERNELS.iter())
            .chain(super::OPS)
            .chain(super::stream_ops())
            .map(|(n, _)| *n)
            .collect();
        names.sort_unstable();
        assert_eq!(files, names);
    }
}
