//! The wafer program's CSL kernels and the op modules they import, compiled into the crate: a
//! build needs no source tree, and their digest keys every compiled program
//! ([`super::build::ProgramKey`]).

/// Every kernel source of `programs/wafer`, by file name.
pub const KERNELS: &[(&str, &str)] = &[
    ("ctl_cx.csl", include_str!("../programs/wafer/ctl_cx.csl")),
    (
        "ctl_cx_gemma.csl",
        include_str!("../programs/wafer/ctl_cx_gemma.csl"),
    ),
    (
        "ctl_cx_gemma12.csl",
        include_str!("../programs/wafer/ctl_cx_gemma12.csl"),
    ),
    (
        "ctl_cx_gptoss.csl",
        include_str!("../programs/wafer/ctl_cx_gptoss.csl"),
    ),
    (
        "ctl_cy_dense.csl",
        include_str!("../programs/wafer/ctl_cy_dense.csl"),
    ),
    (
        "ctl_cy_gemma.csl",
        include_str!("../programs/wafer/ctl_cy_gemma.csl"),
    ),
    (
        "ctl_cy_gemma12.csl",
        include_str!("../programs/wafer/ctl_cy_gemma12.csl"),
    ),
    (
        "ctl_cy_gptoss_att.csl",
        include_str!("../programs/wafer/ctl_cy_gptoss_att.csl"),
    ),
    (
        "ctl_cy_gptoss_moe.csl",
        include_str!("../programs/wafer/ctl_cy_gptoss_moe.csl"),
    ),
    (
        "head_ctl.csl",
        include_str!("../programs/wafer/head_ctl.csl"),
    ),
    (
        "head_ctl_gemma.csl",
        include_str!("../programs/wafer/head_ctl_gemma.csl"),
    ),
    (
        "head_gemma12.csl",
        include_str!("../programs/wafer/head_gemma12.csl"),
    ),
    ("head_pe.csl", include_str!("../programs/wafer/head_pe.csl")),
    (
        "head_pe_gptoss.csl",
        include_str!("../programs/wafer/head_pe_gptoss.csl"),
    ),
    (
        "ple_gemma.csl",
        include_str!("../programs/wafer/ple_gemma.csl"),
    ),
    ("relay.csl", include_str!("../programs/wafer/relay.csl")),
    (
        "weight_dense.csl",
        include_str!("../programs/wafer/weight_dense.csl"),
    ),
    (
        "weight_gemma.csl",
        include_str!("../programs/wafer/weight_gemma.csl"),
    ),
    (
        "weight_gemma12.csl",
        include_str!("../programs/wafer/weight_gemma12.csl"),
    ),
    (
        "weight_gptoss_att.csl",
        include_str!("../programs/wafer/weight_gptoss_att.csl"),
    ),
    (
        "weight_gptoss_moe.csl",
        include_str!("../programs/wafer/weight_gptoss_moe.csl"),
    ),
];

/// Every op module of `programs/wafer/ops`, by file name: the ops a role kernel imports with
/// comptime params (`@import_module("ops/<name>", ..)`). A build writes them all beside the kinds.
pub const OPS: &[(&str, &str)] = &[
    (
        "argmax.csl",
        include_str!("../programs/wafer/ops/argmax.csl"),
    ),
    (
        "fabric.csl",
        include_str!("../programs/wafer/ops/fabric.csl"),
    ),
    ("math.csl", include_str!("../programs/wafer/ops/math.csl")),
    ("mv.csl", include_str!("../programs/wafer/ops/mv.csl")),
    ("red.csl", include_str!("../programs/wafer/ops/red.csl")),
    ("rope.csl", include_str!("../programs/wafer/ops/rope.csl")),
    ("seg.csl", include_str!("../programs/wafer/ops/seg.csl")),
];

/// The source of the kernel file `name`.
pub fn get(name: &str) -> Option<&'static str> {
    KERNELS.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

#[cfg(test)]
mod tests {
    /// The csl files of `programs/wafer/<sub>`, sorted.
    fn files(sub: &str) -> Vec<String> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(sub);
        let mut files: Vec<String> = std::fs::read_dir(dir)
            .map(|d| {
                d.map(|e| e.unwrap().file_name().into_string().unwrap())
                    .filter(|n| n.ends_with(".csl"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files
    }

    /// The tables are the directories: a kernel or op added there and not here would not build.
    #[test]
    fn every_kernel_and_op_is_embedded() {
        let names: Vec<&str> = super::KERNELS.iter().map(|(n, _)| *n).collect();
        assert_eq!(files("programs/wafer"), names);
        let ops: Vec<&str> = super::OPS.iter().map(|(n, _)| *n).collect();
        assert_eq!(files("programs/wafer/ops"), ops);
    }
}
