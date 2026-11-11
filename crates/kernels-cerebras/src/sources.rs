//! The wafer program's CSL kernels, compiled into the crate: a build needs no source tree, and
//! their digest keys every compiled program ([`super::cache::ProgramKey`]).

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

/// The source of the kernel file `name`.
pub fn get(name: &str) -> Option<&'static str> {
    KERNELS.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

#[cfg(test)]
mod tests {
    /// The table is the directory: a kernel added there and not here would not build.
    #[test]
    fn every_kernel_is_embedded() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("programs/wafer");
        let mut files: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();
        let names: Vec<&str> = super::KERNELS.iter().map(|(n, _)| *n).collect();
        assert_eq!(files, names);
    }
}
