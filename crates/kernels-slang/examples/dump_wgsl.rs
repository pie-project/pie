//! Every WGSL module this build carries, written out for a browser to compile
//! (`tools/wgsl-check.html` runs them through Chrome's Tint, which checks
//! what naga does not, such as uniform control flow before a barrier).
//!
//! `cargo run -p kernels-slang --features wgsl --example dump_wgsl -- DIR`

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("an output directory"));
    std::fs::create_dir_all(&out).expect("create the output directory");
    let mut index = String::from("[\n");
    for (i, (key, file, wgsl)) in kernels_slang::WGSL.iter().enumerate() {
        let tier = kernels_slang::CENSUS
            .iter()
            .find(|(k, ..)| k == key)
            .map_or("baseline", |(_, _, tier, _)| *tier);
        std::fs::write(out.join(format!("{key}.wgsl")), promote_storage(wgsl))
            .expect("write a shader");
        let comma = if i + 1 == kernels_slang::WGSL.len() {
            ""
        } else {
            ","
        };
        let _ = writeln!(
            index,
            "  {{ \"name\": \"{key}\", \"file\": \"{file}\", \"tier\": \"{tier}\" }}{comma}"
        );
    }
    index.push_str("]\n");
    std::fs::write(out.join("index.json"), index).expect("write the index");
    println!(
        "{} shaders written to {}",
        kernels_slang::WGSL.len(),
        out.display()
    );
}

/// Every storage binding read-write, as engine-wgpu builds its pipelines.
fn promote_storage(wgsl: &str) -> String {
    let mut out = String::with_capacity(wgsl.len() + 64);
    let mut rest = wgsl;
    while let Some(at) = rest.find("var<storage") {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let close = tail.find('>').map_or(tail.len(), |i| i + 1);
        let (decl, after) = tail.split_at(close);
        let inner: String = decl
            .trim_start_matches("var<storage")
            .trim_end_matches('>')
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        match inner.as_str() {
            "" | ",read" | ",read_write" => out.push_str("var<storage, read_write>"),
            _ => out.push_str(decl),
        }
        rest = after;
    }
    out.push_str(rest);
    out
}
