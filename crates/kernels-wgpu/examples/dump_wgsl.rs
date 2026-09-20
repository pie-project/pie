use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("an output directory"));
    std::fs::create_dir_all(&out).expect("create the output directory");
    let mut index = Vec::new();
    for (file, variant) in kernels_wgpu::sources::declared() {
        let expanded = match kernels_wgpu::sources::at(&variant.entrypoint, variant.tier) {
            Ok(expanded) => expanded,
            Err(why) => {
                eprintln!("{}: {why}", variant.entrypoint);
                continue;
            }
        };
        let name = variant.tier.variant(&variant.entrypoint);
        let wgsl = promote_storage(&kernels_wgpu::with_enables(&expanded.wgsl, variant.tier));
        std::fs::write(out.join(format!("{name}.wgsl")), wgsl).expect("write a shader");
        index.push(serde_json::json!({
            "name": name,
            "file": file,
            "tier": variant.tier.tag(),
            "workgroup": expanded.workgroup,
        }));
    }
    std::fs::write(
        out.join("index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .expect("write the index");
    println!("{} shaders written to {}", index.len(), out.display());
}

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
