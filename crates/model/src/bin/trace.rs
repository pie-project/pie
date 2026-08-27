use model_dsl::Platform;

fn main() {
    let mut args = std::env::args().skip(1);
    let sku = args
        .next()
        .expect("usage: trace <sku> [cuda|metal|wgpu|vulkan]");
    let platform = match args.next().as_deref() {
        None | Some("cuda") => Platform::Cuda,
        Some("metal") => Platform::Metal,
        Some("wgpu") => Platform::Wgpu,
        Some("vulkan") => Platform::Vulkan,
        Some(other) => panic!("unknown platform `{other}`"),
    };
    let rows = model::catalog();
    let row = rows
        .iter()
        .find(|(name, ..)| *name == sku)
        .unwrap_or_else(|| {
            let names: Vec<&str> = rows.iter().map(|(n, ..)| *n).collect();
            panic!("`{sku}` is not a catalog row; rows: {names:#?}")
        });
    let plan = row.2(platform);
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
