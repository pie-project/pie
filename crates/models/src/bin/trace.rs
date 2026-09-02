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
    let row = models::sku(&sku).unwrap_or_else(|| {
        let names: Vec<&str> = models::skus().map(|row| row.name.as_str()).collect();
        panic!("`{sku}` is not a catalog row; rows: {names:#?}")
    });
    let plan = (row.trace)(platform);
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
