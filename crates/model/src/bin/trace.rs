use model_dsl::Plane;

fn main() {
    let mut args = std::env::args().skip(1);
    let sku = args
        .next()
        .expect("usage: trace <sku> [cuda|metal|wgpu|vulkan]");
    let plane = match args.next().as_deref() {
        None | Some("cuda") => Plane::Cuda,
        Some("metal") => Plane::Metal,
        Some("wgpu") => Plane::Wgpu,
        Some("vulkan") => Plane::Vulkan,
        Some(other) => panic!("unknown plane `{other}`"),
    };
    let rows = model::catalog();
    let row = rows
        .iter()
        .find(|(name, _)| *name == sku)
        .unwrap_or_else(|| {
            let names: Vec<&str> = rows.iter().map(|(n, _)| *n).collect();
            panic!("`{sku}` is not a catalog row; rows: {names:#?}")
        });
    let plan = row.1(plane);
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
