use poem::Platform;

fn main() {
    let mut args = std::env::args().skip(1);
    let deployment = args
        .next()
        .expect("usage: trace <deployment> [cuda|metal|wgpu|vulkan|xla]");
    let platform = match args.next().as_deref() {
        None | Some("cuda") => Platform::Cuda,
        Some("metal") => Platform::Metal,
        Some("wgpu") => Platform::Wgpu,
        Some("vulkan") => Platform::Vulkan,
        Some("xla") => Platform::Xla,
        Some(other) => panic!("unknown platform `{other}`"),
    };
    let row = models::deployment(&deployment).unwrap_or_else(|| {
        let names: Vec<&str> = models::deployments().map(|row| row.name.as_str()).collect();
        panic!("`{deployment}` is not a catalog row; rows: {names:#?}")
    });
    let plan = row.trace(platform);
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
