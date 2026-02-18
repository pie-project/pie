use inferlet::prelude::*;
use inferlet::{adapter::Adapter, parse_args, runtime, Result};
use inferlet::wstd::time::Duration;

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = parse_args(args);
    let name: String = args.value_from_str("--name").map_err(|e| e.to_string())?;
    let rank: u32 = args.value_from_str("--rank").map_err(|e| e.to_string())?;
    let alpha: f32 = args.value_from_str("--alpha").map_err(|e| e.to_string())?;
    let population_size: u32 = args.value_from_str("--population-size").map_err(|e| e.to_string())?;
    let mu_fraction: f32 = args.value_from_str("--mu-fraction").map_err(|e| e.to_string())?;
    let initial_sigma: f32 = args.value_from_str("--initial-sigma").map_err(|e| e.to_string())?;
    let upload: Option<String> = args.opt_value_from_str("--upload").map_err(|e| e.to_string())?;

    // Load the first available model.
    let model_name = runtime::models().into_iter().next()
        .ok_or_else(|| "No models available".to_string())?;
    let model = Model::load(&model_name)?;

    // Check if the adapter already exists; create + initialize if not.
    let adapter = if let Some(existing) = Adapter::open(&model, &name) {
        println!("🔧 Existing adapter found. Using adapter '{}'.", name);
        existing
    } else {
        println!("🔧 Initializing new adapter '{}'...", name);
        let adapter = Adapter::create(&model, &name)?;
        inferlet::zo::zo::initialize(&adapter, rank, alpha, population_size, mu_fraction, initial_sigma)?;
        adapter
    };

    // If --upload was provided, load weights from the given path.
    if let Some(path) = upload {
        if !path.is_empty() {
            println!("📥 Loading weights into adapter '{}' from '{}'...", name, path);
            adapter.load(&path)?;
        }
    }

    inferlet::wstd::task::sleep(Duration::from_millis(100)).await;
    println!("✅ Adapter '{}' created or imported successfully.", name);

    Ok(format!("Adapter '{}' ready", name))
}
