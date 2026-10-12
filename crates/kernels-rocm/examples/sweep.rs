//! Compiles every carried unit for the device's architecture and prints the
//! first errors of each one that fails: the to-do list of the HIP port.
//! `cargo run -p kernels-rocm --features rocm --example sweep [filter]`
fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let verbose = std::env::var_os("SWEEP_VERBOSE").is_some();
    let (mut ok, mut bad) = (0, 0);
    for header in kernels_rocm::source::LIBRARY {
        let name = header.name;
        if !name.ends_with(".cuh") || !name.contains(&filter) || name.starts_with("flashinfer/") {
            continue;
        }
        match kernels_rocm::jit::probe(name) {
            Ok(()) => {
                ok += 1;
                println!("ok    {name}");
            }
            Err(log) => {
                bad += 1;
                let errors: Vec<&str> = log.lines().filter(|l| l.contains("error")).collect();
                println!("FAIL  {name}  ({} errors)", errors.len());
                for line in errors.iter().take(if verbose { 400 } else { 4 }) {
                    println!("        {}", line.trim());
                }
            }
        }
    }
    println!("\n{ok} compile, {bad} fail");
}
