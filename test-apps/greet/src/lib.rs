use inferlet::{Args, Result};

// Generate WIT bindings for importing greet-lib
wit_bindgen::generate!({
    path: "wit",
    world: "greet",
    generate_all,
});

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let name: String = args.value_from_str(["-n", "--name"])?;
    Ok(greet::lib::greet::greet(&name))
}
