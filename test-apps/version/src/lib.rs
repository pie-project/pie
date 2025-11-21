use inferlet::{Args, Result};

#[inferlet::main]
async fn main(_: Args) -> Result<String> {
    Ok(inferlet::get_version())
}
