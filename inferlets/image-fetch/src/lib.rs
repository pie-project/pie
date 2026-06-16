//! Demonstrates fetching images over HTTP.
//!
//! This example fetches an image from a URL with `inferlet::http::fetch` (which
//! follows redirects) and decodes it with the `image` crate.

use image::{DynamicImage, load_from_memory};
use inferlet::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_url")]
    url: String,
}

fn default_url() -> String { "https://www.ilankelman.org/stopsigns/australia.jpg".to_string() }

/// Asynchronously fetches an image from the given URL and decodes it.
pub async fn fetch_image(url: &str) -> Result<DynamicImage> {
    let bytes = inferlet::http::fetch(url).await?;
    load_from_memory(&bytes).map_err(|e| e.to_string())
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let url = input.url;

    println!("Fetching image from: {}", url);
    let image = fetch_image(&url).await?;
    println!(
        "Successfully fetched image: {}x{} pixels",
        image.width(),
        image.height()
    );

    Ok(String::new())
}
