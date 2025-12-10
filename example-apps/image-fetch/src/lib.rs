//! Demonstrates fetching images over HTTP.
//!
//! This example shows how to use the HTTP client to fetch an image from a URL
//! and decode it using the `image` crate.

use image::{DynamicImage, load_from_memory};
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{AsyncRead, empty};
use inferlet::{Args, Result};

const HELP: &str = "\
Usage: image-fetch [OPTIONS]

A program to fetch and decode an image from a URL.

Options:
  -u, --url <URL>  The URL of the image to fetch [default: https://www.ilankelman.org/stopsigns/australia.jpg]
  -h, --help       Prints this help message";

/// Asynchronously fetches an image from the given URL.
pub async fn fetch_image(url: &str) -> Result<DynamicImage> {
    let client = Client::new();

    let request = Request::builder()
        .uri(url)
        .method(Method::GET)
        .body(empty())?;

    let response = client.send(request).await?;

    let mut body = response.into_body();
    let mut buf = Vec::new();
    body.read_to_end(&mut buf).await?;

    let img = load_from_memory(&buf)?;

    Ok(img)
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let url: String = args
        .value_from_str(["-u", "--url"])
        .unwrap_or_else(|_| "https://www.ilankelman.org/stopsigns/australia.jpg".to_string());

    println!("Fetching image from: {}", url);
    let image = fetch_image(&url).await?;
    println!(
        "Successfully fetched image: {}x{} pixels",
        image.width(),
        image.height()
    );

    Ok(())
}
