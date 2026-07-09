//! pie:core/http - host-side async HTTP fetch (client).
//!
//! A single buffered request, run host-side on `reqwest` (async). Redirects
//! are NOT followed here (the SDK resolves `location` + re-requests), matching
//! charlie's guest redirect loop. This sidesteps the wasi:io pollable wall that
//! a guest-side wasi:http client would hit under component-model-async.

use crate::api::pie;
use crate::instance::InstanceState;
use anyhow::Result;
use std::sync::OnceLock;
use wasmtime::component::{Accessor, HasSelf};

use pie::core::http::{Request, Response};

static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build host http client")
    })
}

impl pie::core::http::Host for InstanceState {}

impl pie::core::http::HostWithStore<InstanceState> for HasSelf<InstanceState> {
    async fn fetch(
        accessor: &Accessor<InstanceState, Self>,
        request: Request,
    ) -> Result<Result<Response, String>> {
        if !accessor.with(|mut access| access.get().network_allowed()) {
            return Ok(Err("network access is disabled for this inferlet".to_string()));
        }

        let method = match reqwest::Method::from_bytes(request.method.as_bytes()) {
            Ok(m) => m,
            Err(_) => return Ok(Err(format!("invalid HTTP method: {}", request.method))),
        };

        let mut rb = http_client().request(method, &request.url);
        for (k, v) in &request.headers {
            rb = rb.header(k, v);
        }
        if let Some(body) = request.body {
            rb = rb.body(body);
        }

        match rb.send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let headers = resp
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or_default().to_string()))
                    .collect();
                match resp.bytes().await {
                    Ok(body) => Ok(Ok(Response { status, headers, body: body.to_vec() })),
                    Err(e) => Ok(Err(format!("failed to read response body: {e}"))),
                }
            }
            Err(e) => Ok(Err(format!("http request failed: {e}"))),
        }
    }
}
