//! WASI 0.3 host: the clocks. The filesystem and http interfaces are
//! declared (their resource types, which the runtime's world bindings name)
//! but not linked.

mod clocks;
mod filesystem;
mod http;

pub use filesystem::Descriptor;
pub use http::{Fields, Request, RequestOptions, Response};

use wasmtime::component::Linker;

use crate::{WasiWeb, WasiWebView};

/// Generated bindings for the 0.3 world. The runtime's pie-world bindgen maps
/// `wasi:http`, `wasi:clocks` and `wasi:filesystem` onto these modules.
pub mod bindings {
    mod generated {
        wasmtime::component::bindgen!({
            path: "wit/p3",
            world: "wasi-web:p3/imports",
            imports: {
                "wasi:http/client.send": store | trappable,
                "wasi:http/types.[drop]request": store | trappable,
                "wasi:http/types.[drop]response": store | trappable,
                "wasi:http/types.[static]request.consume-body": store | trappable,
                "wasi:http/types.[static]request.new": store | trappable,
                "wasi:http/types.[static]response.consume-body": store | trappable,
                "wasi:http/types.[static]response.new": store | trappable,
                "wasi:filesystem/types.[method]descriptor.read-via-stream": store | trappable,
                "wasi:filesystem/types.[method]descriptor.write-via-stream": store | trappable,
                "wasi:filesystem/types.[method]descriptor.append-via-stream": store | trappable,
                "wasi:filesystem/types.[method]descriptor.read-directory": store | trappable,
                default: trappable,
            },
            require_store_data_send: true,
            with: {
                "wasi:filesystem/types.descriptor": crate::p3::Descriptor,
                "wasi:http/types.fields": crate::p3::Fields,
                "wasi:http/types.request": crate::p3::Request,
                "wasi:http/types.request-options": crate::p3::RequestOptions,
                "wasi:http/types.response": crate::p3::Response,
            },
        });
    }
    pub use generated::wasi::*;
}

/// Register the 0.3 clocks (plus the `insecure-seed` rc shim the runtime
/// links today).
pub fn add_to_linker<T: WasiWebView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    use bindings::clocks;

    clocks::types::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    clocks::monotonic_clock::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    clocks::system_clock::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;

    // Pre-release guests (wit-bindgen's std shim) seed their hash maps from
    // this rc interface; wasmtime never semver-matches pre-releases, so it
    // needs its exact name.
    linker
        .root()
        .instance("wasi:random/insecure-seed@0.3.0-rc-2026-03-15")?
        .func_wrap("get-insecure-seed", |mut store, (): ()| {
            Ok((store.data_mut().wasi_web().ctx.insecure_seed,))
        })?;
    Ok(())
}
