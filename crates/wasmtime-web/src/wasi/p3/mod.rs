mod clocks;
mod filesystem;
mod http;

pub use filesystem::Descriptor;
pub use http::{Fields, Request, RequestOptions, Response};

use wasmtime::component::Linker;

use crate::wasi::{WasiWeb, WasiWebView};

pub mod bindings {
    mod generated {
        wasmtime::component::bindgen!({
            inline: r#"
                package wasi-web:p3;

                world imports {
                    import wasi:clocks/monotonic-clock@0.3.0;
                    import wasi:filesystem/types@0.3.0;
                    import wasi:filesystem/preopens@0.3.0;
                    import wasi:http/client@0.3.0;
                }
            "#,
            path: "../inferlet/wit",
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
                "wasi:filesystem/types.descriptor": crate::wasi::p3::Descriptor,
                "wasi:http/types.fields": crate::wasi::p3::Fields,
                "wasi:http/types.request": crate::wasi::p3::Request,
                "wasi:http/types.request-options": crate::wasi::p3::RequestOptions,
                "wasi:http/types.response": crate::wasi::p3::Response,
            },
        });
    }
    pub use generated::wasi::*;
}

pub fn add_to_linker<T: WasiWebView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    use bindings::clocks;

    clocks::types::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    clocks::monotonic_clock::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    clocks::system_clock::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;

    // wasmtime never semver-matches pre-releases: the rc interface needs its exact name.
    linker
        .root()
        .instance("wasi:random/insecure-seed@0.3.0-rc-2026-03-15")?
        .func_wrap("get-insecure-seed", |mut store, (): ()| {
            Ok((store.data_mut().wasi_web().ctx.insecure_seed,))
        })?;
    Ok(())
}
