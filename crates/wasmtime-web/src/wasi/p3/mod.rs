mod cli;
mod clocks;
mod filesystem;
mod http;
mod random;

pub use cli::{TerminalInput, TerminalOutput};
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
                    import wasi:cli/environment@0.3.0;
                    import wasi:cli/exit@0.3.0;
                    import wasi:cli/stdin@0.3.0;
                    import wasi:cli/stdout@0.3.0;
                    import wasi:cli/stderr@0.3.0;
                    import wasi:cli/terminal-stdin@0.3.0;
                    import wasi:cli/terminal-stdout@0.3.0;
                    import wasi:cli/terminal-stderr@0.3.0;
                    import wasi:random/random@0.3.0;
                    import wasi:random/insecure@0.3.0;
                    import wasi:random/insecure-seed@0.3.0;
                    import wasi:clocks/monotonic-clock@0.3.0;
                    import wasi:filesystem/types@0.3.0;
                    import wasi:filesystem/preopens@0.3.0;
                    import wasi:http/client@0.3.0;
                }
            "#,
            path: "../inferlet/wit",
            world: "wasi-web:p3/imports",
            imports: {
                "wasi:cli/stdin": store | trappable,
                "wasi:cli/stdout": store | trappable,
                "wasi:cli/stderr": store | trappable,
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
                "wasi:cli/terminal-input.terminal-input": crate::wasi::p3::TerminalInput,
                "wasi:cli/terminal-output.terminal-output": crate::wasi::p3::TerminalOutput,
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
    cli::add_to_linker(linker)?;
    random::add_to_linker(linker)?;
    Ok(())
}
