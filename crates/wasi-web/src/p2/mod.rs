//! WASI 0.2 host: cli, clocks, random; io from wasmtime-wasi-io. The
//! filesystem interfaces are declared (their resource types) but not linked.

mod cli;
mod clocks;
mod filesystem;
mod random;

pub use cli::{TerminalInput, TerminalOutput};
pub use filesystem::{Descriptor, DirectoryEntryStream};

use wasmtime::component::Linker;

use crate::{WasiWeb, WasiWebIo, WasiWebView};

/// Generated bindings for the 0.2 world; `wasi:io` types come from wasmtime-wasi-io.
pub mod bindings {
    mod generated {
        wasmtime::component::bindgen!({
            path: "wit/p2",
            world: "wasi-web:p2/imports",
            imports: { default: trappable },
            require_store_data_send: true,
            with: {
                "wasi:io/poll": wasmtime_wasi_io::bindings::wasi::io::poll,
                "wasi:io/streams": wasmtime_wasi_io::bindings::wasi::io::streams,
                "wasi:io/error": wasmtime_wasi_io::bindings::wasi::io::error,
                "wasi:cli/terminal-input.terminal-input": crate::p2::TerminalInput,
                "wasi:cli/terminal-output.terminal-output": crate::p2::TerminalOutput,
                "wasi:filesystem/types.descriptor": crate::p2::Descriptor,
                "wasi:filesystem/types.directory-entry-stream": crate::p2::DirectoryEntryStream,
            },
        });
    }
    pub use generated::wasi::*;
}

/// Register only the 0.2 interfaces.
pub fn add_to_linker<T: WasiWebView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    use bindings::{cli, clocks, random};
    use wasmtime_wasi_io::bindings::wasi::io;

    io::error::add_to_linker::<T, WasiWebIo>(linker, crate::table)?;
    io::poll::add_to_linker::<T, WasiWebIo>(linker, crate::table)?;
    io::streams::add_to_linker::<T, WasiWebIo>(linker, crate::table)?;

    cli::environment::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::exit::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::stdin::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::stdout::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::stderr::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::terminal_input::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::terminal_output::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::terminal_stdin::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::terminal_stdout::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    cli::terminal_stderr::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    clocks::monotonic_clock::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    clocks::wall_clock::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    random::random::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    random::insecure::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    random::insecure_seed::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
    Ok(())
}
