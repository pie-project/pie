use std::pin::Pin;
use std::task::{Context, Poll};

use wasmtime::StoreContextMut;
use wasmtime::component::{
    Access, Destination, FutureReader, Resource, Source, StreamConsumer, StreamProducer,
    StreamReader, StreamResult,
};

use super::bindings::cli::types::ErrorCode;
use super::bindings::cli::{
    environment, exit, stderr, stdin, stdout, terminal_input, terminal_output, terminal_stderr,
    terminal_stdin, terminal_stdout, types,
};
use crate::wasi::ctx::SharedSink;
use crate::wasi::{I32Exit, WasiWeb, WasiWebCtxView, WasiWebView};

pub struct TerminalInput;
pub struct TerminalOutput;

impl types::Host for WasiWebCtxView<'_> {}

impl environment::Host for WasiWebCtxView<'_> {
    fn get_environment(&mut self) -> wasmtime::Result<Vec<(String, String)>> {
        Ok(self.ctx.env.clone())
    }

    fn get_arguments(&mut self) -> wasmtime::Result<Vec<String>> {
        Ok(self.ctx.args.clone())
    }

    fn get_initial_cwd(&mut self) -> wasmtime::Result<Option<String>> {
        Ok(None)
    }
}

impl exit::Host for WasiWebCtxView<'_> {
    fn exit(&mut self, status: Result<(), ()>) -> wasmtime::Result<()> {
        Err(wasmtime::format_err!(I32Exit(status.map_or(1, |()| 0))))
    }

    fn exit_with_code(&mut self, status_code: u8) -> wasmtime::Result<()> {
        Err(wasmtime::format_err!(I32Exit(status_code.into())))
    }
}

/// A browser tab has no stdin: its stream ends at once.
struct Closed;

impl<D> StreamProducer<D> for Closed {
    type Item = u8;
    type Buffer = Option<u8>;

    fn poll_produce<'a>(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _store: StoreContextMut<'a, D>,
        _dst: Destination<'a, Self::Item, Self::Buffer>,
        _finish: bool,
    ) -> Poll<wasmtime::Result<StreamResult>> {
        Poll::Ready(Ok(StreamResult::Dropped))
    }
}

/// Writes what the guest sends to a sink, as it comes.
struct ToSink(SharedSink);

impl<D> StreamConsumer<D> for ToSink {
    type Item = u8;

    fn poll_consume(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        store: StoreContextMut<D>,
        src: Source<Self::Item>,
        _finish: bool,
    ) -> Poll<wasmtime::Result<StreamResult>> {
        let mut src = src.as_direct(store);
        let written = {
            let bytes = src.remaining();
            if !bytes.is_empty() {
                (self.0.lock().unwrap())(bytes);
            }
            bytes.len()
        };
        src.mark_read(written);
        Poll::Ready(Ok(StreamResult::Completed))
    }
}

fn write_to<U>(
    mut store: Access<'_, U, WasiWeb>,
    sink: SharedSink,
    data: StreamReader<u8>,
) -> wasmtime::Result<FutureReader<Result<(), ErrorCode>>> {
    data.pipe(&mut store, ToSink(sink))?;
    FutureReader::new(&mut store, async { wasmtime::error::Ok(Ok(())) })
}

impl<U> stdin::HostWithStore<U> for WasiWeb {
    fn read_via_stream(
        mut store: Access<U, Self>,
    ) -> wasmtime::Result<(StreamReader<u8>, FutureReader<Result<(), ErrorCode>>)> {
        let stream = StreamReader::new(&mut store, Closed)?;
        let done = FutureReader::new(&mut store, async { wasmtime::error::Ok(Ok(())) })?;
        Ok((stream, done))
    }
}

impl<U> stdout::HostWithStore<U> for WasiWeb {
    fn write_via_stream(
        mut store: Access<'_, U, Self>,
        data: StreamReader<u8>,
    ) -> wasmtime::Result<FutureReader<Result<(), ErrorCode>>> {
        let sink = store.get().ctx.stdout.clone();
        write_to(store, sink, data)
    }
}

impl<U> stderr::HostWithStore<U> for WasiWeb {
    fn write_via_stream(
        mut store: Access<'_, U, Self>,
        data: StreamReader<u8>,
    ) -> wasmtime::Result<FutureReader<Result<(), ErrorCode>>> {
        let sink = store.get().ctx.stderr.clone();
        write_to(store, sink, data)
    }
}

impl stdin::Host for WasiWebCtxView<'_> {}
impl stdout::Host for WasiWebCtxView<'_> {}
impl stderr::Host for WasiWebCtxView<'_> {}

impl terminal_input::Host for WasiWebCtxView<'_> {}
impl terminal_output::Host for WasiWebCtxView<'_> {}

impl terminal_input::HostTerminalInput for WasiWebCtxView<'_> {
    fn drop(&mut self, rep: Resource<TerminalInput>) -> wasmtime::Result<()> {
        self.table.delete(rep)?;
        Ok(())
    }
}

impl terminal_output::HostTerminalOutput for WasiWebCtxView<'_> {
    fn drop(&mut self, rep: Resource<TerminalOutput>) -> wasmtime::Result<()> {
        self.table.delete(rep)?;
        Ok(())
    }
}

/// None of a tab's streams is a terminal.
impl terminal_stdin::Host for WasiWebCtxView<'_> {
    fn get_terminal_stdin(&mut self) -> wasmtime::Result<Option<Resource<TerminalInput>>> {
        Ok(None)
    }
}

impl terminal_stdout::Host for WasiWebCtxView<'_> {
    fn get_terminal_stdout(&mut self) -> wasmtime::Result<Option<Resource<TerminalOutput>>> {
        Ok(None)
    }
}

impl terminal_stderr::Host for WasiWebCtxView<'_> {
    fn get_terminal_stderr(&mut self) -> wasmtime::Result<Option<Resource<TerminalOutput>>> {
        Ok(None)
    }
}

pub(super) fn add_to_linker<T: WasiWebView + 'static>(
    linker: &mut wasmtime::component::Linker<T>,
) -> wasmtime::Result<()> {
    use super::bindings::cli;

    cli::types::add_to_linker::<T, WasiWeb>(linker, T::wasi_web)?;
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
    Ok(())
}
