use wasmtime::component::Resource;
use wasmtime_wasi_io::streams::{DynInputStream, DynOutputStream};

use super::bindings::cli::{
    environment, exit, stderr, stdin, stdout, terminal_input, terminal_output, terminal_stderr,
    terminal_stdin, terminal_stdout,
};
use crate::stdio::{ClosedInput, SinkStream};
use crate::{I32Exit, WasiWebCtxView};

/// Never handed out: stdio is never a terminal here.
pub struct TerminalInput;
pub struct TerminalOutput;

impl environment::Host for WasiWebCtxView<'_> {
    fn get_environment(&mut self) -> wasmtime::Result<Vec<(String, String)>> {
        Ok(self.ctx.env.clone())
    }

    fn get_arguments(&mut self) -> wasmtime::Result<Vec<String>> {
        Ok(self.ctx.args.clone())
    }

    fn initial_cwd(&mut self) -> wasmtime::Result<Option<String>> {
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

impl stdin::Host for WasiWebCtxView<'_> {
    fn get_stdin(&mut self) -> wasmtime::Result<Resource<DynInputStream>> {
        let stream: DynInputStream = Box::new(ClosedInput);
        Ok(self.table.push(stream)?)
    }
}

impl stdout::Host for WasiWebCtxView<'_> {
    fn get_stdout(&mut self) -> wasmtime::Result<Resource<DynOutputStream>> {
        let stream: DynOutputStream = Box::new(SinkStream(self.ctx.stdout.clone()));
        Ok(self.table.push(stream)?)
    }
}

impl stderr::Host for WasiWebCtxView<'_> {
    fn get_stderr(&mut self) -> wasmtime::Result<Resource<DynOutputStream>> {
        let stream: DynOutputStream = Box::new(SinkStream(self.ctx.stderr.clone()));
        Ok(self.table.push(stream)?)
    }
}

impl terminal_input::Host for WasiWebCtxView<'_> {}
impl terminal_input::HostTerminalInput for WasiWebCtxView<'_> {
    fn drop(&mut self, r: Resource<TerminalInput>) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}

impl terminal_output::Host for WasiWebCtxView<'_> {}
impl terminal_output::HostTerminalOutput for WasiWebCtxView<'_> {
    fn drop(&mut self, r: Resource<TerminalOutput>) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}

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
