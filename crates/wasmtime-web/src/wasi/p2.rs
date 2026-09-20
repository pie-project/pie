use wasmtime::StoreContextMut;
use wasmtime::component::{ComponentType, Lift, Linker, Lower, Resource, ResourceType};
use wasmtime_wasi_io::poll::subscribe;
use wasmtime_wasi_io::streams::{DynInputStream, DynOutputStream};

use crate::wasi::clock::{self, Deadline};
use crate::wasi::random as rng;
use crate::wasi::stdio::{ClosedInput, SinkStream};
use crate::wasi::{I32Exit, WasiWebIo, WasiWebView};

pub struct TerminalInput;
pub struct TerminalOutput;

#[derive(ComponentType, Lower, Lift)]
#[component(record)]
struct Datetime {
    seconds: u64,
    nanoseconds: u32,
}

const VERSION: &str = "0.2.12";

pub fn add_to_linker<T: WasiWebView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    use wasmtime_wasi_io::bindings::wasi::io;

    io::error::add_to_linker::<T, WasiWebIo>(linker, crate::wasi::table)?;
    io::poll::add_to_linker::<T, WasiWebIo>(linker, crate::wasi::table)?;
    io::streams::add_to_linker::<T, WasiWebIo>(linker, crate::wasi::table)?;

    let mut root = linker.root();
    let name = |interface: &str| format!("wasi:{interface}@{VERSION}");

    let mut i = root.instance(&name("cli/environment"))?;
    i.func_wrap(
        "get-environment",
        |mut store: StoreContextMut<'_, T>, (): ()| {
            Ok((store.data_mut().wasi_web().ctx.env.clone(),))
        },
    )?;
    i.func_wrap(
        "get-arguments",
        |mut store: StoreContextMut<'_, T>, (): ()| {
            Ok((store.data_mut().wasi_web().ctx.args.clone(),))
        },
    )?;
    i.func_wrap("initial-cwd", |_: StoreContextMut<'_, T>, (): ()| {
        Ok((None::<String>,))
    })?;

    let mut i = root.instance(&name("cli/exit"))?;
    i.func_wrap(
        "exit",
        |_: StoreContextMut<'_, T>, (status,): (Result<(), ()>,)| -> wasmtime::Result<()> {
            Err(wasmtime::format_err!(I32Exit(status.map_or(1, |()| 0))))
        },
    )?;
    i.func_wrap(
        "exit-with-code",
        |_: StoreContextMut<'_, T>, (code,): (u8,)| -> wasmtime::Result<()> {
            Err(wasmtime::format_err!(I32Exit(code.into())))
        },
    )?;

    let mut i = root.instance(&name("cli/stdin"))?;
    i.func_wrap("get-stdin", |mut store: StoreContextMut<'_, T>, (): ()| {
        let stream: DynInputStream = Box::new(ClosedInput);
        Ok((store.data_mut().wasi_web().table.push(stream)?,))
    })?;
    let mut i = root.instance(&name("cli/stdout"))?;
    i.func_wrap("get-stdout", |mut store: StoreContextMut<'_, T>, (): ()| {
        let view = store.data_mut().wasi_web();
        let stream: DynOutputStream = Box::new(SinkStream(view.ctx.stdout.clone()));
        Ok((view.table.push(stream)?,))
    })?;
    let mut i = root.instance(&name("cli/stderr"))?;
    i.func_wrap("get-stderr", |mut store: StoreContextMut<'_, T>, (): ()| {
        let view = store.data_mut().wasi_web();
        let stream: DynOutputStream = Box::new(SinkStream(view.ctx.stderr.clone()));
        Ok((view.table.push(stream)?,))
    })?;

    let mut i = root.instance(&name("cli/terminal-input"))?;
    i.resource(
        "terminal-input",
        ResourceType::host::<TerminalInput>(),
        |_, _| Ok(()),
    )?;
    let mut i = root.instance(&name("cli/terminal-output"))?;
    i.resource(
        "terminal-output",
        ResourceType::host::<TerminalOutput>(),
        |_, _| Ok(()),
    )?;
    let mut i = root.instance(&name("cli/terminal-stdin"))?;
    i.func_wrap("get-terminal-stdin", |_: StoreContextMut<'_, T>, (): ()| {
        Ok((None::<Resource<TerminalInput>>,))
    })?;
    let mut i = root.instance(&name("cli/terminal-stdout"))?;
    i.func_wrap(
        "get-terminal-stdout",
        |_: StoreContextMut<'_, T>, (): ()| Ok((None::<Resource<TerminalOutput>>,)),
    )?;
    let mut i = root.instance(&name("cli/terminal-stderr"))?;
    i.func_wrap(
        "get-terminal-stderr",
        |_: StoreContextMut<'_, T>, (): ()| Ok((None::<Resource<TerminalOutput>>,)),
    )?;

    let mut i = root.instance(&name("clocks/monotonic-clock"))?;
    i.func_wrap("now", |_: StoreContextMut<'_, T>, (): ()| {
        Ok((clock::now(),))
    })?;
    i.func_wrap("resolution", |_: StoreContextMut<'_, T>, (): ()| {
        Ok((1u64,))
    })?;
    i.func_wrap(
        "subscribe-instant",
        |mut store: StoreContextMut<'_, T>, (when,): (u64,)| {
            let table = store.data_mut().wasi_web().table;
            let deadline = table.push(Deadline(Some(when)))?;
            Ok((subscribe(table, deadline)?,))
        },
    )?;
    i.func_wrap(
        "subscribe-duration",
        |mut store: StoreContextMut<'_, T>, (when,): (u64,)| {
            let table = store.data_mut().wasi_web().table;
            let deadline = table.push(Deadline(clock::now().checked_add(when)))?;
            Ok((subscribe(table, deadline)?,))
        },
    )?;

    let mut i = root.instance(&name("clocks/wall-clock"))?;
    i.func_wrap("now", |_: StoreContextMut<'_, T>, (): ()| {
        let ns = clock::wall_now();
        Ok((Datetime {
            seconds: ns / 1_000_000_000,
            nanoseconds: (ns % 1_000_000_000) as u32,
        },))
    })?;
    i.func_wrap("resolution", |_: StoreContextMut<'_, T>, (): ()| {
        Ok((Datetime {
            seconds: 0,
            nanoseconds: 1_000_000,
        },))
    })?;

    for interface in ["random/random", "random/insecure"] {
        let insecure = interface.ends_with("insecure");
        let mut i = root.instance(&name(interface))?;
        i.func_wrap(
            if insecure {
                "get-insecure-random-bytes"
            } else {
                "get-random-bytes"
            },
            |_: StoreContextMut<'_, T>, (len,): (u64,)| Ok((rng::bytes(len)?,)),
        )?;
        i.func_wrap(
            if insecure {
                "get-insecure-random-u64"
            } else {
                "get-random-u64"
            },
            |_: StoreContextMut<'_, T>, (): ()| Ok((rng::u64()?,)),
        )?;
    }
    let mut i = root.instance(&name("random/insecure-seed"))?;
    i.func_wrap(
        "insecure-seed",
        |mut store: StoreContextMut<'_, T>, (): ()| {
            Ok((store.data_mut().wasi_web().ctx.insecure_seed,))
        },
    )?;
    Ok(())
}
