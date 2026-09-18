//! A minimal WASI host for wasmtime that also builds for `wasm32-unknown-unknown`.
//!
//! `wasmtime-wasi` and `wasmtime-wasi-http` lean on cap-std, rustix and tokio's
//! networking, none of which exist in a browser tab. This crate hosts the WASI
//! 0.2 interfaces a `wasm32-wasip2` guest actually imports (cli, io, clocks,
//! random) plus the WASI 0.3 clocks the pie inferlet world declares, with a
//! host-pluggable clock and no threads, files or sockets. The filesystem and
//! http interfaces exist only as the resource types the world's bindings
//! name; a guest importing them is refused at instantiation, by name.
//!
//! Store data implements [`WasiWebView`]; [`add_to_linker`] registers everything.

pub mod clock;
mod ctx;
pub mod p2;
pub mod p3;
mod random;
mod stdio;

pub use clock::{fire_due_timers, now, set_clock, set_wall_clock};
pub use ctx::{I32Exit, Sink, WasiWebCtx, WasiWebCtxBuilder, WasiWebCtxView, WasiWebView};

/// `wasmtime-wasi`'s names for the same things, so a crate written against
/// that crate can be pointed at this one on wasm32 (`wasmtime-wasi = { package
/// = "wasi-web" }`) and keep its `use wasmtime_wasi::WasiView` lines.
pub type WasiCtx = WasiWebCtx;
pub type WasiCtxView<'a> = WasiWebCtxView<'a>;

/// `wasmtime_wasi::WasiView`'s shape; anything implementing it is a
/// [`WasiWebView`] through the blanket impl below.
pub trait WasiView: Send {
    fn ctx(&mut self) -> WasiCtxView<'_>;
}

impl<T: WasiView> WasiWebView for T {
    fn wasi_web(&mut self) -> WasiWebCtxView<'_> {
        self.ctx()
    }
}

use wasmtime::component::{HasData, Linker, ResourceTable};

/// The [`HasData`] marker the generated bindings are instantiated with.
///
/// Host trait impls live on [`WasiWebCtxView`]; `HostWithStore` impls (p3
/// functions that need the store) live on this type.
pub struct WasiWeb;

impl HasData for WasiWeb {
    type Data<'a> = WasiWebCtxView<'a>;
}

/// [`HasData`] marker for the `wasi:io` implementation, which only wants the table.
pub(crate) struct WasiWebIo;

impl HasData for WasiWebIo {
    type Data<'a> = &'a mut ResourceTable;
}

pub(crate) fn table<T: WasiWebView>(t: &mut T) -> &mut ResourceTable {
    t.wasi_web().table
}

/// Register every interface this crate hosts (WASI 0.2 and 0.3) into `linker`.
/// Interfaces it does not host (filesystem, http, sockets) stay undefined, so
/// a guest importing them is refused at instantiation, by name.
///
/// The linker's engine must have `async_support` enabled: `wasi:io/poll` and
/// the 0.3 clock waits are async host functions.
pub fn add_to_linker<T: WasiWebView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    p2::add_to_linker(linker)?;
    p3::add_to_linker(linker)?;
    Ok(())
}
