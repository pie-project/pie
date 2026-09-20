mod clock;
mod ctx;
pub mod p2;
pub mod p3;
mod random;
mod stdio;

pub use clock::{fire_due_timers, now, set_clock, set_wall_clock};
pub use ctx::{I32Exit, Sink, WasiWebCtx, WasiWebCtxBuilder, WasiWebCtxView, WasiWebView};

pub type WasiCtx = WasiWebCtx;
pub type WasiCtxView<'a> = WasiWebCtxView<'a>;

pub trait WasiView: Send {
    fn ctx(&mut self) -> WasiCtxView<'_>;
}

impl<T: WasiView> WasiWebView for T {
    fn wasi_web(&mut self) -> WasiWebCtxView<'_> {
        self.ctx()
    }
}

use wasmtime::component::{HasData, Linker, ResourceTable};

pub struct WasiWeb;

impl HasData for WasiWeb {
    type Data<'a> = WasiWebCtxView<'a>;
}

pub(crate) struct WasiWebIo;

impl HasData for WasiWebIo {
    type Data<'a> = &'a mut ResourceTable;
}

pub(crate) fn table<T: WasiWebView>(t: &mut T) -> &mut ResourceTable {
    t.wasi_web().table
}

pub fn add_to_linker<T: WasiWebView + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
    p2::add_to_linker(linker)?;
    p3::add_to_linker(linker)?;
    Ok(())
}

const HOSTED: &[&str] = &[
    "wasi:io/error",
    "wasi:io/poll",
    "wasi:io/streams",
    "wasi:cli/environment",
    "wasi:cli/exit",
    "wasi:cli/stdin",
    "wasi:cli/stdout",
    "wasi:cli/stderr",
    "wasi:cli/terminal-input",
    "wasi:cli/terminal-output",
    "wasi:cli/terminal-stdin",
    "wasi:cli/terminal-stdout",
    "wasi:cli/terminal-stderr",
    "wasi:clocks/monotonic-clock",
    "wasi:clocks/wall-clock",
    "wasi:clocks/system-clock",
    "wasi:clocks/types",
    "wasi:random/random",
    "wasi:random/insecure",
    "wasi:random/insecure-seed",
];

pub fn stub_unhosted<T: 'static>(
    linker: &mut Linker<T>,
    engine: &wasmtime::Engine,
    component: &wasmtime::component::Component,
) -> wasmtime::Result<()> {
    use wasmtime::component::types::ComponentItem;
    use wasmtime::component::{ResourceType, Val};

    let ty = component.component_type();
    let imports: Vec<_> = ty.imports(engine).collect();
    let unversioned = |name: &str| name.split('@').next().unwrap_or(name).to_string();
    let mut hosted_resources: Vec<ResourceType> = Vec::new();
    for (name, item) in &imports {
        if !HOSTED.contains(&unversioned(name).as_str()) {
            continue;
        }
        if let ComponentItem::ComponentInstance(instance) = &item.ty {
            for (_, item) in instance.exports(engine) {
                if let ComponentItem::Resource(ty) = item.ty {
                    hosted_resources.push(ty);
                }
            }
        }
    }

    for (name, item) in &imports {
        if !name.starts_with("wasi:") {
            continue;
        }
        let unversioned = unversioned(name);
        if HOSTED.contains(&unversioned.as_str()) {
            continue;
        }
        let ComponentItem::ComponentInstance(instance) = &item.ty else {
            continue;
        };
        let interface = name.to_string();
        let mut root = linker.root();
        let mut stub = root.instance(name)?;
        for (export, item) in instance.exports(engine) {
            let what = format!("{interface}#{export}");
            match item.ty {
                ComponentItem::ComponentFunc(_)
                    if unversioned == "wasi:filesystem/preopens" && export == "get-directories" =>
                {
                    stub.func_new(export, |_, _, _, results| {
                        results[0] = Val::List(Vec::new());
                        Ok(())
                    })?;
                }
                ComponentItem::ComponentFunc(func) if func.async_() => {
                    stub.func_new_concurrent(export, move |_, _, _, _| {
                        let what = what.clone();
                        Box::pin(async move {
                            wasmtime::bail!("{what} is not available in a browser tab")
                        })
                    })?;
                }
                ComponentItem::ComponentFunc(_) => {
                    stub.func_new(export, move |_, _, _, _| {
                        wasmtime::bail!("{what} is not available in a browser tab")
                    })?;
                }
                ComponentItem::Resource(ty) if hosted_resources.contains(&ty) => {}
                ComponentItem::Resource(_) => {
                    stub.resource(export, ResourceType::host::<()>(), |_, _| Ok(()))?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}
