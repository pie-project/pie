mod common;

use std::path::PathBuf;

use wasmtime::Store;
use wasmtime::component::types::ComponentItem;
use wasmtime::component::{Component, Linker, ResourceType};

fn guest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/target/wasm32-wasip2/release/text_completion.wasm")
}

fn stub_pie_world(
    engine: &wasmtime::Engine,
    linker: &mut Linker<common::State>,
    component: &Component,
) {
    for (name, item) in component.component_type().imports(engine) {
        if name.starts_with("wasi:") {
            continue;
        }
        let ComponentItem::ComponentInstance(instance) = item.ty else {
            panic!("unexpected non-instance import {name}");
        };
        let mut root = linker.root();
        let mut stub = root.instance(name).expect("stub instance");
        for (export, item) in instance.exports(engine) {
            let qualified = format!("{name}#{export}");
            let label = qualified.clone();
            match item.ty {
                ComponentItem::ComponentFunc(func) if func.async_() => {
                    stub.func_new_concurrent(export, move |_, _, _, _| {
                        let label = label.clone();
                        Box::pin(async move { wasmtime::bail!("stubbed import {label}") })
                    })
                }
                ComponentItem::ComponentFunc(_) => stub.func_new(export, move |_, _, _, _| {
                    wasmtime::bail!("stubbed import {label}")
                }),
                ComponentItem::Resource(_) => {
                    stub.resource(export, ResourceType::host::<()>(), |_, _| Ok(()))
                }
                _ => Ok(()),
            }
            .unwrap_or_else(|e| panic!("stub {qualified}: {e}"));
        }
    }
}

#[test]
fn text_completion_links_and_instantiates() {
    let path = guest_path();
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!(
            "skipping: {} not built (cd examples && cargo build -p text-completion --release --target wasm32-wasip2)",
            path.display()
        );
        return;
    };

    let engine = common::engine();
    let component = Component::new(&engine, &bytes).expect("text-completion component");
    let mut linker = Linker::<common::State>::new(&engine);
    wasmtime_web::add_to_linker(&mut linker).expect("add_to_linker");
    stub_pie_world(&engine, &mut linker, &component);

    let wasi_imports: Vec<String> = component
        .component_type()
        .imports(&engine)
        .map(|(name, _)| name.to_string())
        .filter(|name| name.starts_with("wasi:"))
        .collect();
    assert!(
        !wasi_imports.is_empty(),
        "guest imports no wasi interfaces?"
    );

    let (state, _stdout, _stderr) = common::state();
    let mut store = Store::new(&engine, state);
    common::block_on(async {
        linker
            .instantiate_async(&mut store, &component)
            .await
            .unwrap_or_else(|e| {
                panic!("instantiate text-completion (wasi imports: {wasi_imports:?}): {e:?}")
            });
    });
}
