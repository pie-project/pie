//! Generated host bindings for the JSON-only PLEX v0.2 component world.

wasmtime::component::bindgen!({
    path: "../../interface/plex/wit",
    world: "plex-policy",
    anyhow: true,
});
