//! Generated host bindings for the fixed PLEX v0.1 component world.

wasmtime::component::bindgen!({
    path: "../../interface/plex/wit",
    world: "plex-policy",
    anyhow: true,
    imports: {
        "pie:plex/maps.get": store | trappable,
        "pie:plex/telemetry.emit": store | trappable,
    },
});
