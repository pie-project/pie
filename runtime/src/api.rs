pub mod audio_out;
pub mod http;
pub mod inference;
pub mod media;
pub mod messaging;
pub mod model;
pub mod kv_working_set;
pub mod runtime;
pub mod rs_working_set;
pub mod session;
pub mod types;

pub mod instruct;

use wasmtime::component::HasSelf;
use crate::instance::InstanceState;

wasmtime::component::bindgen!({
    path: "../interface/inferlet",
    world: "inferlet",
    // wasmtime 46 split `wasmtime::Error` from `anyhow::Error`; keep the
    // generated host traits on `anyhow::Result` so the existing api/*.rs
    // impls (anyhow `?`/`bail!`/`anyhow!`) continue to compile unchanged.
    anyhow: true,
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        "wasi:filesystem/types": wasmtime_wasi::p2::bindings::filesystem::types,
        "wasi:filesystem/preopens": wasmtime_wasi::p2::bindings::filesystem::preopens,
        "wasi:clocks/wall-clock": wasmtime_wasi::p2::bindings::clocks::wall_clock,
        "wasi:io/streams": wasmtime_wasi::p2::bindings::io::streams,
        "wasi:random/random": wasmtime_wasi::p2::bindings::random::random,
        "wasi:random/insecure": wasmtime_wasi::p2::bindings::random::insecure,
        "wasi:random/insecure-seed": wasmtime_wasi::p2::bindings::random::insecure_seed,
        // pie:core/working-set (kv); rs-working-set below
        "pie:core/working-set.kv-working-set": crate::working_set::kv::KvWorkingSet,
        // pie:core/inference
        "pie:core/inference.grammar": inference::Grammar,
        "pie:core/inference.matcher": inference::Matcher,
        // pie:core/ptir (thrust-3 P2b) — first-class channels + forward-pass
        // submission (the registry surface folded into forward-pass.new).
        "pie:core/ptir.channel": crate::ptir::ptir_host::Channel,
        "pie:core/ptir.forward-pass": crate::ptir::ptir_host::ForwardPass,
        "pie:core/ptir.pipeline": crate::ptir::ptir_host::Pipeline,
        // pie:core/working-set (rs)
        "pie:core/working-set.rs-working-set": crate::working_set::rs::RsWorkingSet,
        // pie:core/media
        "pie:core/media.image": media::Image,
        "pie:core/media.video": media::Video,
        "pie:core/media.audio": media::Audio,
        // pie:core/audio-out
        "pie:core/audio-out.speech": audio_out::Speech,
        // pie:instruct
        "pie:instruct/chat.decoder": instruct::chat::Decoder,
        "pie:instruct/tool-use.decoder": instruct::tool_use::Decoder,
        "pie:instruct/reasoning.decoder": instruct::reasoning::Decoder,
    },
    imports: {
        // subscribe returns a host-created stream<string>, which needs store
        // access to build -> generate it on HostWithStore (store flag), matching
        // how wasmtime_wasi p3 handles stream-returning funcs.
        "pie:core/messaging.subscribe": store | async | trappable,
        default: async | trappable,
    },
    exports: { default: async },
});

pub fn add_to_linker(
    linker: &mut wasmtime::component::Linker<InstanceState>,
) -> Result<(), wasmtime::Error> {
    // Concrete on InstanceState: the async-func imports (execute/receive/pull/
    // subscribe) are generated on `HostWithStore` traits implemented for
    // `HasSelf<InstanceState>`, so the linker `D` type must be concrete.
    type D = HasSelf<InstanceState>;
    pie::core::types::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::working_set::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::http::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::model::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::inference::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::ptir::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::messaging::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::session::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::media::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::audio_out::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::core::runtime::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::instruct::chat::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::instruct::tool_use::add_to_linker::<InstanceState, D>(linker, |s| s)?;
    pie::instruct::reasoning::add_to_linker::<InstanceState, D>(linker, |s| s)?;

    Ok(())
}
