pub mod types;
pub mod context;
pub mod model;
pub mod inference;
pub mod messaging;
pub mod session;
pub mod adapter;
pub mod runtime;

pub mod mcp;
pub mod zo;
pub mod instruct;

use wasmtime::component::HasSelf;

wasmtime::component::bindgen!({
    path: "wit",
    world: "inferlet",
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        // pie:core/types
        "pie:core/types/future-bool": types::FutureBool,
        "pie:core/types/future-string": types::FutureString,
        "pie:core/types/future-blob": types::FutureBlob,
        // pie:core/context
        "pie:core/context/context": context::Context,
        // pie:core/model
        "pie:core/model/model": model::Model,
        "pie:core/model/tokenizer": model::Tokenizer,
        // pie:core/inference
        "pie:core/inference/forward-pass": inference::ForwardPass,
        "pie:core/inference/future-output": inference::FutureOutput,
        "pie:core/inference/grammar": inference::Grammar,
        "pie:core/inference/matcher": inference::Matcher,
        // pie:core/messaging
        "pie:core/messaging/subscription": messaging::Subscription,
        // pie:core/adapter
        "pie:core/adapter/adapter": adapter::Adapter,
        // pie:mcp/client
        "pie:mcp/client/session": mcp::Session,
        // pie:instruct
        "pie:instruct/chat/decoder": instruct::chat::Decoder,
        "pie:instruct/tool-use/decoder": instruct::tool_use::Decoder,
        "pie:instruct/reasoning/decoder": instruct::reasoning::Decoder,
    },
    imports: { default: async | trappable },
    exports: { default: async },
});

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: pie::core::types::Host
        + pie::core::context::Host
        + pie::core::model::Host
        + pie::core::inference::Host
        + pie::core::messaging::Host
        + pie::core::session::Host
        + pie::core::adapter::Host
        + pie::core::runtime::Host
        + pie::mcp::types::Host
        + pie::mcp::client::Host
        + pie::zo::zo::Host
        + pie::instruct::chat::Host
        + pie::instruct::tool_use::Host
        + pie::instruct::reasoning::Host,
{
    pie::core::types::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::context::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::model::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::inference::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::messaging::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::session::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::adapter::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::core::runtime::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::mcp::types::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::mcp::client::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::zo::zo::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::instruct::chat::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::instruct::tool_use::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::instruct::reasoning::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;

    Ok(())
}
