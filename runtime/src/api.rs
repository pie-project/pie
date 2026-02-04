pub mod core;
pub mod image;

pub mod adapter;
pub mod zo;
pub mod actor;

use wasmtime::component::HasSelf;

wasmtime::component::bindgen!({
    path: "wit",
    world: "imports",
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        "inferlet:core/common/blob-result": core::BlobResult,
        "inferlet:core/common/model": core::Model,
        "inferlet:core/common/queue": core::Queue,
        "inferlet:core/common/blob": core::Blob,
        "inferlet:core/common/debug-query-result": core::DebugQueryResult,
        "inferlet:core/common/synchronization-result": core::SynchronizationResult,
        "inferlet:core/message/subscription": core::message::Subscription,
        "inferlet:core/message/receive-result": core::message::ReceiveResult,
        "inferlet:core/forward/forward-pass": core::forward::ForwardPass,
        "inferlet:core/forward/forward-pass-result": core::forward::ForwardPassResult,
        "inferlet:core/tokenize/tokenizer": core::tokenize::Tokenizer,
        "inferlet:actor/common/global-context": actor::GlobalContext,
        "inferlet:actor/common/adapter": actor::Adapter,
        "inferlet:actor/common/optimizer": actor::Optimizer,
    },
    imports: { default: async | trappable },
    exports: { default: async },
});

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: inferlet::core::common::Host
        + inferlet::core::forward::Host
        + inferlet::core::tokenize::Host
        + inferlet::core::runtime::Host
        + inferlet::core::kvs::Host
        + inferlet::core::message::Host
        + inferlet::adapter::common::Host
        + inferlet::zo::evolve::Host
        + inferlet::image::image::Host
        + inferlet::actor::common::Host,
{
    inferlet::core::common::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::core::forward::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::core::tokenize::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::core::runtime::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::core::kvs::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::core::message::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::adapter::common::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::zo::evolve::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::image::image::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    inferlet::actor::common::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;

    Ok(())
}
