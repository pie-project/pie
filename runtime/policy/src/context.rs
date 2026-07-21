use wasmtime::{Store, StoreLimits, StoreLimitsBuilder};

use crate::engine::InvocationPermit;

pub(crate) struct InvocationContext {
    limits: StoreLimits,
    _permit: InvocationPermit,
}

impl InvocationContext {
    pub(crate) fn store(
        engine: &wasmtime::Engine,
        memory_bytes: usize,
        permit: InvocationPermit,
    ) -> Store<Self> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_bytes)
            .table_elements(1024)
            .instances(4)
            .tables(4)
            .memories(1)
            .build();
        let mut store = Store::new(
            engine,
            Self {
                limits,
                _permit: permit,
            },
        );
        store.limiter(|context| &mut context.limits);
        store
    }
}
