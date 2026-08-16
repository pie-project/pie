//! The PTIR channel plane — re-exported, not owned. This file is
//! `pub use driver::*` and nothing else; it sits at the crate root so that
//! every caller under `gpu/` points down at it and no cycle forms.

pub use driver::*;

// A glob re-export cannot fail loudly — it succeeds and exports the wrong
// names — so this names two types that exist ONLY in `driver`. It fails on the
// re-export instead of at twenty unresolved imports under `gpu/`.
const _: fn() = || {
    fn from_the_pipeline_crate<T>(_: Option<&T>) {}
    from_the_pipeline_crate::<self::PassInputs<'static>>(None);
    from_the_pipeline_crate::<self::Registry>(None);
};
