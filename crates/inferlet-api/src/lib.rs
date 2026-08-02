//! The `pie:inferlet` guest contract: the WIT package, and the bindings
//! generated from it.
//!
//! This crate is the **canonical** `wit/` — the host reads it by path
//! (`engine`'s `wasmtime::component::bindgen!` points at `../inferlet-api/wit`),
//! the Python and JavaScript SDK toolchains read it by path, and here it is
//! also compiled, by `wit_bindgen::generate!`, into the guest-side bindings.
//!
//! ## Why the generator lives with the WIT
//!
//! It used to not, and that cost a mirror. `inferlet` held the
//! `wit_bindgen::generate!` site and a byte-identical copy of every `.wit`
//! file next to it, kept in step by `scripts/sync-wit.sh` and policed by a CI
//! drift job. The copy was not laziness: `inferlet` is **published to
//! crates.io** (`.github/workflows/release-cargo.yml`), and `generate!`'s
//! `path` is a filesystem path resolved at macro expansion, so a `.crate`
//! archive can only reach a `wit/` inside its own package directory.
//! `path: "../inferlet-api/wit"` compiles in a git checkout and breaks the
//! moment the package is packed.
//!
//! Moving the generator to the crate that OWNS the WIT dissolves that: the
//! path is `"wit"`, a plain subdirectory, and the bindings reach `inferlet`
//! the way anything else reaches a crate — as a dependency. This is the
//! `wasi` crate's shape, and it is why one vendored copy remains rather than
//! two (bakery's, which is a Python package and cannot depend on a rlib).
//!
//! ## What is here and what is not
//!
//! Only the generated bindings, raw. The ergonomics — `Context`, the `ptir`
//! authoring bridge, `chat`, `mask`, the module aliases that let an author
//! write `model::encode` — are `inferlet`'s, and stay there. The split is
//! contract versus convenience, which is the same line every other `-api`
//! crate in this workspace draws.
//!
//! One consequence worth stating: this is a **wasm32 guest** crate. It sits
//! outside `default-members` (rule 8) even though `engine` reads its `wit/`,
//! because what `engine` reads is the directory, not the rlib.

// Re-exported so `inferlet` can hand it on: a guest that writes its own inline
// `wit_bindgen::generate!` for a private world must use the SAME wit-bindgen
// as the one that generated this world, or the two sets of generated
// `cabi_realloc`/runtime glue disagree.
pub use wit_bindgen;

// With no `async:` option, the WIT's own `async func` annotations drive async
// generation: only run/execute/receive become `async fn`
// (component-model-async); sync funcs (model::encode, chat::*, …) stay sync.
// wit-bindgen generates the wasi:io bindings itself with versioned
// cabi_realloc symbols so it doesn't collide with std's copy.
//
// `pub_export_macro` is what makes the cross-crate split work at all: it emits
// `export!` as a `#[macro_export]` macro taking `with_types_in <path>`, so the
// leaf inferlet can name a re-export path (`::inferlet`) rather than this
// crate, and never has to know the bindings moved.
wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    generate_all,
});
