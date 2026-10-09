# pagable (patched)

buck2's `pagable` at `03a23abe`, which starlark (`crates/poem`) depends on.
The workspace's `[patch]` swaps it in for the git crate.

One change, in `src/typetag/platform.rs`: generic typetag registration
emits its constructor record on Android (the same ELF `.init_array` as Linux)
and on every 64-bit Apple platform (the same Mach-O `__mod_init_func` as
macOS). Upstream refuses those targets with a `compile_error!`, which keeps
pie off Android, iOS and visionOS. Drop this copy once upstream widens its
cfg.
