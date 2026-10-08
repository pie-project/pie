# pie-shim

C-ABI staticlib that embeds the Pie standalone (controller, gateway and
worker composed in one process, Metal engine) for the iOS app under
`ios/`. The boot path is the one `pie run` uses: the config TOML is a
standalone config exactly as `pie config init` writes it, with
`[server] port` forced to 0 so the OS picks a loopback port.

Exported symbols (the full contract is documented in `src/lib.rs`):

    typedef void (*pie_stream_cb)(const char* chunk, void* ctx);
    char* pie_ios_run_stream(const char* config_path, const char* wasm_path,
                             const char* version /* nullable */,
                             const char* input_json, pie_stream_cb cb, void* ctx);
    void pie_ios_free(char* s);

The engine boots on the first call and stays warm for the life of the
process; a failed boot is final until the app relaunches. Each call
installs the component once per process, launches it, streams its
stdout to `cb` on the calling thread and returns its return value, or a
string starting with `PIE ERROR:` (a panic on the calling thread is
reported the same way). A turn that has not returned after 240 s is
terminated and reported as an error.

Build (the crate shares the repository's `target/` via `.cargo/config.toml`):

    cd ios/pie-shim
    cargo build --release --target aarch64-apple-ios        # device
    cargo build --release --target aarch64-apple-ios-sim    # simulator

The library lands at `target/<triple>/release/libpie_ios_shim.a` under
the repository root. Engine diagnostics go to stderr; `RUST_LOG`
overrides the default `info,tarpc=warn` filter.
