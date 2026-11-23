# plex-sdk: provenance

`src/lib.rs` is plex `sdk-1.1/src/lib.rs` at `d32445219` (unchanged up to
the host commits `crates/plex-host` names), with `wit_bindgen::generate!`
reading `../plex-host/wit` (the same normative WIT the host is built from)
instead of `../wit-1.1`, and formatted with pie's rustfmt. The `ext`
module (pie's `pie:plex-ext` extension) is pie's own.
