# Pie Runtime

## Build

```bash
cargo build
```

For the Python extension module:

```bash
cargo build --release
```

## Tests

### Unit tests

```bash
cargo test --lib
```

### Integration tests

Requires test inferlets to be compiled to `wasm32-wasip3` (handled automatically
on first run). `wasm32-wasip3` is a Tier-3 target, so run the one-time setup
first (see the repo `README.md` "wasip3 toolchain" section):

```bash
./scripts/setup-wasip3.sh   # once: wasm32-wasip2 libc donor + linker wrapper
cargo test
```

Run a specific test file:

```bash
cargo test --test e2e
cargo test --test program
cargo test --test smoke
```

### With output

```bash
cargo test -- --nocapture
```

## Test Inferlets

Source lives in `tests/inferlets/`. To build manually:

```bash
cd tests/inferlets
cargo build --target wasm32-wasip2
```

Available test inferlets:

| Name | Purpose |
|---|---|
| `echo` | Returns args joined as output |
| `context` | Exercises model, tokenizer, and context host APIs |
| `error` | Always returns an error |
