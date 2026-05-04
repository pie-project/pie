# http-server

Example inferlet that exports `wasi:http/incoming-handler` instead of the
usual `inferlet:core/run` entrypoint.

Endpoints:

- `/`
- `/wait`
- `/echo`
- `/echo-headers`
- `/info`

## Build

```bash
cargo build -p http-server --target wasm32-wasip2 --release
```

The E2E test launches it as a daemon through the Pie client and then probes
the HTTP endpoints:

```bash
uv run python tests/inferlets/test_http_server.py --dummy
```
