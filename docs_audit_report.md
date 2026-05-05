# Documentation Audit Report

Audit baseline: `main` at `07e386ed`, with an already-dirty working tree. Scope covered public website docs/pages plus public project READMEs. Internal vendored/build/`node_modules`/virtualenv READMEs were excluded.

## Remediation Status

This report's detailed findings below are the historical audit baseline. The high-priority documentation and snippet issues have been addressed in the current working tree:

- `pie run` and `pie-client submit` examples now forward inferlet arguments after `--`.
- The generated Rust Bakery scaffold builds against the current Rust SDK.
- JavaScript client docs use the local package name (`pie-client`) and await async `close()` calls.
- JavaScript client package metadata declares `@noble/hashes`, uses Node's built-in test runner, and installs/tests cleanly.
- Constrained-decoding docs use the current `JsonSchema` / `AnyJson` / `Regex` / `Ebnf` and `.constrain_with(...)` APIs.
- The tutorial no longer claims a Python Bakery scaffold exists.
- Configuration and SGLang references now match current defaults/options.
- Website typecheck failures from `docusaurus.config.ts` and `whypieagent.tsx` are fixed.
- Runnable bash fences no longer contain placeholder angle-bracket arguments or ellipsis prompts.
- Usage templates that are not literal shell commands are fenced as `text`, not `bash`.
- The JavaScript `Process` API sketch is now valid TypeScript instead of invalid JavaScript.
- `pie run --path ... -- --prompt ...` now preserves the first post-`--` flag as inferlet input instead of treating it as the optional inferlet-name positional.

Current verification:

| Check | Result |
|---|---|
| Bash / TOML / JSON / Python fenced-block parse pass | Pass, 248 blocks |
| JavaScript / TypeScript syntax pass | Pass, 111 blocks |
| Rust fenced-block syntax pass (`rustfmt`, file or function-body context) | Pass, 146 blocks |
| `npm run typecheck` in `website` | Pass |
| `npm run build` in `website` | Pass |
| `cargo build -p pie-server` | Pass, existing warnings only |
| `cargo check --workspace` | Pass, existing warnings only |
| `cargo test -p pie-server run_cmd -- --nocapture` | Pass, 4 tests |
| `pie run` with a built `helloworld` inferlet and a temporary dummy-driver config | Pass; prints `{"message":"Hello World!!",...}` and exits cleanly |
| `pie run` with built `text-completion`, `--path`, trailing `-- --prompt ... --max-tokens 4`, and a dummy-driver config | Pass; accepts the documented argument shape and exits cleanly after four dummy requests |
| `bakery build inferlets/python-example -o /tmp/pie-doc-python-example.wasm` | Pass |
| `bakery build inferlets/py-concurrency-test -o /tmp/pie-doc-py-concurrency-test.wasm` | Pass |
| `pie run` with built `py-concurrency-test`, default `python_snapshot = true`, and a dummy-driver config | Pass; returns `done` and exits cleanly after 40 dummy requests |
| `pie run` with built `py-concurrency-test`, `runtime.python_snapshot = false`, and a dummy-driver config | Pass; returns `done` and exits cleanly after 40 dummy requests |
| `pie run` with built `python-example`, default `python_snapshot = true`, and a dummy-driver config | Pass; exits cleanly after 256 dummy requests |
| `npm install` in `client/javascript` | Pass |
| `npm test` in `client/javascript` | Pass, 3 tests |
| `uv run --project client/python --with pytest pytest` | Pass, 3 tests |
| Rust `bakery create` then `bakery build` with `PIE_SDK=/Users/ingim/Workspace/pie/sdk` | Pass |
| TypeScript `bakery create --ts` then `bakery build` with `PIE_SDK=/Users/ingim/Workspace/pie/sdk` | Pass |
| `bakery build inferlets/text-completion -o /tmp/pie-doc-text-completion.wasm` | Pass |
| `uv run --project client/python pie-client submit --help` | Shows `-- --prompt` examples |

Runtime-gated check:

| Check | Result |
|---|---|
| `pie run` with a built `helloworld` inferlet and a temporary dummy-driver config | Pass after making Python shared-module compilation lazy for non-Python inferlets; prints `{"message":"Hello World!!",...}` and exits cleanly. |

Root cause of the original timeout: engine startup eagerly compiled `$PIE_HOME/py-runtime/shared/*.wasm` through Wasmtime before serving any component. A macOS process sample showed the main thread inside `pie_server::serve::start_engine` -> `pie::program::python::runtime::load_shared_modules` -> `wasmtime::Module::new`, waiting on Wasmtime compilation workers. This affected even Rust-only inferlets. The runtime now records the Python runtime directory at startup and compiles shared Python modules only when a Python component path asks for them.

## Original Executive Summary

- Audited 92 public documentation files: 68 MDX pages and 24 repo README/standalone Markdown files.
- Inventoried 1140 fenced blocks: 145 bash, 146 Rust, 118 Python, 105 TypeScript, 29 TOML, plus plain/output fences.
- `npm run build` in `website` passes, but `npm run typecheck` fails.
- Host Rust workspace check passes: `cargo check --workspace`.
- Python SDK/client tests pass when pytest is supplied: 56 SDK tests and 3 client tests.
- Current Rust inferlet examples checked (`text-completion`, `text-completion-spec`, `raw-completion`) pass `cargo check --target wasm32-wasip2`.
- TypeScript bakery scaffold builds, but Rust bakery scaffold generated by `bakery create` fails to compile against the current SDK.
- TOML and JSON fenced examples parse successfully.

## Original High-Priority Findings

1. Website typecheck fails even though the static Docusaurus build succeeds.
   - Evidence: `npm run typecheck` in `website`.
   - `website/docusaurus.config.ts:77` passes the dark palette into a helper typed as the light palette literal type.
   - `website/src/pages/whypieagent.tsx:184` returns `JSX.Element`, but the JSX namespace is not available under the current React/TS config.

2. Many `pie run` examples are not accepted by the current CLI parser.
   - Evidence: `target/debug/pie run --path /tmp/nope.wasm --manifest /tmp/Pie.toml --prompt hi --quiet` exits with `unexpected argument '--prompt'`; help says to use `-- --prompt`.
   - Affected examples include:
     - `website/docs/reference/pie.mdx:97`, `:105`
     - `website/docs/reference/manifest.mdx:99`
     - `website/docs/guide/setup.mdx:67`, `:103`
     - `website/docs/guide/tutorial/run.mdx:157`
     - `website/docs/guide/deploy/build-publish.mdx:72`, `:73`
     - `website/docs/guide/dev-env.mdx:104`
     - `website/docs/guide/first-inferlet.mdx:61`, `:152`
     - `website/docs/guide/examples/overview.mdx:19`
     - `server/README.md:31`
   - Fix pattern: use `pie run ... -- --prompt "..."`, or document `--input '{"prompt":"..."}'`.

3. `pie-client submit` examples have the same trailing-argument problem.
   - Evidence: `uv run pie-client submit text-completion --prompt hello --host 127.0.0.1 --port 9` exits with `No such option: --prompt`.
   - Affected: `website/docs/reference/pie-client.mdx:50`, `:53`.
   - Fix pattern: use `pie-client submit text-completion@0.1.0 -- --prompt "hello"` if that path is supported, or document positional `ARGUMENTS` exactly as Typer expects.

4. `bakery create` Rust scaffold does not compile against the current Rust SDK.
   - Evidence: disposable scaffold in `/tmp`, then `PIE_SDK=/Users/ingim/Workspace/pie/sdk uv run --project sdk/tools/bakery bakery build . -o audit-hello.wasm`.
   - Failures originate from `sdk/tools/bakery/src/bakery/templates/rust/lib.rs.template`:
     - `inferlet::Event` no longer exists at root.
     - Template imports `inference::Sampler` instead of `sample::Sampler`.
     - `.with_max_tokens(...)` should be `.max_tokens(...)`.
     - `.decode().with_reasoning()` no longer matches the current generator/parser surface.
   - This invalidates docs that promise `bakery create <name>` creates a buildable Rust inferlet, including `website/docs/guide/first-inferlet.mdx` and `website/docs/guide/dev-env.mdx`.

5. JavaScript client docs do not match local package metadata.
   - Docs say `npm install @pie/client` and `import { PieClient } from '@pie/client'` in:
     - `website/docs/reference/client-javascript.mdx:16`, `:22`
     - `website/docs/guide/deploy/clients.mdx:12`, `:38`, `:87`
     - `website/docs/guide/tutorial/serve.mdx:69`, `:214`
   - Local package is `client/javascript/package.json:2` with `"name": "pie-client"`.
   - The implementation imports `@noble/hashes` at `client/javascript/src/index.js:10`, but `client/javascript/package.json:17` declares `"blake3": "^3.0.0"` instead.
   - Evidence: `npm install` in `client/javascript` fails with `No matching version found for blake3-wasm@2.1.7`; `npm test` cannot run because Jest is unavailable.

6. JavaScript client snippets call async `close()` without awaiting it.
   - Implementation: `client/javascript/src/index.js:265` declares `async close()`.
   - Affected docs:
     - `website/docs/reference/client-javascript.mdx:43`, `:52`, `:172`
     - `website/docs/guide/deploy/clients.mdx:97`
     - `website/docs/guide/tutorial/serve.mdx:252`
   - Fix pattern: `await client.close()` / `await c1.close()`.

7. Tutorial and standalone constrained-decoding docs reference removed schema APIs.
   - `website/docs/guide/tutorial/build.mdx:406` says `Schema::Json` and `Schema.json(...)`.
   - `sdk/CONSTRAINED_DECODING.md:62-66`, `:107` documents `Schema::JsonSchema`, `Schema::Json`, `Schema::Regex`, `Schema::Ebnf`, `Schema::Grammar`, and `.constrain(...)`.
   - Current public Rust API uses structs/trait such as `JsonSchema`, `AnyJson`, `Regex`, `Ebnf`, `GrammarConstraint`, and generator `.constrain_with(...)`.

8. Tutorial contradicts current Bakery CLI about Python scaffolding.
   - `website/docs/guide/tutorial/build.mdx:84` correctly says Bakery has no Python template.
   - `website/docs/guide/tutorial/build.mdx:135` says `bakery create --lang python` writes one automatically.
   - Evidence: `uv run bakery create audit-py --lang python ...` exits with `No such option: --lang`.

9. Configuration reference has at least one default mismatch.
   - `website/docs/reference/configuration.mdx:16` and `:78` say `[server].verbose` defaults to `false`.
   - Current config template emits `verbose = true` at `server/src/cli/config_cmd/template.rs:30`, and `ServerConfig::default()` is also `true`.

10. SGLang driver docs omit a current option.
    - `driver/sglang/src/pie_driver_sglang/config.py` includes `disable_radix_cache: bool = True`.
    - `website/docs/reference/drivers/sglang.mdx` documents other SGLang options but not this one.

## Original Verification Log

| Check | Result |
|---|---|
| `npm run build` in `website` | Pass |
| `npm run typecheck` in `website` | Fail, two TypeScript errors |
| `cargo check --workspace` | Pass, warnings only |
| `target/debug/pie --help` and subcommand help | Pass; docs missing `check` and `smoke` in `website/docs/reference/pie.mdx` overview |
| `uv run pie-client --help` and subcommand help | Pass |
| `uv run bakery --help` and subcommand help | Pass |
| `uv run --with pytest pytest` in `sdk/python` | Pass, 56 tests |
| `uv run --with pytest pytest` in `client/python` | Pass, 3 tests |
| `npm run build` in `sdk/javascript` | Pass |
| `npm test` in `sdk/javascript` | Fail, script points to `src/__tests__` but no test files are present |
| `npm install` in `client/javascript` | Fail, `blake3` dependency resolution error |
| `cargo check --target wasm32-wasip2` in selected Rust inferlets | Pass for `text-completion`, `text-completion-spec`, `raw-completion` |
| `bakery create --ts` then `bakery build` | Pass |
| `bakery create` Rust then `bakery build` | Fail, stale Rust template |
| TOML/JSON fenced blocks | Pass parse check |
| Python fenced blocks | 114/115 parse; one remaining block is an intentional signature fragment |

## Original Page Status Matrix

| Page | Status | Notes |
|---|---|---|
| `website/docs/overview/what-is-pie.mdx` | Static pass | Rust snippets use current `sample::Sampler` style; live model run gated. |
| `website/docs/overview/components.mdx` | Static pass | Component names align at high level. |
| `website/docs/overview/key-features.mdx` | Static pass | Rust snippets align with current SDK shape. |
| `website/docs/overview/comparison.mdx` | Static pass | Conceptual claims not live-verifiable locally. |
| `website/docs/overview/benchmarks.mdx` | Gated | Benchmark reproduction requires benchmark environment/models. |
| `website/docs/overview/faq.mdx` | Static pass | Some production-readiness claims remain editorial. |
| `website/docs/guide/install.mdx` | Gated | Installer/network/GPU checks not executed. |
| `website/docs/guide/setup.mdx` | Issues | `pie run ... --prompt` examples need `--` or `--input`. |
| `website/docs/guide/dev-env.mdx` | Issues | `pie run` examples need `--`; Rust scaffold promise currently false. |
| `website/docs/guide/first-inferlet.mdx` | Issues | Rust scaffold/build path currently broken; `pie run` examples need `--`. |
| `website/docs/guide/tutorial/build.mdx` | Issues | Python scaffold contradiction and stale schema API mention. |
| `website/docs/guide/tutorial/run.mdx` | Issues | `pie run --path ... --question` needs `--` or `--input`. |
| `website/docs/guide/tutorial/serve.mdx` | Issues | JS client package/import and async close snippets are stale. |
| `website/docs/guide/examples/overview.mdx` | Issues | Published-run example needs `--` before inferlet args. |
| `website/docs/guide/examples/chat.mdx` | Static pass | Links only; example existence not exhaustively run. |
| `website/docs/guide/examples/samplers.mdx` | Static pass | Links only. |
| `website/docs/guide/examples/structured.mdx` | Static pass | Links only. |
| `website/docs/guide/examples/kv-cache.mdx` | Static pass | Links only. |
| `website/docs/guide/examples/speculation.mdx` | Static pass | Links only. |
| `website/docs/guide/examples/reasoning.mdx` | Static pass | Links only. |
| `website/docs/guide/examples/agents.mdx` | Static pass | Links only. |
| `website/docs/guide/examples/integration.mdx` | Static pass | Links only. |
| `website/docs/guide/model/loading.mdx` | Static pass | API names align with SDK wrappers in this pass. |
| `website/docs/guide/model/tokenizer.mdx` | Static pass | Tokenizer methods align across SDKs. |
| `website/docs/guide/context/overview.mdx` | Static pass | Context method names align. |
| `website/docs/guide/context/pages.mdx` | Static pass | Internal counters are marked non-public. |
| `website/docs/guide/context/sharing.mdx` | Static pass | Snapshot/open/take APIs align. |
| `website/docs/guide/context/scheduling.mdx` | Static pass | `set_bid`/`setBid`/`idle` APIs align. |
| `website/docs/guide/forward/overview.mdx` | Static pass | Rust forward snippets match current API shape. |
| `website/docs/guide/forward/inputs.mdx` | Static pass | Rust-only snippets not individually compiled. |
| `website/docs/guide/forward/samplers.mdx` | Static pass | Sampler names align. |
| `website/docs/guide/forward/constrained.mdx` | Static pass | Website constrained docs are newer than `sdk/CONSTRAINED_DECODING.md`. |
| `website/docs/guide/forward/adapters.mdx` | Gated | Adapter runtime behavior requires driver/model support. |
| `website/docs/guide/forward/speculation.mdx` | Gated | Speculation examples require live model/driver. |
| `website/docs/guide/decoder/overview.mdx` | Static pass | Examples not live-run. |
| `website/docs/guide/decoder/generator.mdx` | Static pass | Generator API names align in spot checks. |
| `website/docs/guide/decoder/chat.mdx` | Static pass | Parser API names align in spot checks. |
| `website/docs/guide/decoder/tool-calling.mdx` | Gated | Tool-call grammar/model behavior requires live model. |
| `website/docs/guide/decoder/reasoning.mdx` | Gated | Reasoning parser behavior model-dependent. |
| `website/docs/guide/io/overview.mdx` | Static pass | Conceptual page. |
| `website/docs/guide/io/session.mdx` | Static pass | Client/server live flow gated. |
| `website/docs/guide/io/messaging.mdx` | Static pass | API names align in spot checks. |
| `website/docs/guide/io/http.mdx` | Gated | HTTP examples require runtime sandbox execution. |
| `website/docs/guide/io/filesystem.mdx` | Gated | Filesystem examples require runtime sandbox execution. |
| `website/docs/guide/io/mcp.mdx` | Gated | MCP examples require local MCP server/client bridge. |
| `website/docs/guide/deploy/overview.mdx` | Static pass | Conceptual page. |
| `website/docs/guide/deploy/build-publish.mdx` | Issues | `pie run` examples need `--`; publish flow gated. |
| `website/docs/guide/deploy/serve.mdx` | Gated | Server/model/GPU examples not live-run. |
| `website/docs/guide/deploy/clients.mdx` | Issues | JS package/import and async close snippets stale. |
| `website/docs/guide/deploy/profiling.mdx` | Gated | Benchmark and monitor flows not live-run. |
| `website/docs/reference/sdk-rust.mdx` | Static pass | Rust SDK reference largely aligns; snippets not exhaustively compiled. |
| `website/docs/reference/sdk-python.mdx` | Static pass | Python package tests pass; one signature fence is intentionally non-runnable. |
| `website/docs/reference/sdk-javascript.mdx` | Static pass | SDK builds; SDK test script lacks tests. |
| `website/docs/reference/manifest.mdx` | Issues | `pie run` example needs `--`; manifest TOML parses. |
| `website/docs/reference/client-rust.mdx` | Static pass | Rust client compiles in workspace. |
| `website/docs/reference/client-python.mdx` | Static pass | Python client tests pass. |
| `website/docs/reference/client-javascript.mdx` | Issues | Package name/import/deps and async close snippets stale. |
| `website/docs/reference/pie.mdx` | Issues | `pie run` examples need `--`; missing `check`/`smoke` commands in overview. |
| `website/docs/reference/bakery.mdx` | Static pass | Bakery CLI commands align; help text under-describes actual auto-detect build support. |
| `website/docs/reference/pie-client.mdx` | Issues | `submit --prompt` examples fail parser. |
| `website/docs/reference/configuration.mdx` | Issues | `server.verbose` default mismatch. |
| `website/docs/reference/drivers/cuda.mdx` | Gated | Config keys align in spot check; hardware behavior gated. |
| `website/docs/reference/drivers/portable.mdx` | Gated | Config keys align in spot check; backend behavior gated. |
| `website/docs/reference/drivers/vllm.mdx` | Gated | Config keys align in spot check; vLLM install/model behavior gated. |
| `website/docs/reference/drivers/sglang.mdx` | Issues | Missing `disable_radix_cache`; runtime behavior gated. |
| `website/src/pages/models.mdx` | Gated | Model availability claims are environment/release dependent. |
| `website/src/pages/roadmap.mdx` | Static pass | Roadmap is editorial/future-facing. |
| `website/src/pages/community.mdx` | Static pass | Links not externally crawled. |

## Original README / Standalone Markdown Notes

- `server/README.md` has a stale `pie run text-completion --prompt ...` example.
- `sdk/CONSTRAINED_DECODING.md` is stale relative to current constrained-decoding APIs.
- Driver READMEs generally mirror website driver pages; live install/model/GPU behavior remains gated.
- `client/javascript` package metadata prevents dependency installation, so JS client README/reference examples cannot currently be proven runnable from a clean install.

## Original Recommended Fix Order

1. Fix `pie run` / `pie-client submit` argument examples everywhere; this is the broadest user-facing breakage.
2. Fix the Bakery Rust template, then rerun the first-inferlet/dev-env/tutorial flows.
3. Decide the published JavaScript client package name, update docs or `package.json`, and fix the hash dependency.
4. Fix website TypeScript errors so docs CI can include `npm run typecheck`.
5. Update stale constrained-decoding docs and tutorial schema wording.
6. Correct config defaults and add missing SGLang option.
7. Add automated doc-snippet checks for CLI examples and scaffold builds so these regressions do not return.
