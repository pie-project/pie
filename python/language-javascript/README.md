# pie-language-javascript

The javascript language component of pie, as a wheel: one wasm that
`pie.server.Server` installs when this package is importable.

```bash
pip install "pie-server[javascript]"
```

The wasm is what `pie language install` puts under `$PIE_HOME/languages`;
`scripts/build-languages.sh` builds it here.
