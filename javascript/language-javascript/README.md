# @pie-project/language-javascript

The javascript language component of pie, as an npm package: one wasm, registered
with `@pie-project/server` on import.

```js
import { Server } from '@pie-project/server';
import '@pie-project/language-javascript';        // javascript inferlets run

const server = await Server.start({ model: 'Qwen/Qwen3.5-0.8B' });
```

A server already running when the import happens installs it as well. The
wasm is what `pie language install` puts under `$PIE_HOME/languages`;
`scripts/build-languages.sh` builds it here.
