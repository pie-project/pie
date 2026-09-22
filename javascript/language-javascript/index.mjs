// `import "@pie-project/language-javascript"`: the javascript language component registers
// itself, and every @pie-project/server started (or running) installs it.
import { registerLanguage } from '@pie-project/server/languages';

registerLanguage('javascript', new URL('./javascript.wasm', import.meta.url));
