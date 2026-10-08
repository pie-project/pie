// `import "@pie-project/language-python"`: the python language component registers
// itself, and every @pie-project/server started (or running) installs it.
import { registerLanguage } from '@pie-project/server/languages';

registerLanguage('python', new URL('./python.wasm', import.meta.url));
