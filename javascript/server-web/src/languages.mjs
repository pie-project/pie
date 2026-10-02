// The language components a server installs when it starts. A language
// package registers itself on import (`import "@pie-project/language-python"`);
// `Server.start()` installs every registered language, and a registration
// after the start lands on the running servers too. The registry is one per
// process, however many copies of this module are loaded.

const registry = (globalThis[Symbol.for("pie.languages")] ??= { languages: new Map(), servers: new Set() });

/** Register a language component by the URL of its wasm. */
export function registerLanguage(language, url) {
  registry.languages.set(language, url);
  for (const server of registry.servers) {
    server.installLanguage(language, url).catch((error) => console.error(`pie: ${language}: ${error.message ?? error}`));
  }
}

/** The registered languages, language → URL. */
export function registeredLanguages() {
  return new Map(registry.languages);
}

/** Install every registered language on `server`, and every one registered later while it runs. */
export async function attachLanguages(server) {
  registry.servers.add(server);
  for (const [language, url] of registry.languages) await server.installLanguage(language, url);
  return () => registry.servers.delete(server);
}
