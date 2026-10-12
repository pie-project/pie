export function registerLanguage(language: string, url: URL | string): void;
export function registeredLanguages(): Map<string, URL | string>;
export function attachLanguages(server: {
  installLanguage(language: string, source: URL | string): Promise<string>;
}): Promise<() => void>;
