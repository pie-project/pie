// Alias target for 'pie:inferlet/chat' under vitest. See ../stubs.ts.
import { chatStub } from '../stubs.js';
export const {
  system, firstUser, user, systemUser, assistant, cue, seal, stopTokens,
  createDecoder,
} = chatStub;
