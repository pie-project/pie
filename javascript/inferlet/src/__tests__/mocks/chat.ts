// Alias target for 'pie:inferlet/chat' under vitest. See ../stubs.ts.
import { chatStub } from '../stubs.js';
export const {
  prefix, system, firstUser, user, systemUser, assistant, cue, seal, stopTokens,
  Decoder,
} = chatStub;
