// Alias target for 'pie:inferlet/model' under vitest. See ../stubs.ts.
export * from '../stubs.js';
import { modelStub } from '../stubs.js';
export const {
  name, architecture, defaultSystemSpeculation, isLinear, passKind,
  outputVocabSize, kvPageSize, frameSize, channelCapacity, maxEmbedLength,
  rsStateSize, rsBufferPageSize, rsFoldGranularity, arenaBlockSize,
} = modelStub;
