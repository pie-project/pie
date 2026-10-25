// Alias target for 'pie:inferlet/model' under vitest. See ../stubs.ts.
export * from '../stubs.js';
import { modelStub } from '../stubs.js';
export const {
  name, architecture, defaultSystemSpeculation,
  outputVocabSize, kvPageSize, frameSize, channelCapacity,
  rsStateSize,
  submitDeadlineUs, runAheadWindow, draftBlock, switches, lane,
} = modelStub;
