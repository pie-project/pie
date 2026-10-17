// Alias target for 'pie:inferlet/model' under vitest. See ../stubs.ts.
export * from '../stubs.js';
import { modelStub } from '../stubs.js';
export const {
  name, architecture, defaultSystemSpeculation, passKind,
  outputVocabSize, kvPageSize, frameSize, channelCapacity, maxReadoutRows,
  rsStateSize,
  mtpDepth, submitDeadlineUs, runAheadWindow, draftBlock, canvas,
} = modelStub;
