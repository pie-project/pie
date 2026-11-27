// Alias target for 'pie:inferlet/model' under vitest. See ../stubs.ts.
export * from '../stubs.js';
import { modelStub } from '../stubs.js';
export const {
  name, architecture,
  outputVocabSize, kvPageSize, frameSize, channelCapacity,
  rsStateSize,
  runAheadWindow, draftBlock, draftDepth, switches, lane,
} = modelStub;
