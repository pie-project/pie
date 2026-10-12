// `inferlet/eta` — the ETA authoring surface for JavaScript inferlets: the
// tracing eDSL (`Tensor` + ops), the container encoder, and the WIT bridge
// (`Channel`, `WorkingSet`, `ForwardPass`, `Pipeline`, `runAhead`).
//
// A stage body is an ordinary function traced once at first submit:
//
//     import { eta } from '@pie-project/inferlet';
//     const { Channel, ForwardPass, dtype, intrinsics, reduceArgmax, reshape } = eta;
//
//     const tokOut = fwd.epilogue(() => reduceArgmax(intrinsics.logits()));
//     pipe.submit(fwd);
//     const token = tokOut.takeScalar();
//
// The emitted container bytes are identical to the Rust `inferlet` crate's for the same
// program, so a JS inferlet and a Rust inferlet share the host's program
// cache.

export * as diffusion from './diffusion.js';
export * as intrinsics from './intrinsics.js';
export {
  TOKEN_PAD,
  Channel,
  ForwardPass,
  InferletError,
  PageGrant,
  Pipeline,
  RsWorkingSet,
  WorkingSet,
  channelCapacity,
  frameSize,
  kvPageSize,
  maxEmbedLength,
  mm,
  padTokens,
  prefillChunkHint,
  prefillChunks,
  scale,
  submitDeadlineUs,
  unpadTokens,
} from './bridge.js';
export type { ForwardKind, KvBinding, KvGeometry, PageDecl, PageRange, RsGeometry } from './bridge.js';
export { Dtype, Port, Stage, dtype } from './ir.js';
export type { Shape } from './ir.js';
export { TraceError } from './trace.js';
export {
  ConstData,
  Tensor,
  abs,
  add,
  and,
  broadcast,
  cast,
  causalMask,
  constant,
  cos,
  cummassLe,
  cumprod,
  cumsum,
  div,
  entropy,
  entropyFromLogprobs,
  eq,
  exp,
  gather,
  gatherRow,
  ge,
  gt,
  gumbel,
  gumbelMax,
  indptr,
  iota,
  l2norm,
  le,
  log,
  logSoftmax,
  lt,
  maskApply,
  maskedArgmax,
  matmul,
  maxElem,
  minElem,
  mul,
  ne,
  neg,
  normal,
  not,
  nucleusSample,
  or,
  pivotThreshold,
  probGe,
  rankLe,
  recip,
  reduceArgmax,
  reduceMax,
  reduceMin,
  reduceSum,
  rem,
  reshape,
  rng,
  rsqrt,
  rowMembership,
  scalarGather,
  scatterAdd,
  scatterSet,
  select,
  sign,
  sin,
  sinkWindowMask,
  slidingWindowMask,
  softmax,
  sortDesc,
  sqrt,
  sub,
  topK,
  transpose,
} from './value.js';
export type { Operand } from './value.js';
