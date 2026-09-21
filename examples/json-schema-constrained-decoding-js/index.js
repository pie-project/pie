// JSON generation constrained by a JSON Schema — the JavaScript twin of
// `json-schema-constrained-decoding`.
//
// The host grammar matcher advances after every accepted token and supplies
// the next allowed-token mask to an ETA `maskedArgmax` epilogue. Exercises
// the `grammar` + `mask` modules and a host-writer mask channel.

import { chat, eta, grammar, mask, model } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, RsWorkingSet, WorkingSet, channelCapacity, dtype, indptr, intrinsics, kvPageSize, maskedArgmax, reshape } = eta;

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const divCeil = (a, b) => Math.floor((a + b - 1) / b);

const DEFAULT_SCHEMA = `{
    "type": "object",
    "properties": {
        "name": { "type": "string", "minLength": 1 },
        "age": { "type": "integer", "minimum": 0, "maximum": 150 },
        "skills": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        }
    },
    "required": ["name", "age", "skills"],
    "additionalProperties": false
}`;

export function main(input) {
  const promptText = input.prompt ?? 'Generate a profile for a fictional software engineer named Alice.';
  const schema = input.schema ?? DEFAULT_SCHEMA;
  const maxTokens = Number(input.max_tokens ?? 512);
  if (maxTokens < 1) throw new Error('max_tokens must be at least 1');

  const vocab = model.outputVocabSize();
  const kind = model.passKind();
  const ws = new WorkingSet();
  const rsWs = kind !== 'attention' ? [new RsWorkingSet()] : [];
  const pageSize = kvPageSize();
  const constraint = new grammar.Matcher(grammar.Grammar.fromJsonSchema(schema));

  let prompt = [
    ...chat.systemUser('Generate only the requested JSON value, with no markdown or explanation.', promptText),
    ...chat.cue(),
  ];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;
  const maxPages = Math.max(divCeil(n + maxTokens + 1, pageSize), 1);
  ws.reserve(maxPages);

  const promptTokens = Channel.from(prompt, dtype.i32);
  const prefillIndptr = Channel.from([0, n], dtype.u32).named('prefill_indptr');
  const prefillPositions = Channel.from(range(0, n), dtype.u32).named('prefill_positions');
  const prefillPages = Channel.from(range(0, maxPages), dtype.u32).named('prefill_pages');
  const prefillPageIndptr = Channel.from([0, divCeil(n, pageSize)], dtype.u32).named('prefill_page_indptr');
  const prefillWSlot = Channel.from(range(0, n).map((p) => Math.floor(p / pageSize)), dtype.u32).named('prefill_w_slot');
  const prefillWOff = Channel.from(range(0, n).map((p) => p % pageSize), dtype.u32).named('prefill_w_off');
  const prefillMask = new Channel([vocab], dtype.bool).named('prefill_mask');
  const firstOut = new Channel([1], dtype.i32).named('first_token');
  const prefillKvLen = Channel.from([n], dtype.u32).named('prefill_kv_len');

  const prefill = new ForwardPass(kind);
  prefill.embed(promptTokens, prefillIndptr);
  prefill.bindState(
    ws,
    { kvLen: prefillKvLen, pages: prefillPages, pageIndptr: prefillPageIndptr, wSlot: prefillWSlot, wOff: prefillWOff, positions: prefillPositions },
    rsWs,
  );
  prefill.epilogue(() => {
    const allowed = prefillMask.take();
    firstOut.put(reshape(maskedArgmax(intrinsics.logits(), allowed), [1]));
  });

  prefillMask.put(mask.unpackMask(constraint.mask(), vocab));
  const pipeline = new Pipeline();
  prefill.submit(pipeline);
  const first = firstOut.takeScalar();

  const generated = [first];
  constraint.acceptTokens([first]);

  if (!constraint.isTerminated() && generated.length < maxTokens) {
    const tokenIn = Channel.from([first], dtype.i32).named('token_in');
    const grammarMask = new Channel([vocab], dtype.bool).named('grammar_mask');
    const embedIndptr = Channel.from([0, 1], dtype.u32).named('embed_indptr');
    const positions = Channel.from([n], dtype.u32).named('positions');
    const pages = Channel.from(range(0, maxPages), dtype.u32).named('pages');
    const pageIndptr = Channel.from([0, divCeil(n + 1, pageSize)], dtype.u32).named('page_indptr');
    const wSlot = Channel.from([Math.floor(n / pageSize)], dtype.u32).named('w_slot');
    const wOff = Channel.from([n % pageSize], dtype.u32).named('w_off');
    const tokenOut = new Channel([1], dtype.i32).capacity(channelCapacity()).named('token_out');
    const kvLen = Channel.from([n + 1], dtype.u32).named('kv_len');

    const decode = new ForwardPass(kind);
    decode.embed(tokenIn, embedIndptr);
    decode.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions, writablePages: Math.floor(n / pageSize) }, rsWs);
    decode.epilogue(() => {
      const length = kvLen.take();
      const allowed = grammarMask.take();
      const token = reshape(maskedArgmax(intrinsics.logits(), allowed), [1]);
      const nextLength = length.add(1);
      const pageCount = nextLength.divCeil(pageSize);
      tokenIn.put(token);
      kvLen.put(nextLength);
      positions.put(length);
      wSlot.put(length.div(pageSize));
      wOff.put(length.rem(pageSize));
      pageIndptr.put(indptr(1, pageCount));
      tokenOut.put(token);
    });

    // Depth-1 by nature: fire k+1's mask needs fire k's token.
    const budget = maxTokens - generated.length;
    let submitted = 0;
    while (submitted < budget) {
      grammarMask.put(mask.unpackMask(constraint.mask(), vocab));
      decode.submit(pipeline);
      submitted += 1;
      const token = tokenOut.takeScalar();
      generated.push(token);
      constraint.acceptTokens([token]);
      if (constraint.isTerminated() || generated.length === maxTokens) break;
    }
  }
  pipeline.close();

  if (!constraint.isTerminated()) throw new Error(`JSON generation did not terminate within ${maxTokens} tokens`);

  const text = model.decode(generated);
  try {
    JSON.parse(text);
  } catch (e) {
    throw new Error(`constraint terminated with invalid JSON: ${e}; output=${JSON.stringify(text)}`);
  }
  return text;
}
