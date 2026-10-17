// JSON generation constrained by a JSON Schema — the JavaScript twin of
// `json-schema-constrained-decoding`.
//
// The host grammar matcher advances after every accepted token and supplies
// the next allowed-token mask to an ETA `maskedArgmax` epilogue. Exercises
// the `grammar` + `mask` modules and a host-writer mask channel.

import { chat, eta, grammar, mask, model } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, WorkingSet, channelCapacity, dtype, intrinsics, maskedArgmax, reshape } = eta;

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
  const ws = new WorkingSet();
  const constraint = new grammar.Matcher(grammar.Grammar.fromJsonSchema(schema));

  let prompt = [
    ...chat.systemUser('Generate only the requested JSON value, with no markdown or explanation.', promptText),
    ...chat.cue(),
  ];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;

  const promptTokens = Channel.from(prompt, dtype.i32);
  const prefillIndptr = Channel.from([0, n], dtype.u32).named('prefill_indptr');
  const prefillMask = new Channel([vocab], dtype.bool).named('prefill_mask');
  const firstOut = new Channel([1], dtype.i32).named('first_token');

  const prefill = new ForwardPass();
  prefill.embed(promptTokens, prefillIndptr);
  prefill.state([ws], { fold: 'all' });
  prefill.epilogue(() => {
    const allowed = prefillMask.take();
    firstOut.put(reshape(maskedArgmax(intrinsics.logits(), allowed), [1]));
  });

  prefillMask.put(mask.unpackMask(constraint.mask(), vocab));
  const pipeline = new Pipeline();
  pipeline.submit(prefill);
  const first = firstOut.takeScalar();

  const generated = [first];
  constraint.acceptTokens([first]);

  if (!constraint.isTerminated() && generated.length < maxTokens) {
    const tokenIn = Channel.from([first], dtype.i32).named('token_in');
    const grammarMask = new Channel([vocab], dtype.bool).named('grammar_mask');
    const embedIndptr = Channel.from([0, 1], dtype.u32).named('embed_indptr');
    const tokenOut = new Channel([1], dtype.i32).capacity(channelCapacity()).named('token_out');

    const decode = new ForwardPass();
    decode.embed(tokenIn, embedIndptr);
    decode.state([ws], { fold: 'all' });
    decode.epilogue(() => {
      const allowed = grammarMask.take();
      const token = reshape(maskedArgmax(intrinsics.logits(), allowed), [1]);
      tokenIn.put(token);
      tokenOut.put(token);
    });

    // Depth-1 by nature: fire k+1's mask needs fire k's token.
    const budget = maxTokens - generated.length;
    let submitted = 0;
    while (submitted < budget) {
      grammarMask.put(mask.unpackMask(constraint.mask(), vocab));
      pipeline.submit(decode);
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
