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

  const vocab = model.outputVocabSize();
  const ws = new WorkingSet();
  const constraint = new grammar.Matcher(grammar.Grammar.fromJsonSchema(schema));

  const head = [
    ...chat.systemUser('Generate only the requested JSON value, with no markdown or explanation.', promptText),
    ...chat.cue(),
  ];
  const last = head.pop();
  if (last === undefined) throw new Error('empty prompt');

  const pipeline = new Pipeline();
  if (head.length > 0) {
    const fill = new ForwardPass();
    fill.embed(Channel.from(head, dtype.i32), Channel.from([0, head.length], dtype.u32));
    fill.state([ws], { fold: 'all' });
    pipeline.submit(fill);
  }

  const tokenIn = Channel.from([last], dtype.i32);
  const grammarMask = new Channel([vocab], dtype.bool);
  const embedIndptr = Channel.from([0, 1], dtype.u32);
  const tokenOut = new Channel([1], dtype.i32).capacity(channelCapacity());

  const decode = new ForwardPass();
  decode.embed(tokenIn, embedIndptr);
  decode.state([ws], { fold: 'all' });
  decode.epilogue(() => {
    const allowed = grammarMask.take();
    const token = reshape(maskedArgmax(intrinsics.output(0, 'logits'), allowed), [1]);
    tokenIn.put(token);
    tokenOut.put(token);
  });

  // Depth-1 by nature: fire k+1's mask needs fire k's token.
  const generated = [];
  while (!constraint.isTerminated() && generated.length < maxTokens) {
    grammarMask.put(mask.unpackMask(constraint.mask(), vocab));
    pipeline.submit(decode);
    const token = tokenOut.takeScalar();
    generated.push(token);
    constraint.acceptTokens([token]);
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
