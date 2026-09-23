// Greedy text completion, one forward pass per token, driven from the host.
//
// The shortest complete inferlet: the README's example. `text-completion-js`
// is the same program with chunked prefill and a device loop-carried decode.

import { chat, eta, model } from '@pie-project/inferlet';
const { ForwardPass, Pipeline, WorkingSet, intrinsics, reduceArgmax } = eta;

export function main(input) {
  const tokens = [...chat.prefix(), ...model.encode(input.prompt ?? 'The capital of France is')];
  const maxTokens = Number(input.max_tokens ?? 8);
  const ws = new WorkingSet({ tokens: tokens.length + maxTokens }); // KV pages (and recurrent state on a hybrid model)
  const pipe = new Pipeline();

  let done = 0; // tokens already in the KV cache
  for (let i = 0; i < maxTokens; i++) {
    const fwd = new ForwardPass();
    fwd.embed(tokens.slice(done));                       // the new tokens
    fwd.bindState(ws, ws.geometry(done, tokens.length));
    const out = fwd.epilogue(() => reduceArgmax(intrinsics.logits())); // runs on the device
    pipe.submit(fwd);
    done = tokens.length;
    tokens.push(out.takeScalar());
  }
  pipe.close();

  return { text: model.decode(tokens.slice(-maxTokens)) };
}
