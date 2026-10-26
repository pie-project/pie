// Greedy text completion, one forward pass per token, driven from the host.
//
// The shortest complete inferlet: the README's example. `text-completion-js`
// is the same program with a device loop-carried decode.

import { chat, eta, model } from '@pie-project/inferlet';
const { ForwardPass, Pipeline, WorkingSet, intrinsics, reduceArgmax } = eta;

export function main(input) {
  const tokens = [...chat.prefix(), ...model.encode(input.prompt ?? 'The capital of France is')];
  const maxTokens = Number(input.max_tokens ?? 8);
  const ws = new WorkingSet(); // the sequence's state, whatever the model keeps
  const pipe = new Pipeline();

  let done = 0; // tokens already in the working set
  for (let i = 0; i < maxTokens; i++) {
    const fwd = new ForwardPass();
    fwd.embed(tokens.slice(done));                       // the new tokens
    fwd.state([ws], { fold: 'all' });                    // appended, and final
    const out = fwd.epilogue(() => reduceArgmax(intrinsics.output(0, 'logits'))); // runs on the device
    pipe.submit(fwd);
    done = tokens.length;
    tokens.push(out.takeScalar());
  }
  pipe.close();

  return { text: model.decode(tokens.slice(-maxTokens)) };
}
