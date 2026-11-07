import { eta } from '@pie-project/inferlet';
const { ForwardPass, Pipeline, intrinsics, reduceArgmax } = eta;

export async function main(input) {
  const ws = input.ws;
  const pipe = new Pipeline();
  let tokens = input.tokens;
  const drawn = [];
  for (let i = 0; i < input.max_tokens; i++) {
    const fwd = new ForwardPass();
    fwd.embed(tokens);
    fwd.state([ws], { fold: 'all' });
    const out = fwd.epilogue(() => reduceArgmax(intrinsics.output(0, 'logits')));
    pipe.submit(fwd);
    tokens = [out.takeScalar()];
    drawn.push(tokens[0]);
  }
  pipe.close();
  ws.give('ws');
  return { tokens: drawn };
}
