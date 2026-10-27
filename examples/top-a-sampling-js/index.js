// Top-a sampling — the JavaScript twin of `top-a-sampling`.
//
// `keep(x) iff p(x) >= a · p_max²`: one softmax, one reduceMax, one multiply,
// one comparison over the vocabulary, then a Gumbel-max draw over the masked
// logits.

import { chat, eta, model } from '@pie-project/inferlet';

const {
  Channel, ForwardPass, Pipeline, WorkingSet, broadcast, cast, channelCapacity, constant, dtype, ge, gumbelMax,
  intrinsics, iota, reduceMax, reduceSum, reshape, select, softmax,
} = eta;

/** The top-a keep-mask: [keep, keptCount, keptMass]. */
function topAKeep(logits, vocab, a) {
  const probs = softmax(logits);
  const pMax = reduceMax(probs);
  const threshold = constant(a, dtype.f32).mul(pMax.mul(pMax));
  const keep = ge(probs, broadcast(threshold, [vocab]));
  const zeros = broadcast(constant(0, dtype.f32), [vocab]);
  const keptMass = reshape(reduceSum(select(keep, probs, zeros)), [1]);
  const kept = reshape(reduceSum(cast(keep, dtype.f32)), [1]);
  return [keep, kept, keptMass];
}

function step(logits, vocab, a, temperature, rngState) {
  const scaled = temperature === 1.0 ? logits : logits.div(constant(temperature, dtype.f32));
  const [keep, kept, keptMass] = topAKeep(scaled, vocab, a);
  const negInf = broadcast(-Infinity, [vocab]);
  const masked = select(keep, scaled, negInf);
  return [gumbelMax(masked, rngState), kept, keptMass];
}

export function main(input) {
  const promptText = input.prompt ?? 'Write a short paragraph about top-a sampling.';
  const a = Number(input.a ?? 0.2);
  const temperature = Number(input.temperature ?? 1.0);
  const maxTokens = Number(input.max_tokens ?? 32);
  const seed = Number(input.seed ?? 0x7ce1);
  if (!(a > 0 && a <= 1)) throw new Error('a must be finite and in (0, 1]');
  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error('temperature must be finite and greater than 0');

  const vocab = model.outputVocabSize();
  const ws = new WorkingSet();
  const pipe = new Pipeline();

  // The model's opening (`<bos>` where it has one) before the raw text: a
  // gemma without it answers noise.
  const head = [...chat.prefix(), ...model.encode(promptText)];
  const last = head.pop();
  if (last === undefined) throw new Error('empty prompt');
  if (head.length > 0) {
    const fill = new ForwardPass();
    fill.embed(Channel.from(head, dtype.i32), Channel.from([0, head.length], dtype.u32));
    fill.state([ws], { fold: 'all' });
    pipe.submit(fill);
  }

  const cap = channelCapacity();
  const tokIn = Channel.from([last], dtype.i32);
  const rng = Channel.from([seed, 0], dtype.u32);
  const tokOut = new Channel([1], dtype.i32).capacity(cap);
  const s1Out = new Channel([1], dtype.f32).capacity(cap);
  const s2Out = new Channel([1], dtype.f32).capacity(cap);

  const fwd = new ForwardPass();
  fwd.embed(tokIn, Channel.from([0, 1], dtype.u32));
  fwd.state([ws], { fold: 'all' });
  fwd.epilogue(() => {
    const r = rng.take();
    const logits = intrinsics.output(0, 'logits');
    const [token, kept, mass] = step(logits, vocab, a, temperature, r);
    const rNext = r.add(iota(2));
    tokIn.put(token);
    tokOut.put(token);
    s1Out.put(kept);
    s2Out.put(mass);
    rng.put(rNext);
  });

  const generated = [];
  const s1 = [];
  const s2 = [];
  pipe.runAhead(fwd, maxTokens, () => {
    generated.push(tokOut.takeScalar());
    s1.push(s1Out.takeScalar());
    s2.push(s2Out.takeScalar());
    return true;
  });
  pipe.close();

  const mean = (v) => v.reduce((x, y) => x + y, 0) / Math.max(v.length, 1);
  const minS1 = s1.length ? Math.max(Math.min(...s1), 0) : 0;
  if (s1.length && minS1 === 0) throw new Error('top-a keep-set was empty — the peak token was masked out');
  return {
    sampler: 'top-a',
    text: model.decode(generated),
    count: generated.length,
    a,
    mean_kept: mean(s1),
    min_kept: Math.trunc(minS1),
    mean_mass: mean(s2),
    tokens: generated,
  };
}
