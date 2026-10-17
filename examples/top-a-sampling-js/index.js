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
  if (maxTokens === 0) return { sampler: 'top-a', text: '', count: 0, a, mean_kept: 0, min_kept: 0, mean_mass: 0 };

  // The model's opening (`<bos>` where it has one) before the raw text: a
  // gemma without it answers noise.
  let prompt = [...chat.prefix(), ...model.encode(promptText)];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;

  const generated = [];
  const s1 = [];
  const s2 = [];

  // ── PREFILL FIRE (N-wide) ──
  const toksP = Channel.from(prompt, dtype.i32).named('toks_p');
  const embedIndptrP = Channel.from([0, n], dtype.u32).named('embed_indptr_p');
  const rngP = Channel.from([seed, 0], dtype.u32).named('rng_p');
  const tokOutP = new Channel([1], dtype.i32).named('tok_out_p');
  const s1OutP = new Channel([1], dtype.f32).named('s1_out_p');
  const s2OutP = new Channel([1], dtype.f32).named('s2_out_p');

  const fwdP = new ForwardPass();
  fwdP.embed(toksP, embedIndptrP);
  fwdP.state([ws], { fold: 'all' });
  fwdP.epilogue(() => {
    const r = rngP.take();
    const logits = intrinsics.logits();
    const [token, kept, mass] = step(logits, vocab, a, temperature, r);
    const rNext = r.add(iota(2));
    tokOutP.put(token);
    s1OutP.put(kept);
    s2OutP.put(mass);
    rngP.put(rNext);
  });

  const pipe = new Pipeline();
  pipe.submit(fwdP);
  generated.push(tokOutP.takeScalar());
  s1.push(s1OutP.takeScalar());
  s2.push(s2OutP.takeScalar());

  // ── DECODE LOOP (1-wide, run-ahead) ──
  if (generated.length < maxTokens) {
    const cap = channelCapacity();
    const tokIn = Channel.from([generated[0]], dtype.i32).named('tok_in');
    const rng = Channel.from([seed ^ 0x5bd1, 0], dtype.u32).named('rng');
    const tokOut = new Channel([1], dtype.i32).capacity(cap).named('tok_out');
    const s1Out = new Channel([1], dtype.f32).capacity(cap).named('s1_out');
    const s2Out = new Channel([1], dtype.f32).capacity(cap).named('s2_out');
    const lane1 = Channel.from([0, 1], dtype.u32).named('embed_indptr');

    const fwd = new ForwardPass();
    fwd.embed(tokIn, lane1);
    fwd.state([ws], { fold: 'all' });
    fwd.epilogue(() => {
      const r = rng.take();
      const logits = intrinsics.logits();
      const [token, kept, mass] = step(logits, vocab, a, temperature, r);
      const rNext = r.add(iota(2));
      tokIn.put(token);
      tokOut.put(token);
      s1Out.put(kept);
      s2Out.put(mass);
      rng.put(rNext);
    });

    pipe.runAhead(fwd, maxTokens - 1, () => {
      generated.push(tokOut.takeScalar());
      s1.push(s1Out.takeScalar());
      s2.push(s2Out.takeScalar());
      return true;
    });
  }
  pipe.close();

  const meanS1 = s1.reduce((x, y) => x + y, 0) / s1.length;
  const meanS2 = s2.reduce((x, y) => x + y, 0) / s2.length;
  const minS1 = Math.max(Math.min(...s1), 0);
  if (minS1 === 0) throw new Error('top-a keep-set was empty — the peak token was masked out');
  return {
    sampler: 'top-a',
    text: model.decode(generated),
    count: generated.length,
    a,
    mean_kept: meanS1,
    min_kept: Math.trunc(minS1),
    mean_mass: meanS2,
    tokens: generated,
  };
}
