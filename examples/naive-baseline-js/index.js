// Naive text completion — the JavaScript twin of `naive-baseline`.
//
// One N-wide prefill fire, then a device-carried decode loop driven by
// `runAhead`, which keeps the runtime's run-ahead window full ahead of the
// host drain. The epilogue temperature-scales the logits and draws a
// Gumbel-max sample; `stats` adds the two extra `[1]` f32 drains. Traces to
// the same container bytes as the Rust inferlet.

import { chat, model, eta } from '@pie-project/inferlet';

const {
  Channel,
  ForwardPass,
  Pipeline,
  WorkingSet,
  cast,
  channelCapacity,
  dtype,
  gumbelMax,
  intrinsics,
  iota,
  reshape,
} = eta;

/** One sampling step: temperature, then a Gumbel-max draw over the vocab. */
function step(logits, temperature, rngState) {
  const scaled = temperature === 1.0 ? logits : logits.div(temperature);
  return gumbelMax(scaled, rngState);
}

export function main(input) {
  const promptText = input.prompt ?? 'Write a short paragraph about naive sampling.';
  const temperature = Number(input.temperature ?? 1.0);
  const maxTokens = Number(input.max_tokens ?? 32);
  const seed = Number(input.seed ?? 0x7ce1);
  const wantStats = Boolean(input.stats ?? false);

  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error('temperature must be finite and greater than 0');

  const ws = new WorkingSet();

  if (maxTokens === 0) return { sampler: 'naive-baseline-js', text: '', tokens: [], count: 0, stats: wantStats };

  let prompt = [...chat.prefix(), ...model.encode(promptText)];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;

  const generated = [];
  const pipe = new Pipeline();

  const toksP = Channel.from(prompt, dtype.i32);
  const embedIndptrP = Channel.from([0, n], dtype.u32);
  const rngP = Channel.from([seed, 0], dtype.u32);
  const tokOutP = new Channel([1], dtype.i32);
  const s1OutP = new Channel([1], dtype.f32);
  const s2OutP = new Channel([1], dtype.f32);

  const fwdP = new ForwardPass();
  fwdP.embed(toksP, embedIndptrP);
  fwdP.state([ws], { fold: 'all' });
  fwdP.epilogue(() => {
    const r = rngP.take();
    const logits = intrinsics.output(0, 'logits');
    const token = step(logits, temperature, r);
    const rNext = r.add(iota(2));
    tokOutP.put(token);
    if (wantStats) {
      const mirror = reshape(cast(token, dtype.f32), [1]);
      s1OutP.put(mirror);
      s2OutP.put(mirror);
    }
    rngP.put(rNext);
  });
  pipe.submit(fwdP);
  const g0 = tokOutP.takeScalar();
  if (wantStats) {
    s1OutP.takeHost();
    s2OutP.takeHost();
  }
  generated.push(g0);

  // ── DECODE LOOP (1-wide, run-ahead). ──
  if (generated.length < maxTokens) {
    const cap = channelCapacity();
    const tokIn = Channel.from([g0], dtype.i32);
    const rng = Channel.from([seed ^ 0x5bd1, 0], dtype.u32);
    const tokOut = new Channel([1], dtype.i32).capacity(cap);
    const s1Out = new Channel([1], dtype.f32).capacity(cap);
    const s2Out = new Channel([1], dtype.f32).capacity(cap);
    const lane1 = Channel.from([0, 1], dtype.u32);

    const fwd = new ForwardPass();
    fwd.embed(tokIn, lane1);
    fwd.state([ws], { fold: 'all' });
    fwd.epilogue(() => {
      const r = rng.take();
      const logits = intrinsics.output(0, 'logits');
      const token = step(logits, temperature, r);

      const rNext = r.add(iota(2));

      tokIn.put(token);
      tokOut.put(token);
      if (wantStats) {
        const mirror = reshape(cast(token, dtype.f32), [1]);
        s1Out.put(mirror);
        s2Out.put(mirror);
      }
      rng.put(rNext);
    });

    const budget = maxTokens - 1;
    pipe.runAhead(fwd, budget, () => {
      const t = tokOut.takeScalar();
      if (wantStats) {
        s1Out.takeHost();
        s2Out.takeHost();
      }
      generated.push(t);
      return true;
    });
  }
  pipe.close();

  return {
    sampler: 'naive-baseline-js',
    text: model.decode(generated),
    tokens: generated,
    count: generated.length,
    stats: wantStats,
  };
}
