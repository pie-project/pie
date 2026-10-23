use super::*;

pub fn argmax(xs: &[&Value]) -> Value {
    let first = xs.first().expect("an argmax over at least one value");
    let r = first.rec();
    let y = r.fresh(tensor(first.rows(), xs.len() as u64, Dtype::I32));
    r.push(
        Sampling::Argmax {
            xs: xs.iter().map(|x| x.id()).collect(),
            y: y.id(),
        },
        xs,
    );
    y
}

pub fn top_k(x: &Value, k: u32) -> (Value, Value) {
    let r = x.rec();
    let values = r.fresh(tensor(x.rows(), u64::from(k), Dtype::F32));
    let indices = r.fresh(tensor(x.rows(), u64::from(k), Dtype::I32));
    r.push(
        Sampling::TopK {
            x: x.id(),
            k,
            values: values.id(),
            indices: indices.id(),
        },
        &[x],
    );
    (values, indices)
}

/// DSpark's Markov head over a drafted block's readouts
/// (`sampling.markov_walk`): the logits as biased, each row by the token
/// the row before it picked, and each row's pick (i32).
#[allow(clippy::too_many_arguments)]
pub fn markov_walk(
    logits: &Value,
    tokens: &Value,
    positions: &Value,
    ends: &Value,
    embed: &Weight,
    head: &Weight,
    block: u32,
) -> (Value, Value) {
    let r = logits.rec();
    let biased = r.fresh(logits.ty().clone());
    let picks = r.fresh(tensor(logits.rows(), 1u64, Dtype::I32));
    r.push(
        Sampling::MarkovWalk {
            logits: logits.id(),
            tokens: tokens.id(),
            positions: positions.id(),
            ends: ends.id(),
            embed: r.weight(embed),
            head: r.weight(head),
            block,
            biased: biased.id(),
            picks: picks.id(),
        },
        &[logits, tokens, positions, ends],
    );
    (biased, picks)
}

pub fn selector_walk(
    cand: &Value,
    unary: &Value,
    hp: Option<&Value>,
    tokens: &Value,
    pred: &Weight,
    succ: &Weight,
    first: u32,
) -> Value {
    let r = cand.rec();
    let picks = r.fresh(tensor(cand.rows(), 1u64, Dtype::I32));
    let mut deps: Vec<&Value> = vec![cand, unary];
    deps.extend(hp);
    deps.push(tokens);
    r.push(
        Sampling::SelectorWalk {
            cand: cand.id(),
            unary: unary.id(),
            hp: hp.map(Value::id),
            tokens: tokens.id(),
            pred: r.weight(pred),
            succ: r.weight(succ),
            first,
            picks: picks.id(),
        },
        &deps,
    );
    picks
}
