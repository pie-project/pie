//! A layout's weights named by where they sit in it: `layers.3.q` is the
//! `q` of the fourth of `layers`. A package states what a weight is and
//! where it goes; its name follows.

use std::cell::RefCell;
use std::collections::HashMap;

use starlark::values::dict::{AllocDict, DictRef};
use starlark::values::list::{AllocList, ListRef};
use starlark::values::structs::{AllocStruct, StructRef};
use starlark::values::tuple::{AllocTuple, TupleRef};
use starlark::values::{Heap, Value as Star, ValueLike};

use crate::star::values::WeightValue;

thread_local! {
    /// While set, each renaming: a weight's stated name and its path.
    pub(crate) static RENAMED: RefCell<Option<Vec<(String, String)>>> =
        const { RefCell::new(None) };
}

/// `f`, and every weight it named by path: the name it was stated under and
/// the path it now goes by.
pub fn renamed<R>(f: impl FnOnce() -> R) -> (R, Vec<(String, String)>) {
    RENAMED.with(|r| *r.borrow_mut() = Some(Vec::new()));
    let out = f();
    let log = RENAMED.with(|r| r.borrow_mut().take()).unwrap_or_default();
    (out, log)
}

/// What makes two weights one: the object, or a name a Rust caller stated.
#[derive(PartialEq, Eq, Hash)]
enum Same {
    Object(u64),
    Stated(String),
}

/// Which layout value `v` is, as a weight turned by it holds it.
pub(crate) fn origin(v: Star<'_>) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    std::hash::Hash::hash(&v.identity(), &mut h);
    std::hash::Hasher::finish(&h)
}

fn same(w: &crate::Weight, origin: u64) -> Same {
    if w.name.is_empty() {
        Same::Object(origin)
    } else {
        Same::Stated(w.name.clone())
    }
}

/// The layout `m` with each weight named by its path in it. A weight met at
/// two paths (a head tied to its embedding) is one weight: the first path
/// names it. A turned weight's signs go by the name their own path gives.
pub(crate) fn named<'v>(heap: Heap<'v>, m: Star<'v>) -> anyhow::Result<Star<'v>> {
    let mut seen = HashMap::new();
    walk(heap, m, "", &mut |v, w, path| {
        seen.entry(same(w, origin(v)))
            .or_insert_with(|| path.to_string());
        v
    });
    let mut stray = None;
    let m = walk(heap, m, "", &mut |v, w, path| {
        let name = seen[&same(w, origin(v))].clone();
        RENAMED.with(|r| {
            if let Some(log) = r.borrow_mut().as_mut() {
                log.push((w.name.clone(), name.clone()));
            }
        });
        if name == w.name && w.turn.is_none() {
            return v;
        }
        let mut renamed = w.clone();
        renamed.name = name;
        if let Some(t) = renamed.turn.as_deref_mut() {
            match seen.get(&same(&t.signs, t.origin)) {
                Some(signs) => t.signs.name = signs.clone(),
                None => stray = Some(path.to_string()),
            }
        }
        heap.alloc(WeightValue(renamed))
    });
    match stray {
        Some(path) => anyhow::bail!("`{path}` is turned by signs its layout does not hold"),
        None => Ok(m),
    }
}

fn at(path: &str, part: &str) -> String {
    if path.is_empty() {
        part.to_string()
    } else {
        format!("{path}.{part}")
    }
}

/// `v` with each weight in it what `weight` makes of it at its path.
fn walk<'v>(
    heap: Heap<'v>,
    v: Star<'v>,
    path: &str,
    weight: &mut dyn FnMut(Star<'v>, &crate::Weight, &str) -> Star<'v>,
) -> Star<'v> {
    if let Some(w) = v.downcast_ref::<WeightValue>() {
        return weight(v, &w.0, path);
    }
    if let Some(s) = StructRef::from_value(v) {
        let fields: Vec<(String, Star<'v>)> = s
            .iter()
            .map(|(k, x)| {
                let k = k.as_str().to_string();
                let x = walk(heap, x, &at(path, &k), weight);
                (k, x)
            })
            .collect();
        return heap.alloc(AllocStruct(fields));
    }
    if let Some(l) = ListRef::from_value(v) {
        let items: Vec<Star<'v>> = l
            .iter()
            .enumerate()
            .map(|(i, x)| walk(heap, x, &at(path, &i.to_string()), weight))
            .collect();
        return heap.alloc(AllocList(items));
    }
    if let Some(t) = TupleRef::from_value(v) {
        let items: Vec<Star<'v>> = t
            .iter()
            .enumerate()
            .map(|(i, x)| walk(heap, x, &at(path, &i.to_string()), weight))
            .collect();
        return heap.alloc(AllocTuple(items));
    }
    if let Some(d) = DictRef::from_value(v) {
        let items: Vec<(Star<'v>, Star<'v>)> = d
            .iter()
            .map(|(k, x)| {
                let part = k.unpack_str().map_or_else(|| k.to_string(), str::to_string);
                (k, walk(heap, x, &at(path, &part), weight))
            })
            .collect();
        return heap.alloc(AllocDict(items));
    }
    v
}

/// The weights the layout `m` states the values of, each once.
pub(crate) fn stated<'v>(m: Star<'v>) -> Vec<crate::Weight> {
    fn gather<'v>(v: Star<'v>, out: &mut Vec<crate::Weight>) {
        if let Some(w) = v.downcast_ref::<WeightValue>() {
            if w.0.values.is_some() && !out.iter().any(|o| o.name == w.0.name) {
                out.push(w.0.clone());
            }
        } else if let Some(s) = StructRef::from_value(v) {
            s.iter().for_each(|(_, x)| gather(x, out));
        } else if let Some(l) = ListRef::from_value(v) {
            l.iter().for_each(|x| gather(x, out));
        } else if let Some(t) = TupleRef::from_value(v) {
            t.iter().for_each(|x| gather(x, out));
        } else if let Some(d) = DictRef::from_value(v) {
            d.iter().for_each(|(_, x)| gather(x, out));
        }
    }
    let mut out = Vec::new();
    gather(m, &mut out);
    out
}
