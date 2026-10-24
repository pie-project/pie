//! A layout's weights named by where they sit in it: `layers.3.q` is the
//! `q` of the fourth of `layers`. A package states what a weight is and
//! where it goes; its name follows.

use std::cell::RefCell;
use std::collections::HashMap;

use starlark::values::dict::{AllocDict, DictRef};
use starlark::values::list::{AllocList, ListRef};
use starlark::values::structs::{AllocStruct, StructRef};
use starlark::values::tuple::{AllocTuple, TupleRef};
use starlark::values::{Heap, Value as Star, ValueIdentity, ValueLike};

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
enum Same<'v> {
    Object(ValueIdentity<'v>),
    Stated(String),
}

/// The layout `m` with each weight named by its path in it. A weight met at
/// two paths (a head tied to its embedding) is one weight: the first path
/// names it.
pub(crate) fn named<'v>(heap: Heap<'v>, m: Star<'v>) -> Star<'v> {
    walk(heap, m, "", &mut HashMap::new())
}

fn at(path: &str, part: &str) -> String {
    if path.is_empty() {
        part.to_string()
    } else {
        format!("{path}.{part}")
    }
}

fn walk<'v>(
    heap: Heap<'v>,
    v: Star<'v>,
    path: &str,
    seen: &mut HashMap<Same<'v>, String>,
) -> Star<'v> {
    if let Some(w) = v.downcast_ref::<WeightValue>() {
        let same = if w.0.name.is_empty() {
            Same::Object(v.identity())
        } else {
            Same::Stated(w.0.name.clone())
        };
        let name = seen.entry(same).or_insert_with(|| path.to_string()).clone();
        RENAMED.with(|r| {
            if let Some(log) = r.borrow_mut().as_mut() {
                log.push((w.0.name.clone(), name.clone()));
            }
        });
        if name == w.0.name {
            return v;
        }
        let mut renamed = w.0.clone();
        renamed.name = name;
        return heap.alloc(WeightValue(renamed));
    }
    if let Some(s) = StructRef::from_value(v) {
        let fields: Vec<(String, Star<'v>)> = s
            .iter()
            .map(|(k, x)| {
                let k = k.as_str().to_string();
                let x = walk(heap, x, &at(path, &k), seen);
                (k, x)
            })
            .collect();
        return heap.alloc(AllocStruct(fields));
    }
    if let Some(l) = ListRef::from_value(v) {
        let items: Vec<Star<'v>> = l
            .iter()
            .enumerate()
            .map(|(i, x)| walk(heap, x, &at(path, &i.to_string()), seen))
            .collect();
        return heap.alloc(AllocList(items));
    }
    if let Some(t) = TupleRef::from_value(v) {
        let items: Vec<Star<'v>> = t
            .iter()
            .enumerate()
            .map(|(i, x)| walk(heap, x, &at(path, &i.to_string()), seen))
            .collect();
        return heap.alloc(AllocTuple(items));
    }
    if let Some(d) = DictRef::from_value(v) {
        let items: Vec<(Star<'v>, Star<'v>)> = d
            .iter()
            .map(|(k, x)| {
                let part = k.unpack_str().map_or_else(|| k.to_string(), str::to_string);
                (k, walk(heap, x, &at(path, &part), seen))
            })
            .collect();
        return heap.alloc(AllocDict(items));
    }
    v
}
