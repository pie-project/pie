use alloc::vec::Vec;
use core::cell::RefCell;

use eta_ir::registry::OutputDecl;

#[derive(Clone, Debug)]
struct TraceConstants {
    vocab: u32,
    page_size: u32,
    /// Each lane's outputs, in lane order.
    outputs: Vec<Vec<OutputDecl>>,
}

impl Default for TraceConstants {
    fn default() -> Self {
        TraceConstants {
            vocab: 32_000,
            page_size: 16,
            outputs: Vec::new(),
        }
    }
}

thread_local! {
    static MODEL: RefCell<TraceConstants> = RefCell::new(TraceConstants::default());
}

pub(crate) fn with_constants<R>(
    vocab: u32,
    page_size: u32,
    outputs: Vec<Vec<OutputDecl>>,
    f: impl FnOnce() -> R,
) -> R {
    let previous = MODEL.with(|model| {
        model.replace(TraceConstants {
            vocab,
            page_size,
            outputs,
        })
    });
    let result = f();
    MODEL.with(|model| model.replace(previous));
    result
}

pub(crate) fn vocab() -> u32 {
    MODEL.with(|m| m.borrow().vocab)
}
pub(crate) fn page_size() -> u32 {
    MODEL.with(|m| m.borrow().page_size)
}
pub(crate) fn output(lane: u32, name: &str) -> Option<OutputDecl> {
    MODEL.with(|m| {
        m.borrow()
            .outputs
            .get(lane as usize)?
            .iter()
            .find(|decl| decl.name == name)
            .cloned()
    })
}
