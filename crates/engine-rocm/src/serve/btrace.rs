use std::cell::RefCell;
use std::time::Instant;

fn on() -> bool {
    super::diag::on().boundary_trace
}

thread_local! {
    static MARKS: RefCell<Vec<(&'static str, Instant)>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn mark(label: &'static str) {
    if !on() {
        return;
    }
    MARKS.with(|marks| marks.borrow_mut().push((label, Instant::now())));
}

pub(crate) fn flush(seq: u64) {
    if !on() {
        return;
    }
    MARKS.with(|marks| {
        let mut marks = marks.borrow_mut();
        if marks.is_empty() {
            return;
        }
        let origin = marks[0].1;
        let mut line = format!("[boundary] seq={seq}");
        let mut prev = origin;
        for (label, at) in marks.iter().skip(1) {
            let us = at.duration_since(prev).as_micros();
            line.push_str(&format!(" {label}={us}"));
            prev = *at;
        }
        let total = prev.duration_since(origin).as_micros();
        line.push_str(&format!(" total={total}us"));
        eprintln!("{line}");
        marks.clear();
    });
}
