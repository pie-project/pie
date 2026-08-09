//! No-copy buffers over host pages, and buffers owned somewhere else.
//!
//! The claim that matters is that `Mapped` is not a copy. Anything that
//! copied the bytes into a Metal allocation would pass a test that only reads
//! the buffer back, so the test here writes from the GPU and reads the
//! ORIGINAL host allocation -- the pointer `alloc` returned, not
//! `buffer.contents()`.

#![allow(clippy::print_stdout)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;

use driver_metal::metal::page_size;
use driver_metal::{
    Compiler, Context, Error, Externals, Mapped, Pool, Region, Stepper, Tables,
};

const FILL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = out[gid] + gid + 1;
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

/// A page-aligned host allocation, freed when it goes.
///
/// Stands in for the loader's `mmap`. Page-aligned because
/// `newBufferWithBytesNoCopy:` requires it, which is the same requirement an
/// `mmap` satisfies for free.
struct Pages {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new(bytes: usize) -> Self {
        let page = usize::try_from(page_size()).expect("a page fits a usize");
        let layout = Layout::from_size_align(bytes.next_multiple_of(page), page).expect("layout");
        // SAFETY: the layout is non-zero and its alignment is a power of two.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).expect("host pages");
        Self { ptr, layout }
    }

    fn as_u32s(&self, count: usize) -> Vec<u32> {
        // SAFETY: the allocation is zeroed, at least `count * 4` bytes, and
        // `u32` has no invalid bit patterns.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<u32>(), count) }.to_vec()
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: exactly what `alloc_zeroed` was given in `new`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[test]
fn the_gpu_writes_into_the_hosts_own_pages() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler.compile(&context, FILL, "fill").expect("fill");

    let count = 256usize;
    let pages = Pages::new(count * 4);
    let len = pages.layout.size() as u64;

    // SAFETY: `pages` outlives `mapped` -- it is dropped after it, below.
    let mapped = unsafe { Mapped::new(&context, pages.ptr.cast(), len) }.expect("mapped");

    assert_eq!(
        mapped.contents(),
        pages.ptr.cast(),
        "Metal handed back a different address, so this is a copy"
    );
    assert_ne!(mapped.gpu_address(), 0);

    // Seed from the host through the mapping, so the kernel's `out[gid] +`
    // proves it read the host's bytes as well as wrote them.
    let seed: Vec<u8> = (0..count as u32)
        .flat_map(|i| (i * 10).to_le_bytes())
        .collect();
    // SAFETY: nothing is in flight and the slice is inside the mapping.
    unsafe { mapped.write(0, &seed) }.expect("seed");

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, mapped.gpu_address())
        .expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([count, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    // Read the ORIGINAL allocation, not the buffer.
    let want: Vec<u32> = (0..count as u32).map(|i| i * 10 + i + 1).collect();
    assert_eq!(
        pages.as_u32s(count),
        want,
        "the GPU's writes did not land in the host's pages, so the buffer copied them"
    );

    drop(mapped);
    drop(pages);
}

#[test]
fn a_misaligned_or_empty_mapping_is_refused_with_a_reason() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let pages = Pages::new(1);

    // SAFETY: the pointer is inside a live allocation; the call fails before
    // it is used.
    let err = unsafe { Mapped::new(&context, pages.ptr.cast(), 0) }.unwrap_err();
    assert!(format!("{err}").contains("zero bytes"), "{err}");

    let page = page_size();
    // SAFETY: 64 bytes into a page-sized allocation is still inside it, and
    // the call refuses before dereferencing.
    let offset = unsafe { pages.ptr.add(64) };
    let err = unsafe { Mapped::new(&context, offset.cast(), page - 64) }.unwrap_err();
    let text = format!("{err}");
    assert!(text.contains("not aligned"), "{text}");
    assert!(
        text.contains(&page.to_string()),
        "the message should name this host's page size: {text}"
    );
}

#[test]
fn an_external_buffer_is_counted_once_and_released_by_the_last_holder() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // A buffer this registry did not allocate. The pool owns it; `Externals`
    // is only being asked to keep it resident.
    let pool = Pool::new(1 << 20);
    let foreign = pool.acquire(&context, 4096).expect("foreign");
    let other = pool.acquire(&context, 4096).expect("other");

    let externals = Externals::new();
    assert!(externals.is_empty());

    let first = externals.insert(&context, foreign.buffer());
    let second = externals.insert(&context, foreign.buffer());
    assert_eq!(
        externals.len(),
        1,
        "two registrations of one buffer are one allocation in the set"
    );

    let third = externals.insert(&context, other.buffer());
    assert_eq!(externals.len(), 2);

    drop(first);
    assert_eq!(
        externals.len(),
        1 + 1,
        "the first holder to finish dropped the buffer out from under the second"
    );
    drop(second);
    assert_eq!(externals.len(), 1);
    drop(third);
    assert!(externals.is_empty());

    // The registry is shared, not copied: a clone sees the same buffers.
    let clone = externals.clone();
    let held = externals.insert(&context, foreign.buffer());
    assert_eq!(clone.len(), 1);
    drop(held);
    assert!(clone.is_empty());
}
