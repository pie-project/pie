//! Behavioural parity with the C++ `CublasHandle` — gate-cublas.
//!
//! The oracle in `tests/oracle/cublas_handle/` compiles the real
//! `gemm/gemm.cpp` (dispatchers discarded by `--gc-sections`) and drives
//! the handle wrapper through construction, stream rebinding and both
//! failure paths — including the math-mode failure that LEAKS the created
//! handle, which the port reproduces. This test replays the same script
//! and requires the transcripts to be byte-identical.
//!
//! Run `tests/oracle/cublas_handle/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash.

use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda_new::cuda::cublas::{CublasHandle, CublasOps};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x7171daf2be28cd47;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 18;

const SEP: char = '\u{1f}';

struct FakeCublas {
    out: String,
    case: String,
    next_handle: usize,
    fail_create: bool,
    fail_math: bool,
    last_stream: *mut c_void,
}

fn stream_name(s: *mut c_void) -> &'static str {
    match s as usize {
        0 => "s0",
        0xA0 => "sA",
        0xB0 => "sB",
        _ => "s?",
    }
}

impl FakeCublas {
    fn new() -> Self {
        Self {
            out: String::new(),
            case: String::new(),
            next_handle: 0,
            fail_create: false,
            fail_math: false,
            last_stream: std::ptr::null_mut(),
        }
    }

    fn note(&mut self, body: &str) {
        let case = self.case.clone();
        let _ = writeln!(self.out, "{case}{SEP}{body}");
    }
}

impl CublasOps for FakeCublas {
    type Handle = usize;

    fn create(&mut self) -> Result<usize, i32> {
        if self.fail_create {
            self.fail_create = false;
            self.note("create FAIL");
            return Err(1);
        }
        let h = self.next_handle;
        self.next_handle += 1;
        self.note(&format!("create h#{h}"));
        Ok(h)
    }

    fn destroy(&mut self, handle: usize) {
        self.note(&format!("destroy h#{handle}"));
    }

    fn set_stream(&mut self, handle: &usize, stream: *mut c_void) -> Result<(), i32> {
        self.note(&format!("set-stream h#{handle} {}", stream_name(stream)));
        self.last_stream = stream;
        Ok(())
    }

    fn get_stream(&mut self, handle: &usize) -> *mut c_void {
        let s = self.last_stream;
        self.note(&format!("get-stream h#{handle} -> {}", stream_name(s)));
        s
    }

    fn set_math_mode_tensor_op(&mut self, handle: &usize) -> Result<(), i32> {
        if self.fail_math {
            self.fail_math = false;
            self.note("math-mode FAIL");
            return Err(1);
        }
        self.note(&format!("math-mode h#{handle} mode=1"));
        Ok(())
    }
}

fn transcript() -> String {
    let mut ops = FakeCublas::new();
    let s_a = 0xA0 as *mut c_void;
    let s_b = 0xB0 as *mut c_void;

    // a. Default construction: no set-stream call at all.
    ops.case = "a-default".into();
    {
        ops.last_stream = std::ptr::null_mut();
        let mut h = CublasHandle::create(&mut ops, std::ptr::null_mut()).unwrap();
        let name = *h.handle().unwrap();
        ops.note(&format!("handle=h#{name}"));
        let s = stream_name(h.stream(&mut ops));
        ops.note(&format!("stream={s}"));
        h.release(&mut ops);
    }

    // b. Stream-bound construction, rebind, and the getter.
    ops.case = "b-stream".into();
    {
        let mut h = CublasHandle::create(&mut ops, s_a).unwrap();
        h.set_stream(&mut ops, s_b).unwrap();
        let s = stream_name(h.stream(&mut ops));
        ops.note(&format!("stream={s}"));
        h.release(&mut ops);
    }

    // c. Create fails: the error carries the status and the call name.
    ops.case = "c-create-fail".into();
    {
        ops.fail_create = true;
        match CublasHandle::<usize>::create(&mut ops, std::ptr::null_mut()) {
            Ok(mut h) => {
                h.release(&mut ops);
                ops.note("no-throw");
            }
            Err(e) => ops.note(&format!("threw {e}")),
        }
    }

    // d. Math mode fails: the created handle leaks — a create row with no
    //    destroy row, the C++'s own behaviour reproduced.
    ops.case = "d-math-fail".into();
    {
        ops.fail_math = true;
        match CublasHandle::<usize>::create(&mut ops, std::ptr::null_mut()) {
            Ok(mut h) => {
                h.release(&mut ops);
                ops.note("no-throw");
            }
            Err(e) => ops.note(&format!("threw {e}")),
        }
    }

    ops.out
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_port_reproduces_the_cpp_transcript() {
    let text = transcript();
    let rows = text.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — script shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("cublas_handle_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
