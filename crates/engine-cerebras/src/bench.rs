//! A bench for kernel tests: records what kernel entries emit into one
//! program, compiles it with `cslc`, runs it on the fabric simulator through
//! the SDK binding, and hands the written buffers back as f32.
//!
//! `run` returns `Ok(false)` when the SDK toolchain is not on this box, so a
//! test returns early instead of failing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use dtype::Dtype;
use kernels_cerebras::plan::Plan;
use kernels_cerebras::{Cx, Emit, Error, Tensor};

use crate::fabric::{Emitting, Fabric, Operands, Placed, Planner, Program, Rendered};
use crate::sdk::{Arch, Compile, Cslc, Sdk};

/// One simulator at a time on this box: runs are serialised across threads
/// here and each one gets its own `fabric-run` process, because the SDK's
/// fabric model can be created only once per process.
static SIMULATOR: Mutex<()> = Mutex::new(());

fn simulator_lock() -> MutexGuard<'static, ()> {
    SIMULATOR.lock().expect("simulator lock poisoned")
}

/// Rounds to bf16, nearest even, and back.
pub fn round_bf16(v: f32) -> f32 {
    let u = v.to_bits();
    let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

/// Asserts `got` is within `atol + rtol * |want|` of `want`, elementwise.
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, want {w} (tol {tol})\n got: {got:?}\nwant: {want:?}"
        );
    }
}

/// Host-side buffers keyed by handle, plus the last rendered program.
#[derive(Default)]
pub struct Bench {
    next: u32,
    /// Raw 32-bit words per handle: f32 bits or i32 bits by dtype.
    data: HashMap<u32, Vec<u32>>,
    /// The sources of the last `run`, for a failing test to print.
    pub last: Option<Rendered>,
    /// The last program's phases as rendered for the fabric (what ran).
    pub phases: Vec<Rendered>,
    /// Where the last program was compiled, kept when `PIE_CEREBRAS_KEEP` is set.
    pub last_dir: Option<PathBuf>,
}

impl Bench {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc(&mut self, dtype: Dtype, rows: u32, width: u32, words: Vec<u32>) -> Tensor {
        assert_eq!(words.len(), rows as usize * width as usize, "data length");
        self.next += 1;
        let t = Tensor::new(self.next, rows, width, dtype);
        self.data.insert(t.buf, words);
        t
    }

    /// A bf16 tensor holding `data` rounded to bf16.
    pub fn bf16(&mut self, rows: u32, width: u32, data: &[f32]) -> Tensor {
        let v = data.iter().map(|x| round_bf16(*x).to_bits()).collect();
        self.alloc(Dtype::Bf16, rows, width, v)
    }

    pub fn f32(&mut self, rows: u32, width: u32, data: &[f32]) -> Tensor {
        self.alloc(
            Dtype::F32,
            rows,
            width,
            data.iter().map(|x| x.to_bits()).collect(),
        )
    }

    pub fn i32(&mut self, rows: u32, width: u32, data: &[i32]) -> Tensor {
        self.alloc(
            Dtype::I32,
            rows,
            width,
            data.iter().map(|x| *x as u32).collect(),
        )
    }

    /// A byte tensor; each byte travels as one 32-bit word on this backend.
    pub fn u8(&mut self, rows: u32, width: u32, data: &[u8]) -> Tensor {
        self.alloc(
            Dtype::U8,
            rows,
            width,
            data.iter().map(|x| u32::from(*x)).collect(),
        )
    }

    pub fn zeros(&mut self, dtype: Dtype, rows: u32, width: u32) -> Tensor {
        self.alloc(dtype, rows, width, vec![0; rows as usize * width as usize])
    }

    /// The host copy of a float `t` (after `run`, what the device wrote).
    pub fn read_f32(&self, t: Tensor) -> Vec<f32> {
        self.data[&t.buf]
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect()
    }

    /// The host copy of an integer `t`.
    pub fn read_i32(&self, t: Tensor) -> Vec<i32> {
        self.data[&t.buf].iter().map(|w| *w as i32).collect()
    }

    /// Traces `body`, compiles and runs it. `Ok(false)` means no toolchain.
    pub fn run(
        &mut self,
        body: impl FnOnce(&kernels_cerebras::Ctx<'_>) -> Result<(), Error>,
    ) -> Result<bool, String> {
        let recorder = Recorder::default();
        body(&recorder).map_err(|e| e.to_string())?;
        let program = recorder.into_program();
        let rendered = program.render();
        let phases = program.render_phases();
        self.last = Some(rendered);
        self.phases = phases.clone();

        if !Cslc::available() {
            return Ok(false);
        }
        if !Sdk::available() {
            eprintln!("no SDK runtime at {}", Sdk::default_lib_dir().display());
            return Ok(false);
        }

        let _one_at_a_time = simulator_lock();
        let dir = tempfile::Builder::new()
            .prefix("pie-cerebras-bench-")
            .tempdir()
            .map_err(|e| e.to_string())?;
        let root = dir.path().to_path_buf();
        let cslc = Cslc::find().map_err(|e| e.to_string())?;
        let mut compiled = Vec::new();
        for (at, r) in phases.iter().enumerate() {
            if r.manifest.host.is_some() {
                compiled.push((PathBuf::new(), r.manifest.clone()));
                continue;
            }
            let pdir = root.join(format!("phase{at}"));
            std::fs::create_dir_all(&pdir).map_err(|e| e.to_string())?;
            std::fs::write(pdir.join("gather.csl"), &r.layout).map_err(|e| e.to_string())?;
            std::fs::write(pdir.join("pe.csl"), &r.pe).map_err(|e| e.to_string())?;
            let out = pdir.join("out");
            let (w, h) = r.manifest.rect;
            let compile = Compile::memcpy(Arch::Wse3, pdir.join("gather.csl"), w, h, &out);
            cslc.compile(&compile)
                .map_err(|e| format!("{e}\n--- layout.csl\n{}\n--- pe.csl\n{}", r.layout, r.pe))?;
            compiled.push((out, r.manifest.clone()));
        }

        // The host state: every buffer's words.
        let mut state: HashMap<String, Vec<u32>> = HashMap::new();
        for e in program.exports() {
            // Buffers the kernels declared themselves (lane headers) start as zeros.
            let words = self
                .data
                .get(&e.buf)
                .cloned()
                .unwrap_or_else(|| vec![0; e.rows as usize * e.width as usize]);
            state.insert(e.name.clone(), words);
        }
        crate::device::run_phases(
            &crate::device::fabric_run().map_err(|e| e.to_string())?,
            "wse3",
            None,
            &compiled,
            &mut state,
            &root,
            None,
            &std::collections::HashSet::new(),
            &[],
        )
        .map_err(|e| {
            format!(
                "{e}\n--- pe.csl (whole)\n{}",
                self.last.as_ref().map_or("", |r| r.pe.as_str())
            )
        })?;
        for e in program.exports() {
            if let (Some(words), true) = (state.remove(&e.name), self.data.contains_key(&e.buf)) {
                self.data.insert(e.buf, words);
            }
        }

        if std::env::var_os("PIE_CEREBRAS_KEEP").is_some() {
            let kept = dir.keep();
            eprintln!("kept {}", kept.display());
            self.last_dir = Some(kept);
        }
        Ok(true)
    }
}

/// The bench's operands: every handle is one exported buffer, `b{handle}`.
#[derive(Debug)]
pub struct Roots {
    pub fabric: Fabric,
}

impl Operands for Roots {
    fn read(&mut self, p: &mut Program, t: Tensor) -> Result<Placed, Error> {
        let whole = p.read("read", t)?;
        Ok(Placed {
            offset: 0,
            rows: whole.rows,
            whole,
        })
    }

    fn write(&mut self, p: &mut Program, t: Tensor) -> Result<Placed, Error> {
        let whole = p.write("write", t)?;
        Ok(Placed {
            offset: 0,
            rows: whole.rows,
            whole,
        })
    }

    /// Every bf16 handle a kernel asks for packed is packed (the bench
    /// standing in for the engine's weights): a weight under its own
    /// packed name, a pool (asked through `Cx::pool_half`) under its plain
    /// name, read and written.
    fn read_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<Placed>, Error> {
        if !self.fabric.half_weights || t.dtype != Dtype::Bf16 || !t.width.is_multiple_of(2) {
            return Ok(None);
        }
        let whole = p.declare_packed(&format!("b{}h", t.buf), t.dtype, t.rows, t.width)?;
        Ok(Some(Placed {
            offset: 0,
            rows: whole.rows,
            whole,
        }))
    }

    fn pool_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<Placed>, Error> {
        if !self.fabric.half_pool || t.dtype != Dtype::Bf16 || !t.width.is_multiple_of(2) {
            return Ok(None);
        }
        let whole = p.declare_packed(&format!("b{}", t.buf), t.dtype, t.rows, t.width)?;
        Ok(Some(Placed {
            offset: 0,
            rows: whole.rows,
            whole,
        }))
    }
}

/// A recording sink over one program: what a bench or a probe emits into.
/// [`Emit::emit`] opens a phase per call.
pub struct Recorder {
    program: std::cell::RefCell<Program>,
    operands: std::cell::RefCell<Box<dyn Operands>>,
    plan: Planner,
}

impl Recorder {
    pub fn new(operands: impl Operands + 'static, fabric: Fabric) -> Self {
        Recorder {
            program: std::cell::RefCell::new(Program::new()),
            operands: std::cell::RefCell::new(Box::new(operands)),
            plan: Planner { fabric },
        }
    }

    pub fn into_program(self) -> Program {
        self.program.into_inner()
    }
}

impl Default for Recorder {
    /// Over the bench's operands, on the terms the environment sets.
    fn default() -> Self {
        let fabric = crate::fabric::terms();
        Self::new(Roots { fabric }, fabric)
    }
}

impl Emit for Recorder {
    fn plan(&self) -> &dyn Plan {
        &self.plan
    }

    fn emit(&self, body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), Error>) -> Result<(), Error> {
        let mut program = self.program.borrow_mut();
        let mut operands = self.operands.borrow_mut();
        let mut sink = Emitting::new(&mut program, operands.as_mut(), &self.plan);
        let mut cx = Cx::new(&mut sink, "op");
        body(&mut cx)?;
        cx.finish();
        Ok(())
    }
}
