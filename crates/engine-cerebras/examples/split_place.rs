//! The role-split placement on a recording device: every copy as
//! `sym x y w h n bits sha256(words)` a line, to compare two placements
//! (or a placement against a reference log) copy for copy.
//!
//! `split_place <prep dir> <prefix> <overrides k=v,...> <out log>`

use std::io::Write;

use engine_cerebras::sdk::{self, DataType, Order, Rect};
use engine_cerebras::spatial::device::Device;
use engine_cerebras::spatial::split::{SplitSpec, place};
use kernels_cerebras::spatial::emit::meta::Meta;
use kernels_cerebras::spatial::program::Program;
use sha2::{Digest, Sha256};

struct Log(std::io::BufWriter<std::fs::File>);

impl Log {
    fn line(&mut self, sym: &str, data: &[u32], r: Rect, n: usize, dt: DataType) {
        let mut h = Sha256::new();
        for w in data {
            h.update(w.to_le_bytes());
        }
        let bits = if dt == DataType::Bits16 { 16 } else { 32 };
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        writeln!(
            self.0,
            "{sym} {} {} {} {} {n} {bits} {hex}",
            r.x, r.y, r.w, r.h
        )
        .unwrap();
    }
}

impl Device for Log {
    fn memcpy_h2d(
        &mut self,
        s: &str,
        d: &[u32],
        r: Rect,
        n: usize,
        _: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        assert_eq!(d.len(), r.w as usize * r.h as usize * n, "{s}");
        self.line(s, d, r, n, dt);
        Ok(())
    }
    fn memcpy_d2h(
        &mut self,
        _: &str,
        out: &mut [u32],
        _: Rect,
        _: usize,
        _: Order,
        _: DataType,
    ) -> sdk::Result<()> {
        out.fill(0);
        Ok(())
    }
    fn memcpy_h2d_rowbcast(
        &mut self,
        s: &str,
        _: &[u32],
        _: Rect,
        _: usize,
        _: DataType,
    ) -> sdk::Result<()> {
        // (the role-split placement makes no row broadcasts)
        Err(sdk::Error::WrongKind(s.to_string(), sdk::RpcKind::Var))
    }
    fn memcpy_h2d_nonblock(
        &mut self,
        s: &str,
        d: Vec<u32>,
        r: Rect,
        n: usize,
        o: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.memcpy_h2d(s, &d, r, n, o, dt)
    }
    fn launch(&mut self, _: &str, _: &[u32]) -> sdk::Result<()> {
        Ok(())
    }
    fn launch_nonblock(&mut self, _: &str, _: &[u32]) -> sdk::Result<()> {
        Ok(())
    }
    fn wait_pending(&mut self) {}
    fn has_symbol(&self, _: &str) -> bool {
        true
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let spec = SplitSpec::parse(&format!("{}:{}:{}", a[1], a[2], a[3])).unwrap();
    let t0 = std::time::Instant::now();
    let meta = Meta {
        map: spec.meta.clone().into_iter().collect(),
    };
    let prog = Program::from_meta(&meta).unwrap();
    eprintln!(
        "program {:.1} s ({} x {})",
        t0.elapsed().as_secs_f64(),
        prog.floor.w,
        prog.floor.h
    );
    let mut log = Log(std::io::BufWriter::new(
        std::fs::File::create(&a[4]).unwrap(),
    ));
    place::place(&mut log, &prog, &spec.prep, &spec.prefix).unwrap();
    eprintln!("placed {:.1} s", t0.elapsed().as_secs_f64());
}
