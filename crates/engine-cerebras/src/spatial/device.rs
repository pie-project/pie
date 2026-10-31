//! The wafer as the host drives it: one trait for every host↔wafer call the
//! spatial engine makes (copies in and out at 16 or 32 bits, launches, and
//! what the program exports), implemented by the SDK runtime and, in tests,
//! by a recorder that keeps the call sequence and a memory of what was
//! written.
//!
//! Everything that touches the device takes `&mut dyn Device`: weight
//! placement, route tables, page copies, the VM's loads and every launch. A
//! test can then run the host side of any design without a fabric and pin
//! exactly what it would send.

use crate::sdk::{self, DataType, Order, Rect, Runtime};

type R<T> = std::result::Result<T, String>;

/// A loaded wafer program the host copies to and from and launches.
///
/// The copy methods take the SDK's arguments as they are (`elems_per_pe`
/// elements on each PE of `rect`, row-major); a 16-bit copy moves one
/// element a word, in the word's low half.
pub trait Device {
    /// `data` into the exported variable `symbol` on every PE of `rect`.
    fn memcpy_h2d(
        &mut self,
        symbol: &str,
        data: &[u32],
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> sdk::Result<()>;

    /// `symbol` from every PE of `rect` into `out`.
    fn memcpy_d2h(
        &mut self,
        symbol: &str,
        out: &mut [u32],
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> sdk::Result<()>;

    /// Row `r` of `data` into `symbol` on every PE of row `r` of `rect`.
    fn memcpy_h2d_rowbcast(
        &mut self,
        symbol: &str,
        data: &[u32],
        rect: Rect,
        elems_per_pe: usize,
        data_type: DataType,
    ) -> sdk::Result<()>;

    /// [`Device::memcpy_h2d`] queued without waiting.
    fn memcpy_h2d_nonblock(
        &mut self,
        symbol: &str,
        data: Vec<u32>,
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> sdk::Result<()>;

    /// Launches the exported function `name` and waits for it.
    fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()>;

    /// [`Device::launch`] queued without waiting.
    fn launch_nonblock(&mut self, name: &str, args: &[u32]) -> sdk::Result<()>;

    /// Waits for every queued call.
    fn wait_pending(&mut self);

    /// Whether the program exports `name` (a variable or a function).
    fn has_symbol(&self, name: &str) -> bool;

    /// The blocking calls' totals so far, for the logs.
    fn io_summary(&self) -> String {
        String::new()
    }

    /// A 32-bit copy of placed data (the error names the symbol).
    fn put(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize, words: &[u32]) -> R<()> {
        self.memcpy_h2d(
            symbol,
            words,
            rect,
            elems_per_pe,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| format!("placing {symbol}: {e}"))
    }

    /// A 16-bit copy of placed data: each word's low half is one element.
    fn put16(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize, words: &[u32]) -> R<()> {
        self.memcpy_h2d(
            symbol,
            words,
            rect,
            elems_per_pe,
            Order::RowMajor,
            DataType::Bits16,
        )
        .map_err(|e| format!("placing {symbol}: {e}"))
    }

    /// `elems_per_pe` 32-bit elements a PE of `rect`, read.
    fn get(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize) -> R<Vec<u32>> {
        let mut out = vec![0u32; rect.w as usize * rect.h as usize * elems_per_pe];
        self.memcpy_d2h(
            symbol,
            &mut out,
            rect,
            elems_per_pe,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// `elems_per_pe` 16-bit elements a PE of `rect`, read, one a word.
    fn get16(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize) -> R<Vec<u32>> {
        let mut out = vec![0u32; rect.w as usize * rect.h as usize * elems_per_pe];
        self.memcpy_d2h(
            symbol,
            &mut out,
            rect,
            elems_per_pe,
            Order::RowMajor,
            DataType::Bits16,
        )
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// A 32-bit write of device state (the error is the SDK's).
    fn write(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize, words: &[u32]) -> R<()> {
        self.memcpy_h2d(
            symbol,
            words,
            rect,
            elems_per_pe,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())
    }

    /// A 16-bit write of device state (the error is the SDK's).
    fn write16(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize, words: &[u32]) -> R<()> {
        self.memcpy_h2d(
            symbol,
            words,
            rect,
            elems_per_pe,
            Order::RowMajor,
            DataType::Bits16,
        )
        .map_err(|e| e.to_string())
    }

    /// A blocking launch (the error is the SDK's).
    fn call(&mut self, name: &str, args: &[u32]) -> R<()> {
        self.launch(name, args).map_err(|e| e.to_string())
    }
}

impl Device for Runtime {
    fn memcpy_h2d(
        &mut self,
        symbol: &str,
        data: &[u32],
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> sdk::Result<()> {
        Runtime::memcpy_h2d(self, symbol, data, rect, elems_per_pe, order, data_type)
    }

    fn memcpy_d2h(
        &mut self,
        symbol: &str,
        out: &mut [u32],
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> sdk::Result<()> {
        Runtime::memcpy_d2h(self, symbol, out, rect, elems_per_pe, order, data_type)
    }

    fn memcpy_h2d_rowbcast(
        &mut self,
        symbol: &str,
        data: &[u32],
        rect: Rect,
        elems_per_pe: usize,
        data_type: DataType,
    ) -> sdk::Result<()> {
        Runtime::memcpy_h2d_rowbcast(self, symbol, data, rect, elems_per_pe, data_type)
    }

    fn memcpy_h2d_nonblock(
        &mut self,
        symbol: &str,
        data: Vec<u32>,
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> sdk::Result<()> {
        Runtime::memcpy_h2d_nonblock(self, symbol, data, rect, elems_per_pe, order, data_type)
    }

    fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        Runtime::launch(self, name, args)
    }

    fn launch_nonblock(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        Runtime::launch_nonblock(self, name, args)
    }

    fn wait_pending(&mut self) {
        Runtime::wait_pending(self);
    }

    fn has_symbol(&self, name: &str) -> bool {
        self.rpc().get(name).is_ok()
    }

    fn io_summary(&self) -> String {
        self.io.to_string()
    }
}

#[cfg(test)]
pub use recorder::{Call, Recorder};

#[cfg(test)]
mod recorder {
    //! A [`Device`] that records every call (what, where, how many and a
    //! sha256 of the words) and remembers what was written, PE by PE, so
    //! reads answer as a device would. Unwritten words read as `fill`'s.

    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use std::rc::Rc;

    use sha2::{Digest, Sha256};

    use super::Device;
    use crate::sdk::{self, DataType, Order, Rect};

    /// One recorded call, as a line: `op symbol x,y,w,h n bits words sha`.
    pub type Call = String;

    type Fill = fn(&str, i32, i32, usize) -> u32;

    #[derive(Default)]
    struct State {
        calls: Vec<Call>,
        /// Words a PE holds of a symbol, as last written (or preset).
        memory: HashMap<(String, i32, i32), Vec<u32>>,
    }

    /// The recording device. Clones share one log and memory.
    #[derive(Clone)]
    pub struct Recorder {
        state: Rc<RefCell<State>>,
        /// Whether writes are kept for later reads (off for huge shapes
        /// whose reads never come back to them).
        remember: bool,
        fill: Fill,
        missing: Rc<HashSet<String>>,
    }

    fn zero(_: &str, _: i32, _: i32, _: usize) -> u32 {
        0
    }

    fn digest(words: &[u32]) -> String {
        let mut h = Sha256::new();
        for w in words {
            h.update(w.to_le_bytes());
        }
        h.finalize()[..12]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn bits(dt: DataType) -> u32 {
        match dt {
            DataType::Bits32 => 32,
            DataType::Bits16 => 16,
        }
    }

    impl Default for Recorder {
        fn default() -> Recorder {
            Recorder {
                state: Rc::default(),
                remember: true,
                fill: zero,
                missing: Rc::default(),
            }
        }
    }

    impl Recorder {
        /// Unwritten words read as `fill(symbol, x, y, element)`.
        #[must_use]
        pub fn with_fill(mut self, fill: Fill) -> Recorder {
            self.fill = fill;
            self
        }

        /// Writes are not kept (reads answer `fill`'s words).
        #[must_use]
        pub fn forgetful(mut self) -> Recorder {
            self.remember = false;
            self
        }

        /// The program exports no `names`.
        #[must_use]
        pub fn without(mut self, names: &[&str]) -> Recorder {
            self.missing = Rc::new(names.iter().map(|s| (*s).to_string()).collect());
            self
        }

        /// `words` as PE (x, y)'s `symbol`, as if the program held them.
        pub fn preset(&self, symbol: &str, x: i32, y: i32, words: &[u32]) {
            self.state
                .borrow_mut()
                .memory
                .insert((symbol.to_string(), x, y), words.to_vec());
        }

        /// The calls so far, and the log emptied.
        pub fn take(&self) -> Vec<Call> {
            std::mem::take(&mut self.state.borrow_mut().calls)
        }

        /// A call on a symbol the program does not export: logged
        /// (`missing <name>`) and the SDK's error.
        fn unknown(&self, name: &str) -> sdk::Error {
            self.log(format!("missing {name}"));
            sdk::Error::UnknownSymbol(name.to_string())
        }

        fn log(&self, line: String) {
            self.state.borrow_mut().calls.push(line);
        }

        fn store(
            &self,
            symbol: &str,
            rect: Rect,
            n: usize,
            row_of: impl Fn(i32, i32) -> usize,
            data: &[u32],
        ) {
            if !self.remember {
                return;
            }
            let mut st = self.state.borrow_mut();
            for y in rect.y..rect.y + rect.h {
                for x in rect.x..rect.x + rect.w {
                    let at = row_of(x, y) * n;
                    st.memory
                        .insert((symbol.to_string(), x, y), data[at..at + n].to_vec());
                }
            }
        }

        fn check(&self, symbol: &str, rect: Rect, n: usize, len: usize) -> sdk::Result<()> {
            if self.missing.contains(symbol) {
                return Err(self.unknown(symbol));
            }
            let expected = rect.w.max(0) as usize * rect.h.max(0) as usize * n;
            if len != expected {
                return Err(sdk::Error::Size {
                    symbol: symbol.to_string(),
                    expected,
                    actual: len,
                });
            }
            Ok(())
        }

        fn h2d(
            &self,
            op: &str,
            symbol: &str,
            data: &[u32],
            rect: Rect,
            n: usize,
            dt: DataType,
        ) -> sdk::Result<()> {
            self.check(symbol, rect, n, data.len())?;
            self.log(format!(
                "{op} {symbol} {},{},{},{} n{n} b{} w{} {}",
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                bits(dt),
                data.len(),
                digest(data)
            ));
            let w = rect.w as usize;
            self.store(
                symbol,
                rect,
                n,
                |x, y| (y - rect.y) as usize * w + (x - rect.x) as usize,
                data,
            );
            Ok(())
        }
    }

    impl Device for Recorder {
        fn memcpy_h2d(
            &mut self,
            symbol: &str,
            data: &[u32],
            rect: Rect,
            elems_per_pe: usize,
            _order: Order,
            data_type: DataType,
        ) -> sdk::Result<()> {
            self.h2d("h2d", symbol, data, rect, elems_per_pe, data_type)
        }

        fn memcpy_d2h(
            &mut self,
            symbol: &str,
            out: &mut [u32],
            rect: Rect,
            elems_per_pe: usize,
            _order: Order,
            data_type: DataType,
        ) -> sdk::Result<()> {
            self.check(symbol, rect, elems_per_pe, out.len())?;
            {
                let st = self.state.borrow();
                let mut at = 0;
                for y in rect.y..rect.y + rect.h {
                    for x in rect.x..rect.x + rect.w {
                        let held = st.memory.get(&(symbol.to_string(), x, y));
                        for i in 0..elems_per_pe {
                            let w = held
                                .and_then(|h| h.get(i).copied())
                                .unwrap_or_else(|| (self.fill)(symbol, x, y, i));
                            out[at] = if data_type == DataType::Bits16 {
                                w & 0xffff
                            } else {
                                w
                            };
                            at += 1;
                        }
                    }
                }
            }
            self.log(format!(
                "d2h {symbol} {},{},{},{} n{elems_per_pe} b{} w{} {}",
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                bits(data_type),
                out.len(),
                digest(out)
            ));
            Ok(())
        }

        fn memcpy_h2d_rowbcast(
            &mut self,
            symbol: &str,
            data: &[u32],
            rect: Rect,
            elems_per_pe: usize,
            data_type: DataType,
        ) -> sdk::Result<()> {
            if self.missing.contains(symbol) {
                return Err(self.unknown(symbol));
            }
            let expected = rect.h.max(0) as usize * elems_per_pe;
            if data.len() != expected {
                return Err(sdk::Error::Size {
                    symbol: symbol.to_string(),
                    expected,
                    actual: data.len(),
                });
            }
            self.log(format!(
                "rowbcast {symbol} {},{},{},{} n{elems_per_pe} b{} w{} {}",
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                bits(data_type),
                data.len(),
                digest(data)
            ));
            self.store(
                symbol,
                rect,
                elems_per_pe,
                |_, y| (y - rect.y) as usize,
                data,
            );
            Ok(())
        }

        fn memcpy_h2d_nonblock(
            &mut self,
            symbol: &str,
            data: Vec<u32>,
            rect: Rect,
            elems_per_pe: usize,
            _order: Order,
            data_type: DataType,
        ) -> sdk::Result<()> {
            self.h2d("h2d_nb", symbol, &data, rect, elems_per_pe, data_type)
        }

        fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
            if self.missing.contains(name) {
                return Err(self.unknown(name));
            }
            self.log(format!("launch {name} {args:?}"));
            Ok(())
        }

        fn launch_nonblock(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
            if self.missing.contains(name) {
                return Err(self.unknown(name));
            }
            self.log(format!("launch_nb {name} {args:?}"));
            Ok(())
        }

        fn wait_pending(&mut self) {
            self.log("wait".to_string());
        }

        fn has_symbol(&self, name: &str) -> bool {
            !self.missing.contains(name)
        }
    }
}
