//! The wafer as the host drives it: one trait for every host-wafer call the engine makes.

use crate::sdk::{self, DataType, Order, Rect, Runtime};

type R<T> = std::result::Result<T, String>;

/// A loaded wafer program the host copies to and from and launches. Copies take the SDK's
/// arguments; a 16-bit copy moves one element a word, in its low half.
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

    /// Launches the exported function `name` and waits for it.
    fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()>;

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

    fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        Runtime::launch(self, name, args)
    }
}
