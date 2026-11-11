//! What the runtime is told about a load besides the wafer program: the trace's lane classes, its
//! inputs, the float output the readout rows read, and the host's paging. The model forward runs
//! on the wafer; these follow from the trace and the budgets alone.

use engine::load::Budgets as LoadBudgets;
use poem_compiler::{Budget, Budgets, CompiledModel, DeviceProfile, compile_axes};
use poem_exec::feeds::{ClassOutput, Feeds};
use poem_exec::store::kv::Paging;
use poem_ir::{CacheRow, Dtype, Trace};

type R<T> = std::result::Result<T, String>;

/// The host pages: they hold tokens (the wafer holds the cache), so they cost no device memory
/// and the pool is whole from the load; the memory governor has nothing to commit or release.
const FILL_PAGES: u64 = 65536;

/// The output the readout rows read when the trace names it.
const LOGITS: &str = "logits";

pub struct Facts {
    trace: Trace,
    compiled: CompiledModel,
    exports: Vec<ClassOutput>,
    paging: Paging,
}

impl Facts {
    pub fn of(trace: Trace, budgets: &LoadBudgets, slots: u32) -> R<Facts> {
        let mut budget = Budget {
            max_lanes: budgets.max_lanes,
            max_tokens: budgets.max_tokens,
            buckets: budgets.buckets.clone(),
            max_adapters: budgets.max_adapters,
            readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
        };
        if budget.buckets.is_empty() {
            // the row ladder a fire is padded up to: powers of two to `max_tokens`
            budget.buckets = std::iter::successors(Some(1u32), |rung| rung.checked_mul(2))
                .take_while(|rung| *rung < budget.max_tokens)
                .chain(std::iter::once(budget.max_tokens))
                .collect();
        }
        let compiled = compile_axes(&trace, &Budgets::of(budget), &DeviceProfile::default())
            .map_err(|e| e.to_string())?;
        let exports = poem_exec::feeds::class_outputs(&trace, &compiled)?;
        if exports.is_empty() {
            return Err("the trace has no output, so a fire would compute nothing".into());
        }
        let least = u64::from(budgets.max_context.div_ceil(budgets.page_size.max(1)));
        let cells = u64::from(u32::MAX) / u64::from(budgets.page_size.max(1)) - 1;
        let window = trace
            .caches
            .iter()
            .filter_map(|row| match row {
                CacheRow::Kv { window, .. } => *window,
                CacheRow::State { .. } => None,
            })
            .max();
        let paging = Paging::of(
            budgets.page_size,
            budgets.max_context,
            slots,
            FILL_PAGES.max(least).min(cells),
        )
        .map_err(|e| e.to_string())?
        .windowed(window, budgets.max_tokens);
        Ok(Facts {
            trace,
            compiled,
            exports,
            paging,
        })
    }

    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// The name and width of the float output the readout rows read: `logits` where the trace
    /// has it, else its only float readout.
    pub fn readout(&self) -> Option<(String, u32)> {
        let float = |out: &&ClassOutput| {
            out.decl.readout && matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32)
        };
        let out = match self
            .exports
            .iter()
            .filter(float)
            .find(|o| o.decl.name == LOGITS)
        {
            Some(out) => out,
            None => {
                let mut floats = self.exports.iter().filter(float);
                match (floats.next(), floats.next()) {
                    (Some(only), None) => only,
                    _ => return None,
                }
            }
        };
        Some((out.decl.name.clone(), out.decl.width))
    }

    /// Whether the trace keeps recurrent state (its lanes' state replays from tokens here).
    pub fn has_state(&self) -> bool {
        self.trace
            .caches
            .iter()
            .any(|row| matches!(row, CacheRow::State { .. }))
    }

    /// Bytes of one slot of every recurrent state row (i32 rows as i32, the rest as f32: four
    /// bytes an element either way).
    pub fn state_slot_bytes(&self) -> u64 {
        self.trace
            .caches
            .iter()
            .map(|row| match row {
                CacheRow::State { slab, .. } => slab.iter().product::<u64>() * 4,
                CacheRow::Kv { .. } => 0,
            })
            .sum()
    }

    pub fn input_decls(&self) -> Vec<poem_ir::InputDecl> {
        Feeds::of(&self.trace, &self.compiled).decls(&self.trace, &self.compiled)
    }

    pub fn classes(&self) -> poem_ir::LaneClasses {
        poem_exec::feeds::lane_classes(&self.trace, &self.compiled, &self.exports)
    }
}
