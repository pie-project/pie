//! The guest program plane as the shared engine drives it: registering
//! programs and channels, binding instances, moving cells, and what a fire
//! reads of an instance. Each backend's plane runs its device halves its own
//! way.

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use eta_compiler::codegen::launch::LaunchPackage;

use crate::serve::{InputCell, Serve};

#[derive(Debug)]
pub enum Refusal {
    Closed { what: &'static str, id: u64 },

    Program(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Closed { what, id } => write!(f, "{what} {id} is closed"),
            Refusal::Program(why) => f.write_str(why),
        }
    }
}

impl From<String> for Refusal {
    fn from(why: String) -> Refusal {
        Refusal::Program(why)
    }
}

/// A backend's program plane.
pub trait Programs<B: Serve>: Default {
    fn register(&mut self, registration: &ProgramRegistration) -> Result<ProgramId, Refusal>;

    fn package(&self, program: ProgramId) -> Option<&LaunchPackage>;

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel, Refusal>;

    fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal>;

    /// Binds an instance; its device half on `device`, where one is loaded.
    fn bind(
        &mut self,
        binding: &InstanceBinding,
        device: Option<&B::Context>,
    ) -> Result<BoundInstance, Refusal>;

    fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal>;

    /// The oldest cell guest input channel `id` of `instance` holds.
    fn input_cell(&self, instance: InstanceId, id: ChannelId) -> Result<InputCell<B>, Refusal>;

    /// The rows a lane of `instance` folds, where the instance states them.
    fn envelope_fold(&self, instance: InstanceId) -> Result<Option<u32>, Refusal>;

    fn publish(&mut self, instance: InstanceId, channel: u32, cell: &[u8])
    -> Result<bool, Refusal>;

    fn take(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>, Refusal>;

    /// The outputs `instance` reads, each with the lane of its attachment.
    fn reads(&self, instance: InstanceId) -> Vec<(String, u32)>;
}

/// The elements a cell of `shape` holds (a scalar holds one).
#[must_use]
pub fn numel_of(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>().max(1)
}

/// A wire cell of `dtype` and `shape` as a host value; `None` where its
/// bytes do not fit the cell or the dtype has no host form.
#[must_use]
pub fn decode(
    bytes: &[u8],
    dtype: eta_ir::container::ChanDType,
    shape: &[u32],
) -> Option<eta_exec::Value> {
    use eta_ir::types::Dtype;
    let dtype = eta_exec::concrete_dtype(dtype);
    let numel = numel_of(shape);
    if bytes.len() != eta_exec::wire_cell_bytes(dtype, numel) {
        return None;
    }
    let words = || {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2], c[3]])
    };
    Some(match dtype {
        Dtype::Bool => {
            eta_exec::Value::Bool((0..numel).map(|j| (bytes[j / 8] >> (j % 8)) & 1).collect())
        }
        Dtype::I32 => eta_exec::Value::I32(words().map(i32::from_le_bytes).collect()),
        Dtype::U32 => eta_exec::Value::U32(words().map(u32::from_le_bytes).collect()),
        Dtype::F32 => eta_exec::Value::F32(words().map(f32::from_le_bytes).collect()),
        _ => return None,
    })
}
