//! The device half: an instance, a queue, and pipelines that actually fire.
//!
//! Only compiled under the `native` feature, and only because it needs a GPU to
//! be PRESENT — every line of it builds anywhere, which is why the feature
//! gates the tests rather than the compilation.
//!
//! # What this is for, beyond wrapping `ash`
//!
//! `kernels-vulkan`'s docs list what a shell that runs its modules has to do,
//! and each item on that list is there because something got it wrong:
//!
//! * A layout must cover every binding the MODULE decorates, not every operand
//!   the row names. 292 entrypoints name no operands at all, and a layout built
//!   from such a row segfaults inside `vkCreateComputePipelines` rather than
//!   returning an error. [`Pipelines`] builds from
//!   [`crate::spirv::Declared::bindings`] for exactly this reason.
//! * A push range may not be wider than the block the module declares, and it
//!   may not be narrower than what the shader reads.
//! * `robustBufferAccess` must be on. `quant/qmm_t.slang` accumulates over its
//!   whole tile and guards only the store, so at a ragged shape it deliberately
//!   fetches outside the matrix and needs those reads defined as zero.
//! * A descriptor's offset must be a multiple of
//!   `minStorageBufferOffsetAlignment`, which has no Metal counterpart worth
//!   the name and which an arena allocator meets only if it asks. See
//!   [`Bound`].
//! * A binding must be given at least the bytes the shader's block reads.
//!   `robustBufferAccess` makes the tail read as ZERO rather than fault, so a
//!   short parameter block is a plausible number and not an error. See
//!   [`crate::spirv::Declared::block_bytes`].
//! * A descriptor's range should be the operand's extent and not
//!   `VK_WHOLE_SIZE`. In an arena the two differ by every tensor allocated
//!   after it, and `robustBufferAccess` -- already required above -- makes the
//!   narrow one CONFINE an overrun rather than merely describe it.
//! * The grid is a count of WORKGROUPS. See [`crate::geometry`].
//!
//! # The validation layer
//!
//! Enabled when the loader can find it, and an error ends the process. That is
//! not politeness: without it this driver answers a malformed request by
//! crashing or hanging instead of failing, so a green test without the layer is
//! not evidence that a dispatch was legal. It is a soft dependency because a
//! build machine will not have one.

use crate::geometry::{self, Dims, Module, Rule, Ungeometric};
use crate::spirv::{self, Declared};
use ash::vk;
use kernels_vulkan::Capability;
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::Mutex;

/// Why there is no device to run on.
///
/// A distinct type from [`Ungeometric`] because the two mean opposite things to
/// a caller: this one is the environment, and a machine with no GPU is the
/// normal state of a build host rather than a defect.
#[derive(Clone, Debug)]
pub struct Unavailable(pub String);

impl core::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Unavailable {}

/// Why a dispatch could not be made.
///
/// Compared by value because [`crate::binding`] carries one inside its own
/// refusal, and a test that asserts WHICH refusal came back is the only way
/// an alignment failure stays distinguishable from a length one.
#[derive(Clone, Debug, PartialEq)]
pub enum Failed {
    /// The launch shape could not be worked out.
    Geometry(Ungeometric),
    /// The module could not be read.
    Module(spirv::Malformed),
    /// The caller bound a different number of buffers than the module declares.
    ///
    /// A short set is the one that matters: Vulkan reads a descriptor the
    /// shader uses and the set never filled, which is undefined rather than
    /// empty.
    Bindings {
        /// What the module decorates, one past its highest.
        module: u32,
        /// What the caller bound.
        bound: usize,
    },
    /// The push block is a different size than the pipeline's range.
    ///
    /// Both directions are refused. Too wide overruns the range; too narrow
    /// leaves the shader reading push memory that was never written, which the
    /// layer does not always catch and which reads as whatever the last
    /// dispatch left.
    Push {
        /// The range the pipeline was built with.
        range: u32,
        /// The bytes the caller offered.
        given: usize,
    },
    /// A sub-range starts at an offset the device will not address from.
    ///
    /// `minStorageBufferOffsetAlignment` is a hardware granularity, not a
    /// preference: a descriptor written at an offset it does not divide is
    /// invalid, and the layer says so. Without the layer it is undefined and
    /// on this device it happens to read the right bytes, which is the worst
    /// of the outcomes because it makes the defect a property of the machine
    /// it was tested on.
    Unaligned {
        /// The offset asked for.
        offset: u64,
        /// What the device requires it to be a multiple of.
        alignment: u64,
    },
    /// A sub-range runs past the end of the buffer holding it.
    ///
    /// The one refusal here that Vulkan itself would also catch -- but only
    /// with a layer, and only sometimes: `VK_WHOLE_SIZE` is legal, so a range
    /// that overruns is a number the driver may clamp, may honour, or may
    /// fault on, and the shader reads whichever happened.
    Overrun {
        /// Where it starts.
        offset: u64,
        /// How much it asked for.
        len: u64,
        /// What the buffer holds.
        size: u64,
    },
    /// A binding was given fewer bytes than the block the shader reads.
    ///
    /// The defect with no symptom at all, and the reason this refusal exists
    /// rather than a comment saying to be careful. `robustBufferAccess` is on
    /// -- the tiled GEMM needs it -- so a read past the bound range returns
    /// ZERO. A parameter block one word short does not fault and does not
    /// return garbage: the missing scalar is zero, which for a pitch or a flag
    /// is an entirely plausible value, and nothing downstream can tell it from
    /// one that was meant.
    ///
    /// `driver-metal` was found packing exactly this defect into two blocks,
    /// and there it was LOUDER: Metal's params region is written per dispatch,
    /// so the shader read the next dispatch's scalars instead of zeros.
    Short {
        /// Which descriptor.
        binding: u32,
        /// What the module's block needs.
        needs: u32,
        /// What the caller bound.
        given: u64,
    },
    /// A grid is wider on one axis than the device will dispatch.
    ///
    /// Refused rather than clamped or split. Clamping computes part of an
    /// output and says nothing; splitting is a decision about what a launch
    /// MEANS -- whether its workgroup index may restart -- and a driver does
    /// not get to make that on a kernel's behalf.
    Grid {
        /// 0, 1 or 2: x, y or z.
        axis: u32,
        /// What the geometry asked for.
        groups: u32,
        /// What `maxComputeWorkGroupCount` allows on that axis.
        limit: u32,
    },
    /// The device would not give this allocation memory, right now.
    ///
    /// Separated from [`Self::Vulkan`] because the CALLER'S next move
    /// differs, which is the only reason any of these variants exist. Every
    /// other failure here is a fault: the frame is wrong, or the module is,
    /// and repeating it repeats the failure. This one is a scheduling fact.
    /// The same frame, posted after something else is evicted, succeeds.
    ///
    /// It is reachable in ordinary service rather than only under abuse.
    /// [`Device::budget`] reports a heap's SIZE, not what is free in it, so
    /// [`crate::resources::Pool::ceiling`] admits any frame the device could
    /// hold if it were empty -- and the device is never empty, because the
    /// model's weights are in it. A frame under the ceiling and over the free
    /// space is the normal shape of a busy server, not a bug.
    OutOfMemory {
        /// What was asked for, in bytes.
        bytes: u64,
        /// Which call refused, for the log. Not matched on.
        during: &'static str,
    },
    /// A Vulkan call failed.
    Vulkan(String),
}

impl Failed {
    /// Classify a Vulkan result: out of memory, or a fault.
    ///
    /// The two out-of-memory codes are one answer here. A caller cannot act
    /// on the difference -- it evicts and retries either way -- and treating
    /// only the device-local one as retryable would make the host-visible
    /// heap, which is the one this shell allocates from, the case that
    /// wrongly kills a request.
    #[must_use]
    pub fn of_vulkan(result: ash::vk::Result, during: &'static str, bytes: u64) -> Self {
        if matches!(
            result,
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY
        ) {
            Self::OutOfMemory { bytes, during }
        } else {
            Self::Vulkan(format!("{during}: {result}"))
        }
    }

    /// Whether the device refused for want of memory rather than for a fault.
    #[must_use]
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::OutOfMemory { .. })
    }
}

impl core::fmt::Display for Failed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Grid {
                axis,
                groups,
                limit,
            } => write!(
                f,
                "a grid of {groups} workgroups on axis {axis} is past this \
                 device's limit of {limit}"
            ),
            Self::Geometry(e) => write!(f, "no launch geometry: {e}"),
            Self::Module(e) => write!(f, "the module is malformed: {e}"),
            Self::Bindings { module, bound } => write!(
                f,
                "the module decorates {module} bindings but {bound} buffers were bound"
            ),
            Self::Push { range, given } => write!(
                f,
                "the pipeline's push range is {range} bytes and {given} were pushed"
            ),
            Self::Unaligned { offset, alignment } => write!(
                f,
                "this device addresses a storage buffer every {alignment} bytes \
                 and the range starts at {offset}"
            ),
            Self::Overrun { offset, len, size } => write!(
                f,
                "a range of {len} bytes at {offset} runs past the {size} the \
                 buffer holds"
            ),
            Self::Short {
                binding,
                needs,
                given,
            } => write!(
                f,
                "binding {binding} reads a {needs}-byte block and was given \
                 {given} bytes, whose tail reads as zero"
            ),
            Self::OutOfMemory { bytes, during } => write!(
                f,
                "this device would not give {bytes} bytes to `{during}` right now"
            ),
            Self::Vulkan(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for Failed {}

impl From<Ungeometric> for Failed {
    fn from(e: Ungeometric) -> Self {
        Self::Geometry(e)
    }
}

impl From<spirv::Malformed> for Failed {
    fn from(e: spirv::Malformed) -> Self {
        Self::Module(e)
    }
}

/// End the process on a validation error, printing what the layer said.
///
/// # Running with the layer at all
///
/// Every command in this repository's history disables it -- `VK_LAYER_PATH=`
/// is set empty in the gate invocations and in the sweep -- so this callback
/// went a long time without ever firing. Ubuntu has no validation package
/// installed by default either. To actually use it:
///
/// ```sh
/// cd /tmp && apt-get download vulkan-validationlayers
/// dpkg-deb -x vulkan-validationlayers_*.deb /tmp/vvl
/// VK_LAYER_PATH=/tmp/vvl/usr/share/vulkan/explicit_layer.d \
/// LD_LIBRARY_PATH=/tmp/vvl/usr/lib/x86_64-linux-gnu \
/// VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
///   cargo test -p driver-vulkan --features native --test device -- --nocapture
/// ```
///
/// `--nocapture` matters: without it the abort below takes the process down
/// before the harness prints what this callback wrote, and the failure looks
/// like an unexplained SIGABRT. That is exactly how the first real find --
/// two tests passing one `VkDevice`'s buffers to another's commands --
/// presented itself.
///
/// Measured clean this way, with GPU-assisted validation on: the 40 kernel
/// proofs, the 71 device tests (329 s under the layer), all twelve end-to-end
/// gates, and the full 40-program inferlet sweep, which is the broadest thing
/// there is to point it at -- 36/40, the four being the model-gated attention
/// intrinsics that refuse on every backend. The layer was confirmed loaded in
/// the server process rather than assumed, since a clean run under a layer
/// that never loaded says nothing at all.
///
/// Re-measured after the shader tree moved from GLSL to Slang, and that is
/// the reason to re-measure rather than to trust the paragraph above. The
/// bodies did not change what they do, but the COMPILER changed what they
/// declare: `slangc` spells 8-bit storage access with the wider
/// `UniformAndStorageBuffer8BitAccess` capability, and a module may not
/// declare a capability whose feature is off. Two devices had to ask for it
/// -- this one, and the separate one `kernels-vulkan`'s harness builds -- and
/// neither omission was visible without the layer: this NVIDIA driver loads
/// the module regardless, so all 250 driver tests, all 40 proofs and 36/40 of
/// the sweep passed while both were wrong.
///
/// # Safety
///
/// Called by the Vulkan loader with a `p_callback_data` valid for the duration
/// of the call. Nothing here outlives it.
unsafe extern "system" fn fail_on_validation_error(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kinds: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let said = unsafe { data.as_ref() }
        .and_then(|d| (!d.p_message.is_null()).then(|| unsafe { CStr::from_ptr(d.p_message) }))
        .map_or_else(
            || "<no message>".to_string(),
            |m| m.to_string_lossy().into_owned(),
        );
    eprintln!(
        "\nthe Vulkan validation layer reported an ERROR:\n\n{said}\n\nThis \
         driver treats one as fatal, because without the layer the same \
         mistakes are a crash inside the driver or a wrong answer rather than \
         a message.\n"
    );
    // Abort and not panic: this is called across an `extern \"system\"`
    // boundary, where an unwind is undefined.
    std::process::abort();
}

/// Where a fire's time actually goes, per kernel, measured on the device.
///
/// Every performance claim in this crate before this existed was wall-clock
/// around a `queue_submit`/`wait_for_fences` pair. That answers "how long did
/// the step take" and nothing else: a decode is four hundred and fifty
/// dispatches, and which of them the milliseconds belong to has never been
/// askable here. Deciding what to optimise from a single number is deciding
/// from a guess, and the guesses this crate has had to throw away -- the model
/// was in system RAM, the coopmat modules were unreachable, the logits
/// readback was uncached -- were all found by measuring a level down.
///
/// Off unless `PIE_VULKAN_TIMING` is set in the environment, because it is not
/// free: two `vkCmdWriteTimestamp`s per dispatch, and a `vkGetQueryPoolResults`
/// that blocks on the same fence the fire already waited for.
///
/// ## What it costs, which is more than it looks
///
/// Measured, release, the 4-bit qwen3-0.6b at twenty-four tokens: the
/// submit-and-wait around one decode fire is **3.47 ms with this off and 5.78
/// ms with it on**. Two thirds added to the very thing being measured. The
/// pool reset is 453 queries of it, but most of it is the timestamps
/// themselves: a `BOTTOM_OF_PIPE` write after every dispatch is a point the
/// card cannot finish early, so neighbouring dispatches that had been
/// overlapping stop.
///
/// So the absolute milliseconds this reports are an upper bound and not a
/// benchmark, and a claim of the form "this kernel takes N ms" must not be
/// built on them. What survives the perturbation is the SHARES, and only
/// because the differences that matter here are far larger than it: attention
/// goes from a fifth of device time to three quarters between short and long
/// context, which two thirds of added overhead cannot manufacture.
///
/// It also means the honest way to measure the driver's own overhead is with
/// this off, timing the submit against the wall.
///
/// ## What the numbers mean, and what they do not
///
/// One timestamp is written at `TOP_OF_PIPE` before the first dispatch and one
/// at `BOTTOM_OF_PIPE` after each, so slot `i + 1` minus slot `i` is the time
/// from "everything before this dispatch had finished" to "this dispatch had
/// finished" -- which includes any barrier recorded in front of it. That is
/// the honest unit: a barrier exists because of the dispatch after it, and
/// charging it there is what makes the per-symbol totals add up to the fire.
///
/// The attribution is only exact where the dispatches cannot overlap. Two
/// neighbours with no barrier between them may run at once on this card, and
/// then the first one's slice absorbs work the second one did. So a symbol's
/// total is an upper bound on its own time and the SUM over a fire is right;
/// treat a single row as "this much time had passed by here", not as an
/// isolated kernel benchmark.
struct Timing {
    pool: vk::QueryPool,
    /// Nanoseconds per tick, from `VkPhysicalDeviceLimits::timestampPeriod`.
    period: f32,
    /// How many queries the pool holds. A fire needing more is skipped whole
    /// rather than reported in part, and counted in `skipped`.
    slots: u32,
    /// Symbol to (total ticks, times dispatched).
    per_symbol: Mutex<HashMap<String, (u64, u32)>>,
    /// Fires that had more dispatches than the pool has slots.
    skipped: std::sync::atomic::AtomicU32,
}

/// An open Vulkan device with a compute queue.
///
/// Owns everything it creates and destroys it in reverse order on drop, so a
/// caller cannot leak a pipeline by dropping the device first.
pub struct Device {
    entry: ash::Entry,
    /// Kept alive for the instance's lifetime: dropping the messenger early
    /// silences exactly the calls that are about to be made.
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    name: String,
    max_push: u32,
    /// `minStorageBufferOffsetAlignment`.
    ///
    /// The limit with no Metal counterpart worth the name. `setBuffer:offset:`
    /// takes a byte offset and Apple silicon asks 4 of it; Vulkan will not
    /// address a storage buffer from an offset this does not divide, and a
    /// driver that binds a sub-range of an arena is doing nothing else.
    min_storage_offset: u64,
    /// `maxComputeWorkGroupCount`, per axis.
    ///
    /// The limit with the widest spread in Vulkan. This card answers
    /// 2147483647 on every axis; the specification GUARANTEES only 65535, and
    /// devices that answer exactly that are common. A dispatch past it is not
    /// clamped and not an error the queue returns -- it is undefined, which on
    /// a card that runs the first 65535 workgroups is a plan that computed
    /// part of its output and reported success.
    max_groups: [u32; 3],
    /// Is host memory and device memory the same memory?
    ///
    /// Read from `deviceType`, not from the heaps. A discrete card exposes a
    /// small heap that is both DEVICE_LOCAL and HOST_VISIBLE -- the resizable
    /// BAR -- and a driver that concluded "unified" from finding one would be
    /// wrong about every allocation that did not fit in it.
    unified: bool,

    /// Whether the device reports itself as a CPU implementation.
    ///
    /// Narrower than [`Self::unified`] on purpose: that one is about where
    /// memory lives and is true of an integrated GPU as well, which is a real
    /// device with a real clock. This is about whether there is a GPU at all.
    /// It exists because a wall-clock BUDGET is meaningful on hardware and
    /// meaningless on `llvmpipe`, and a suite run there should say which it is
    /// rather than fail a regression guard that was calibrated elsewhere.
    software: bool,
    validated: bool,
    /// The tiers this device can actually load, best first.
    ///
    /// Derived from what was ENABLED rather than from what was reported: a
    /// feature the device has and the driver did not turn on is a feature a
    /// module may not declare a capability for.
    tiers: Vec<Capability>,
    /// The objects a fire needs and does not need a new one of.
    ///
    /// Behind a lock because [`Self::run_all`] takes `&self` -- the driver
    /// holds one device and fires from it -- and because a command pool and a
    /// queue are externally synchronised objects. The lock is not a
    /// concession: it is the same serialisation the queue already imposed.
    scratch: Mutex<Scratch>,
    /// How many buffers this device has allocated, ever.
    ///
    /// Kept for [`Device::allocations`]. Every buffer is its own
    /// `vkAllocateMemory`, and `maxMemoryAllocationCount` is a hard ceiling
    /// -- 4096 on a good many devices -- so how often a fire asks is not a
    /// matter of speed alone. A test that counts is the only way to say the
    /// difference between one allocation per fire and one per scalar block,
    /// since both answer correctly.
    allocations: std::sync::atomic::AtomicU32,
    /// How many pipeline barriers the fires on this device have recorded.
    ///
    /// A number, not a duration, for the reason every counter here is one: a
    /// `hazards` that answered `true` for everything would record the same
    /// right answers this driver recorded before it existed, only slowly, and
    /// on a shared machine a duration measures the neighbours. What a test
    /// can hold is that a fire of `n` dispatches records FEWER than `n - 1`
    /// of these and more than none.
    barriers: std::sync::atomic::AtomicU32,
    /// The staging buffer [`Device::read_at`] DMAs into, kept between reads.
    ///
    /// Grown to the largest read asked for and never shrunk, which is the
    /// rule `Scratch`'s descriptor pool follows for the same reason: a
    /// server's reads are the same two or three sizes forever, so a cache
    /// reaches the largest of them and then stops being an allocation at all.
    /// Measured, and the reason it exists: a create-plus-allocate-plus-free
    /// costs about 260 us on this card, which is a twentieth of a decode.
    ///
    /// `None` until the first staged read, so a device that never reads back
    /// -- and every device on a machine with no checkpoint -- allocates
    /// nothing.
    staging: Mutex<Option<Buffer>>,
    /// Reads that went through the copy engine rather than through a mapping.
    ///
    /// Counted because the difference between the two is thirty seconds on a
    /// prefill and nothing at all in the answer, so a change that quietly
    /// stopped staging would show up only as a slow server. See
    /// [`Device::read_at`].
    staged: std::sync::atomic::AtomicU32,
    /// How many buffers this device has freed, ever.
    ///
    /// The other half of [`Device::live_buffers`]. A path that returns early
    /// and forgets what it took leaks device memory silently -- the card has
    /// twenty-four gigabytes and a scalar block is tens of bytes, so nothing
    /// downstream ever fails -- and this is what makes that countable.
    frees: std::sync::atomic::AtomicU32,
    /// How many BYTES this device has uploaded through [`Device::write`].
    ///
    /// Bytes rather than calls, because the question this answers is about
    /// bus traffic and the calls differ by five orders of magnitude: a fire
    /// table is tens of bytes and a prefill's arena was 233 megabytes.
    ///
    /// It exists because "the device made these bytes itself" is a claim
    /// about a route, and this crate has now had the same route mistake three
    /// times without a single test going red -- the contents were always
    /// correct. Counting the bus makes the claim checkable without timing
    /// anything, which on a shared box is the difference between a tripwire
    /// and a flaky test. See `turns::arena_for`.
    uploaded: std::sync::atomic::AtomicU64,
    /// Per-kernel device time, when `PIE_VULKAN_TIMING` asked for it.
    timing: Option<Timing>,
}

/// What one fire allocates, kept between fires.
///
/// Measured on an RTX 4090, in microseconds per fire, made and destroyed each
/// time against reused, over 300 fires after a warm-up:
///
/// | dispatches per fire | fresh | reused |
/// | --- | --- | --- |
/// | 1 | 421 | 35 |
/// | 8 | 407 | 64 |
/// | 64 | 582 | 220 |
///
/// The dispatches are a 256-wide RMS norm, which is microseconds of work, so
/// the first column is very nearly the cost of creating and destroying a
/// descriptor pool, a command buffer and a fence -- and it barely moves
/// between one dispatch and eight, which is what a fixed cost looks like. A
/// driver that decodes one token pays it once per fire, for every layer of
/// every step.
///
/// Nothing here is a cache with a policy. The command buffer and the fence
/// are one object each, reset before use. The descriptor pool GROWS -- when a
/// fire wants more sets or more descriptors than the pool was built for, the
/// pool is destroyed and a bigger one takes its place -- so a steady state
/// stops allocating entirely, and a fire that is bigger than every fire
/// before it pays once.
struct Scratch {
    /// Reset, not freed, at the start of every fire.
    pool: vk::DescriptorPool,
    /// How many sets this pool was built to hold.
    sets: u32,
    /// How many storage descriptors this pool was built to hold.
    descriptors: u32,
    /// One primary buffer, reset before each recording.
    cmd: vk::CommandBuffer,
    /// One fence, reset before each submit.
    fence: vk::Fence,
    /// How many descriptor pools this device has made, ever.
    ///
    /// Kept for [`Device::pools_made`], which is what lets a test state that
    /// a steady state stops allocating. Without it, growing the pool to the
    /// high-water mark and rebuilding it for every fire are indistinguishable
    /// -- both answer correctly, and only one of them is the point.
    made: u32,
}

impl Scratch {
    /// Make the objects a fire reuses.
    ///
    /// The pool starts empty -- zero sets, zero descriptors -- because the
    /// first fire's size is the only honest guess and it has not happened
    /// yet. `for_run` grows it.
    ///
    /// # Safety
    ///
    /// `pool` must be a command pool of `device` created with
    /// `RESET_COMMAND_BUFFER`, and the caller owns destroying the result with
    /// [`Self::destroy`] before `device` goes.
    unsafe fn new(device: &ash::Device, pool: vk::CommandPool) -> Result<Self, vk::Result> {
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                unsafe { device.free_command_buffers(pool, &[cmd]) };
                return Err(e);
            }
        };
        Ok(Self {
            pool: vk::DescriptorPool::null(),
            sets: 0,
            descriptors: 0,
            cmd,
            fence,
            made: 0,
        })
    }

    /// A descriptor pool this fire fits in, emptied of the last fire's sets.
    ///
    /// Grown to what is asked and never shrunk. A driver's fires are the same
    /// few shapes over and over, so the pool reaches the largest of them and
    /// then stops being an allocation at all; shrinking would turn a steady
    /// state back into churn to save memory measured in kilobytes.
    ///
    /// # Safety
    ///
    /// No command buffer using a set from this pool may still be executing.
    /// The single fence [`Device::run_all`] waits on before returning is what
    /// makes that true here.
    unsafe fn for_run(
        &mut self,
        device: &ash::Device,
        sets: u32,
        descriptors: u32,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        if self.pool != vk::DescriptorPool::null()
            && sets <= self.sets
            && descriptors <= self.descriptors
        {
            // Resetting frees every set at once, which is why the sets are
            // not tracked individually: there is nothing to free them from.
            unsafe {
                device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            }?;
            return Ok(self.pool);
        }
        // The high-water mark rather than the request, in both dimensions.
        // For one pipeline the two agree -- sets and descriptors climb
        // together -- so no test here can tell them apart, and this is a
        // deliberate survivor: it costs a comparison and stops a fire that is
        // wide in sets and narrow in descriptors from shrinking the pool out
        // from under the next fire that is the other way round.
        let want_sets = sets.max(self.sets).max(1);
        let want_descriptors = descriptors.max(self.descriptors).max(1);
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(want_descriptors)];
        let fresh = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(want_sets)
                    .pool_sizes(&sizes),
                None,
            )
        }?;
        if self.pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.pool, None) };
        }
        self.pool = fresh;
        self.made += 1;
        self.sets = want_sets;
        self.descriptors = want_descriptors;
        Ok(fresh)
    }

    /// Give everything back.
    ///
    /// # Safety
    ///
    /// `pool` must be the command pool the buffer came from, and no
    /// submission may still be in flight.
    unsafe fn destroy(&self, device: &ash::Device, pool: vk::CommandPool) {
        unsafe {
            if self.pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.pool, None);
            }
            device.destroy_fence(self.fence, None);
            device.free_command_buffers(pool, &[self.cmd]);
        }
    }
}

/// One physical device the loader offered, and everything needed to
/// decide whether to run on it.
///
/// This exists because the previous line of code was `devices.first()`,
/// and the Vulkan specification says nothing whatsoever about the order
/// `vkEnumeratePhysicalDevices` returns. On the machine this crate was
/// developed on that call returns the discrete card at index 0 and a
/// `llvmpipe` software rasteriser at index 1 -- so every number ever
/// measured here was measured on the card, and none of it was because the
/// code asked. Reorder the ICD manifests, run on a laptop whose iGPU
/// enumerates first, or use a loader build without the device-select
/// sort, and the whole suite would have moved onto a CPU implementation,
/// passed every correctness test it has, and got slower by two orders of
/// magnitude while saying nothing.
///
/// So the choice is made here, from the device's own reported type, and
/// the software adapter is chosen only when it is the only thing that can
/// compute at all.
struct Candidate {
    handle: vk::PhysicalDevice,
    props: vk::PhysicalDeviceProperties,
    name: String,
    /// `None` means this device cannot run a compute shader, which is why
    /// picking one is not simply a matter of ranking types. A device with
    /// no compute queue is not a worse choice, it is not a choice.
    compute_family: Option<u32>,
}

impl Candidate {
    fn read(instance: &ash::Instance, handle: vk::PhysicalDevice) -> Self {
        let props = unsafe { instance.get_physical_device_properties(handle) };
        let name = props
            .device_name_as_c_str()
            .map_or_else(|_| "<unnamed>".to_string(), |s| s.to_string_lossy().into());
        let compute_family =
            unsafe { instance.get_physical_device_queue_family_properties(handle) }
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);
        Self {
            handle,
            props,
            name,
            compute_family,
        }
    }

    /// Lower is better. The order is the one an inference runtime wants and
    /// not the one the enum happens to declare: a real card, then a shared
    /// one, then a virtualised one, then whatever `OTHER` is, and a CPU
    /// implementation last of all because it is a correctness tool rather
    /// than a device.
    /// The reported type as a word. `PhysicalDeviceType` is a newtype over an
    /// integer with no `Debug`, and a bare number in the one message a user
    /// with no working device ever sees would tell them nothing.
    fn kind(&self) -> &'static str {
        match self.props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => "software",
            _ => "other",
        }
    }

    fn rank(&self) -> u8 {
        match self.props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 4,
            _ => 3,
        }
    }

    /// The best device that can compute, or `None` if none can.
    ///
    /// `PIE_VULKAN_DEVICE` overrides the ranking with a case-insensitive
    /// substring of the device name -- which is how the software adapter
    /// gets deliberately selected, and the only reason this crate can
    /// prove the ranking works rather than assert it. An override that
    /// matches nothing that can compute is a refusal, not a fallback: a
    /// run that was asked for a named device and silently got another one
    /// would be a worse failure than not starting.
    fn choose(seen: &[Self]) -> Option<&Self> {
        Self::choose_from(seen, std::env::var("PIE_VULKAN_DEVICE").ok().as_deref())
    }

    /// The decision itself, with the override passed in rather than read.
    ///
    /// Split out so that it is a pure function of its arguments, which is the
    /// only way the ranking can be tested against device shapes this machine
    /// does not have -- and, less obviously, the only way those tests can run
    /// in PARALLEL. A `choose` that reads the environment can only be tested
    /// by a test that writes it, and two such tests in one process race each
    /// other rather than the code. This one was written the other way first
    /// and the ranking test failed intermittently for exactly that reason.
    fn choose_from<'a>(seen: &'a [Self], want: Option<&str>) -> Option<&'a Self> {
        let usable = || seen.iter().filter(|c| c.compute_family.is_some());
        match want.map(str::trim).filter(|w| !w.is_empty()) {
            Some(want) => {
                let want = want.to_ascii_lowercase();
                usable().find(|c| c.name.to_ascii_lowercase().contains(&want))
            }
            None => usable().min_by_key(|c| c.rank()),
        }
    }

    /// Every device seen and why it was or was not eligible, for the one
    /// message a user gets when nothing here can run.
    fn roster(seen: &[Self]) -> String {
        if seen.is_empty() {
            return "nothing at all".to_string();
        }
        seen.iter()
            .map(|c| {
                let q = if c.compute_family.is_some() {
                    "compute"
                } else {
                    "NO compute queue"
                };
                format!("{} ({}, {q})", c.name, c.kind())
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The one `(M, N, K, types, scope)` every `@coopmat` module in this tree uses.
///
/// `quant/qmm_t.slang` declares `CoopMat<half, Subgroup, 16, 16, MatrixA>`,
/// the same for B, and `CoopMat<float, Subgroup, 16, 16, MatrixAccumulator>`.
/// Every other coopmat module here matches it. One shape, so one query.
const OUR_MATRIX: (u32, u32, u32) = (16, 16, 16);

/// Whether the device advertises the configuration the coopmat tier needs.
///
/// The point of this function is that the FEATURE BIT DOES NOT IMPLY IT.
/// `VK_KHR_cooperative_matrix` guarantees no configuration whatsoever; the
/// device publishes a list and anything off it is undefined behaviour. On the
/// machine this was written on, the discrete card publishes fifteen entries
/// including this one, and Mesa's `lavapipe` -- which advertises the
/// extension, the feature, `shaderFloat16` and the memory model, and passes
/// every other admission test this crate had -- publishes four, all 8x8x8.
/// Handing it a 16x16x16 matrix segfaulted inside `vkCreateComputePipelines`
/// while the validation layer reported nothing, which is the correct
/// behaviour of a validation layer: undefined is not invalid.
fn advertises_the_matrix_this_tree_uses(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let ext = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let props = unsafe { ext.get_physical_device_cooperative_matrix_properties(physical) }
        .unwrap_or_default();
    let (m, n, k) = OUR_MATRIX;
    props.iter().any(|c| {
        c.m_size == m
            && c.n_size == n
            && c.k_size == k
            // A and B are `half`, the accumulator and the result `float`. All
            // four are checked because a device may publish the shape with
            // other component types and that is a different matrix.
            && c.a_type == vk::ComponentTypeKHR::FLOAT16
            && c.b_type == vk::ComponentTypeKHR::FLOAT16
            && c.c_type == vk::ComponentTypeKHR::FLOAT32
            && c.result_type == vk::ComponentTypeKHR::FLOAT32
            // Subgroup scope, because that is what the typealias says. A
            // workgroup-scoped entry of the same shape is not this matrix.
            && c.scope == vk::ScopeKHR::SUBGROUP
    })
}

/// How long a fence wait may take before the driver calls it a failure.
///
/// This was `fence_timeout_ns()` written out three times, with nothing saying what
/// the number was for or what it cost. It is a deadlock guard: a `vkQueueSubmit`
/// whose fence never signals would otherwise wedge the calling thread forever,
/// and a scheduler blocked in an un-timed wait cannot even report that it is
/// stuck. Ten seconds is enormous next to the ~5 ms a decode step takes on the
/// card this was written against.
///
/// The cost is that a device merely SLOW is indistinguishable from a device
/// that is gone. That is not hypothetical and it is not new -- `fire`'s
/// recovery path already documents a prefill tile missing this wait in a debug
/// build under two validation layers, and how the timeout was then buried
/// under the fault it caused. Measured again, and much harder, on Mesa's
/// `llvmpipe`: with the coopmat tier correctly declined, 59 of this crate's 72
/// device tests pass there and 13 fail, none of them on an answer. They fail
/// on this number. A CPU implementation running a real model's prefill does
/// not finish a tile in ten seconds and never will.
///
/// So the deadline is now named, stated once, and can be raised for a device
/// that deserves more time. `PIE_VULKAN_FENCE_TIMEOUT_SECS` takes whole
/// seconds; a value that does not parse, or is zero, is ignored in favour of
/// the default, because a submit with no deadline at all is the hang this
/// exists to prevent and a typo should not buy one.
fn fence_timeout_ns() -> u64 {
    const DEFAULT_SECS: u64 = 10;
    let secs = std::env::var("PIE_VULKAN_FENCE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(DEFAULT_SECS);
    // Saturating, because `u64::MAX` nanoseconds is how Vulkan spells "wait
    // forever" and nobody should reach it by writing a large number of
    // seconds. Clamped an hour below it so the multiplication cannot land
    // there by accident.
    secs.saturating_mul(1_000_000_000).min(u64::MAX - 1)
}

/// What a device that ran out of time is told, and it is not `TIMEOUT`.
///
/// The deadline above is raisable, which is worth nothing to the person who
/// hits it, because the three wait sites reported `wait: ERROR_TIMEOUT` and
/// stopped there. That names neither how long the driver actually waited nor
/// the one environment variable that would have let it wait longer. The
/// knob existed and was reachable only by reading this file.
///
/// It is the timeout arm specifically that gets the longer message.
/// `ERROR_DEVICE_LOST` is a different event with a different remedy, and
/// telling someone whose GPU fell off the bus to raise a deadline would send
/// them the wrong way.
fn waited_too_long(during: &str, e: vk::Result, ns: u64) -> Failed {
    if e == vk::Result::TIMEOUT {
        let secs = ns / 1_000_000_000;
        return Failed::Vulkan(format!(
            "{during}: wait: the device did not signal within {secs}s. That is a \
             deadlock guard, not a device limit -- a slow device (a software \
             adapter running a real model's prefill, say) needs more than the \
             default. Raise it with PIE_VULKAN_FENCE_TIMEOUT_SECS=<whole seconds>."
        ));
    }
    Failed::Vulkan(format!("{during}: wait: {e}"))
}

impl Device {
    /// Open the first device with a compute queue.
    ///
    /// # Errors
    ///
    /// [`Unavailable`] when there is no loader, no device, or no compute queue.
    /// None of those is a defect — they are what a machine without a GPU looks
    /// like — so this returns rather than panics.
    pub fn open() -> Result<Self, Unavailable> {
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| Unavailable(format!("no loader: {e}")))?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"driver-vulkan")
            .api_version(vk::API_VERSION_1_3);

        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let validation = c"VK_LAYER_KHRONOS_validation";
        let validated = layers
            .iter()
            .any(|l| l.layer_name_as_c_str().is_ok_and(|s| s == validation));
        let enabled_layers: Vec<*const c_char> = if validated {
            vec![validation.as_ptr()]
        } else {
            Vec::new()
        };
        let enabled_exts: Vec<*const c_char> = if validated {
            vec![ash::ext::debug_utils::NAME.as_ptr()]
        } else {
            Vec::new()
        };

        // Beyond the default checks, which only read what the API was ASKED to
        // do. Synchronization validation tracks real hazards between
        // dispatches -- a missing barrier gives the right answer on this device
        // most of the time, so no comparison of numbers will ever find one.
        let wanted = [
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
        ];
        let mut features = vk::ValidationFeaturesEXT::default();
        if validated {
            features = features.enabled_validation_features(&wanted);
        }

        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_exts);
        if validated {
            info = info.push_next(&mut features);
        }
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|e| Unavailable(format!("no instance: {e}")))?;

        let messenger = validated.then(|| {
            let debug = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let create = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                )
                .pfn_user_callback(Some(fail_on_validation_error));
            unsafe { debug.create_debug_utils_messenger(&create, None) }
                .ok()
                .map(|m| (debug, m))
        });
        let messenger = messenger.flatten();

        // From here on a failure has to destroy the instance, so the body is a
        // closure and the cleanup happens once.
        match Self::finish(entry, instance, messenger, validated) {
            Ok(d) => Ok(d),
            Err((e, instance, messenger)) => {
                unsafe {
                    if let Some((debug, m)) = messenger {
                        debug.destroy_debug_utils_messenger(m, None);
                    }
                    instance.destroy_instance(None);
                }
                Err(e)
            }
        }
    }

    /// Pick a device and create it, handing the instance back if it cannot.
    #[allow(clippy::type_complexity)]
    // The `Err` is large because it CARRIES the instance and the messenger.
    // They were created by the caller, this function is the only thing that
    // could have consumed them, and on failure the caller is the only thing
    // left that can destroy them. Boxing to shrink the variant would put an
    // allocation between the loader and its own teardown; making the error
    // small by dropping them would leak an instance on every machine without a
    // compute queue.
    #[allow(clippy::result_large_err)]
    fn finish(
        entry: ash::Entry,
        instance: ash::Instance,
        messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
        validated: bool,
    ) -> Result<
        Self,
        (
            Unavailable,
            ash::Instance,
            Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
        ),
    > {
        macro_rules! bail {
            ($($t:tt)*) => {
                return Err((Unavailable(format!($($t)*)), instance, messenger))
            };
        }

        let devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(d) => d,
            Err(e) => bail!("cannot enumerate devices: {e}"),
        };
        if devices.is_empty() {
            bail!("the loader found no physical device")
        }
        let seen: Vec<Candidate> = devices
            .iter()
            .map(|&d| Candidate::read(&instance, d))
            .collect();
        let Some(chosen) = Candidate::choose(&seen) else {
            bail!(
                "no device here can compute. The loader offered {}",
                Candidate::roster(&seen)
            )
        };
        let physical = chosen.handle;
        let name = chosen.name.clone();
        let family = chosen.compute_family.expect("choose() only returns those");
        let props = chosen.props;

        // Ask what the device has, then enable the subset the shader tree
        // needs. Naming them one by one rather than handing back everything
        // reported is what makes this list documentation: these are the
        // non-core things every module here assumes.
        // Whether the matrix unit is even offered. Asked of the device rather
        // than assumed, and asked BEFORE the feature query, because pushing
        // `PhysicalDeviceCooperativeMatrixFeaturesKHR` into a chain on a
        // device without the extension is not a question it can answer.
        let extensions = match unsafe { instance.enumerate_device_extension_properties(physical) } {
            Ok(e) => e,
            Err(e) => bail!("cannot enumerate device extensions: {e}"),
        };
        let has_coopmat = extensions.iter().any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|s| s == ash::khr::cooperative_matrix::NAME)
        });

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        {
            // Its own scope, and fresh structs below. `push_next` leaves each
            // struct's `p_next` pointing at the previous one, so feeding the
            // same structs into a second chain closes a cycle and the loader
            // walks it until it overflows.
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12);
            if has_coopmat {
                query = query.push_next(&mut fcm);
            }
            unsafe { instance.get_physical_device_features2(physical, &mut query) };
        }
        let core = unsafe { instance.get_physical_device_features(physical) };

        if core.robust_buffer_access != vk::TRUE {
            // Refusing rather than continuing. `quant/qmm_t.slang` fetches
            // outside its matrix on purpose and needs those reads defined as
            // zero; without this the spec allows neighbouring memory, which is
            // a wrong answer and not a crash.
            bail!("{name} has no robustBufferAccess, which the tiled GEMM depends on");
        }

        let mut e11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f11.storage_buffer16_bit_access == vk::TRUE)
            .uniform_and_storage_buffer16_bit_access(
                f11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
            );
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(f12.shader_float16 == vk::TRUE)
            .shader_int8(f12.shader_int8 == vk::TRUE)
            .storage_buffer8_bit_access(f12.storage_buffer8_bit_access == vk::TRUE)
            // The UNIFORM half is asked for as well, and it is not redundant.
            // The shaders only ever put `uint8_t` in a storage buffer, and
            // `glslc` used to emit exactly `StorageBuffer8BitAccess` for that.
            // `slangc` emits the wider `UniformAndStorageBuffer8BitAccess`
            // instead -- the same access, spelled with the capability that
            // also covers UBOs -- and a module may not DECLARE a capability
            // whose feature is off, whatever it goes on to do with it
            // (`VUID-VkShaderModuleCreateInfo-pCode-08740`, which the
            // validation layer reported on the first module the device loads).
            .uniform_and_storage_buffer8_bit_access(
                f12.uniform_and_storage_buffer8_bit_access == vk::TRUE,
            )
            .vulkan_memory_model(f12.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE);
        let mut ecm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(fcm.cooperative_matrix == vk::TRUE);
        let mut enable = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(core.shader_int16 == vk::TRUE)
                    .robust_buffer_access(true),
            )
            .push_next(&mut e11)
            .push_next(&mut e12);

        // Every name `Capability::requires()` lists, checked together rather
        // than one at a time. The tier's A/B operands are `float16_t`, so a
        // matrix unit with no `shaderFloat16` behind it is not enough; and
        // `GL_KHR_cooperative_matrix` pulls in `GL_KHR_memory_scope_semantics`,
        // so all 146 modules in the tier declare `OpCapability
        // VulkanMemoryModel`, which a module may not do unless the feature is
        // on (`VUID-VkShaderModuleCreateInfo-pCode-08740`).
        //
        // `vulkanMemoryModelDeviceScope` is the subtle one and it is not
        // optional: turning the memory model on changes the rules for EVERY
        // module the device loads, and `moe/route.slang` -- a BASELINE module
        // that has nothing to do with matrices -- counts token histograms with
        // a device-scoped `atomicAdd`. Enabling the model without the scope
        // breaks it.
        //
        // And the CONFIGURATION, which the feature bit does not imply.
        // `VK_KHR_cooperative_matrix` promises no shape at all: a device
        // advertises a list of `(M, N, K, types, scope)` tuples and a
        // `coopmat` outside that list is undefined behaviour, not a slow path.
        // Asking only the feature was measured wrong rather than argued:
        // Mesa's `lavapipe` advertises the extension, the feature, `float16`
        // and the memory model, and advertises exactly four configurations,
        // all of them 8x8x8. This tree's matrices are 16x16x16 -- see
        // `quant/qmm_t.slang`, whose `MatA`/`MatB`/`MatAcc` fix that shape --
        // so admitting the tier there handed the driver a matrix it had never
        // claimed, and `vkCreateComputePipelines` segfaulted with the
        // validation layer reporting nothing at all. It reports nothing
        // because nothing illegal happened: undefined is not invalid.
        let coopmat = has_coopmat
            && fcm.cooperative_matrix == vk::TRUE
            && f12.shader_float16 == vk::TRUE
            && f12.vulkan_memory_model == vk::TRUE
            && f12.vulkan_memory_model_device_scope == vk::TRUE
            && advertises_the_matrix_this_tree_uses(&entry, &instance, physical);

        let mut tiers = vec![Capability::Baseline];
        if f12.shader_float16 == vk::TRUE {
            tiers.push(Capability::Fp16);
        }
        if coopmat {
            tiers.push(Capability::Coopmat);
            enable = enable.push_next(&mut ecm);
        }
        tiers.sort_unstable();
        tiers.reverse();

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        // The EXTENSION as well as the feature. A feature struct pushed into
        // the create chain for an extension that was not enabled is ignored,
        // silently, and the tier would then load a module declaring a
        // capability that is not there.
        let device_exts: Vec<*const c_char> = if coopmat {
            vec![ash::khr::cooperative_matrix::NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&device_exts)
            .push_next(&mut enable);
        let device = match unsafe { instance.create_device(physical, &create, None) } {
            Ok(d) => d,
            Err(e) => bail!("cannot create a device on {name}: {e}"),
        };
        let queue = unsafe { device.get_device_queue(family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                unsafe { device.destroy_device(None) };
                bail!("cannot create a command pool on {name}: {e}")
            }
        };

        // The command buffer and the fence a fire reuses, made once here so
        // that no fire has to. A failure at this point is a device that
        // cannot record at all, which is the same class of failure as not
        // having a queue.
        let scratch = match unsafe { Scratch::new(&device, pool) } {
            Ok(s) => s,
            Err(e) => {
                unsafe {
                    device.destroy_command_pool(pool, None);
                    device.destroy_device(None);
                }
                bail!("cannot prepare a recording on {name}: {e}")
            }
        };

        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        // Timestamps, if asked for and if the device can answer.
        //
        // `timestampComputeAndGraphics` is the one check worth making: the
        // specification says that when it is true, every queue family that
        // supports graphics or compute reports 64 valid timestamp bits. A
        // device that says false may still support them on some family, and
        // this driver does not go looking -- a measurement tool that reports
        // on some cards and silently mis-reports on others is worse than one
        // that says it is unavailable.
        let timing = (std::env::var_os("PIE_VULKAN_TIMING").is_some()
            && props.limits.timestamp_compute_and_graphics == vk::TRUE
            && props.limits.timestamp_period > 0.0)
            .then(|| {
                let slots = 4096u32;
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(slots);
                unsafe { device.create_query_pool(&info, None) }
                    .ok()
                    .map(|pool| Timing {
                        pool,
                        period: props.limits.timestamp_period,
                        slots,
                        per_symbol: Mutex::new(HashMap::new()),
                        skipped: std::sync::atomic::AtomicU32::new(0),
                    })
            })
            .flatten();
        Ok(Self {
            entry,
            messenger,
            instance,
            device,
            queue,
            pool,
            memory,
            name,
            max_push: props.limits.max_push_constants_size,
            min_storage_offset: props.limits.min_storage_buffer_offset_alignment,
            max_groups: props.limits.max_compute_work_group_count,
            unified: matches!(
                props.device_type,
                vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
            ),
            software: props.device_type == vk::PhysicalDeviceType::CPU,
            validated,
            tiers,
            scratch: Mutex::new(scratch),
            allocations: std::sync::atomic::AtomicU32::new(0),
            barriers: std::sync::atomic::AtomicU32::new(0),
            staging: Mutex::new(None),
            staged: std::sync::atomic::AtomicU32::new(0),
            frees: std::sync::atomic::AtomicU32::new(0),
            uploaded: std::sync::atomic::AtomicU64::new(0),
            timing,
        })
    }

    /// Where the device time went, per kernel, since this device was opened.
    ///
    /// Empty unless `PIE_VULKAN_TIMING` was set in the environment when the
    /// device was opened -- see [`Timing`], which also states what the numbers
    /// do and do not mean. Sorted longest first, as milliseconds and the
    /// number of dispatches that made them up.
    #[must_use]
    pub fn timings(&self) -> Vec<(String, f64, u32)> {
        let Some(t) = self.timing.as_ref() else {
            return Vec::new();
        };
        let per = t
            .per_symbol
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut rows: Vec<(String, f64, u32)> = per
            .iter()
            .map(|(k, (ticks, n))| (k.clone(), *ticks as f64 * f64::from(t.period) / 1.0e6, *n))
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }

    /// How many fires were too big to time, and so are missing from
    /// [`Device::timings`].
    ///
    /// Public because a total that silently omits the largest fire is the one
    /// way this tool could mislead, and a reader has to be able to rule it out.
    #[must_use]
    pub fn timings_skipped(&self) -> u32 {
        self.timing
            .as_ref()
            .map_or(0, |t| t.skipped.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// How many descriptor pools the fires on this device have needed.
    ///
    /// One per fire would mean the pool is not being reused; a number that
    /// stops climbing means it is. Public so a test can say which of those
    /// is happening, since both compute the right answer.
    #[must_use]
    pub fn pools_made(&self) -> u32 {
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .made
    }

    /// How many buffers this device has been asked for, ever.
    ///
    /// Counts allocations that succeeded. See the field for why the number,
    /// and not just the elapsed time, is what a test should hold.
    #[must_use]
    pub fn allocations(&self) -> u32 {
        self.allocations.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many bytes this device has uploaded through [`Device::write`].
    ///
    /// Cumulative over the device's life. See the field for why bytes and not
    /// calls.
    #[must_use]
    pub fn uploaded(&self) -> u64 {
        self.uploaded.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many pipeline barriers this device's fires have recorded.
    ///
    /// Cumulative over the device's life. See the field, and [`hazards`] for
    /// what decides them.
    #[must_use]
    pub fn barriers(&self) -> u32 {
        self.barriers.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many reads went through the copy engine.
    ///
    /// See [`Device::read_at`]: a read that does NOT is a read of uncached
    /// write-combined memory, which on this card runs at ten megabytes a
    /// second.
    #[must_use]
    pub fn staged(&self) -> u32 {
        self.staged.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many buffers this device holds that nothing has freed.
    ///
    /// A fire that returns -- with an answer or with a refusal -- should leave
    /// this where it found it.
    #[must_use]
    pub fn live_buffers(&self) -> u32 {
        self.allocations()
            .saturating_sub(self.frees.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// What the device calls itself.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Is a validation layer watching?
    ///
    /// Worth asking rather than assuming: a test that means to prove a dispatch
    /// is LEGAL proves nothing without one, and should say so instead of
    /// passing.
    #[must_use]
    pub fn validated(&self) -> bool {
        self.validated
    }

    /// The tiers this device can load, best first.
    ///
    /// What was ENABLED, not what was reported. The distinction matters
    /// because a module may not declare a capability whose feature is off, so
    /// a device that has `shaderFloat16` and a driver that did not turn it on
    /// is a device that cannot load the fp16 tier -- and answering from the
    /// report would say otherwise.
    ///
    /// [`Capability::Baseline`] is always first-or-last and always present:
    /// every entrypoint has a baseline module, so a device offering nothing
    /// optional still resolves all 480.
    #[must_use]
    pub fn tiers(&self) -> &[Capability] {
        &self.tiers
    }

    /// The best module for `entrypoint` this device can load, and its tier.
    ///
    /// Walks the tiers best-first and takes the first one `dir` actually has.
    /// Both halves are required: a device may support a tier this build did
    /// not compile, and a build may have a tier this device cannot load. The
    /// answer is `None` only when even the baseline module is missing, which
    /// means the directory is not one a `native` build wrote.
    #[must_use]
    pub fn module_for(
        &self,
        dir: &std::path::Path,
        entrypoint: &str,
    ) -> Option<(PathBuf, Capability)> {
        self.tiers.iter().find_map(|&tier| {
            let path = dir.join(tier.module(entrypoint));
            path.exists().then_some((path, tier))
        })
    }

    /// `maxPushConstantsSize`.
    #[must_use]
    pub fn max_push(&self) -> u32 {
        self.max_push
    }

    /// `maxComputeWorkGroupCount`: the widest grid this device will dispatch.
    ///
    /// Per axis, because they differ: the specification's floor is 65535 on
    /// all three, and a card that raises the first one has not necessarily
    /// raised the others.
    #[must_use]
    pub fn max_groups(&self) -> [u32; 3] {
        self.max_groups
    }

    /// `minStorageBufferOffsetAlignment`: the granularity a sub-range may
    /// start at.
    ///
    /// Always a power of two and at least 1, per the specification, so
    /// [`Bound::at`] may mask rather than divide.
    #[must_use]
    pub fn min_storage_offset(&self) -> u64 {
        self.min_storage_offset
    }

    /// Is host memory and device memory the same memory?
    ///
    /// True on an integrated GPU and on a software implementation, false on a
    /// discrete card. See the field for why the heaps do not answer this.
    #[must_use]
    pub fn unified(&self) -> bool {
        self.unified
    }

    /// Whether this device is a CPU implementation of Vulkan.
    ///
    /// Public so that a test asserting a TIME can say what it is timing. Every
    /// correctness check in this crate applies unchanged to a software
    /// adapter -- and 59 of the 72 device tests already passed on `llvmpipe`
    /// the first time it was tried -- but a ceiling in milliseconds is a
    /// statement about a particular piece of hardware, and enforcing it
    /// against an LLVM JIT measures the host's cores.
    #[must_use]
    pub fn software(&self) -> bool {
        self.software
    }

    /// Does the device expose memory the host cannot see?
    ///
    /// The second signal for [`Self::unified`], and independent of it: that one
    /// reads `deviceType`, this one reads the heaps. A part where every
    /// device-local type is also host-visible has one pool of memory; a part
    /// with a device-local type the host cannot map has two. They must agree,
    /// and `tests/device.rs` is where that is held.
    #[must_use]
    pub fn device_only_memory(&self) -> bool {
        self.memory.memory_types[..self.memory.memory_type_count as usize]
            .iter()
            .any(|t| {
                t.property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !t
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
    }

    /// The largest heap this driver could allocate a buffer out of, in bytes.
    ///
    /// Every buffer here is host-visible and coherent (see [`Self::buffer`]),
    /// so the answer is the largest heap backing a HOST_VISIBLE type and not
    /// the largest heap on the part -- on a discrete card those differ by the
    /// whole of VRAM, and the wrong one turns an allocation that will never
    /// succeed into one a caller waits for.
    ///
    /// An upper bound and not a promise: the heap is shared with the weights,
    /// with every other process on the device, and with whatever the
    /// allocator has fragmented. It is used for one thing -- telling a
    /// scheduler that a demand can never be met apart from one that cannot be
    /// met NOW -- and for that, a bound that is too generous merely turns a
    /// permanent refusal into a retried one.
    #[must_use]
    pub fn budget(&self) -> u64 {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        types
            .iter()
            .filter(|t| {
                t.property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
            .filter_map(|t| self.memory.memory_heaps.get(t.heap_index as usize))
            .map(|h| h.size)
            .max()
            .unwrap_or(0)
    }

    /// A host-visible storage buffer holding `bytes`.
    ///
    /// Host-visible and coherent throughout. This is a correctness shell: being
    /// able to read a result back without a staging copy is worth more here
    /// than the bandwidth a device-local heap would add.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the allocation fails or the device exposes no
    /// host-visible memory type.
    pub fn buffer(&self, bytes: &[u8]) -> Result<Buffer, Failed> {
        // At least four bytes: a zero-sized buffer cannot be created, and an
        // operand a variant never reads still needs a descriptor to point at.
        let buffer = self.allocate(bytes.len().max(4) as u64)?;
        if !bytes.is_empty() {
            self.write(&buffer, bytes)?;
        }
        Ok(buffer)
    }

    /// A buffer of `size` bytes whose contents are UNDEFINED.
    ///
    /// The allocation half of [`Device::buffer`] without the write half, for
    /// a caller that is about to fill the whole thing on the device. Filling
    /// it from the host instead would mean holding `size` bytes of host
    /// memory to do it, which for a KV pool is the pool's own size again.
    ///
    /// Undefined, not zero: Vulkan does not clear a fresh allocation, and
    /// pretending otherwise is the kind of assumption that reads correctly
    /// for a year. A caller that needs zeros asks for [`Device::zero`].
    ///
    /// # Errors
    ///
    /// As [`Device::buffer`].
    pub fn empty(&self, size: u64) -> Result<Buffer, Failed> {
        self.allocate(size.max(4))
    }

    /// The allocation both of the above share.
    fn allocate(&self, size: u64) -> Result<Buffer, Failed> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            // TRANSFER_SRC as well as STORAGE_BUFFER, because a buffer in
            // mappable VRAM is write-combined and reading one back through
            // its mapping is uncached: measured on this card, 334 megabytes
            // of a prefill's arena took 33 SECONDS to copy out, which is ten
            // megabytes a second on a bus that does twelve gigabytes. The
            // copy engine reads the same memory at bus speed, so `read_at`
            // DMAs into a host-cached staging buffer instead -- and it can
            // only do that if the buffer it reads was created as a transfer
            // source.
            // TRANSFER_DST as well, so a buffer can be the DESTINATION of a
            // device-side copy. That is what lets a pool resize move the
            // pages it keeps without either side touching host memory: see
            // [`Device::copy_between`], and the measurement beside
            // `Pool::resize`.
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { self.device.create_buffer(&info, None) }
            .map_err(|e| Failed::of_vulkan(e, "create buffer", size))?;
        let need = unsafe { self.device.get_buffer_memory_requirements(handle) };

        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // DEVICE_LOCAL FIRST, and this one line was the whole performance
        // story of this driver.
        //
        // Every buffer here is host-visible, because this driver writes
        // weights, page tables and scalars straight into mapped memory and
        // has no staging path. That requirement is real. What was NOT
        // examined is that a discrete card offers SEVERAL host-visible types,
        // and this took the first one it found. On the 4090 the list is:
        //
        // | type | flags | heap |
        // |---|---|---|
        // | 1 | `DEVICE_LOCAL` | 24 GB, VRAM |
        // | 2 | `HOST_VISIBLE \| HOST_COHERENT` | 47 GB, system RAM |
        // | 3 | + `HOST_CACHED` | 47 GB, system RAM |
        // | 4 | `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | 24 GB, VRAM |
        //
        // The first match is type 2. So every weight, every KV page and every
        // activation lived in SYSTEM MEMORY, and each of the 452 dispatches
        // in a decode step reached across PCIe for all of it. Type 4 is the
        // same memory the card computes out of, mappable across its whole
        // twenty-four gigabytes because resizable BAR is on.
        //
        // This is what "about 12 GB/s on a card that does roughly a thousand"
        // was, at every launch size, for every kernel, uniformly -- the
        // uniformity being the clue that went unread for three refuted
        // hypotheses about the shaders. PCIe 4.0 x16 is 32 GB/s of theory and
        // twelve of practice. The kernels were never the ceiling; the bus
        // was.
        //
        // Falling back rather than requiring it, in both directions. A part
        // without a device-local host-visible type -- an older card without
        // resizable BAR exposes 256 MB of one, an integrated part has only
        // the one pool -- still gets the type it always got. And a device-
        // local allocation that FAILS falls back too, because the mappable
        // VRAM heap is smaller than system memory and shared with every other
        // process on the card: a model that no longer fits should get slower,
        // not refused.
        let prefers = |flags: vk::MemoryPropertyFlags| {
            (0..self.memory.memory_type_count).find(|i| {
                need.memory_type_bits & (1 << i) != 0
                    && self.memory.memory_types[*i as usize]
                        .property_flags
                        .contains(flags)
            })
        };
        let local = prefers(want | vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let Some(index) = local.or_else(|| prefers(want)) else {
            unsafe { self.device.destroy_buffer(handle, None) };
            return Err(Failed::Vulkan("no host-visible memory type".into()));
        };

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        let memory = match unsafe { self.device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            // The device-local heap is the one that runs out. Ask again for
            // the plain host-visible type before reporting a failure, so that
            // a card whose VRAM is full serves slowly instead of refusing.
            Err(_) if local == Some(index) => {
                let Some(fallback) = prefers(want) else {
                    unsafe { self.device.destroy_buffer(handle, None) };
                    return Err(Failed::Vulkan("no host-visible memory type".into()));
                };
                let alloc = vk::MemoryAllocateInfo::default()
                    .allocation_size(need.size)
                    .memory_type_index(fallback);
                match unsafe { self.device.allocate_memory(&alloc, None) } {
                    Ok(m) => m,
                    Err(e) => {
                        unsafe { self.device.destroy_buffer(handle, None) };
                        return Err(Failed::of_vulkan(e, "allocate", need.size));
                    }
                }
            }
            Err(e) => {
                unsafe { self.device.destroy_buffer(handle, None) };
                return Err(Failed::of_vulkan(e, "allocate", need.size));
            }
        };
        if let Err(e) = unsafe { self.device.bind_buffer_memory(handle, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(handle, None);
            }
            return Err(Failed::Vulkan(format!("bind: {e}")));
        }

        let buffer = Buffer {
            handle,
            memory,
            size,
            mapped: need.size,
            local: self.memory.memory_types[index as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
        };
        self.allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(buffer)
    }

    /// Overwrite a buffer's first `bytes.len()` bytes.
    ///
    /// # Why this one stays on the host when the copies did not
    ///
    /// Because the write-combined penalty is one-directional, and that is
    /// worth stating where somebody auditing the mappings will read it.
    /// Write-combining exists to make streaming stores fast; it is the LOAD
    /// side that runs uncached. Measured on this card, through the same
    /// mapping [`Device::copy_within`] was abandoning:
    ///
    /// ```text
    ///    64 KiB    10.1 GB/s
    ///     1 MiB    10.3 GB/s
    ///    16 MiB    10.2 GB/s
    /// ```
    ///
    /// Against about thirty megabytes a second for a host READ of the same
    /// memory -- a factor of three hundred and forty between the directions.
    /// So an upload has nothing to gain from a staging buffer and would pay
    /// an extra copy for it, while a download stages ([`Device::read_at`])
    /// and an in-buffer move goes to the copy engine. Three different answers
    /// from one measurement, which is why the number lives here rather than
    /// in a commit message.
    ///
    /// This is the per-step path as well as the load path:
    /// [`crate::resources::Pool::state`] rewrites a fire's tables in place
    /// through it whenever the size is unchanged.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the mapping fails, or if `bytes` is longer than
    /// the buffer — which is refused rather than truncated, since a short write
    /// leaves the tail holding the previous fire's numbers and every kernel
    /// here reads its whole operand.
    pub fn write(&self, buffer: &Buffer, bytes: &[u8]) -> Result<(), Failed> {
        if bytes.len() as u64 > buffer.size {
            return Err(Failed::Vulkan(format!(
                "{} bytes into a {}-byte buffer",
                bytes.len(),
                buffer.size
            )));
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.mapped, vk::MemoryMapFlags::empty())
                .map_err(|e| Failed::Vulkan(format!("map: {e}")))?
                .cast::<u8>();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            self.device.unmap_memory(buffer.memory);
        }
        self.uploaded
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Copy one range of a buffer over another range of the SAME buffer.
    ///
    /// # Why one buffer and not two
    ///
    /// The cache is one buffer per layer holding every page, so moving a
    /// conversation's history is a move within a buffer and never between two.
    /// A general two-buffer copy would be a second thing to test for the one
    /// caller that does not exist.
    ///
    /// # Which route, and why it is not always the same one
    ///
    /// Every buffer this driver allocates is host-visible and coherent -- see
    /// [`Device::buffer`] -- so a `memmove` through a mapping is a copy with
    /// no command buffer, no barrier and nothing in flight when it returns.
    /// That was the whole implementation, and being CORRECT is the only thing
    /// it was: host-visible is not the same as fast to read. This card's
    /// mappable VRAM is write-combined, so the load side of that `memmove`
    /// runs uncached, and the copy never needed to leave the card at all.
    /// Measured here, one range onto another of the same size:
    ///
    /// ```text
    ///     1 KiB   mapping 17.8 us    copy engine 41.0 us
    ///     2 KiB   mapping 35.1 us    copy engine 15.9 us
    ///    32 KiB   mapping 1.04 ms    copy engine 22.9 us
    ///   256 KiB   mapping 8.33 ms    copy engine 27.0 us
    ///     1 MiB   mapping 33.37 ms   copy engine 27.4 us
    /// ```
    ///
    /// The mapping route is LINEAR in the bytes -- about thirty megabytes a
    /// second -- and the copy engine's cost is the submission, flat at some
    /// twenty microseconds. They cross at about a kilobyte and a half. At a
    /// megabyte it is a thousand times, and this is the path a prefix share
    /// and a fork take: [`crate::resources::Pool::copy_plan`] calls it once
    /// per layer per half for every page the engine asks to move. It is the
    /// same defect [`crate::resources::Pool::resize`] had and for the same
    /// reason, found by measuring the route rather than by any test going red.
    ///
    /// There is no size threshold, and the table is why one would be false
    /// economy: below the crossover the mapping saves twenty-three
    /// microseconds, and above it the copy engine saves thirty-three
    /// milliseconds. A branch that has to be right about a page's size in
    /// order to pay for itself is worth less than the line it costs.
    ///
    /// So the copy engine takes it whenever the two ranges are DISJOINT,
    /// which is every move a pool actually makes: a whole-page move names two
    /// different pages, and a cell move is one row. The mapping stays for the
    /// overlapping case, because a `vkCmdCopyBuffer` whose source and
    /// destination regions overlap within one buffer is undefined -- and the
    /// promise below is worth keeping rather than quietly narrowing.
    ///
    /// Overlapping ranges are allowed and move correctly: this is a `memmove`,
    /// not a `memcpy`. Stated rather than left to the reader because a page
    /// compaction moves pages DOWN, and a move by less than one page-size
    /// stride overlaps.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if either range leaves the buffer, or if the
    /// mapping or the submission fails. A range that left the buffer would
    /// otherwise be a write past an allocation, which this card does not
    /// report.
    pub fn copy_within(
        &self,
        buffer: &Buffer,
        from: u64,
        to: u64,
        bytes: u64,
    ) -> Result<(), Failed> {
        let ends = |at: u64| at.checked_add(bytes).is_some_and(|e| e <= buffer.size);
        if !ends(from) || !ends(to) {
            return Err(Failed::Vulkan(format!(
                "{bytes} bytes from {from} to {to} in a {}-byte buffer",
                buffer.size
            )));
        }
        if bytes == 0 || from == to {
            return Ok(());
        }
        // Disjoint, so the copy engine can have it. `from + bytes` and
        // `to + bytes` are both known not to overflow by the bounds check
        // above.
        if from + bytes <= to || to + bytes <= from {
            return self.submit_once("copy within a buffer", |device, cmd| {
                let region = [vk::BufferCopy::default()
                    .src_offset(from)
                    .dst_offset(to)
                    .size(bytes)];
                unsafe { device.cmd_copy_buffer(cmd, buffer.handle, buffer.handle, &region) };
            });
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.mapped, vk::MemoryMapFlags::empty())
                .map_err(|e| Failed::Vulkan(format!("map: {e}")))?
                .cast::<u8>();
            // `copy`, not `copy_nonoverlapping`: see above.
            std::ptr::copy(ptr.add(from as usize), ptr.add(to as usize), bytes as usize);
            self.device.unmap_memory(buffer.memory);
        }
        Ok(())
    }

    /// Copy `len` bytes from one buffer into another, ON THE DEVICE.
    ///
    /// One `vkCmdCopyBuffer` on the transfer path, one fence, and nothing
    /// through a mapping. That matters for the same reason [`Device::read_at`]
    /// stages: mappable VRAM is write-combined, so a host-side move of a
    /// device buffer pays an uncached read for every byte -- ten megabytes a
    /// second, measured on this card. The copy engine reads the same memory
    /// at the bus's rate and, for a device-to-device move, never leaves the
    /// card at all.
    ///
    /// Unlike [`Device::copy_within`], the two buffers must be DIFFERENT
    /// allocations; a `vkCmdCopyBuffer` whose regions overlap within one
    /// buffer is undefined, and the in-buffer move that a page compaction
    /// needs is what `copy_within` is for.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if either range leaves its buffer, or if the
    /// submission fails. Not a silent fallback like [`Device::stage`]: a
    /// caller moving a pool's contents cannot carry on without the bytes.
    pub fn copy_between(
        &self,
        src: &Buffer,
        src_at: u64,
        dst: &Buffer,
        dst_at: u64,
        len: u64,
    ) -> Result<(), Failed> {
        if src_at.checked_add(len).is_none_or(|e| e > src.size) {
            return Err(Failed::Vulkan(format!(
                "copy of {len} bytes from {src_at} in a {}-byte source",
                src.size
            )));
        }
        if dst_at.checked_add(len).is_none_or(|e| e > dst.size) {
            return Err(Failed::Vulkan(format!(
                "copy of {len} bytes to {dst_at} in a {}-byte destination",
                dst.size
            )));
        }
        if len == 0 {
            return Ok(());
        }
        self.submit_once("copy between buffers", |device, cmd| {
            let region = [vk::BufferCopy::default()
                .src_offset(src_at)
                .dst_offset(dst_at)
                .size(len)];
            unsafe { device.cmd_copy_buffer(cmd, src.handle, dst.handle, &region) };
        })
    }

    /// Fill `len` bytes of a buffer from `at` with zeros, on the device.
    ///
    /// The tail of a grown pool, without holding the tail in host memory to
    /// write it. `vkCmdFillBuffer` needs both the offset and the length to be
    /// multiples of four, which every pool extent here is -- a page holds
    /// whole elements of at least two bytes and at least two of them -- so a
    /// range that is not is a caller's arithmetic slip and says so.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the range leaves the buffer, is not
    /// four-byte-aligned in both offset and length, or if the submission
    /// fails.
    pub fn zero(&self, buffer: &Buffer, at: u64, len: u64) -> Result<(), Failed> {
        if at.checked_add(len).is_none_or(|e| e > buffer.size) {
            return Err(Failed::Vulkan(format!(
                "zero of {len} bytes from {at} in a {}-byte buffer",
                buffer.size
            )));
        }
        if !at.is_multiple_of(4) || !len.is_multiple_of(4) {
            return Err(Failed::Vulkan(format!(
                "zero of {len} bytes from {at} is not four-byte aligned"
            )));
        }
        if len == 0 {
            return Ok(());
        }
        self.submit_once("zero a buffer", |device, cmd| unsafe {
            device.cmd_fill_buffer(cmd, buffer.handle, at, len, 0);
        })
    }

    /// Record one transfer command, submit it, and wait for it.
    ///
    /// The scratch command buffer and fence are the same pair
    /// [`Device::stage`] uses, and the lock is what keeps the two off each
    /// other. Waited on rather than left in flight because every caller here
    /// reads or frees what it just moved.
    fn submit_once(
        &self,
        during: &'static str,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Failed> {
        let scratch = self
            .scratch
            .lock()
            .map_err(|_| Failed::Vulkan(format!("{during}: the scratch lock is poisoned")))?;
        let (cmd, fence) = (scratch.cmd, scratch.fence);
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| Failed::Vulkan(format!("{during}: begin: {e}")))?;
            record(&self.device, cmd);
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| Failed::Vulkan(format!("{during}: end: {e}")))?;
            self.device
                .reset_fences(&[fence])
                .map_err(|e| Failed::Vulkan(format!("{during}: reset: {e}")))?;
            let bufs = [cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            self.device
                .queue_submit(self.queue, &submits, fence)
                .map_err(|e| Failed::Vulkan(format!("{during}: submit: {e}")))?;
            let ns = fence_timeout_ns();
            self.device
                .wait_for_fences(&[fence], true, ns)
                .map_err(|e| waited_too_long(during, e, ns))?;
        }
        Ok(())
    }

    /// Read a buffer's contents back.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the mapping fails.
    pub fn read(&self, buffer: &Buffer) -> Result<Vec<u8>, Failed> {
        self.read_at(buffer, 0, buffer.size)
    }

    /// Read `len` bytes of a buffer from `at`.
    ///
    /// # Why this is not a mapped copy
    ///
    /// Every buffer this driver allocates prefers the memory type that is
    /// both `DEVICE_LOCAL` and `HOST_VISIBLE` -- see [`Device::buffer`],
    /// where that one line was worth five times the decode rate. Mappable
    /// VRAM is WRITE-COMBINED: writes through the mapping are fast because
    /// they coalesce, and reads through it are uncached, unprefetched, and
    /// one PCIe round trip deep.
    ///
    /// Measured on this card, on a 1024-token prefill of qwen3-0.6B:
    ///
    /// | phase | mapped | staged |
    /// |---|---|---|
    /// | allocate and zero the 334 MB arena | 82 ms | 82 ms |
    /// | every dispatch of every layer | 588 ms | 588 ms |
    /// | **read the answer back** | **32 967 ms** | **220 ms** |
    /// | widen 155 M bf16 logits to f32 | 278 ms | 278 ms |
    /// | the whole step | 33 847 ms | 1 107 ms |
    ///
    /// Ten megabytes a second, and ninety-eight per cent of the step. The
    /// dispatches -- the part a driver is for -- were under a sixtieth of it.
    ///
    /// So the read goes through the copy engine: a staging buffer in
    /// host-cached system memory, one `vkCmdCopyBuffer`, one fence, and then
    /// a cached `memcpy` out of it. The DMA reads VRAM at the bus's own rate
    /// and the host reads system RAM at the cache's.
    ///
    /// Two fallbacks, both to the mapping this used to be: a buffer whose
    /// memory is NOT device-local is already in system memory and staging it
    /// would be a copy for nothing, and a staging path that cannot allocate
    /// or cannot submit should be slow rather than a failure.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the range leaves the buffer, or if the mapping
    /// fails.
    pub fn read_at(&self, buffer: &Buffer, at: u64, len: u64) -> Result<Vec<u8>, Failed> {
        if at.checked_add(len).is_none_or(|e| e > buffer.size) {
            return Err(Failed::Vulkan(format!(
                "{len} bytes from {at} in a {}-byte buffer",
                buffer.size
            )));
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        if buffer.local
            && let Some(out) = self.stage(buffer, at, len)
        {
            self.staged
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(out);
        }
        let mut out = vec![0u8; len as usize];
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.mapped, vk::MemoryMapFlags::empty())
                .map_err(|e| Failed::Vulkan(format!("map: {e}")))?
                .cast::<u8>();
            std::ptr::copy_nonoverlapping(ptr.add(at as usize), out.as_mut_ptr(), out.len());
            self.device.unmap_memory(buffer.memory);
        }
        Ok(out)
    }

    /// `len` bytes from `at`, through the copy engine, or `None` to fall back.
    ///
    /// Every failure answers `None` rather than an error, because every one of
    /// them means "this could not be done the fast way" and the slow way is
    /// still there. The only thing that would make a failure here fatal is a
    /// partial copy, and there is no partial copy: the fence is waited on
    /// before anything is read out of the staging buffer.
    fn stage(&self, buffer: &Buffer, at: u64, len: u64) -> Option<Vec<u8>> {
        let mut held = self.staging.lock().ok()?;
        if held.as_ref().is_none_or(|b| b.size < len) {
            if let Some(old) = held.take() {
                // Waited on, because the buffer being replaced was the
                // destination of a copy this device submitted. Every such copy
                // was waited on below before its bytes were read, so nothing
                // is in flight -- but a free that depends on that reasoning
                // rather than on a wait is a use-after-free the day the wait
                // moves.
                unsafe { self.device.destroy_buffer(old.handle, None) };
                unsafe { self.device.free_memory(old.memory, None) };
            }
            *held = Some(self.staging_of(len)?);
        }
        let staging = held.as_ref()?;

        let copied = {
            let Ok(scratch) = self.scratch.lock() else {
                return None;
            };
            let (cmd, fence) = (scratch.cmd, scratch.fence);
            unsafe {
                self.device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .and_then(|()| {
                        let region = [vk::BufferCopy::default()
                            .src_offset(at)
                            .dst_offset(0)
                            .size(len)];
                        self.device
                            .cmd_copy_buffer(cmd, buffer.handle, staging.handle, &region);
                        self.device.end_command_buffer(cmd)
                    })
                    .and_then(|()| self.device.reset_fences(&[fence]))
                    .and_then(|()| {
                        let bufs = [cmd];
                        let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
                        self.device.queue_submit(self.queue, &submits, fence)
                    })
                    .and_then(|()| {
                        self.device
                            .wait_for_fences(&[fence], true, fence_timeout_ns())
                    })
            }
        };
        copied.ok()?;

        let mut out = vec![0u8; len as usize];
        let ptr = unsafe {
            self.device.map_memory(
                staging.memory,
                0,
                staging.mapped,
                vk::MemoryMapFlags::empty(),
            )
        }
        .ok()?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), out.len());
            self.device.unmap_memory(staging.memory);
        }
        Some(out)
    }

    /// A host-cached buffer of at least `len` bytes to DMA into.
    ///
    /// Not [`Device::buffer`], which prefers the memory this exists to avoid:
    /// that one asks for `DEVICE_LOCAL` first because the shaders read it, and
    /// an uncached staging buffer would move the uncached read from one side
    /// of the copy to the other.
    fn staging_of(&self, len: u64) -> Option<Buffer> {
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { self.device.create_buffer(&info, None) }.ok()?;
        let need = unsafe { self.device.get_buffer_memory_requirements(handle) };
        let prefers = |flags: vk::MemoryPropertyFlags| {
            (0..self.memory.memory_type_count).find(|i| {
                need.memory_type_bits & (1 << i) != 0
                    && self.memory.memory_types[*i as usize]
                        .property_flags
                        .contains(flags)
            })
        };
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // HOST_CACHED first, and it is the whole point.
        let index = prefers(want | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|| prefers(want));
        let Some(index) = index else {
            unsafe { self.device.destroy_buffer(handle, None) };
            return None;
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        let Ok(memory) = (unsafe { self.device.allocate_memory(&alloc, None) }) else {
            unsafe { self.device.destroy_buffer(handle, None) };
            return None;
        };
        if unsafe { self.device.bind_buffer_memory(handle, memory, 0) }.is_err() {
            unsafe {
                self.device.destroy_buffer(handle, None);
                self.device.free_memory(memory, None);
            }
            return None;
        }
        Some(Buffer {
            handle,
            memory,
            size: len,
            mapped: need.size,
            // Host memory by construction, and the flag is what stops a read
            // OF this buffer from trying to stage it into another one.
            local: false,
        })
    }

    /// Destroy a buffer.
    ///
    /// Explicit rather than a `Drop` on [`Buffer`], because freeing needs the
    /// device and a handle that carried one would make every buffer as large as
    /// a reference and impossible to store beside the device that owns it.
    pub fn free(&self, buffer: Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.handle, None);
            self.device.free_memory(buffer.memory, None);
        }
        self.frees
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Run one dispatch to completion and wait for it.
    ///
    /// Synchronous on purpose. This is the shell a correctness test drives, and
    /// a pipelined submission would make a wrong answer a race rather than a
    /// wrong answer.
    ///
    /// # Errors
    ///
    /// [`Failed::Bindings`] or [`Failed::Push`] when the call does not match
    /// what the module declares, [`Failed::Geometry`] when the rule cannot
    /// answer, and [`Failed::Vulkan`] for a failed call.
    pub fn run(
        &self,
        pipeline: &Pipeline,
        buffers: &[Bound<'_>],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<(), Failed> {
        // A run of one, rather than a second path that records a dispatch.
        // The two used to be written out separately, and the separate one
        // kept its own descriptor pool, command buffer and fence per call --
        // so every improvement to a fire had to be made twice or silently
        // was not.
        self.run_all(&[Recorded {
            symbol: "run_one",
            pipeline,
            buffers,
            writes: &[],
            push,
            groups,
        }])
        .map_err(|(_, e)| e)
    }

    /// Record a run of dispatches into one command buffer and submit once.
    ///
    /// [`Self::run`] is one dispatch, one command buffer, one submit and one
    /// fence wait, which is right for a test and wrong for a fire: a real
    /// plan states thousands of rectangles -- six texts here state 6584 --
    /// and one round trip to the queue per rectangle is most
    /// of the time a small model spends.
    ///
    /// Between every pair a full compute-to-compute memory barrier. Vulkan
    /// gives NO ordering between dispatches in one command buffer -- they may
    /// overlap, and a plan's launches are chained through the arena, so the
    /// second one reading the first one's output is the normal case rather
    /// than the exception. This is the single thing that makes a recorded run
    /// mean what the plan says, and its absence is silent: the layer does not
    /// complain, every call returns success, and the numbers are whatever the
    /// scheduler happened to produce.
    ///
    /// Each dispatch is checked exactly as [`Self::run`] checks one, and the
    /// whole run is refused if any of them is -- nothing is submitted, so a
    /// caller never has to reason about a partially executed plan.
    ///
    /// # Errors
    ///
    /// [`Failed`], with the index of the dispatch that produced it.
    pub fn run_all(&self, run: &[Recorded<'_, '_>]) -> Result<(), (usize, Failed)> {
        for (at, one) in run.iter().enumerate() {
            self.check(one).map_err(|e| (at, e))?;
        }
        if run.is_empty() {
            return Ok(());
        }
        // A poisoned lock means a fire panicked mid-recording. The objects
        // behind it are handles, not state a panic can leave half written --
        // and the alternative is a device that answers nothing forever -- so
        // the next fire takes them anyway.
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let descriptors = run.iter().map(|r| r.pipeline.bindings).sum::<u32>();
        // Safe because the previous fire waited on `scratch.fence` before it
        // returned, so no command buffer holding one of these sets is still
        // running.
        let pool = unsafe { scratch.for_run(&self.device, run.len() as u32, descriptors) }
            .map_err(|e| (0, Failed::Vulkan(format!("descriptor pool: {e}"))))?;
        // Safe because `pool` came from `scratch` a line ago and the
        // recording is the only user of either until it returns.
        let fired = unsafe { self.record_all(run, pool, &scratch) };
        if fired.is_err() {
            // A failed fire is the DANGEROUS case, not the harmless one.
            //
            // The caller's next act after a refusal is to give the scalar
            // block back -- `serve::fire` frees it on both paths -- and the
            // sets recorded a moment ago still name it. If the failure was the
            // fence timing out, the queue may also still be reading it. So the
            // device is brought to a halt before anything is freed.
            //
            // Found by asking for a decode over a thousand tokens of history
            // in a debug build under two validation layers, where one prefill
            // tile does not finish inside the ten-second wait. The timeout was
            // reported correctly and then buried: the free that followed it
            // tripped "vkDestroyBuffer(): can't be called on VkBuffer ...
            // currently in use by VkDescriptorSet", which this driver treats
            // as fatal, so the process aborted on the consequence and never
            // printed the cause. An hour went into the wrong bug.
            let _ = unsafe { self.device.device_wait_idle() };
        }
        // The sets are freed at the end of the fire that used them, rather
        // than at the start of the next one.
        //
        // `for_run` already resets, and for a long time that was the only
        // reset, which is a different claim than it looks: it means a fire's
        // descriptor sets outlive the fire, still naming its buffers, until
        // some later fire happens to want the pool. The scalar block is freed
        // as soon as the fire returns, so the window is every gap between two
        // fires.
        //
        // Safe for the same reason `for_run`'s reset is -- the fence was
        // waited on inside `record_all` -- and on the failing path because of
        // the idle above.
        let _ = unsafe {
            self.device
                .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
        };
        fired
    }

    /// Everything [`Self::run`] refuses a dispatch for, without recording it.
    fn check(&self, one: &Recorded<'_, '_>) -> Result<(), Failed> {
        let pipeline = one.pipeline;
        // One per slot in the layout, less the module's HOLES.
        //
        // Two different things make a layout wider than the bindings a module
        // decorates, and they pull opposite ways:
        //
        // * a hole, where `slangc` dropped a binding in the MIDDLE of the set.
        //   `affine_qmv_routed` has seven slots and one hole, and a real
        //   lowering states exactly six operands for it. Demanding seven
        //   would mean demanding a buffer for a binding no shader reads and
        //   the plan does not name -- the caller would have to invent one.
        //
        // * a caller whose row lists MORE buffers than the module decorates,
        //   which happens for eleven entrypoints and is legal.
        //   `layer_scalar_mul_bfloat16` lists four against a module of three,
        //   and `Pipelines::get` widens the layout to four so the call is not
        //   refused for being right.
        //
        // Subtracting holes from the layout answers both: four for the
        // second, six for the first. Counting only the decorated bindings
        // answered the first and broke the second, which is how the two were
        // found to be different questions.
        let real = pipeline.bindings as usize - pipeline.declared.holes();
        if one.buffers.len() != real {
            return Err(Failed::Bindings {
                module: real as u32,
                bound: one.buffers.len(),
            });
        }
        // Both directions. A short push leaves the shader reading bytes nothing
        // wrote, which is the previous dispatch's block and reads as a
        // plausible number.
        if one.push.len() != pipeline.push as usize {
            return Err(Failed::Push {
                range: pipeline.push,
                given: one.push.len(),
            });
        }
        if one.groups.contains(&0) {
            return Err(Failed::Vulkan(format!(
                "a dispatch of {:?} workgroups would run nothing and report success",
                one.groups
            )));
        }
        // And the other end of the same argument. A grid past
        // `maxComputeWorkGroupCount` is undefined rather than refused: the
        // card may dispatch the part that fits and return success, which is
        // an output computed for some of its rows and stale for the rest --
        // fluent, plausible, wrong. Named here, before anything is recorded,
        // because the alternative is a plan that runs on this card and
        // silently truncates on one whose limit is the specification's floor.
        for (axis, (groups, limit)) in one.groups.iter().zip(self.max_groups).enumerate() {
            if *groups > limit {
                return Err(Failed::Grid {
                    axis: axis as u32,
                    groups: *groups,
                    limit,
                });
            }
        }
        for (binding, bound) in slots(pipeline).zip(one.buffers) {
            let Some(Some(needs)) = pipeline.declared.block_bytes.get(binding) else {
                continue;
            };
            if bound.len < u64::from(*needs) {
                return Err(Failed::Short {
                    binding: binding as u32,
                    needs: *needs,
                    given: bound.len,
                });
            }
        }
        Ok(())
    }

    unsafe fn record_all(
        &self,
        run: &[Recorded<'_, '_>],
        pool: vk::DescriptorPool,
        scratch: &Scratch,
    ) -> Result<(), (usize, Failed)> {
        let device = &self.device;
        // Every set allocated and written BEFORE any recording. A descriptor
        // set is read when the command executes, not when it is bound, so a
        // set rewritten between two `cmd_bind_descriptor_sets` in one command
        // buffer would give both dispatches the second write. One set each is
        // the only arrangement that means what it looks like.
        let mut sets = Vec::with_capacity(run.len());
        for (at, one) in run.iter().enumerate() {
            let layouts = [one.pipeline.set_layout];
            let allocated = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            }
            .map_err(|e| (at, Failed::Vulkan(format!("descriptor set: {e}"))))?;
            let set = allocated[0];
            let infos: Vec<_> = one
                .buffers
                .iter()
                .map(|b| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.buffer.handle)
                        .offset(b.offset)
                        // The range, never `WHOLE_SIZE`. `WHOLE_SIZE` means
                        // "to the end of the buffer", so a sub-range written
                        // that way binds its own start and everything after
                        // it, and a shader that runs one row too far reads
                        // the NEXT tensor instead of faulting. The extent is
                        // the half of an operand that makes the overrun
                        // visible, and discarding it here would discard it at
                        // the only point where the device could act on it.
                        .range(b.len)
                })
                .collect();
            // Only the bindings the module actually decorates.
            //
            // 165 of this tree's 665 modules leave a hole -- 358 of them in
            // all -- because `slangc` drops the declaration of a buffer a
            // variant never reads, and `kv_append_paged` holes 10 and 11 on
            // purpose to keep Metal's ring-ABI slots. A hole is free on
            // Metal, where an argument index nothing is set at is one the
            // shader does not read; the question here was whether Vulkan
            // agrees, since the SET still needs a slot at every number up to
            // the highest.
            //
            // It does, and the specification says so in the VUID this would
            // otherwise trip: descriptors "must be valid IF THEY ARE
            // ACCESSED". Measured under GPU-assisted validation rather than
            // assumed -- dispatching with both holes of a 7-binding module
            // unwritten succeeds and the layer stays silent, while leaving a
            // decorated one unwritten reports VUID-vkCmdDispatch-None-08114
            // by name.
            //
            // Skipping them is not merely allowed, it is the only thing this
            // driver can do: a hole has no operand in the plan, so there is
            // no buffer to put there and inventing one would bind an
            // unrelated tensor to a slot on the theory that nothing reads it.
            let writes: Vec<_> = slots(one.pipeline)
                .zip(&infos)
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            if !writes.is_empty() {
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            sets.push(set);
        }

        // Reset rather than allocated. `begin_command_buffer` on a buffer in
        // the executable state is an implicit reset, but only for a pool made
        // with `RESET_COMMAND_BUFFER`, and saying it here is what ties this
        // code to that flag rather than to a memory of it.
        let cmd = scratch.cmd;
        unsafe { device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| (0, Failed::Vulkan(format!("command buffer: {e}"))))?;

        let result = (|| -> Result<(), Failed> {
            unsafe {
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| Failed::Vulkan(format!("begin: {e}")))?;
                // The ranges written, and the ranges touched at all, since
                // the last barrier. See `hazards`.
                let mut pending_writes: Vec<Span> = Vec::new();
                let mut pending_reads: Vec<Span> = Vec::new();
                // Timestamps only when the pool can hold one per dispatch plus
                // the opening one. A partial answer would be read as a whole
                // one, so a fire that does not fit is left out of the totals
                // and counted instead.
                let clock = self.timing.as_ref().filter(|t| {
                    let fits = run.len() < t.slots as usize;
                    if !fits {
                        t.skipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    fits
                });
                if let Some(t) = clock {
                    device.cmd_reset_query_pool(cmd, t.pool, 0, run.len() as u32 + 1);
                    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, t.pool, 0);
                }
                for (at, one) in run.iter().enumerate() {
                    if at > 0 && hazards(one, &pending_writes, &pending_reads) {
                        pending_writes.clear();
                        pending_reads.clear();
                        self.barriers
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // One global barrier rather than a buffer barrier per
                        // operand. Every buffer barrier this could state would
                        // name one of two or three buffers -- the arena, a KV
                        // page, a fire table -- and a driver coalesces them
                        // into the same stall, so the finer barrier would buy
                        // the finer PLACEMENT and nothing else. The placement
                        // is what `hazards` decides, above.
                        let barrier = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                    }
                    // A mask that is not exactly as long as the bindings is
                    // one this code cannot index, so every slot counts as a
                    // write -- which is what the driver did for every slot
                    // before any mask existed.
                    let blind = one.writes.len() != one.buffers.len();
                    for (i, b) in one.buffers.iter().enumerate() {
                        let span = Span::of(b);
                        if blind || one.writes[i] {
                            pending_writes.push(span);
                        }
                        pending_reads.push(span);
                    }
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        one.pipeline.pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        one.pipeline.layout,
                        0,
                        &[sets[at]],
                        &[],
                    );
                    if !one.push.is_empty() {
                        device.cmd_push_constants(
                            cmd,
                            one.pipeline.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            one.push,
                        );
                    }
                    device.cmd_dispatch(cmd, one.groups[0], one.groups[1], one.groups[2]);
                    if let Some(t) = clock {
                        device.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            t.pool,
                            at as u32 + 1,
                        );
                    }
                }
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| Failed::Vulkan(format!("end: {e}")))?;

                // The fire's own fence, unsignalled again. Reset HERE and
                // not after the wait, so that a fire which fails between the
                // reset and the submit leaves it in the state the next fire
                // resets anyway rather than in one that would hang it.
                let fence = scratch.fence;
                device
                    .reset_fences(&[fence])
                    .map_err(|e| Failed::Vulkan(format!("fence: {e}")))?;
                let cmd_bufs = [cmd];
                let submits = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
                let submitted = device
                    .queue_submit(self.queue, &submits, fence)
                    .map_err(|e| Failed::Vulkan(format!("submit: {e}")));
                submitted.and_then(|()| {
                    let ns = fence_timeout_ns();
                    device
                        .wait_for_fences(&[fence], true, ns)
                        .map_err(|e| waited_too_long("fire", e, ns))
                })?;

                // The fence has been waited on, so every query is available
                // and `WAIT` costs nothing; it is asked for anyway because
                // "available" is a property of the query and not of the fence,
                // and a read that assumed otherwise would return stale ticks
                // with no way to tell.
                if let Some(t) = clock {
                    let mut ticks = vec![0u64; run.len() + 1];
                    if device
                        .get_query_pool_results(
                            t.pool,
                            0,
                            &mut ticks,
                            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                        )
                        .is_ok()
                    {
                        let mut per = t
                            .per_symbol
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        for (at, one) in run.iter().enumerate() {
                            let slice = ticks[at + 1].saturating_sub(ticks[at]);
                            let e = per.entry(one.symbol.to_string()).or_insert((0, 0));
                            e.0 += slice;
                            e.1 += 1;
                        }
                    }
                }
                Ok(())
            }
        })();

        result.map_err(|e| (0, e))
    }
}

/// A range of one buffer, in the terms a hazard is decided in.
///
/// The buffer HANDLE and not the `&Buffer`, for the reason [`Bound`]'s
/// `PartialEq` states: two borrows of one buffer are the same memory, and a
/// comparison that said otherwise would miss the hazard between them.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Span {
    buffer: vk::Buffer,
    from: u64,
    to: u64,
}

impl Span {
    fn of(b: &Bound<'_>) -> Self {
        Self {
            buffer: b.buffer.handle,
            from: b.offset,
            to: b.offset + b.len,
        }
    }

    fn meets(self, other: Self) -> bool {
        self.buffer == other.buffer && self.from < other.to && other.from < self.to
    }
}

/// Does this dispatch have to wait for the ones recorded since the last
/// barrier?
///
/// # Why a fire is not one long chain
///
/// It looks like one -- a plan threads a hidden state through a few hundred
/// rectangles -- and for a long time this driver recorded it as one, with a
/// full pipeline barrier between every pair. That is not free: measured on
/// this card, a qwen3-0.6b decode of 452 rectangles spends 4.6 milliseconds
/// inside the submit, and 3.8 of those go away when the barriers do. Eight
/// microseconds each, which is what a compute-to-compute flush costs when the
/// dispatch it separates is a single row of a small model and takes two.
///
/// Most neighbouring pairs do not touch the same bytes. A layer's Q, K and V
/// projections all read the same normed rows and write three different
/// places; the per-head rotary writes are disjoint; a norm's scratch is not
/// the next norm's. What makes those visible is that this driver binds RANGES
/// and not whole buffers -- see [`Bound`], and the note there on why
/// `WHOLE_SIZE` is never used -- so two rectangles of one arena are two spans
/// that can be compared.
///
/// # The three hazards, and why reads are tracked too
///
/// A barrier goes in when this dispatch would otherwise race one already
/// recorded:
///
///   - **read-after-write**: it reads a range something pending wrote;
///   - **write-after-write**: it writes a range something pending wrote;
///   - **write-after-read**: it writes a range something pending READ, which
///     is the one it would be easy to leave out. Without a barrier the two
///     dispatches may overlap, so the earlier one's reads can land after the
///     later one's write and see the new bytes.
///
/// An operand the kernel row marks [`kernels::Ty::BufMut`] counts as both,
/// since "may write through" does not say it does not also read.
///
/// The write-after-read case is the one this crate cannot demonstrate, and
/// that is recorded rather than hidden. Dropping it entirely leaves the
/// byte-for-byte comparison against the one-submission-per-dispatch reference
/// passing on three runs of all six texts, and changes the count on exactly
/// one of them -- olmo2-1b, 227 barriers to 211. So it is here because the
/// specification requires it and not because a measurement caught it.
///
/// # Which of the three a decode actually spends
///
/// All three were counted on a warm qwen3-0.6b decode, classifying every
/// barrier by the first case that fired for it. The 311 barriers of a step
/// are **283 read-after-write, 28 write-after-write, and 0 write-after-read**
/// -- the three sum to the count exactly, which is the check that the
/// classification and the decision are reading the same thing.
///
/// This was measured to answer a specific suspicion, and refuted it. Barriers
/// are about 72% of a decode, and the cheap explanation would have been false
/// serialization: an arena that hands two independent operations overlapping
/// ranges makes them look dependent, and a transformer decode has plenty of
/// independent pairs -- the three projections, the gate and the up. If that
/// were happening it would show up as write-after-read and write-after-write,
/// because a false dependency is precisely one where no value flows.
///
/// It is not happening. 91% of the barriers are read-after-write: one
/// dispatch reads what the one before it wrote, which is what a forward pass
/// IS. There is no bookkeeping left to fix here, and the way to fewer
/// barriers is fewer kernels.
///
/// # What this trusts
///
/// The kernel table's operand types. A row that calls an output `Buf` would
/// have this omit a barrier the fire needs, and the result is not a fault --
/// it is a race, so it is a number that is right most of the time. That is
/// why the claim is checked against the coarse recording rather than
/// reasoned about: see the test that fires a real decode both ways and
/// compares the arenas byte for byte.
fn hazards(one: &Recorded<'_, '_>, wrote: &[Span], read: &[Span]) -> bool {
    // A pending list that has grown this long is one where the scan costs
    // more than the barrier it might save. Nothing in this tree reaches it --
    // a decode's longest barrier-free run is a few dozen rectangles -- and it
    // is here so that a fire which does cannot become quadratic.
    if read.len() > 512 {
        return true;
    }
    let blind = one.writes.len() != one.buffers.len();
    one.buffers.iter().enumerate().any(|(i, b)| {
        let span = Span::of(b);
        let writes = blind || one.writes[i];
        wrote.iter().any(|w| span.meets(*w)) || (writes && read.iter().any(|r| span.meets(*r)))
    })
}

/// One dispatch in a recorded run.
///
/// The same four things [`Device::run`] takes, named rather than positional
/// because a run states many of them and a swapped pair of arguments in a
/// list of hundreds is not something a reader would catch.
#[derive(Clone, Copy)]
pub struct Recorded<'a, 'b> {
    /// The entrypoint's name, for [`Device::timings`] only.
    ///
    /// Nothing in the recording reads it -- the pipeline below is what runs --
    /// and it is here because a per-dispatch duration with no name attached is
    /// a list of numbers. `serve` already has the symbol borrowed at the point
    /// it builds this, so carrying it costs a pointer and no allocation.
    pub symbol: &'a str,
    /// The compiled module and its layout.
    pub pipeline: &'a Pipeline,
    /// One range per binding the module reads, less its holes.
    pub buffers: &'a [Bound<'b>],
    /// Which of [`Self::buffers`] the shader may write through.
    ///
    /// Parallel to `buffers`. An empty slice means "no idea", which is read
    /// as "all of them" -- see [`Device::run_all`]'s barriers.
    pub writes: &'a [bool],
    /// The push block, empty if the module has none.
    pub push: &'a [u8],
    /// Workgroups in each dimension, none of them zero.
    pub groups: [u32; 3],
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Everything in flight must finish before anything it touched is
            // destroyed. A device that is dropped while a submission is live is
            // undefined, and the layer reports it as a use-after-free with no
            // obvious connection to the test that caused it.
            let _ = self.device.device_wait_idle();
            if let Some(t) = self.timing.as_ref() {
                self.device.destroy_query_pool(t.pool, None);
            }
            self.scratch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .destroy(&self.device, self.pool);
            if let Some(b) = self
                .staging
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                self.device.destroy_buffer(b.handle, None);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            if let Some((debug, m)) = self.messenger.take() {
                debug.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
        // `entry` owns the loaded loader and is dropped after, by field order.
        let _ = &self.entry;
    }
}

/// A storage buffer and the memory behind it.
///
/// Plain data with no device reference, so a caller can keep a table of these
/// beside the [`Device`] that made them. Freed with [`Device::free`].
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    /// What the caller asked for, which is what [`Device::read`] returns.
    size: u64,
    /// What was actually allocated, which is what may be mapped. The driver
    /// rounds up, and mapping past this is invalid while mapping only `size`
    /// is not always allowed for a coherent range.
    mapped: u64,
    /// Whether the memory it was given is device-local as well as mappable.
    ///
    /// Which is the same question as "is reading this back through its
    /// mapping uncached", and [`Device::read_at`] asks it to decide whether a
    /// staged copy is worth making. A buffer in plain system memory is read
    /// fastest by reading it.
    local: bool,
}

impl Buffer {
    /// Bytes the caller asked for.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// A buffer of a stated size that names no device allocation.
    ///
    /// Binding produces offsets and lengths AGAINST a buffer and never
    /// dereferences its handle, so every question in [`crate::binding`] can be
    /// asked without a GPU -- and asking them without one is why the arena
    /// arithmetic is tested at all, since the machines that change a lowering
    /// are not the machines that have a Vulkan device.
    ///
    /// Passing one of these to [`Device::run`] would bind a null handle, so it
    /// is `#[doc(hidden)]`: `tests/arena.rs` needs it to put a real plan
    /// through the real binder on a machine with no GPU, and nothing else has
    /// a reason to reach for it.
    #[doc(hidden)]
    #[must_use]
    pub fn placeholder(size: u64) -> Self {
        Self {
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
            mapped: size,
            local: false,
        }
    }
}

/// What one descriptor addresses: a buffer, and which part of it.
///
/// # Why this is not `&Buffer`
///
/// A driver does not allocate a buffer per tensor. It allocates an arena and
/// hands out offsets into it, because a fire's activations are hundreds of
/// values whose lifetimes nest and whose sizes are known together --
/// `driver-metal`'s binder resolves an operand to `Slice { address, bytes }`
/// for exactly this reason.
///
/// Metal can take that literally: `setBuffer:offset:` moves the base by a byte
/// count and the shader addresses from there. **Vulkan cannot.** A descriptor
/// carries an offset the device must be able to address from, and
/// `minStorageBufferOffsetAlignment` says how coarsely. So the arena model
/// needs a type that carries the offset and can be refused, rather than a
/// reference that cannot.
///
/// The extent travels with the address for the reason the Metal binder gives:
/// an arena reused across fires can be smaller than the new one needs, and an
/// operand whose length lives in a neighbouring field is a bound two call
/// sites have to agree about.
#[derive(Clone, Copy, Debug)]
pub struct Bound<'a> {
    buffer: &'a Buffer,
    offset: u64,
    len: u64,
}

/// Two bounds are the same range when they name the same MEMORY, not the
/// same `&Buffer`.
///
/// Written out rather than derived because the derived version would compare
/// the reference, and a caller holding two borrows of one buffer would find
/// its two identical ranges unequal. What a test asks of a dispatch is where
/// it points, and the answer is the handle, the offset and the length.
impl PartialEq for Bound<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.buffer.handle == other.buffer.handle
            && self.offset == other.offset
            && self.len == other.len
    }
}

impl Eq for Bound<'_> {}

impl<'a> Bound<'a> {
    /// The whole buffer.
    ///
    /// Offset zero, which every alignment divides, so this cannot be refused
    /// and does not need the device to say so.
    #[must_use]
    pub fn whole(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            len: buffer.size,
        }
    }

    /// `len` bytes at `offset`.
    ///
    /// # Errors
    ///
    /// [`Failed::Unaligned`] when the device cannot address from there, and
    /// [`Failed::Overrun`] when the range leaves the buffer.
    ///
    /// A zero-length range is [`Failed::Overrun`] with `len` zero rather than a
    /// variant of its own: it is illegal Vulkan, and it is also always the
    /// same defect -- a width computed from a shape that came out empty --
    /// so the numbers that produced it are the useful part of the message.
    pub fn at(device: &Device, buffer: &'a Buffer, offset: u64, len: u64) -> Result<Self, Failed> {
        Self::within(buffer, offset, len, device.min_storage_offset)
    }

    /// `len` bytes at `offset`, against a stated alignment.
    ///
    /// The same rule as [`Bound::at`] with the device's one number passed in
    /// rather than read out. It exists because binding a plan is arithmetic
    /// over offsets and extents, and the machines that CHANGE a plan -- the
    /// ones running `model-compiler`'s tests -- do not have a Vulkan device to
    /// ask. Splitting the number out is what lets `crate::binding` be tested
    /// where the numbers are produced instead of only where they are used.
    ///
    /// It is also how a check can be made against the SPECIFICATION's 256
    /// rather than the local card's 16, which is the difference between
    /// knowing a plan binds here and knowing it binds anywhere.
    ///
    /// # Errors
    ///
    /// As [`Bound::at`].
    pub fn within(
        buffer: &'a Buffer,
        offset: u64,
        len: u64,
        alignment: u64,
    ) -> Result<Self, Failed> {
        // `max(1)` because the specification's guarantee is a promise from the
        // driver, and dividing by a zero one would panic where refusing is the
        // whole job. Every offset is a multiple of 1, so a device that reports
        // nothing constrains nothing.
        //
        // UNWITNESSED, and unwitnessably so from here: deleting it changes no
        // test because the alignment comes from the device's own limits and
        // this card reports 16. Reaching it needs a driver that reports zero,
        // which is the case it exists for and not one a test can produce. Kept
        // rather than removed -- unlike the dead clamp in `geometry.rs`, this
        // one guards a division and its absence is a panic, not a wrong
        // answer.
        let alignment = alignment.max(1);
        if !offset.is_multiple_of(alignment) {
            return Err(Failed::Unaligned { offset, alignment });
        }
        // Checked rather than added: an offset near `u64::MAX` would wrap to a
        // small sum and pass a bound it is nowhere near.
        let end = offset.checked_add(len);
        if len == 0 || end.is_none_or(|e| e > buffer.size) {
            return Err(Failed::Overrun {
                offset,
                len,
                size: buffer.size,
            });
        }
        Ok(Self {
            buffer,
            offset,
            len,
        })
    }

    /// Where the range starts in its buffer.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Bytes the range covers.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Is the range empty? Never, by construction -- both constructors refuse
    /// it -- but clippy asks for it beside `len` and a caller may prefer to ask.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer the range lives in.
    #[must_use]
    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }
}

impl<'a> From<&'a Buffer> for Bound<'a> {
    fn from(buffer: &'a Buffer) -> Self {
        Self::whole(buffer)
    }
}

/// A compute pipeline and the layout it was built with.
pub struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    /// Descriptors the layout has: the module's own count or the caller's,
    /// whichever is larger. See [`Pipelines::get`] for why neither alone.
    bindings: u32,
    /// Bytes the push range covers.
    push: u32,
    /// What the module said about itself, kept so a caller can build the grid
    /// without reading the file twice.
    declared: Declared,
}

impl Pipeline {
    /// What the module declared.
    #[must_use]
    pub fn declared(&self) -> &Declared {
        &self.declared
    }

    /// Descriptors this pipeline's set layout has.
    #[must_use]
    pub fn bindings(&self) -> u32 {
        self.bindings
    }

    /// Bytes of push constants this pipeline expects.
    #[must_use]
    pub fn push(&self) -> u32 {
        self.push
    }
}

/// Compiled pipelines, one per entrypoint, built on first use.
///
/// Separate from [`Device`] rather than a field on it, so that the borrow of a
/// pipeline does not borrow the device — a caller needs both at once on every
/// dispatch.
pub struct Pipelines {
    /// Keyed by entrypoint AND tier, because they are different modules.
    ///
    /// A key of the name alone is the bug this is written to avoid: the same
    /// entrypoint has up to three modules, they are compiled from different
    /// bodies with different extensions, and the first one built would then
    /// answer for all of them. That is not a slow path or a wrong tier -- it
    /// is a pipeline whose SPIR-V does not match the layout the caller sized
    /// from the tier it thinks it selected.
    built: HashMap<(String, Capability), Pipeline>,
}

impl Default for Pipelines {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipelines {
    /// How many pipelines this cache holds.
    ///
    /// Not `len`, because a cache is not a collection the caller iterates and
    /// `is_empty` would be the next thing a lint asked for. This exists so
    /// that "the second fire built nothing new" is a number rather than a
    /// claim about how long something took.
    #[must_use]
    pub fn built(&self) -> usize {
        self.built.len()
    }

    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            built: HashMap::new(),
        }
    }

    /// The pipeline for `entrypoint`, building it from `code` if new.
    ///
    /// The layout is built from what the MODULE declares, not from the row.
    /// This is the crate's central hazard: 292 of the 480 entrypoints name no
    /// operands, and a layout of no descriptors under a module that decorates
    /// several is a segfault inside `vkCreateComputePipelines` rather than an
    /// error return — so there is nothing to check after the fact.
    ///
    /// `push` is the range in bytes. Passing the row's `push_size` is right for
    /// a row that states its scalars; for one that does not,
    /// [`Device::max_push`] is the only safe answer, since any block the module
    /// declares fits inside it.
    ///
    /// # Errors
    ///
    /// [`Failed::Module`] if `code` is not a module this tree built, and
    /// [`Failed::Vulkan`] if a Vulkan call fails.
    ///
    /// `descriptors` is the count the CALLER will bind, and it is a floor
    /// under what the module declares rather than an alternative to it. Both
    /// numbers are needed because they disagree, and measurement says which
    /// way: over the 188 stated entrypoints 177 agree and 11 have a module
    /// declaring exactly one binding FEWER than the row does, because slangc
    /// drops the `OpDecorate Binding` of a buffer the shader never reads.
    /// `layer_scalar_mul_bfloat16` is one -- the row lists four buffers and
    /// the compiled module decorates three.
    ///
    /// Taking the module's number alone would then refuse those eleven at
    /// `run`, a legitimate call rejected. Taking the caller's alone is the
    /// SIGSEGV: a module reading `binding = 11` under a layout that stops at
    /// 10 does not return an error from `vkCreateComputePipelines`, it takes
    /// the process down. A descriptor declared and never read costs nothing,
    /// so the maximum is the only answer that is safe in both directions.
    pub fn get(
        &mut self,
        device: &Device,
        entrypoint: &str,
        code: &[u8],
        push: u32,
        descriptors: u32,
        tier: Capability,
    ) -> Result<&Pipeline, Failed> {
        let key = (entrypoint.to_string(), tier);
        if !self.built.contains_key(&key) {
            let built = Self::build(device, code, push, descriptors)?;
            self.built.insert(key.clone(), built);
        }
        Ok(&self.built[&key])
    }

    /// A pipeline already built, without the borrow that building takes.
    ///
    /// [`Self::get`] needs `&mut self` because it may build, which means a
    /// caller cannot hold a reference to one pipeline while asking for the
    /// next. A caller recording a whole plan needs exactly that -- one
    /// reference per launch, all alive at once -- so it builds every distinct
    /// module first and then asks for them.
    #[must_use]
    pub fn peek(&self, entrypoint: &str, tier: Capability) -> Option<&Pipeline> {
        self.built.get(&(entrypoint.to_string(), tier))
    }

    /// Build one pipeline.
    fn build(
        device: &Device,
        code: &[u8],
        push: u32,
        descriptors: u32,
    ) -> Result<Pipeline, Failed> {
        let words = spirv::words(code)?;
        let declared = spirv::declared(&words)?;
        let count = declared.bindings.max(descriptors);

        let bindings: Vec<_> = (0..count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let d = &device.device;
        unsafe {
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| Failed::Vulkan(format!("set layout: {e}")))?;

            let set_layouts = [set_layout];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push)];
            let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if push > 0 {
                info = info.push_constant_ranges(&ranges);
            }

            let layout = match d.create_pipeline_layout(&info, None) {
                Ok(l) => l,
                Err(e) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(Failed::Vulkan(format!("pipeline layout: {e}")));
                }
            };

            let module = match d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_pipeline_layout(layout, None);
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(Failed::Vulkan(format!("shader module: {e}")));
                }
            };

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let made = d.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            );
            // The module is only needed while the pipeline is being made.
            d.destroy_shader_module(module, None);

            match made {
                Ok(pipelines) => Ok(Pipeline {
                    pipeline: pipelines[0],
                    layout,
                    set_layout,
                    bindings: count,
                    push,
                    declared,
                }),
                Err((_, e)) => {
                    d.destroy_pipeline_layout(layout, None);
                    d.destroy_descriptor_set_layout(set_layout, None);
                    Err(Failed::Vulkan(format!("compute pipeline: {e}")))
                }
            }
        }
    }

    /// Destroy everything built so far.
    ///
    /// Explicit and not a `Drop`, for the same reason [`Device::free`] is: the
    /// device is what destroys these, and a cache that borrowed it could not be
    /// held beside it.
    pub fn clear(&mut self, device: &Device) {
        let d = &device.device;
        unsafe {
            let _ = d.device_wait_idle();
            for (_, p) in self.built.drain() {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.set_layout, None);
            }
        }
    }
}

/// The workgroup count for a fire, from the rule and the module it will run.
///
/// The one place [`geometry`] and this module meet, and the reason the loaded
/// module is what answers: the divisor and the GEMM tile are the module's, and
/// asking it is what keeps them from being assumed.
///
/// # Errors
///
/// [`Failed::Geometry`] when the rule cannot answer for these dimensions.
pub fn groups_for(
    entrypoint: &str,
    rule: Rule,
    dims: Dims,
    pipeline: &Pipeline,
) -> Result<[u32; 3], Failed> {
    Ok(geometry::groups(
        rule,
        dims,
        Module::loaded(entrypoint, &pipeline.declared),
    )?)
}

/// The binding numbers a caller's buffers go to, in order.
///
/// Every slot of the layout except the module's holes. A slot at or past
/// `declared.bindings` is not a hole -- it is a descriptor the caller asked
/// the layout for, and skipping it would shift every buffer after it.
fn slots(pipeline: &Pipeline) -> impl Iterator<Item = usize> + '_ {
    (0..pipeline.bindings as usize)
        .filter(|i| pipeline.declared.used.get(*i).copied().unwrap_or(true))
}

#[cfg(test)]
mod tests {
    use super::{Candidate, Failed};
    use ash::vk;

    /// The classification, which is the whole of the retry decision.
    ///
    /// It is a pure function of a result code, and it is tested as one
    /// because the alternative -- proving it on the device -- means genuinely
    /// exhausting the machine's memory. This tree runs on a box shared with
    /// other work, and a test that provokes the OOM killer to prove a `match`
    /// arm is a bad trade: the arm is one comparison, and the consequence of
    /// running the box out of memory lands on somebody else's job.
    ///
    /// See `Shell::admit` for what is done with the answer.
    #[test]
    fn a_device_that_is_out_of_memory_is_told_apart_from_one_that_faulted() {
        for code in [
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ] {
            let e = Failed::of_vulkan(code, "allocate", 4096);
            assert!(
                e.is_out_of_memory(),
                "{code:?} is a scheduling fact and was classified as a fault, so \
                 a frame the scheduler could serve after evicting fails the \
                 request instead"
            );
            assert!(
                matches!(e, Failed::OutOfMemory { bytes: 4096, during } if during == "allocate"),
                "the size and the call are what the log needs to say which \
                 allocation refused"
            );
        }
    }

    /// A fault must NOT be retried, which is the failure the other direction.
    ///
    /// `ERROR_DEVICE_LOST` is the case that matters: it repeats forever, so a
    /// scheduler told to evict and re-post would spin on it rather than
    /// surfacing it.
    #[test]
    fn a_lost_device_is_a_fault_and_not_something_to_retry() {
        let e = Failed::of_vulkan(vk::Result::ERROR_DEVICE_LOST, "submit", 0);
        assert!(
            !e.is_out_of_memory(),
            "a lost device was classified as out of memory, so the scheduler \
             would evict and re-post forever against a device that is gone"
        );
        assert!(
            e.to_string().contains("submit"),
            "a fault's text must name the call that failed: {e}"
        );
    }

    /// A synthetic candidate, so the ranking can be tested on machines and
    /// shapes this box does not have.
    fn candidate(name: &str, ty: vk::PhysicalDeviceType, compute: bool) -> Candidate {
        Candidate {
            handle: vk::PhysicalDevice::null(),
            props: vk::PhysicalDeviceProperties {
                device_type: ty,
                ..Default::default()
            },
            name: name.to_string(),
            compute_family: compute.then_some(0),
        }
    }

    /// The device is chosen by what it IS, not by where the loader put it.
    ///
    /// This is the test for the bug that was here: `devices.first()`. The
    /// Vulkan specification does not order `vkEnumeratePhysicalDevices`, and
    /// this very machine offers a discrete card and a `llvmpipe` software
    /// rasteriser -- so the entire suite ran on the card by the loader's grace
    /// and would have run on a CPU implementation, silently and correctly and
    /// a hundred times slower, had that order ever changed.
    ///
    /// Every case puts the wanted device LAST, because a ranking that is
    /// really still `first()` passes any test that puts it first.
    #[test]
    fn the_device_chosen_is_the_best_one_and_not_the_first_one() {
        let software = || candidate("llvmpipe", vk::PhysicalDeviceType::CPU, true);
        let cases: [(&str, Vec<Candidate>, &str); 4] = [
            (
                "a software adapter must not beat a card",
                vec![
                    software(),
                    candidate("card", vk::PhysicalDeviceType::DISCRETE_GPU, true),
                ],
                "card",
            ),
            (
                "an integrated part must not beat a card",
                vec![
                    candidate("igpu", vk::PhysicalDeviceType::INTEGRATED_GPU, true),
                    candidate("card", vk::PhysicalDeviceType::DISCRETE_GPU, true),
                ],
                "card",
            ),
            (
                "a card that cannot compute is not a choice, it is not eligible",
                vec![
                    candidate("blind card", vk::PhysicalDeviceType::DISCRETE_GPU, false),
                    software(),
                ],
                "llvmpipe",
            ),
            (
                "software is right when it is the only thing there",
                vec![software()],
                "llvmpipe",
            ),
        ];
        for (why, seen, want) in cases {
            let got = Candidate::choose_from(&seen, None).map(|c| c.name.clone());
            assert_eq!(
                got.as_deref(),
                Some(want),
                "{why}. Saw {}",
                Candidate::roster(&seen)
            );
        }

        let none = vec![candidate(
            "display only",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            false,
        )];
        assert!(
            Candidate::choose_from(&none, None).is_none(),
            "a device with no compute queue was chosen, and the queue index \
             that follows would be invented"
        );
        assert!(
            Candidate::roster(&none).contains("NO compute queue"),
            "the one message a user with no usable device gets must say WHY \
             each device was passed over: {}",
            Candidate::roster(&none)
        );
    }

    /// `PIE_VULKAN_DEVICE` overrides the ranking, and refuses rather than
    /// falls back.
    ///
    /// The override is what makes the ranking testable on a real machine --
    /// it is how the software adapter sitting next to this box's card gets
    /// opened deliberately -- so it has to be exact. A name that matches
    /// nothing usable must not quietly hand back the device the ranking would
    /// have picked: a run that asked for one device and got another is a
    /// worse outcome than a run that did not start, because it looks like a
    /// measurement.
    #[test]
    fn a_named_device_is_taken_at_its_word_or_refused() {
        let card = "NVIDIA GeForce RTX 4090";
        let pipe = "llvmpipe (LLVM 21.1.8, 256 bits)";
        let seen = vec![
            candidate(card, vk::PhysicalDeviceType::DISCRETE_GPU, true),
            candidate(pipe, vk::PhysicalDeviceType::CPU, true),
        ];
        for (set, want, why) in [
            (
                Some("llvmpipe"),
                Some(pipe),
                "a named device must beat the ranking",
            ),
            (
                Some("LLVMPIPE"),
                Some(pipe),
                "the match is case-insensitive",
            ),
            (
                Some("4090"),
                Some(card),
                "any substring of the name will do",
            ),
            (
                Some("no such device"),
                None,
                "a name that matches nothing must REFUSE",
            ),
            (
                Some("   "),
                Some(card),
                "blank means unset, not 'match everything'",
            ),
            (None, Some(card), "unset is the ranking"),
        ] {
            let got = Candidate::choose_from(&seen, set).map(|c| c.name.clone());
            assert_eq!(
                got.as_deref(),
                want,
                "PIE_VULKAN_DEVICE={set:?} chose {got:?}: {why}"
            );
        }
    }

    /// A device that ran out of time says which knob would have given it more.
    ///
    /// `PIE_VULKAN_FENCE_TIMEOUT_SECS` is only useful to somebody who knows it
    /// exists, and the three wait sites used to report `wait: ERROR_TIMEOUT`
    /// and nothing else. On `llvmpipe` that is the single most likely failure
    /// this crate produces, and it sent the reader looking for a bug in the
    /// shader rather than at a deadline written for a 4090.
    ///
    /// The other half of the assertion matters as much. `ERROR_DEVICE_LOST` is
    /// not a slow device and raising a deadline will not help it, so it must
    /// NOT collect the timeout advice -- an error message that offers the
    /// wrong remedy costs more than one that offers none.
    #[test]
    fn a_wait_that_ran_out_of_time_names_the_knob_that_buys_more() {
        let ns = 7_000_000_000u64;
        let Failed::Vulkan(timed_out) = super::waited_too_long("fire", vk::Result::TIMEOUT, ns)
        else {
            panic!("a timeout is a Vulkan failure");
        };
        assert!(
            timed_out.contains("PIE_VULKAN_FENCE_TIMEOUT_SECS"),
            "the one refusal a slow device produces must name the variable that \
             raises the deadline, or the knob is reachable only by reading this \
             file: {timed_out}"
        );
        assert!(
            timed_out.contains("7s"),
            "and it must say how long it actually waited, because the deadline \
             is configurable and the default is not the whole story: {timed_out}"
        );

        let Failed::Vulkan(lost) =
            super::waited_too_long("fire", vk::Result::ERROR_DEVICE_LOST, ns)
        else {
            panic!("a lost device is a Vulkan failure");
        };
        assert!(
            !lost.contains("PIE_VULKAN_FENCE_TIMEOUT_SECS"),
            "a device that fell off the bus is not a slow one, and telling its \
             owner to wait longer sends them the wrong way: {lost}"
        );
    }

    /// The fence deadline is one number, and a bad one cannot buy a hang.
    ///
    /// The whole point of the constant is that a submit has a deadline at
    /// all: an un-timed `vkWaitForFences` on a device that has stopped
    /// signalling wedges the calling thread with no way to report it. So the
    /// interesting cases are not the ones that parse -- they are `0`, the
    /// typo, and the enormous value, each of which is a way to end up with no
    /// deadline if the parsing is written carelessly.
    ///
    /// `u64::MAX` nanoseconds is how Vulkan spells "wait forever", so the
    /// clamp is not tidiness: without it, `PIE_VULKAN_FENCE_TIMEOUT_SECS`
    /// set to something around 18 billion multiplies straight into it and the
    /// driver silently loses its only protection against a hang.
    #[test]
    fn the_fence_deadline_is_bounded_however_it_is_asked_for() {
        let ten = 10_000_000_000u64;
        let _hold = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (set, want, why) in [
            (
                None,
                Some(ten),
                "unset is ten seconds, the value this replaced",
            ),
            (
                Some("30"),
                Some(30_000_000_000),
                "a plain number is seconds",
            ),
            (
                Some(" 30 "),
                Some(30_000_000_000),
                "surrounding space is not a parse error",
            ),
            (
                Some("0"),
                Some(ten),
                "zero is not 'no deadline', it is a mistake",
            ),
            (Some("soon"), Some(ten), "a typo must not disarm the guard"),
            (Some("-5"), Some(ten), "nor may a negative"),
            (Some(""), Some(ten), "nor may an empty value"),
            (
                Some("99999999999999999999"),
                Some(ten),
                "nor may an overflowing one",
            ),
            (
                Some("18446744074"),
                None,
                "a value that saturates the multiply is clamped",
            ),
        ] {
            unsafe {
                match set {
                    Some(v) => std::env::set_var("PIE_VULKAN_FENCE_TIMEOUT_SECS", v),
                    None => std::env::remove_var("PIE_VULKAN_FENCE_TIMEOUT_SECS"),
                }
            }
            let got = super::fence_timeout_ns();
            assert!(
                got < u64::MAX,
                "PIE_VULKAN_FENCE_TIMEOUT_SECS={set:?} produced u64::MAX, which \
                 Vulkan reads as 'wait forever'. The deadline is the only thing \
                 standing between a stopped device and a wedged thread"
            );
            if let Some(want) = want {
                assert_eq!(got, want, "PIE_VULKAN_FENCE_TIMEOUT_SECS={set:?}: {why}");
            }
        }
        unsafe { std::env::remove_var("PIE_VULKAN_FENCE_TIMEOUT_SECS") };
    }

    /// Serialises the one test that writes the environment. `set_var` is
    /// unsound only when another thread READS the environment concurrently,
    /// and this is the only test here that touches it -- the device choice is
    /// tested through `choose_from`, which takes the override as an argument
    /// for exactly this reason.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
