//! What only a device can answer.
//!
//! Everything else in this crate's test suite is structural: the entrypoint set
//! matches Metal's, the bindings match the table, every module compiles. None
//! of it can catch a wrong formula, because none of it ever runs a shader. This
//! file runs them.
//!
//! # How to run it
//!
//! ```sh
//! cargo test -p kernels-vulkan --features native --test gpu -- --nocapture
//! ```
//!
//! `native` is required because a test cannot dispatch Slang -- it needs the
//! SPIR-V that only a `native` build produces. Without it, or without a Vulkan
//! device, every test here SKIPS rather than fails. That is deliberate: this
//! suite has to stay green on the machines that build `model-ir`, which
//! have no GPU and no `slangc`, while still being the thing that proves the
//! shaders on a machine that has one. A skip prints why.
//!
//! ## Why CI does not hand this suite a software device
//!
//! The obvious way to make these run on a runner is Mesa's `lavapipe`, a CPU
//! Vulkan implementation -- it is exactly what the `kernels-wgpu` suite uses
//! to be genuinely runnable in CI. It does not work here, and the failure is
//! worth writing down so nobody spends the afternoon rediscovering it: Mesa
//! 26.0's `lavapipe` presents a device, reports the tier as loadable, and
//! then SIGSEGVs inside the ICD partway through
//! `affine_qmm_t_is_right_at_every_tile_shape_and_quantization_point`.
//! Reproducibly, in isolation, on a shader that is correct on real hardware.
//!
//! So the CI job installs `slangc` and no driver, every test here skips, and
//! what CI proves is that all 666 modules COMPILE. Running them is the
//! hardware gates' job.
//!
//! # What a failure here means
//!
//! A structural test failing means the table and the tree disagree. A test
//! HERE failing means the arithmetic is wrong, or the ABI is wrong in a way the
//! static checks cannot see -- a descriptor bound to the right index holding
//! the wrong bytes, a push field read at the right offset with the wrong
//! meaning. Those are the failures worth having a GPU for.

// Nearly every loop here uses its counter to index several arrays at once and
// to build a flat offset besides, so the iterator rewrite clippy wants would
// hide the addressing that is the actual subject of these tests.
#![allow(clippy::needless_range_loop)]

use ash::vk;
use kernels_vulkan::Capability;
use std::ffi::CStr;

/// Where a `native` build left the modules, or `None` if this is not one.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

// ---------------------------------------------------------------------------
// bf16, on the host
// ---------------------------------------------------------------------------

/// The same narrowing `common/bf16.slang` does, in Rust.
///
/// Round to nearest even, with the NaN case broken out -- a truncating
/// `(bits >> 16) as u16` would agree on most inputs and disagree on exactly the
/// ones a tolerance check is least likely to notice.
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// Widening is exact: bf16 IS the top half of an f32.
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
        .collect()
}

fn bf16_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// How far two bf16 results may sit apart and still be the same answer.
///
/// bf16 keeps 8 significand bits, so one ulp is about 2^-8 relative. This is a
/// few ulp, which is the room a different summation ORDER needs -- and a
/// different order is exactly what the GPU is doing against the scalar
/// reference below. A tighter bound would be testing the reduction tree rather
/// than the arithmetic.
const BF16_TOLERANCE: f32 = 0.02;

#[track_caller]
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    // The scale an error is measured against is the ROW's largest magnitude,
    // not `max(|w|, 1.0)`. Both are ways of tolerating cancellation -- an
    // element that came out near zero because big terms cancelled carries the
    // absolute error of the big terms, so dividing by its own tiny value would
    // reject correct arithmetic -- but the constant 1.0 picks the row's scale
    // out of the air.
    //
    // It happens to be far too generous for most of this tree. Attention
    // values run 0.1-0.3 and router weights sit near 0.125, so a floor of 1.0
    // turned a 2% claim into a flat absolute 0.02, which is 7-20% of the
    // answer: enough to pass a kernel that is meaningfully wrong. Using the
    // row's own maximum keeps the cancellation slack where it is needed and
    // tightens everything else by however much the data is smaller than one.
    let scale = want
        .iter()
        .fold(0.0f32, |m, w| m.max(w.abs()))
        .max(f32::MIN_POSITIVE);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= BF16_TOLERANCE * scale,
            "{what}: element {i} is {g}, reference says {w} \
             (tolerance {} against a row scale of {scale})",
            BF16_TOLERANCE * scale,
        );
    }
}

/// A component type, spelled the way the shader spells it.
fn component(t: vk::ComponentTypeKHR) -> String {
    match t {
        vk::ComponentTypeKHR::FLOAT16 => "float16_t".into(),
        vk::ComponentTypeKHR::FLOAT32 => "float32_t".into(),
        vk::ComponentTypeKHR::FLOAT64 => "float64_t".into(),
        vk::ComponentTypeKHR::SINT8 => "int8_t".into(),
        vk::ComponentTypeKHR::SINT16 => "int16_t".into(),
        vk::ComponentTypeKHR::SINT32 => "int32_t".into(),
        vk::ComponentTypeKHR::UINT8 => "uint8_t".into(),
        vk::ComponentTypeKHR::UINT32 => "uint32_t".into(),
        other => format!("<{}>", other.as_raw()),
    }
}

// ---------------------------------------------------------------------------
// the device
// ---------------------------------------------------------------------------

/// A Vulkan device held open for the length of the suite, plus the one
/// operation this file needs from it: dispatch a module over some buffers.
struct Gpu {
    _entry: ash::Entry,
    /// Kept alive for the instance's lifetime. Dropping the messenger early
    /// would silence exactly the calls a test is about to make.
    _messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    name: String,
    /// `maxPushConstantsSize`. Used as the range for a module whose row
    /// states no scalars, since any block it declares fits inside it.
    max_push: u32,
    /// Whether `VK_KHR_cooperative_matrix` is present AT ALL, which is a
    /// different question from whether the tier was admitted -- and the gap
    /// between the two is exactly what
    /// `the_coopmat_tier_is_offered_only_for_a_matrix_the_device_advertises`
    /// exists to check. Kept because querying the configuration list through
    /// a loader wrapper for an extension the device does not have is a null
    /// function pointer, not an empty answer.
    has_coopmat_ext: bool,
    /// Which tiers this device can actually load. The claim
    /// `Capability::requires` makes, checked against real hardware.
    tiers: Vec<Capability>,
}

/// End the process on a validation error, printing what the layer said.
///
/// # Safety
///
/// Called by the Vulkan loader with a `p_callback_data` that is valid for the
/// duration of the call. Nothing here outlives it.
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
        "\nthe Vulkan validation layer reported an ERROR, which this suite \
         treats as a failure:\n\n{said}\n\nA passing test is not evidence \
         that a dispatch was legal -- this driver creates malformed pipelines \
         without complaint. Run with `VK_LAYER_PATH` pointing at the layer \
         manifests to reproduce.\n"
    );
    std::process::abort();
}

/// Why the suite is not running, phrased for someone reading test output.
fn unavailable() -> Option<&'static str> {
    if SPV_DIR.is_none() {
        return Some("built without --features native, so there are no SPIR-V modules");
    }
    None
}

/// Every test's hand-packed push block, checked against the row it is for.
///
/// The push block is std430, so a scalar is aligned to its own width and an
/// eight-byte stride after a lone `int` starts at byte 8. Getting that wrong is
/// silent in the worst way: the shader reads two halves of two different
/// numbers and returns something, and one test in this file packs exactly that
/// shape by hand. So rather than trusting each call site, every dispatch is
/// held to `kernels_vulkan::push_layout` -- the same function a driver will
/// use, which is the point.
///
/// A test may push FEWER bytes than the row states, and several do: the qmm
/// helper writes `k` and `n` and stops, because the entrypoint it is aimed at
/// reads nothing further. That is allowed, but it has to stop on a field
/// boundary; a length that lands mid-field means some scalar was written with
/// half a value, which is the mistake this exists to catch. An UNSTATED row has
/// no layout to check against, and the two dispatches that use one say so.
fn check_push_against_the_row(entrypoint: &str, push: &[u8]) {
    // A NO-OP, and it was one before this body was deleted.
    //
    // It used to resolve `kernels::sig_in(kernels_vulkan::KERNELS, entrypoint)`
    // and require the pushed length to end on one of the row block's field
    // boundaries -- a length landing mid-field means some scalar was written
    // with half a value. Then families started crossing to routines, a guard
    // was added to return early for a RETIRED entrypoint, and every entrypoint
    // retired. The function returned on all 481 without reading a byte.
    //
    // That is the shape worth naming rather than tidying away: a check does
    // not go FALSE when its reference implementation retires, it goes SILENT,
    // and a silent check reads exactly like a passing one. The block's fields
    // are stated by the routine's own signature now and `tests/routines.rs`
    // is what holds a body to them.
    //
    // The three call sites stay, naming the entrypoint they push for, so that
    // whoever writes the routine-side equivalent finds the places that want
    // it.
    let _ = (entrypoint, push);
}

/// Read a `.spv` file as a word stream, or explain which file.
fn spv_words(path: &std::path::Path) -> Vec<u32> {
    let code =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    assert!(
        code.len().is_multiple_of(4),
        "{} is not a SPIR-V word stream",
        path.display()
    );
    code.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// How many descriptors a layout must have for this module to be legal.
///
/// The highest `binding = N` the module decorates, plus one -- not the COUNT of
/// them, because the set is often not contiguous. 79 of the 292 unstated
/// modules have holes, and the holes are benign: they are `slangc` eliminating
/// a buffer the variant does not read. `affine_qmm_t_fp16_precast` declares
/// 0, 1, 2, 4, 7 because under that macro `load_x` reads `half_in` at 7 and
/// never touches `x` at 3, so the declaration is dropped from the SPIR-V. The
/// layout still needs a descriptor at 3; the shader simply ignores it.
///
/// A few shaders hole their bindings on purpose as well -- `kv_append_paged`
/// declares 0..3 and then 10 and 11, since the row keeps Metal's ring-ABI
/// placeholder slots. Either way, a layout has to cover the highest number.
fn declared_binding_count(words: &[u32]) -> u32 {
    // `OpDecorate` is opcode 71 and `Binding` is decoration 33; the literal
    // follows. Walking every instruction rather than stopping at the first
    // non-annotation, because decorations sit in their own section and this is
    // cheap enough not to need the shortcut.
    let mut highest: Option<u32> = None;
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        if count == 0 || i + count > words.len() {
            break;
        }
        if words[i] & 0xffff == 71 && count == 4 && words[i + 2] == 33 {
            highest = Some(highest.map_or(words[i + 3], |h: u32| h.max(words[i + 3])));
        }
        i += count;
    }
    highest.map_or(0, |h| h + 1)
}

impl Gpu {
    /// Open the first device with a compute queue, or explain why not.
    ///
    /// Returns `Err` rather than panicking for the whole no-GPU case, because
    /// "there is no Vulkan here" is the normal state of a build machine and not
    /// a test failure.
    fn open() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader: {e}"))?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"kernels-vulkan tests")
            .api_version(vk::API_VERSION_1_3);

        // Enable `VK_LAYER_KHRONOS_validation` when the loader can see it.
        //
        // This is not optional politeness. Without it a driver answers a
        // malformed request by crashing or hanging rather than returning an
        // error -- an empty descriptor layout under a module that declares
        // bindings segfaults inside `vkCreateComputePipelines`, which is how
        // the unstated-row hazard was found. With the layer, the same mistake
        // is a message with the entrypoint and the binding in it.
        //
        // It is a soft dependency because a build machine will not have it and
        // "there is no validation layer here" must not be a test failure. Set
        // `VK_LAYER_PATH` to a directory of layer manifests to use one that is
        // not installed system-wide.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let validation = c"VK_LAYER_KHRONOS_validation";
        let has_validation = layers.iter().any(|l| {
            l.layer_name_as_c_str()
                .map(|s| s == validation)
                .unwrap_or(false)
        });
        let enabled: Vec<*const std::ffi::c_char> = if has_validation {
            vec![validation.as_ptr()]
        } else {
            Vec::new()
        };
        // `VK_EXT_debug_utils` is what turns those messages from stderr noise
        // into a process that stops. It is only asked for alongside the layer,
        // since without one there is nothing to report.
        let extensions: Vec<*const std::ffi::c_char> = if has_validation {
            vec![ash::ext::debug_utils::NAME.as_ptr()]
        } else {
            Vec::new()
        };

        // Two validation features beyond the default checks, because the
        // default checks only read what the API was ASKED to do.
        //
        // SYNCHRONIZATION_VALIDATION tracks real hazards between dispatches.
        // Nothing else here can: a missing barrier between a kernel that
        // writes a buffer and one that reads it produces the right answer on
        // this device most of the time, and a suite that compares numbers will
        // never see the race.
        //
        // GPU_ASSISTED instruments the shaders themselves and reports an
        // out-of-range access from inside the dispatch. That is the one that
        // closes a hole this suite documented and could not fix: the device
        // enables `robustBufferAccess`, which turns an overrunning store into a
        // defined discard, so deleting a tail guard still passes every
        // comparison. With this on, the guard is observable.
        let mut features = vk::ValidationFeaturesEXT::default();
        let wanted = [
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
        ];
        if has_validation {
            features = features.enabled_validation_features(&wanted);
        }

        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&enabled)
            .enabled_extension_names(&extensions);
        if has_validation {
            info = info.push_next(&mut features);
        }
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|e| format!("no Vulkan instance: {e}"))?;

        // A validation message that only PRINTS is a message nobody reads. The
        // errors this layer found -- a capability declared without its feature,
        // a push block wider than its range -- had all been passing tests for
        // weeks, because a green suite is what a reader looks at. So an error
        // ends the process.
        //
        // It aborts rather than panics because this is called from the driver
        // across an `extern "system"` boundary, where an unwind is undefined.
        // The message names the entrypoint and the VUID, which is enough to
        // find the test; the alternative -- routing it back to a specific test
        // -- cannot work anyway, since these run in parallel threads and the
        // layer reports against the whole process.
        let messenger = if has_validation {
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
        } else {
            None
        };

        // NOT `.first()`. Nothing in the Vulkan specification orders
        // `vkEnumeratePhysicalDevices`, and this box offers two devices: an
        // RTX 4090 and a `llvmpipe` software rasteriser, from an
        // `lvp_icd.json` that sorts BEFORE `nvidia_icd.json` in the loader's
        // manifest directory. Every number this file has ever proved was
        // proved on the card by the loader's grace. `driver-vulkan`'s
        // `Device::finish` had the same line and the same reasoning is
        // written out at length there.
        //
        // `PIE_VULKAN_DEVICE` names one by a case-insensitive substring and
        // refuses if it matches nothing that can compute -- which is how
        // these proofs get run against llvmpipe on purpose, a second SPIR-V
        // compiler and a second scheduler for the same modules.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("cannot enumerate devices: {e}"))?;
        let seen: Vec<(vk::PhysicalDevice, String, u8, Option<u32>)> = devices
            .iter()
            .map(|&d| {
                let props = unsafe { instance.get_physical_device_properties(d) };
                let name = props
                    .device_name_as_c_str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "<unnamed>".into());
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 4,
                    _ => 3,
                };
                let family = unsafe { instance.get_physical_device_queue_family_properties(d) }
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|i| i as u32);
                (d, name, rank, family)
            })
            .collect();
        let pin = std::env::var("PIE_VULKAN_DEVICE").ok();
        let pin = pin.as_deref().map(str::trim).filter(|p| !p.is_empty());
        let usable = || seen.iter().filter(|(_, _, _, f)| f.is_some());
        let chosen = match pin {
            Some(want) => {
                let want = want.to_ascii_lowercase();
                usable().find(|(_, n, _, _)| n.to_ascii_lowercase().contains(&want))
            }
            None => usable().min_by_key(|(_, _, r, _)| *r),
        };
        let Some((physical, name, _, family)) = chosen else {
            let roster: Vec<&str> = seen.iter().map(|(_, n, _, _)| n.as_str()).collect();
            return Err(match pin {
                Some(want) => format!("no device here matches {want:?}. Saw {roster:?}"),
                None => format!("no device here can compute. Saw {roster:?}"),
            });
        };
        let (physical, name) = (*physical, name.clone());
        let family = family.expect("only devices with a compute family are chosen");
        let props = unsafe { instance.get_physical_device_properties(physical) };

        // Ask the device what it has, then hand the SAME structs back when
        // creating it. Enabling exactly what was reported is the shortest way
        // to be sure a shader's declared capability has its feature behind it -- and
        // the tier machinery is precisely about features that may be absent, so
        // hard-coding a list here would test the list and not the device.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(|e| format!("cannot enumerate extensions: {e}"))?;
        let has_coopmat = extensions.iter().any(|e| {
            e.extension_name_as_c_str()
                .map(|s| s == ash::khr::cooperative_matrix::NAME)
                .unwrap_or(false)
        });

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        // The query chain lives in its own scope so it releases the structs
        // before the answer is read out of them. Feeding these SAME structs
        // back into a second chain is what a first draft did, and it hangs:
        // `push_next` leaves each struct's `p_next` pointing at the last one,
        // so pushing it again closes a cycle and ash walks it until the index
        // overflows. The enable chain below therefore uses fresh structs.
        {
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12);
            if has_coopmat {
                query = query.push_next(&mut fcm);
            }
            unsafe { instance.get_physical_device_features2(physical, &mut query) };
        }
        let core = unsafe { instance.get_physical_device_features(physical) };

        // What the device says about the two optional things the tiers name.
        let mut tiers = vec![Capability::Baseline];
        if f12.shader_float16 == vk::TRUE {
            tiers.push(Capability::Fp16);
        }
        // Both names, because `Capability::Coopmat::requires()` states both:
        // the tier's A/B operands are `float16_t`, so a matrix unit without
        // `shaderFloat16` behind it is not enough. And `vulkanMemoryModel`,
        // because `GL_KHR_cooperative_matrix` pulls in
        // `GL_KHR_memory_scope_semantics` and so every module in the tier
        // declares `OpCapability VulkanMemoryModel` -- which may not be
        // declared unless the feature is enabled.
        // And the CONFIGURATION. `coopmat_configs()` below already says why
        // -- the list is the contract, and a matrix off it is undefined
        // behaviour -- but until this line nothing ASKED. Mesa's `lavapipe`
        // advertises the extension, the feature, `float16` and the memory
        // model, and publishes four configurations, all 8x8x8, against the
        // 16x16x16 every `@coopmat` module here declares. It was admitted to
        // the tier and `vkCreateComputePipelines` segfaulted, with the
        // validation layer silent, because undefined is not invalid.
        if has_coopmat
            && fcm.cooperative_matrix == vk::TRUE
            && f12.shader_float16 == vk::TRUE
            && f12.vulkan_memory_model == vk::TRUE
            && f12.vulkan_memory_model_device_scope == vk::TRUE
            && {
                let ext = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
                let props =
                    unsafe { ext.get_physical_device_cooperative_matrix_properties(physical) }
                        .unwrap_or_default();
                props.iter().any(|c| {
                    c.m_size == 16
                        && c.n_size == 16
                        && c.k_size == 16
                        && c.a_type == vk::ComponentTypeKHR::FLOAT16
                        && c.b_type == vk::ComponentTypeKHR::FLOAT16
                        && c.c_type == vk::ComponentTypeKHR::FLOAT32
                        && c.result_type == vk::ComponentTypeKHR::FLOAT32
                        && c.scope == vk::ScopeKHR::SUBGROUP
                })
            }
        {
            tiers.push(Capability::Coopmat);
        }

        // Enable what the SHADER TREE needs, named one by one rather than by
        // handing back everything the device reported. The list is short and it
        // is documentation: these are the non-core things every module in this
        // crate assumes, and a device missing one of the first three cannot run
        // the baseline at all.
        //
        //   - 16-bit storage + shaderInt16: bf16 is stored as `uint16_t`
        //     throughout, per `common/bf16.slang`.
        //   - shaderFloat16: the `@fp16` tier's genuine `float16_t` math.
        //   - cooperativeMatrix: the `@coopmat` tier.
        //   - vulkanMemoryModel: also the `@coopmat` tier, and it was missing
        //     until a validation layer said so. `GL_KHR_cooperative_matrix`
        //     requires `GL_KHR_memory_scope_semantics`, so all 146 modules in
        //     that tier declare `OpCapability VulkanMemoryModel`, and a module
        //     may not declare a capability whose feature is off. This driver
        //     built the pipeline regardless, which is precisely why the gap
        //     survived until something checked.
        //   - vulkanMemoryModelDeviceScope: enabling the memory model above
        //     re-reads every OTHER module too, and `moe/route.slang` -- which is
        //     baseline, and nothing to do with matrices -- histograms tokens
        //     with a device-scoped `atomicAdd`. Turning the tier on without
        //     this name breaks that kernel.
        let mut e11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f11.storage_buffer16_bit_access == vk::TRUE)
            .uniform_and_storage_buffer16_bit_access(
                f11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
            );
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(f12.shader_float16 == vk::TRUE)
            .shader_int8(f12.shader_int8 == vk::TRUE)
            .storage_buffer8_bit_access(f12.storage_buffer8_bit_access == vk::TRUE)
            // And the uniform half, which `slangc` makes necessary: the
            // shaders put `uint8_t` in storage buffers only, but Slang
            // declares that access as `UniformAndStorageBuffer8BitAccess`
            // where `glslc` declared the narrow `StorageBuffer8BitAccess`,
            // and a module may not declare a capability whose feature is off.
            //
            // This block is a deliberate SECOND copy of `driver_vulkan`'s
            // device setup -- this crate does not depend on that one -- which
            // is exactly why the omission survived being fixed there: the
            // validation layer reported it here on the next run.
            .uniform_and_storage_buffer8_bit_access(
                f12.uniform_and_storage_buffer8_bit_access == vk::TRUE,
            )
            .vulkan_memory_model(f12.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE);
        let mut ecm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(fcm.cooperative_matrix == vk::TRUE);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(core.shader_int16 == vk::TRUE)
                    // `quant/qmm_t.slang` accumulates over the whole BM x BN
                    // tile and guards only the STORE, so at a ragged shape it
                    // deliberately fetches weights and activations outside the
                    // matrix and expects them to read as zero. Without
                    // `robustBufferAccess` that is undefined behaviour, not a
                    // wasted fetch: the spec allows it to return neighbouring
                    // memory, which would be a silently wrong answer rather
                    // than a crash. It happened to be benign on the device
                    // these tests run on, which is the worst way for a
                    // dependency to be satisfied. It is a core Vulkan 1.0
                    // feature, so asking for it costs nothing and makes the
                    // contract the shaders already rely on an explicit one.
                    .robust_buffer_access(core.robust_buffer_access == vk::TRUE),
            )
            .push_next(&mut e11)
            .push_next(&mut e12);
        assert!(
            core.robust_buffer_access == vk::TRUE,
            "the device does not support robustBufferAccess, which the tiled \
             GEMM's out-of-range fetches depend on"
        );
        if has_coopmat {
            features = features.push_next(&mut ecm);
        }

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut enabled: Vec<*const std::ffi::c_char> = Vec::new();
        if has_coopmat {
            enabled.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&enabled)
            .push_next(&mut features);
        let device = unsafe { instance.create_device(physical, &create, None) }
            .map_err(|e| format!("cannot create a device on {name}: {e}"))?;
        let queue = unsafe { device.get_device_queue(family, 0) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };

        Ok(Self {
            _entry: entry,
            _messenger: messenger,
            instance,
            physical,
            device,
            queue,
            family,
            memory,
            name,
            max_push: props.limits.max_push_constants_size,
            tiers,
            has_coopmat_ext: has_coopmat,
        })
    }

    /// Every `(M, N, K, types)` the device's matrix unit implements.
    ///
    /// The list is the contract. A `coopmat` whose element type or shape is not
    /// on it is undefined behaviour, not a slow path.
    fn coopmat_configs(&self) -> Vec<(u32, u32, u32, String, String)> {
        let ext = ash::khr::cooperative_matrix::Instance::new(&self._entry, &self.instance);
        let props = unsafe { ext.get_physical_device_cooperative_matrix_properties(self.physical) }
            .unwrap_or_default();
        // Copied out rather than returned: the properties borrow the loader
        // wrapper, which does not outlive this call.
        props
            .iter()
            .map(|p| {
                (
                    p.m_size,
                    p.n_size,
                    p.k_size,
                    component(p.a_type),
                    component(p.c_type),
                )
            })
            .collect()
    }

    /// A host-visible storage buffer holding `bytes`, sized at least one byte
    /// so that an empty operand still has something to bind.
    fn buffer(&self, bytes: &[u8]) -> (vk::Buffer, vk::DeviceMemory, u64) {
        let size = bytes.len().max(4) as u64;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&info, None) }.expect("create buffer");
        let need = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // HOST_VISIBLE | HOST_COHERENT throughout: a correctness harness wants
        // to read results back without a staging copy, and none of these
        // dispatches is large enough for device-local to matter.
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = (0..self.memory.memory_type_count)
            .find(|i| {
                need.memory_type_bits & (1 << i) != 0
                    && self.memory.memory_types[*i as usize]
                        .property_flags
                        .contains(want)
            })
            .expect("a host-visible memory type");
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }.expect("allocate");
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.expect("bind");

        if !bytes.is_empty() {
            unsafe {
                let ptr = self
                    .device
                    .map_memory(memory, 0, need.size, vk::MemoryMapFlags::empty())
                    .expect("map") as *mut u8;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                self.device.unmap_memory(memory);
            }
        }
        (buffer, memory, size)
    }

    /// Dispatch one module and hand back what the buffers hold afterwards.
    ///
    /// `operands` is the descriptor set in the table's order, and `push` is the
    /// push block as bytes. Both come from `kernels_vulkan::bindings`, which is
    /// the point of the harness: if the shader and the table disagree about
    /// where something rides, the answer comes out wrong here.
    /// Build a compute pipeline for `path` under a layout of `buffers` storage
    /// descriptors and a push range of `push` bytes, then throw it away.
    ///
    /// Everything is destroyed before returning, because the caller runs this
    /// over every module in the tree and a leak per module is 665 live objects
    /// against a driver's pool.
    fn build_pipeline(
        &self,
        path: &std::path::Path,
        buffers: u32,
        push: usize,
    ) -> Result<(), String> {
        let code =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if !code.len().is_multiple_of(4) {
            return Err("not a SPIR-V word stream".into());
        }
        let words: Vec<u32> = code
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let bindings: Vec<_> = (0..buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        unsafe {
            let set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| format!("descriptor set layout: {e}"))?;

            let set_layouts = [set_layout];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push as u32)];
            let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if push > 0 {
                info = info.push_constant_ranges(&ranges);
            }
            let layout = match self.device.create_pipeline_layout(&info, None) {
                Ok(l) => l,
                Err(e) => {
                    self.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(format!("pipeline layout: {e}"));
                }
            };

            let module = match self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_pipeline_layout(layout, None);
                    self.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(format!("shader module: {e}"));
                }
            };

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let result = self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            );

            let answer = match result {
                Ok(pipelines) => {
                    self.device.destroy_pipeline(pipelines[0], None);
                    Ok(())
                }
                Err((_, e)) => Err(format!("compute pipeline: {e}")),
            };
            self.device.destroy_shader_module(module, None);
            self.device.destroy_pipeline_layout(layout, None);
            self.device.destroy_descriptor_set_layout(set_layout, None);
            answer
        }
    }

    fn dispatch(
        &self,
        entrypoint: &str,
        tier: Capability,
        operands: &[Vec<u8>],
        push: &[u8],
        groups: [u32; 3],
    ) -> Vec<Vec<u8>> {
        check_push_against_the_row(entrypoint, push);
        let path = std::path::Path::new(SPV_DIR.expect("checked by the caller"))
            .join(tier.module(entrypoint));
        let code =
            std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        assert!(
            code.len().is_multiple_of(4),
            "{} is not a SPIR-V word stream",
            path.display()
        );
        let words: Vec<u32> = code
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let allocated: Vec<_> = operands.iter().map(|b| self.buffer(b)).collect();

        let bindings: Vec<_> = (0..operands.len() as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("descriptor set layout");

        let set_layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push.len().max(4) as u32)];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if !push.is_empty() {
            layout_info = layout_info.push_constant_ranges(&ranges);
        }
        let layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .expect("pipeline layout");

        let module = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .expect("shader module");

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipelines = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
        }
        .unwrap_or_else(|(_, e)| panic!("cannot build a pipeline for {entrypoint} @{tier:?}: {e}"));
        let pipeline = pipelines[0];

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(operands.len().max(1) as u32)];
        let pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .expect("descriptor pool");
        let sets = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
        }
        .expect("descriptor set");
        let set = sets[0];

        let infos: Vec<_> = allocated
            .iter()
            .map(|(b, _, size)| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(*b)
                    .offset(0)
                    .range(*size)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let cmd_pool = unsafe {
            self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.family),
                None,
            )
        }
        .expect("command pool");
        let cmds = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("command buffer");
        let cmd = cmds[0];

        // Held across the whole record-submit-wait, because the queue is shared
        // and `vkQueueSubmit` is externally synchronised.
        let _serialised = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin");
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                self.device
                    .cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            self.device
                .cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            self.device.end_command_buffer(cmd).expect("end");

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("fence");
            let buffers = [cmd];
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&buffers)],
                    fence,
                )
                .expect("submit");
            self.device
                .wait_for_fences(&[fence], true, 10_000_000_000)
                .expect("the dispatch finished within ten seconds");
            self.device.destroy_fence(fence, None);
        }

        let out: Vec<Vec<u8>> = allocated
            .iter()
            .zip(operands)
            .map(|((_, mem, size), original)| unsafe {
                let ptr = self
                    .device
                    .map_memory(*mem, 0, *size, vk::MemoryMapFlags::empty())
                    .expect("map for readback") as *const u8;
                let mut v = vec![0u8; original.len().max(4)];
                std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), v.len());
                self.device.unmap_memory(*mem);
                v.truncate(original.len().max(4));
                v
            })
            .collect();

        unsafe {
            self.device.destroy_command_pool(cmd_pool, None);
            self.device.destroy_descriptor_pool(pool, None);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_shader_module(module, None);
            self.device.destroy_pipeline_layout(layout, None);
            self.device.destroy_descriptor_set_layout(set_layout, None);
            for (b, m, _) in allocated {
                self.device.destroy_buffer(b, None);
                self.device.free_memory(m, None);
            }
        }
        out
    }
}

// No `Drop`. The device is opened once per PROCESS and deliberately outlives
// every test, which is not laziness: an earlier draft opened and closed one per
// test, and the fourteenth `Entry::load` / `dlclose` cycle segfaulted inside
// the loader. Tearing down a Vulkan instance is also not something this suite
// is trying to test. The process exiting is the cleanup.

/// The one device, opened on first use.
///
/// `OnceLock` rather than a per-test open, and a `Mutex` around dispatch
/// because `vkQueueSubmit` requires external synchronisation on the queue and
/// the harness would otherwise be a race the moment two tests ran at once.
static GPU: std::sync::OnceLock<Result<Gpu, String>> = std::sync::OnceLock::new();
static QUEUE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn shared_gpu() -> Result<&'static Gpu, String> {
    GPU.get_or_init(Gpu::open).as_ref().map_err(Clone::clone)
}

/// Open a device, or print why the test is skipping and hand back `None`.
macro_rules! gpu {
    () => {
        match unavailable() {
            Some(why) => {
                eprintln!("SKIP: {why}");
                return;
            }
            None => match shared_gpu() {
                Ok(gpu) => gpu,
                Err(why) => {
                    eprintln!("SKIP: {why}");
                    return;
                }
            },
        }
    };
}

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

/// Whether this run measured anything at all, said out loud.
///
/// Every other test in this file opens the device through `gpu!()`, which
/// prints `SKIP:` and returns when there is none. That is the right shape --
/// a test that cannot run should not fail -- and it has one bad property: a
/// run that proved 46 shaders and a run that touched no GPU whatsoever both
/// print
///
/// ```text
/// test result: ok. 48 passed
/// ```
///
/// and `cargo test` hides a passing test's stdout, so the `SKIP:` lines that
/// would have said which are not shown either. This was not hypothetical.
/// Pointed at a box with a real NVIDIA card, this suite reported 48 passed --
/// in 0.06 seconds, against a container whose `libGLX_nvidia.so.0` is a stub
/// with no `vkCreateInstance` in it. Nothing in the output distinguished that
/// from the 6.5 seconds the same 48 take when they run. Only the clock did.
///
/// This test is not gated, because its whole job is to run everywhere.
/// `PIE_VULKAN_REQUIRE_DEVICE=1` turns the absence into a failure, and is
/// what any job that installs a driver ON PURPOSE should set -- otherwise the
/// install can silently stop working and the suite goes on reading green.
///
/// Both ways of measuring nothing are caught, because both produce the same
/// vacuous green: a build without `native` (no SPIR-V to run) and a build
/// with no device (nothing to run it on).
#[test]
fn the_runner_states_whether_it_has_a_device() {
    let required = std::env::var_os("PIE_VULKAN_REQUIRE_DEVICE").is_some_and(|v| v != "0");
    // Counted off this file rather than written down, so the number cannot
    // become a lie the next time a test is added.
    //
    // The needle is split because an undivided one MATCHES ITSELF: this
    // literal is in the text `include_str!` reads.
    let needle = concat!("= gpu", "!();");
    let gated = include_str!("gpu.rs").matches(needle).count();
    assert!(
        gated >= 40,
        "found {gated} device-gated tests by reading this file, which is not what it contains"
    );

    let why = match unavailable() {
        Some(why) => Some(why.to_string()),
        None => shared_gpu().err(),
    };
    match why {
        None => {
            let gpu = shared_gpu().expect("just checked");
            println!(
                "VULKAN DEVICE: PRESENT ({}). The {gated} device-gated tests here measured real numbers.",
                gpu.name
            );
        }
        Some(why) => {
            println!("VULKAN DEVICE: ABSENT ({why}).");
            println!(
                "All {gated} device-gated tests in this file skipped, so a green `--test gpu` here measured NOTHING."
            );
            assert!(
                !required,
                "PIE_VULKAN_REQUIRE_DEVICE is set and no device opened: {why}. A suite that silently skips is what this test exists to prevent"
            );
        }
    }
}

/// What the device offers, against what the tiers claim to need.
///
/// Not an assertion about any particular GPU -- it prints. The assertion is the
/// one that must hold everywhere: baseline is always available, which is the
/// backward-compatibility promise `Capability` is built around.
#[test]
fn the_device_reports_which_tiers_it_can_load() {
    let gpu = gpu!();
    eprintln!("device: {}", gpu.name);
    for tier in Capability::PREFERENCE {
        let have = gpu.tiers.contains(&tier);
        eprintln!(
            "  {:9} {:3}  requires {:?}",
            tier.tag(),
            if have { "yes" } else { "no" },
            tier.requires(),
        );
    }
    assert!(
        gpu.tiers.contains(&Capability::Baseline),
        "baseline needs nothing optional, so every device must offer it",
    );
}

/// `row_gather_bfloat16`, which is the ABI question this crate got wrong twice.
///
/// # The question, and both of its old answers
///
/// The row stated `count: Ty::InPacked` -- a value the driver must supply that
/// got no slot of its own, because it was the second FIELD of the params struct
/// buffer 3 bound. `bindings()` first folded it into the push block, Metal's
/// rule, which would have pushed a word no shader read and left `p.count`
/// holding whatever was in the buffer. This test then wrote the struct the way
/// `Binding::Packed` said to -- `{width, count}` in buffer 3, nothing pushed.
///
/// Both readings are gone. `width` is a `Const<u32>` mark and `count` is the
/// request count the body asks the fire for, so the pair is an ordinary
/// eight-byte PUSH range built in the order the body passes them -- which is
/// the first answer, arrived at honestly, after the thing that made it wrong
/// (a struct the push block could not also be) stopped existing.
///
/// What the dispatch proves is unchanged: the rows come out gathered, and a
/// count read from the wrong field would write the wrong number of them.
#[test]
fn row_gather_reads_its_width_then_its_count_from_the_push_block() {
    let gpu = gpu!();

    let width = 4usize;
    let rows = 5usize;
    let input: Vec<f32> = (0..rows * width).map(|i| i as f32).collect();
    let pick: [u32; 3] = [4, 0, 2];

    let operands = vec![
        bf16_bytes(&input),
        vec![0u8; pick.len() * width * 2],
        pick.iter().flat_map(|r| r.to_le_bytes()).collect(),
    ];
    // `{width, count}` -- the same two words in the same order the struct held
    // them, in the push range now.
    let push: Vec<u8> = [width as u32, pick.len() as u32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let out = gpu.dispatch(
        "row_gather_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [width.div_ceil(16) as u32, pick.len().div_ceil(16) as u32, 1],
    );

    let got = bf16_read(&out[1]);
    let want: Vec<f32> = pick
        .iter()
        .flat_map(|r| (0..width).map(move |c| (*r as usize * width + c) as f32))
        .collect();
    assert_close(&got, &want, "row_gather_bfloat16");
}

/// `rms_single_row_bfloat16` against a scalar reference.
///
/// The kernel every model runs many times per token, and the first shader in
/// the tree to use both halves of the launch ABI at once: `RmsParams` stays a
/// buffer because the row says `Buf`, while `row_pitch` -- absent in this
/// unstrided variant -- would be a push constant. It also exercises the
/// subgroup reduction, whose width the implementation chooses, so a result that
/// matches here is evidence for the argument in `common/reduce.slang`.
#[test]
fn rms_single_row_matches_a_scalar_reference() {
    let gpu = gpu!();

    let axis = 1024usize;
    // Deliberately not smooth: a ramp would hide an indexing error, since
    // neighbouring elements are nearly equal.
    let x: Vec<f32> = (0..axis)
        .map(|i| ((i * 37 % 71) as f32 - 35.0) / 16.0)
        .collect();
    let w: Vec<f32> = (0..axis).map(|i| 0.5 + (i % 13) as f32 / 32.0).collect();
    let eps = 1e-5f32;

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes()); // axis_size
    params.extend_from_slice(&1u32.to_le_bytes()); // w_stride
    params.extend_from_slice(&0u32.to_le_bytes()); // plus_one
    params.extend_from_slice(&1.0f32.to_le_bytes()); // gain

    let operands = vec![bf16_bytes(&x), bf16_bytes(&w), vec![0u8; axis * 2], params];
    let out = gpu.dispatch(
        "rms_single_row_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [1, 1, 1],
    );

    // The reference reads back the bf16 the DEVICE was given, not the f32 the
    // test started from. Comparing against the f32 would fold the input
    // rounding into the tolerance and quietly widen it.
    let xq = bf16_read(&operands[0]);
    let wq = bf16_read(&operands[1]);
    let mean: f32 = xq.iter().map(|v| v * v).sum::<f32>() / axis as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    let want: Vec<f32> = xq.iter().zip(&wq).map(|(v, g)| g * (v * inv)).collect();

    assert_close(&bf16_read(&out[2]), &want, "rms_single_row_bfloat16");
}

/// `plus_one` is the gemma convention, and it is folded in FLOAT.
///
/// Worth its own case because the difference between `(1 + w)` in float and
/// `1 + w` after a bf16 round is small, one-directional, and exactly the kind
/// of thing a port gets wrong without any test noticing.
#[test]
fn rms_folds_plus_one_before_the_bf16_round() {
    let gpu = gpu!();

    let axis = 256usize;
    let x: Vec<f32> = (0..axis).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let w: Vec<f32> = (0..axis).map(|i| (i % 5) as f32 / 64.0).collect();
    let eps = 1e-6f32;
    let gain = 1.5f32;

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes()); // plus_one
    params.extend_from_slice(&gain.to_le_bytes());

    let operands = vec![bf16_bytes(&x), bf16_bytes(&w), vec![0u8; axis * 2], params];
    let out = gpu.dispatch(
        "rms_single_row_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [1, 1, 1],
    );

    let xq = bf16_read(&operands[0]);
    let wq = bf16_read(&operands[1]);
    let mean: f32 = xq.iter().map(|v| v * v).sum::<f32>() / axis as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    let want: Vec<f32> = xq
        .iter()
        .zip(&wq)
        .map(|(v, g)| gain * (1.0 + g) * (v * inv))
        .collect();

    assert_close(&bf16_read(&out[2]), &want, "rms plus_one");
}

/// The strided form, which is where the push constant appears.
///
/// `row_pitch` is the shader's only scalar, so it is push field 0 and nothing
/// else is in the block. A pitch WIDER than the axis is the point: if the
/// shader read the pitch from anywhere but the push block, the second row
/// would be gathered from the wrong offset and the gap would be read as data.
///
/// One caveat, shared with `sdpa_paged_mma` below and with nothing else here.
/// This row is UNSTATED — the table names no operands for it — so the layout above was
/// read off the shader rather than off the table. That makes this a test of
/// the BODY and not of the ABI: it shows the arithmetic and the pitch handling
/// are right, and it cannot show the driver would bind them the same way,
/// because there is nothing for the driver to read. Every other dispatch here
/// takes its layout from `kernels_vulkan::push_layout`, which derives it from
/// the row.
#[test]
fn rms_strided_row_reads_its_pitch_from_the_push_block() {
    let gpu = gpu!();

    let axis = 128usize;
    let pitch = 192usize; // deliberately > axis, so the gap is visible
    let rows = 3usize;
    let x: Vec<f32> = (0..rows * pitch)
        .map(|i| ((i % 23) as f32 - 11.0) / 8.0)
        .collect();
    let w: Vec<f32> = (0..axis).map(|i| 0.75 + (i % 7) as f32 / 32.0).collect();
    let eps = 1e-5f32;

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());

    let operands = vec![
        bf16_bytes(&x),
        bf16_bytes(&w),
        vec![0u8; rows * pitch * 2],
        params,
    ];
    let out = gpu.dispatch(
        "rms_strided_row_bfloat16",
        Capability::Baseline,
        &operands,
        &(pitch as i32).to_le_bytes(),
        [rows as u32, 1, 1],
    );

    let xq = bf16_read(&operands[0]);
    let wq = bf16_read(&operands[1]);
    let got = bf16_read(&out[2]);
    for row in 0..rows {
        let base = row * pitch;
        let slice = &xq[base..base + axis];
        let mean: f32 = slice.iter().map(|v| v * v).sum::<f32>() / axis as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        let want: Vec<f32> = slice.iter().zip(&wq).map(|(v, g)| g * (v * inv)).collect();
        assert_close(&got[base..base + axis], &want, &format!("row {row}"));
    }
}

// ---------------------------------------------------------------------------
// the quantized path
// ---------------------------------------------------------------------------

/// A deterministic affine-quantized weight matrix, in the layout
/// `common/affine.slang` states: codes little-endian within a 32-bit word,
/// lowest code in the lowest bits; `scales`/`biases` one bf16 per group, laid
/// out `[out_vec_size, in_vec_size / group]`.
///
/// Returns the packed words, the scales, the biases, and the dequantized matrix
/// the reference multiplies by — built from the SAME codes, so the reference
/// tests the shader's arithmetic rather than this function's.
fn affine_weights(
    rows: usize,
    k: usize,
    group: usize,
    bits: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>) {
    let codes_per_word = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let groups = k / group;

    let mut words: Vec<u32> = Vec::with_capacity(rows * k / codes_per_word);
    let mut scales: Vec<f32> = Vec::with_capacity(rows * groups);
    let mut biases: Vec<f32> = Vec::with_capacity(rows * groups);
    let mut dense: Vec<f32> = vec![0.0; rows * k];

    for r in 0..rows {
        for g in 0..groups {
            scales.push(0.02 + ((r * 7 + g * 3) % 11) as f32 / 512.0);
            biases.push(((r + g) % 5) as f32 / 32.0 - 0.06);
        }
        for w0 in 0..k / codes_per_word {
            let mut word = 0u32;
            for c in 0..codes_per_word {
                let kk = w0 * codes_per_word + c;
                let code = ((r * 31 + kk * 17 + c * 5) as u32) & mask;
                word |= code << (c * bits);
                let g = kk / group;
                // bf16 is what the device will read, so the reference has to
                // dequantize from the bf16 too -- a scale kept in f32 here
                // would make every comparison fail by a rounding step.
                let s = bf16_to_f32(f32_to_bf16(scales[r * groups + g]));
                let b = bf16_to_f32(f32_to_bf16(biases[r * groups + g]));
                dense[r * k + kk] = code as f32 * s + b;
            }
            words.push(word);
        }
    }

    (
        words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        bf16_bytes(&scales),
        bf16_bytes(&biases),
        dense,
    )
}

/// `affine_qmv_fast_bfloat16_gs_64_b_4` against a dense reference.
///
/// The quantized matvec is where a decode step spends most of its time, and it
/// is the shader with the most ways to be quietly wrong: the code unpacking,
/// the group indexing, the `scale * accum + sum * bias` factoring, and a push
/// block whose first two fields are the only ones the row states. A dense
/// reference built from the same codes checks all four at once.
#[test]
fn affine_qmv_fast_matches_a_dense_reference() {
    let gpu = gpu!();

    let (group, bits) = (64usize, 4usize);
    let k = 256usize;
    // One workgroup covers 8 output rows, so 16 would be two whole groups and
    // the `row < out_vec_size` bound would never decide anything. 13 leaves
    // three lanes of the second group past the end.
    let rows = 13usize;
    let (w, scales, biases, dense) = affine_weights(rows, k, group, bits);
    let x: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) / 12.0).collect();

    let operands = vec![w, scales, biases, bf16_bytes(&x), vec![0u8; rows * 2]];
    let mut push = Vec::new();
    push.extend_from_slice(&(k as i32).to_le_bytes()); // in_vec_size
    push.extend_from_slice(&(rows as i32).to_le_bytes()); // out_vec_size

    // One workgroup covers 8 output rows; `gl_WorkGroupID.x` is the vector.
    let out = gpu.dispatch(
        "affine_qmv_fast_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &operands,
        &push,
        [1, rows.div_ceil(8) as u32, 1],
    );

    let xq = bf16_read(&operands[3]);
    let want: Vec<f32> = (0..rows)
        .map(|r| (0..k).map(|i| xq[i] * dense[r * k + i]).sum())
        .collect();
    assert_close(&bf16_read(&out[4]), &want, "affine_qmv_fast gs_64 b_4");
}

/// The same, at the other group size and bit width.
///
/// `PIE_GROUP` and `PIE_BITS` are a COORDINATE, not a label — g64/b8 and
/// g128/b4 pack to identical shapes, so a module compiled for the wrong pair
/// returns fluent nonsense rather than failing. Running a second point proves
/// the `-D` pair actually reaches the code that unpacks.
#[test]
fn affine_qmv_fast_is_right_at_a_second_quantization_point() {
    let gpu = gpu!();

    let (group, bits) = (128usize, 8usize);
    let k = 256usize;
    // Under the 8 rows a workgroup covers, so this one runs a single group
    // that is short from the start -- the other corner from 13, where the
    // shortfall is in the last of several.
    let rows = 5usize;
    let (w, scales, biases, dense) = affine_weights(rows, k, group, bits);
    let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 10.0).collect();

    let operands = vec![w, scales, biases, bf16_bytes(&x), vec![0u8; rows * 2]];
    let mut push = Vec::new();
    push.extend_from_slice(&(k as i32).to_le_bytes());
    push.extend_from_slice(&(rows as i32).to_le_bytes());

    let out = gpu.dispatch(
        "affine_qmv_fast_bfloat16_gs_128_b_8",
        Capability::Baseline,
        &operands,
        &push,
        [1, rows.div_ceil(8) as u32, 1],
    );

    let xq = bf16_read(&operands[3]);
    let want: Vec<f32> = (0..rows)
        .map(|r| (0..k).map(|i| xq[i] * dense[r * k + i]).sum())
        .collect();
    assert_close(&bf16_read(&out[4]), &want, "affine_qmv_fast gs_128 b_8");
}

// ---------------------------------------------------------------------------
// the tiers, against each other
// ---------------------------------------------------------------------------

/// Run `affine_qmm_t` at one tier and hand back the result.
///
/// `Y = X · Wᵀ`, with `X` an `m × k` bf16 activation and `W` an `n × k` affine
/// weight. One workgroup owns a `PIE_BM × PIE_BN` output tile, so the grid is
/// exactly `[n / bn, m / bm, 1]` and the shapes are chosen to divide evenly --
/// this test is about the arithmetic, and a ragged edge would be a second
/// question asked at the same time.
#[allow(clippy::too_many_arguments)] // a GEMM's shape is this many numbers
fn qmm_t(
    gpu: &Gpu,
    tier: Capability,
    entrypoint: &str,
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    w: &[u8],
    scales: &[u8],
    biases: &[u8],
    x: &[u8],
) -> Vec<f32> {
    // The shader's push block carries no `m`, so its row overhang cannot be
    // guarded inside the kernel (see `write_out` in `quant/qmm_t.slang`). The
    // contract is that the caller allocates a whole number of `bm` rows; this
    // helper honours it and hands back only the real ones.
    let m_padded = m.div_ceil(bm) * bm;
    let operands = vec![
        w.to_vec(),
        scales.to_vec(),
        biases.to_vec(),
        x.to_vec(),
        vec![0u8; m_padded * n * 2],
    ];
    let mut push = Vec::new();
    push.extend_from_slice(&(k as i32).to_le_bytes()); // k
    push.extend_from_slice(&(n as i32).to_le_bytes()); // n

    let out = gpu.dispatch(
        entrypoint,
        tier,
        &operands,
        &push,
        [n.div_ceil(bn) as u32, m.div_ceil(bm) as u32, 1],
    );
    let mut y = bf16_read(&out[4]);
    y.truncate(m * n);
    y
}

/// The baseline `affine_qmm_t` against a dense reference.
///
/// This is the tiled matmul the prefill path runs, and it has to be right
/// before "the coopmat tier agrees with it" means anything.
#[test]
fn affine_qmm_t_baseline_matches_a_dense_reference() {
    let gpu = gpu!();

    let (group, bits) = (128usize, 4usize);
    let (m, n, k, bm, bn) = (32usize, 32usize, 256usize, 32usize, 32usize);
    let (w, scales, biases, dense) = affine_weights(n, k, group, bits);
    let xf: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) / 24.0)
        .collect();
    let x = bf16_bytes(&xf);

    let got = qmm_t(
        gpu,
        Capability::Baseline,
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32",
        m,
        n,
        k,
        bm,
        bn,
        &w,
        &scales,
        &biases,
        &x,
    );

    let xq = bf16_read(&x);
    let want: Vec<f32> = (0..m)
        .flat_map(|r| {
            let xq = &xq;
            let dense = &dense;
            (0..n).map(move |c| (0..k).map(|i| xq[r * k + i] * dense[c * k + i]).sum())
        })
        .collect();
    assert_close(&got, &want, "affine_qmm_t baseline");
}

/// The `@coopmat` tier answers the same question as its baseline.
///
/// This is the whole backward-compatibility claim made testable. A tier is an
/// ADDITIONAL module for an entrypoint that already exists, so the two must
/// agree — but not bit for bit. `coopMatMulAdd` has its own internal summation
/// order, and the tier restructures the accumulation to keep every matrix op in
/// uniform control flow, so the difference is reassociation and the comparison
/// needs a tolerance. Both are therefore checked against the same dense
/// reference rather than against each other, which is the stronger statement:
/// it catches the case where both tiers are wrong in the same way.
#[test]
fn the_coopmat_tier_agrees_with_its_baseline() {
    let gpu = gpu!();
    if !gpu.tiers.contains(&Capability::Coopmat) {
        eprintln!("SKIP: {} does not offer cooperativeMatrix", gpu.name);
        return;
    }

    let (group, bits) = (128usize, 4usize);
    let (m, n, k, bm, bn) = (32usize, 32usize, 256usize, 32usize, 32usize);
    let entrypoint = "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32";
    let (w, scales, biases, dense) = affine_weights(n, k, group, bits);
    let xf: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) / 24.0)
        .collect();
    let x = bf16_bytes(&xf);

    let xq = bf16_read(&x);
    let want: Vec<f32> = (0..m)
        .flat_map(|r| {
            let xq = &xq;
            let dense = &dense;
            (0..n).map(move |c| (0..k).map(|i| xq[r * k + i] * dense[c * k + i]).sum())
        })
        .collect();

    for tier in [Capability::Baseline, Capability::Coopmat] {
        let got = qmm_t(
            gpu, tier, entrypoint, m, n, k, bm, bn, &w, &scales, &biases, &x,
        );
        assert_close(&got, &want, &format!("affine_qmm_t @{}", tier.tag()));
    }
}

/// What shapes and types the device's matrix unit actually implements.
///
/// `VK_KHR_cooperative_matrix` does not promise any particular configuration:
/// a device advertises a LIST, and using a `coopmat` type outside that list is
/// undefined behaviour which a driver is free to accept and then miscompute.
/// So this prints the list, and the `@coopmat` tier has to be written against
/// it rather than against what reads naturally in the shader.
#[test]
fn the_device_lists_its_cooperative_matrix_configurations() {
    let gpu = gpu!();
    if !gpu.has_coopmat_ext {
        eprintln!("SKIP: {} has no VK_KHR_cooperative_matrix", gpu.name);
        return;
    }
    for (m, n, k, a, c) in gpu.coopmat_configs() {
        eprintln!("  {m}x{n}x{k}  A/B={a}  C/Result={c}");
    }
}

/// The coopmat tier is offered only for a matrix the device advertises.
///
/// The tier used to be admitted on FEATURE BITS -- the extension, the
/// `cooperativeMatrix` feature, `shaderFloat16`, the memory model -- and none
/// of those promises a shape. The list is the contract, and this tree uses
/// exactly one entry of it: `quant/qmm_t.slang` declares 16x16x16 with `half`
/// A and B and a `float` accumulator, at subgroup scope.
///
/// The gap was not hypothetical and was not found by reading. Mesa's
/// `llvmpipe` passes every one of those feature checks and advertises four
/// configurations, all 8x8x8. Admitted to the tier, it segfaulted inside
/// `vkCreateComputePipelines` while the validation layer reported nothing --
/// correctly, because undefined behaviour is not invalid usage, and a layer
/// checks the second. With the list consulted, the tier is declined there and
/// all 47 proofs in this file pass on llvmpipe as well as on the card.
///
/// So this is the check, both ways, and it is meaningful on any machine:
/// wherever the tier is offered the matrix must be on the list, and wherever
/// the matrix is on the list with the features present the tier must be
/// offered -- otherwise the whole tier could quietly stop being tested and
/// every proof in this file would still pass.
#[test]
fn the_coopmat_tier_is_offered_only_for_a_matrix_the_device_advertises() {
    let gpu = gpu!();
    let offered = gpu.tiers.contains(&Capability::Coopmat);
    if !gpu.has_coopmat_ext {
        assert!(
            !offered,
            "{} offers the coopmat tier without the extension",
            gpu.name
        );
        eprintln!("SKIP: {} has no VK_KHR_cooperative_matrix", gpu.name);
        return;
    }
    let configs = gpu.coopmat_configs();
    let ours = configs.iter().any(|(m, n, k, a, c)| {
        (*m, *n, *k) == (16, 16, 16) && a == "float16_t" && c == "float32_t"
    });
    assert_eq!(
        offered,
        ours,
        "{} advertises {} cooperative matrix configurations, ours (16x16x16, \
         A/B float16_t, C float32_t) {} among them, and the tier is {}. A tier \
         offered without the matrix is undefined behaviour the driver may \
         accept and then miscompute; a tier withheld with the matrix present \
         silently stops testing a third of this table. Saw {configs:?}",
        gpu.name,
        configs.len(),
        if ours { "IS" } else { "is NOT" },
        if offered { "offered" } else { "withheld" },
    );
}

// ---------------------------------------------------------------------------
// paged attention
// ---------------------------------------------------------------------------

/// `sdpa_paged_decode_bfloat16_d_64` against a scalar reference.
///
/// The decode step's attention, and the most interesting row in the table for
/// the launch ABI: its operands ALTERNATE between buffers and scalars, so the
/// two runs are `Buffer(0..3)`, `Push(0)`, `Buffer(4..7)`, `Push(1..3)`,
/// `Buffer(8)`, `Push(4)`, `Buffer(9)`, `Push(5)`, `Buffer(10)`. A backend that
/// numbered scalars alongside buffers -- Metal's rule -- would put `page_size`
/// at 5 and be wrong about everything after it. Nothing static catches that;
/// only a number does.
///
/// It also exercises the page table (a logical position becomes a physical slot
/// through `kv_page_indices`/`kv_page_indptr`), grouped-query attention via
/// `gqa_factor`, and the online-softmax recurrence in `attn/sdpa_online.slang`.
/// The pages are deliberately shuffled so a shader that ignored the indirection
/// and read the cache linearly would fail.
#[test]
fn sdpa_paged_decode_matches_a_scalar_reference() {
    let gpu = gpu!();

    let head_dim = 64usize;
    let page_size = 16usize;
    let n_kv_heads = 2usize;
    let gqa = 2usize;
    let n_q_heads = n_kv_heads * gqa;
    let rows = 3usize; // one decode token per request
    let scale = 0.125f32;

    // Each request holds a different history length, and the pages are handed
    // out in a shuffled order so the indirection is load-bearing.
    let lengths = [17usize, 5, 32];
    let pages_per: Vec<usize> = lengths.iter().map(|l| l.div_ceil(page_size)).collect();
    let total_pages: usize = pages_per.iter().sum();
    let physical: Vec<u32> = {
        // A reversed assignment: logical page 0 of request 0 is the LAST
        // physical page, so any code that assumes identity is caught.
        let mut v: Vec<u32> = (0..total_pages as u32).collect();
        v.reverse();
        v
    };
    let mut indptr = vec![0u32];
    for p in &pages_per {
        indptr.push(indptr.last().unwrap() + *p as u32);
    }

    let slots = total_pages * page_size;
    let kv_elems = slots * n_kv_heads * head_dim;
    let kf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 31) as f32 - 15.0) / 40.0)
        .collect();
    let vf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 23) as f32 - 11.0) / 30.0)
        .collect();
    let qf: Vec<f32> = (0..rows * n_q_heads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) / 20.0)
        .collect();

    // The query position IS the history length minus one: a decode step
    // attends to everything up to and including itself.
    let positions: Vec<i32> = lengths.iter().map(|l| *l as i32 - 1).collect();
    let req_of_token: Vec<i32> = (0..rows as i32).collect();

    let operands = vec![
        bf16_bytes(&qf),
        bf16_bytes(&kf),
        bf16_bytes(&vf),
        vec![0u8; rows * n_q_heads * head_dim * 2],
        positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
        req_of_token.iter().flat_map(|r| r.to_le_bytes()).collect(),
        physical.iter().flat_map(|p| p.to_le_bytes()).collect(),
        indptr.iter().flat_map(|p| p.to_le_bytes()).collect(),
        vec![0u8; rows],          // attention_mask, unused while disabled
        vec![0u8; rows],          // attention_mask_enabled: every row off
        vec![0u8; n_q_heads * 2], // sinks, unread without PIE_WITH_SINK
    ];

    let mut push = Vec::new();
    push.extend_from_slice(&(gqa as i32).to_le_bytes()); // gqa_factor
    push.extend_from_slice(&(page_size as i32).to_le_bytes()); // page_size
    push.extend_from_slice(&(n_kv_heads as i32).to_le_bytes()); // n_kv_heads
    push.extend_from_slice(&scale.to_le_bytes()); // scale
    push.extend_from_slice(&0u32.to_le_bytes()); // attention_mask_stride
    push.extend_from_slice(&0i32.to_le_bytes()); // window: 0 disables it

    // `gl_WorkGroupID.x` is the query head and `.y` the row; the workgroup is
    // one lane per head dimension.
    let out = gpu.dispatch(
        "sdpa_paged_decode_bfloat16_d_64",
        Capability::Baseline,
        &operands,
        &push,
        [n_q_heads as u32, rows as u32, 1],
    );

    let q = bf16_read(&operands[0]);
    let k = bf16_read(&operands[1]);
    let v = bf16_read(&operands[2]);
    let slot_of = |req: usize, kp: usize| {
        let phys = physical[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };

    let mut want = vec![0.0f32; rows * n_q_heads * head_dim];
    for (row, &position) in positions.iter().enumerate() {
        let q_pos = position as usize;
        for h in 0..n_q_heads {
            let kv_head = h / gqa;
            let q_base = (row * n_q_heads + h) * head_dim;

            let scores: Vec<f32> = (0..=q_pos)
                .map(|kp| {
                    let k_base = (slot_of(row, kp) * n_kv_heads + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|d| scale * q[q_base + d] * k[k_base + d])
                        .sum::<f32>()
                })
                .collect();

            // Softmax the plain way. The shader folds it online, one key at a
            // time, rescaling the running sum -- algebraically the same thing,
            // and agreeing to a few ulp is the evidence that the fold is right.
            let hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - hi).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for d in 0..head_dim {
                let acc: f32 = (0..=q_pos)
                    .map(|kp| {
                        let v_at = (slot_of(row, kp) * n_kv_heads + kv_head) * head_dim + d;
                        exps[kp] * v[v_at]
                    })
                    .sum();
                want[q_base + d] = acc / denom;
            }
        }
    }

    let out_plain = out[3].clone();
    assert_close(&bf16_read(&out_plain), &want, "sdpa_paged_decode d_64");

    // The sink variant is the same dispatch with one more buffer read. A sink
    // is a learned logit that joins the DENOMINATOR without contributing a
    // value, so it can only shrink the output, and it shrinks each query head
    // by its own amount. The sinks below therefore differ per head and are set
    // near the score range: far below and the effect vanishes into the
    // tolerance, far above and every output collapses to zero, and both of
    // those would pass a test that merely checked "something changed".
    let sinks: Vec<f32> = (0..n_q_heads).map(|h| -0.5 + h as f32 * 0.75).collect();
    let mut with_sink = operands.clone();
    with_sink[3] = vec![0u8; rows * n_q_heads * head_dim * 2];
    with_sink[10] = bf16_bytes(&sinks);
    let out = gpu.dispatch(
        "sdpa_paged_decode_sink_bfloat16_d_64",
        Capability::Baseline,
        &with_sink,
        &push,
        [n_q_heads as u32, rows as u32, 1],
    );

    let sq = bf16_read(&with_sink[10]);
    let got = bf16_read(&out[3]);
    let plain = bf16_read(&out_plain);
    for (row, &position) in positions.iter().enumerate() {
        let q_pos = position as usize;
        for h in 0..n_q_heads {
            let kv_head = h / gqa;
            let q_base = (row * n_q_heads + h) * head_dim;
            let scores: Vec<f32> = (0..=q_pos)
                .map(|kp| {
                    let k_base = (slot_of(row, kp) * n_kv_heads + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|d| scale * q[q_base + d] * k[k_base + d])
                        .sum::<f32>()
                })
                .collect();
            let hi = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .max(sq[h]);
            let exps: Vec<f32> = scores.iter().map(|s| (s - hi).exp()).collect();
            let denom: f32 = exps.iter().sum::<f32>() + (sq[h] - hi).exp();
            let want: Vec<f32> = (0..head_dim)
                .map(|d| {
                    let acc: f32 = (0..=q_pos)
                        .map(|kp| {
                            let v_at = (slot_of(row, kp) * n_kv_heads + kv_head) * head_dim + d;
                            exps[kp] * v[v_at]
                        })
                        .sum();
                    acc / denom
                })
                .collect();
            assert_close(
                &got[q_base..q_base + head_dim],
                &want,
                &format!("sdpa_paged_decode_sink row {row} head {h}"),
            );
            // A sink adds to the denominator and nothing to the numerator, so
            // every element must have moved TOWARD zero relative to the run
            // without one. Checking the direction as well as the value is what
            // rules out a body that read the sink and then divided by it.
            for d in 0..head_dim {
                let (with, without) = (got[q_base + d], plain[q_base + d]);
                assert!(
                    with.abs() <= without.abs() + BF16_TOLERANCE,
                    "sink grew the output at row {row} head {h} dim {d}: \
                     {without} became {with}"
                );
            }
        }
    }
}

/// `sdpa_paged_tiled_bfloat16_d_64` against the same scalar reference.
///
/// The prefill path, which until this test had no direct numeric coverage at
/// all -- it was checked only through the driver, end to end, against a
/// checkpoint's expected tokens. That is a real check but a slow and blunt
/// one, and it was not enough to rewrite the kernel against.
///
/// The rewrite made the key loop cooperative: thirty-two lanes share a row and
/// reduce one score between them, instead of each lane recomputing the whole
/// dot product for every dimension it owns. Doing that needs a workgroup-wide
/// `barrier()` inside the key loop, and a barrier is only legal if every
/// thread reaches it -- which is the hard part here, because a tile spans
/// thirty-two DIFFERENT rows with different positions, different windows and
/// different masks. Three things had to become uniform: the loop bound is the
/// largest position in the TILE rather than each row's own, rows past
/// `n_rows` stay in the loop with `q_pos = -1` instead of returning, and the
/// mask became a predicate on the body rather than a `continue`.
///
/// **This test does not prove those three.** It was written believing it did,
/// and then mutated to find out. Reverting the tile-wide bound to each row's
/// own position passes. Letting the dead rows return early passes. Both do,
/// because on this GPU a row's thirty-two lanes are exactly one subgroup, so
/// lanes never actually wait on a barrier a neighbouring row failed to reach
/// -- the divergence the barrier rule exists to forbid is unobservable here.
/// The uniform shape is still what the specification requires and is what a
/// wider or narrower subgroup would need; this machine simply cannot tell.
/// Saying so is better than a comment claiming coverage that does not exist.
///
/// What it DOES prove, by mutants that fail: the mask and window are applied
/// (dropping `keeps` fails), and each lane accumulates the output dimensions
/// it actually owns (striding them `lane * n + i` instead of
/// `lane + i * 32` fails). Together with the page table, the shuffled
/// physical pages, grouped-query attention and the online-softmax fold, that
/// is the arithmetic of the rewrite. The barrier discipline is argued, not
/// measured, and a machine with a subgroup narrower than 32 would be the
/// place to measure it.
///
/// The reference below is the plain softmax, computed only over the keys the
/// mask and the window keep.
#[test]
fn sdpa_paged_tiled_matches_a_scalar_reference() {
    let gpu = gpu!();

    let head_dim = 64usize;
    let page_size = 16usize;
    let n_kv_heads = 2usize;
    let gqa = 2usize;
    let n_q_heads = n_kv_heads * gqa;
    let scale = 0.125f32;
    let window = 8i32;

    // Two requests prefilled at once: 20 tokens and 15. 35 rows is one full
    // tile plus three, which is the point -- see the doc above.
    let lengths = [20usize, 15];
    let rows: usize = lengths.iter().sum();
    assert!(rows > 32, "the dead-lane case is the interesting one");

    let pages_per: Vec<usize> = lengths.iter().map(|l| l.div_ceil(page_size)).collect();
    let total_pages: usize = pages_per.iter().sum();
    let physical: Vec<u32> = {
        let mut v: Vec<u32> = (0..total_pages as u32).collect();
        v.reverse();
        v
    };
    let mut indptr = vec![0u32];
    for p in &pages_per {
        indptr.push(indptr.last().unwrap() + *p as u32);
    }

    let slots = total_pages * page_size;
    let kv_elems = slots * n_kv_heads * head_dim;
    let kf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 31) as f32 - 15.0) / 40.0)
        .collect();
    let vf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 23) as f32 - 11.0) / 30.0)
        .collect();
    let qf: Vec<f32> = (0..rows * n_q_heads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) / 20.0)
        .collect();

    let mut positions: Vec<i32> = Vec::new();
    let mut req_of_token: Vec<i32> = Vec::new();
    for (req, len) in lengths.iter().enumerate() {
        for t in 0..*len {
            positions.push(t as i32);
            req_of_token.push(req as i32);
        }
    }

    // Row 7 is in the first tile, row 33 in the second and alive; both keep
    // only the keys whose index is not one more than a multiple of three, and
    // their own position is always kept so no row ends up with an empty
    // softmax.
    let stride = 20usize;
    let masked_rows = [7usize, 33];
    let mut mask = vec![0u8; rows * stride];
    let mut mask_on = vec![0u8; rows];
    for &r in &masked_rows {
        mask_on[r] = 1;
        for kp in 0..stride {
            let keep = kp % 3 != 1 || kp == positions[r] as usize;
            mask[r * stride + kp] = u8::from(keep);
        }
    }

    let operands = vec![
        bf16_bytes(&qf),
        bf16_bytes(&kf),
        bf16_bytes(&vf),
        vec![0u8; rows * n_q_heads * head_dim * 2],
        positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
        req_of_token.iter().flat_map(|r| r.to_le_bytes()).collect(),
        physical.iter().flat_map(|p| p.to_le_bytes()).collect(),
        indptr.iter().flat_map(|p| p.to_le_bytes()).collect(),
        mask.clone(),
        mask_on.clone(),
        vec![0u8; n_q_heads * 2],
    ];

    let mut push = Vec::new();
    push.extend_from_slice(&(gqa as i32).to_le_bytes());
    push.extend_from_slice(&(page_size as i32).to_le_bytes());
    push.extend_from_slice(&(n_kv_heads as i32).to_le_bytes());
    push.extend_from_slice(&scale.to_le_bytes());
    push.extend_from_slice(&(stride as u32).to_le_bytes());
    push.extend_from_slice(&window.to_le_bytes());
    push.extend_from_slice(&(rows as i32).to_le_bytes());

    let out = gpu.dispatch(
        "sdpa_paged_tiled_bfloat16_d_64",
        Capability::Baseline,
        &operands,
        &push,
        [n_q_heads as u32, (rows as u32).div_ceil(32), 1],
    );

    let q = bf16_read(&operands[0]);
    let k = bf16_read(&operands[1]);
    let v = bf16_read(&operands[2]);
    let slot_of = |req: usize, kp: usize| {
        let phys = physical[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };

    let mut want = vec![0.0f32; rows * n_q_heads * head_dim];
    let mut kept_total = 0usize;
    for row in 0..rows {
        let req = req_of_token[row] as usize;
        let q_pos = positions[row] as usize;
        let start = if window > 0 && q_pos as i32 >= window {
            q_pos + 1 - window as usize
        } else {
            0
        };
        let keeps: Vec<usize> = (start..=q_pos)
            .filter(|kp| mask_on[row] == 0 || mask[row * stride + kp] != 0)
            .collect();
        kept_total += keeps.len();
        for h in 0..n_q_heads {
            let kv_head = h / gqa;
            let q_base = (row * n_q_heads + h) * head_dim;
            let scores: Vec<f32> = keeps
                .iter()
                .map(|&kp| {
                    let k_base = (slot_of(req, kp) * n_kv_heads + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|d| scale * q[q_base + d] * k[k_base + d])
                        .sum::<f32>()
                })
                .collect();
            let hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - hi).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for d in 0..head_dim {
                let acc: f32 = keeps
                    .iter()
                    .enumerate()
                    .map(|(i, &kp)| {
                        let v_at = (slot_of(req, kp) * n_kv_heads + kv_head) * head_dim + d;
                        exps[i] * v[v_at]
                    })
                    .sum();
                want[q_base + d] = acc / denom;
            }
        }
    }

    // The window and the mask must actually have thrown keys away, or the
    // whole point of the shape above is lost and this is a causal-only test.
    let all_causal: usize = (0..rows).map(|r| positions[r] as usize + 1).sum();
    assert!(
        kept_total < all_causal,
        "the window and mask kept everything ({kept_total} of {all_causal}); \
         this test is not covering what it claims to"
    );

    assert_close(&bf16_read(&out[3]), &want, "sdpa_paged_tiled d_64");
}

// ---------------------------------------------------------------------------
// the remaining common idioms
// ---------------------------------------------------------------------------

/// `silu_mul_bfloat16` — the SwiGLU every layer's MLP runs.
///
/// Small, but the sigmoid is written the MLX way (`1/(1+exp(-|x|))` folded by
/// sign) rather than the naive way, specifically so a large negative input does
/// not overflow `exp`. The reference here is the naive form, and the inputs
/// reach far enough negative that a shader which had transcribed the naive
/// version would differ.
#[test]
fn silu_mul_matches_a_scalar_reference() {
    let gpu = gpu!();

    // 512 would be two whole workgroups of 256, which is a shape where the
    // tail branch never runs at all. 460 leaves 204 lanes past the end.
    let n = 460usize;
    let gate: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) / 2.0).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();

    let operands = vec![bf16_bytes(&gate), bf16_bytes(&up), vec![0u8; n * 2]];
    let out = gpu.dispatch(
        "silu_mul_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [n.div_ceil(256) as u32, 1, 1],
    );

    let g = bf16_read(&operands[0]);
    let u = bf16_read(&operands[1]);
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    assert_close(&bf16_read(&out[2]), &want, "silu_mul_bfloat16");
}

/// `neox_decode_bfloat16` — RoPE, in place.
///
/// Two things worth a device. It writes its INPUT buffer, so a shader that
/// read `x[i2]` after writing `x[i1]` would rotate against a value it had
/// already changed — the pair has to be read before either store, and only
/// running it shows that. And the three push fields are `scale`, `base`,
/// `head_dim`, one of which is a float among ints; a push block that ordered
/// them differently would still typecheck everywhere.
#[test]
fn neox_decode_rotates_pairs_in_place() {
    let gpu = gpu!();

    let head_dim = 64usize;
    let n_head = 2usize;
    let pair_half = head_dim / 2;
    let scale = 1.0f32;
    let base = 13.0f32; // exp2(-d * base) is the frequency; any value will do
    let position = 7i32;

    let xf: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i % 23) as f32 - 11.0) / 8.0)
        .collect();
    let operands = vec![bf16_bytes(&xf), position.to_le_bytes().to_vec()];
    let mut push = Vec::new();
    push.extend_from_slice(&scale.to_le_bytes());
    push.extend_from_slice(&base.to_le_bytes());
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());

    // `.x` is the pair index, `.y` the head; the decode form pins the row to 0.
    let out = gpu.dispatch(
        "neox_decode_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [pair_half as u32, n_head as u32, 1],
    );

    let x = bf16_read(&operands[0]);
    let mut want = x.clone();
    for h in 0..n_head {
        for i in 0..pair_half {
            let d = i as f32 / pair_half as f32;
            let theta = scale * position as f32 * (-d * base).exp2();
            let (c, s) = (theta.cos(), theta.sin());
            let i1 = h * head_dim + i;
            let i2 = i1 + pair_half;
            want[i1] = x[i1] * c - x[i2] * s;
            want[i2] = x[i1] * s + x[i2] * c;
        }
    }
    assert_close(&bf16_read(&out[0]), &want, "neox_decode_bfloat16");
}

/// `router_topk_bfloat16` — which experts a token goes to, and how much of it.
///
/// The one kernel here whose output is partly INTEGER, so a tolerance is the
/// wrong instrument for half of it: the expert ids have to match exactly, and
/// only the weights get a tolerance. Ties are avoided in the fixture because
/// the shader breaks them by first-index and a reference sort need not.
#[test]
fn router_topk_picks_the_right_experts() {
    let gpu = gpu!();

    let n_experts = 8usize;
    let top_k = 2usize;
    let rows = 4usize;
    // Distinct values per row, so "the largest two" is unambiguous.
    let logits: Vec<f32> = (0..rows * n_experts)
        .map(|i| ((i * 7 + (i / n_experts) * 3) % 29) as f32 / 4.0)
        .collect();

    let operands = vec![
        bf16_bytes(&logits),
        vec![0u8; rows * top_k * 4],
        vec![0u8; rows * top_k * 2],
        // RouterParams { n_experts, experts_per_token, softmax_over_all, logits_pitch }
        [n_experts as u32, top_k as u32, 1, n_experts as u32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        vec![0u8; n_experts * 2], // per_expert_scale, unread without PIE_SCALED
    ];

    let out = gpu.dispatch(
        "router_topk_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [1, rows as u32, 1],
    );

    let lg = bf16_read(&operands[0]);
    let ids: Vec<i32> = out[1]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let weights = bf16_read(&out[2]);

    for row in 0..rows {
        let slice = &lg[row * n_experts..(row + 1) * n_experts];
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|a, b| slice[*b].total_cmp(&slice[*a]));

        // `softmax_over_all` is set, so the denominator runs over EVERY expert
        // and not only the chosen ones -- the weights of a token therefore do
        // not sum to one, which is the behaviour being pinned.
        let hi = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = slice.iter().map(|v| (v - hi).exp()).sum();

        for r in 0..top_k {
            let want_id = order[r];
            assert_eq!(
                ids[row * top_k + r] as usize,
                want_id,
                "row {row} rank {r}: expert id"
            );
            let want_w = (slice[want_id] - hi).exp() / denom;
            assert_close(
                &[weights[row * top_k + r]],
                &[want_w],
                &format!("row {row} rank {r}: weight"),
            );
        }
    }
}

/// `affine_qmv_routed_bfloat16_gs_64_b_4` — the MoE expert matvec.
///
/// This is the one shader here that indexes THREE things at once: the weight
/// block is chosen by the expert id, the input row by a pair of strides, and
/// the output by the flat slot. Any of the three can be off by a factor of the
/// others and still produce plausible numbers, because every expert's weights
/// have the same shape. The reference below therefore varies the routing —
/// different rows pick different experts, and one slot is left unrouted — so a
/// shader that ignored `expert_ids` and always read expert 0 would disagree
/// everywhere except by luck.
#[test]
fn routed_matvec_reads_the_expert_the_router_chose() {
    let gpu = gpu!();

    let (group, bits) = (64usize, 4usize);
    let (k, out, experts) = (128usize, 16usize, 4usize);
    let (rows, slots) = (3usize, 2usize);

    // The expert axis is the outermost dimension of the same row-major block
    // the dense matvec uses, so `experts * out` rows generate it exactly.
    let (w, scales, biases, dense) = affine_weights(experts * out, k, group, bits);

    // A slot of -1 is the router declining to route: the shader must return
    // before it writes, leaving the output at whatever was there.
    let expert_ids: Vec<i32> = vec![2, 0, 1, 3, -1, 2];
    assert_eq!(expert_ids.len(), rows * slots);

    // Deliberately not tight: a stride equal to `k` would hide a shader that
    // multiplied the wrong one, since row and slot would then be interchangeable.
    let x_slot_stride = k;
    let x_row_stride = k * slots + 8;
    let xf: Vec<f32> = (0..rows * x_row_stride)
        .map(|i| ((i % 23) as f32 - 11.0) / 17.0)
        .collect();

    let mut operands = vec![
        w,
        scales,
        biases,
        bf16_bytes(&xf),
        vec![0u8; rows * slots * out * 2],
        bf16_bytes(&vec![0.0; experts * out]),
        expert_ids.iter().flat_map(|e| e.to_le_bytes()).collect(),
    ];
    let mut push = Vec::new();
    for v in [
        k as i32,
        out as i32,
        x_slot_stride as i32,
        x_row_stride as i32,
        slots as i32,
    ] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let got = gpu.dispatch(
        "affine_qmv_routed_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &operands,
        &push,
        [rows as u32, out.div_ceil(8) as u32, slots as u32],
    );

    let xq = bf16_read(&operands[3]);
    let mut want = vec![0.0f32; rows * slots * out];
    for row in 0..rows {
        for slot in 0..slots {
            let sel = row * slots + slot;
            let e = expert_ids[sel];
            if e < 0 {
                continue; // stays at the zero the buffer was created with
            }
            let base = row * x_row_stride + slot * x_slot_stride;
            for o in 0..out {
                let wrow = (e as usize) * out + o;
                want[sel * out + o] = (0..k).map(|i| xq[base + i] * dense[wrow * k + i]).sum();
            }
        }
    }
    assert_close(&bf16_read(&got[4]), &want, "affine_qmv_routed");

    // The bias is indexed by EXPERT, not by slot, which is the one indexing
    // the unbiased variant above cannot check at all.
    let bias: Vec<f32> = (0..experts * out)
        .map(|i| ((i % 13) as f32 - 6.0) / 40.0)
        .collect();
    operands[5] = bf16_bytes(&bias);
    let got = gpu.dispatch(
        "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &operands,
        &push,
        [rows as u32, out.div_ceil(8) as u32, slots as u32],
    );

    let biasq = bf16_read(&operands[5]);
    let mut want_bias = want.clone();
    for row in 0..rows {
        for slot in 0..slots {
            let sel = row * slots + slot;
            let e = expert_ids[sel];
            if e < 0 {
                continue;
            }
            for o in 0..out {
                want_bias[sel * out + o] += biasq[(e as usize) * out + o];
            }
        }
    }
    assert_close(&bf16_read(&got[4]), &want_bias, "affine_qmv_routed_bias");
}

/// `embed_gather` — the quantized tied-embedding read, at both its shapes.
///
/// The single-token form and the `_mb` form are different modules with
/// different workgroup shapes, and `_scaled` adds a SECOND push field. All
/// three differences are places the launch can be wired to a body that ignores
/// it, so the test runs the plain module and then the scaled multi-batch one
/// and holds both to the same dequantization.
#[test]
fn embed_gather_dequantizes_the_row_the_id_names() {
    let gpu = gpu!();

    let (group, bits) = (64usize, 4usize);
    let (vocab, hidden) = (12usize, 128usize);
    let (w, scales, biases, dense) = affine_weights(vocab, hidden, group, bits);

    let one = vec![
        w.clone(),
        scales.clone(),
        biases.clone(),
        7i32.to_le_bytes().to_vec(),
        vec![0u8; hidden * 2],
    ];
    let got = gpu.dispatch(
        "embed_gather_4bit_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &one,
        &(hidden as i32).to_le_bytes(),
        [hidden.div_ceil(256) as u32, 1, 1],
    );
    let want: Vec<f32> = (0..hidden).map(|k| dense[7 * hidden + k]).collect();
    assert_close(&bf16_read(&got[4]), &want, "embed_gather single");

    // Ids chosen so no two are adjacent: a shader that used the invocation
    // index where it meant the id would still line up on a run of 0,1,2,...
    let ids: Vec<i32> = vec![9, 2, 11, 0, 5, 5];
    let embed_scale = 1.75f32;
    let many = vec![
        w,
        scales,
        biases,
        ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
        vec![0u8; ids.len() * hidden * 2],
    ];
    let mut push = (hidden as i32).to_le_bytes().to_vec();
    push.extend_from_slice(&embed_scale.to_le_bytes());
    let got = gpu.dispatch(
        "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &many,
        &push,
        [hidden.div_ceil(16) as u32, ids.len().div_ceil(16) as u32, 1],
    );

    let mut want: Vec<f32> = Vec::with_capacity(ids.len() * hidden);
    for &id in &ids {
        for k in 0..hidden {
            want.push(dense[id as usize * hidden + k] * embed_scale);
        }
    }
    assert_close(&bf16_read(&got[4]), &want, "embed_gather scaled mb");
}

/// The other three gated activations in `mlp/gated.slang`.
///
/// One file, one binding contract, five bodies chosen by `-D`. That is exactly
/// the arrangement where a preprocessor typo compiles into the WRONG
/// activation and nothing complains, because every branch reads two buffers
/// and writes one. So each is held to its own closed-form reference rather
/// than to the others.
///
/// The two params structs are the second reason. `gptoss` reads `limit` and
/// `alpha` out of a std430 struct whose first field is unused padding, and the
/// strided geglu reads five extents out of another; both are buffers, not push
/// constants, so a field written at the wrong offset is a silent misread.
#[test]
fn geglu_tanh_matches_its_closed_form() {
    let gpu = gpu!();

    // Not a multiple of the 256-wide workgroup, so the last group runs
    // partly past the end and the guard is on the path.
    let n = 460usize;
    // Wide enough that the tanh approximation is saturated at both ends,
    // where a sign or coefficient error stops cancelling.
    let gate: Vec<f32> = (0..n).map(|i| ((i % 61) as f32 - 30.0) / 4.0).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();

    let operands = vec![
        bf16_bytes(&gate),
        bf16_bytes(&up),
        vec![0u8; n * 2],
        vec![0u8; 4],
    ];
    let out = gpu.dispatch(
        "geglu_tanh_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [n.div_ceil(256) as u32, 1, 1],
    );

    let g = bf16_read(&operands[0]);
    let u = bf16_read(&operands[1]);
    let want: Vec<f32> = g.iter().zip(&u).map(|(g, u)| gelu_tanh(*g) * u).collect();
    assert_close(&bf16_read(&out[2]), &want, "geglu_tanh_bfloat16");
}

fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

/// `geglu_tanh_strided_bfloat16` — the same activation with three pitches.
///
/// The three pitches are deliberately all DIFFERENT here. A shader that used
/// the gate pitch to index the output would agree with the reference for as
/// long as they happened to be equal, which is the usual case in production
/// and therefore the useless case to test.
#[test]
fn geglu_strided_uses_each_of_its_three_pitches() {
    let gpu = gpu!();

    let (width, rows) = (48usize, 5usize);
    let (gate_pitch, up_pitch, out_pitch) = (64usize, 56usize, 80usize);

    let gate: Vec<f32> = (0..rows * gate_pitch)
        .map(|i| ((i % 61) as f32 - 30.0) / 6.0)
        .collect();
    let up: Vec<f32> = (0..rows * up_pitch)
        .map(|i| ((i % 23) as f32 - 11.0) / 5.0)
        .collect();

    let mut params = Vec::new();
    for v in [width, rows, gate_pitch, up_pitch, out_pitch] {
        params.extend_from_slice(&(v as u32).to_le_bytes());
    }
    let operands = vec![
        bf16_bytes(&gate),
        bf16_bytes(&up),
        vec![0u8; rows * out_pitch * 2],
        params,
    ];
    let out = gpu.dispatch(
        "geglu_tanh_strided_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [width.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
    );

    let g = bf16_read(&operands[0]);
    let u = bf16_read(&operands[1]);
    let got = bf16_read(&out[2]);
    // A ROW at a time, not an element at a time. `assert_close` scales its
    // tolerance by the largest magnitude it is handed, so checking single
    // elements makes every element its own scale -- and `gelu_tanh` produces
    // elements where that is meaningless. At `gate[0] = -5` the function is
    // `0.5 * x * (1 + tanh(-8.45))`, and `1 + tanh` there is the difference of
    // two numbers that agree to seven digits: the answer, 6.6e-7, is entirely
    // decided by the last bit of whatever `tanh` the device's compiler
    // emitted. Per element, this asked two independent implementations to
    // agree bit for bit on cancellation noise.
    //
    // Measured, not supposed: this element is the ONE disagreement between
    // the RTX 4090 and Mesa's `llvmpipe` across all 47 proofs in this file.
    // llvmpipe's `tanh` returns exactly -1 there and so the product is 0;
    // neither implementation is wrong, and SPIR-V's `Tanh` does not promise
    // enough for either to be. Against the row's own scale -- around 5, four
    // million times larger -- the question becomes the one worth asking,
    // which is whether the shader read the right three pitches.
    for m in 0..rows {
        let want: Vec<f32> = (0..width)
            .map(|k| gelu_tanh(g[m * gate_pitch + k]) * u[m * up_pitch + k])
            .collect();
        let got_row: Vec<f32> = (0..width).map(|k| got[m * out_pitch + k]).collect();
        assert_close(&got_row, &want, &format!("geglu_strided row {m}"));
    }

    // Past `width` the row is padding the shader must not have touched.
    for m in 0..rows {
        for k in width..out_pitch {
            assert_eq!(
                got[m * out_pitch + k],
                0.0,
                "geglu_strided wrote past width at row {m} col {k}"
            );
        }
    }
}

/// `gptoss_swiglu_bfloat16` — the clamped SwiGLU, which is where the limits are.
///
/// `gate` is clamped from ABOVE only and `up` from both sides, an asymmetry
/// that is easy to write as one `clamp` for both. The inputs below run past
/// the limit in every direction so that a symmetric clamp disagrees, and one
/// input is far enough negative that `exp(-alpha * g)` overflows to infinity —
/// the result must still be a finite zero rather than a NaN.
#[test]
fn gptoss_swiglu_clamps_its_two_inputs_differently() {
    let gpu = gpu!();

    let (limit, alpha) = (7.0f32, 1.702f32);
    let gate: Vec<f32> = vec![
        -120.0, -9.0, -2.5, -0.25, 0.0, 0.25, 2.5, 9.0, 120.0, 6.9, 7.1, 3.0,
    ];
    let up: Vec<f32> = vec![
        -30.0, -7.5, -1.0, 0.5, 3.0, 9.0, 40.0, -0.5, 1.0, 2.0, -2.0, 0.0,
    ];
    let n = gate.len();

    let mut params = 0u32.to_le_bytes().to_vec();
    params.extend_from_slice(&limit.to_le_bytes());
    params.extend_from_slice(&alpha.to_le_bytes());

    let operands = vec![bf16_bytes(&gate), bf16_bytes(&up), vec![0u8; n * 2], params];
    let out = gpu.dispatch(
        "gptoss_swiglu_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [n.div_ceil(256) as u32, 1, 1],
    );

    let g = bf16_read(&operands[0]);
    let u = bf16_read(&operands[1]);
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            g * (1.0 / (1.0 + (-alpha * g).exp())) * (u + 1.0)
        })
        .collect();
    let got = bf16_read(&out[2]);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "gptoss_swiglu produced a non-finite value: {got:?}"
    );
    assert_close(&got, &want, "gptoss_swiglu_bfloat16");
}

/// `kv_append_bfloat16` — the contiguous cache write.
///
/// This is a pure scatter, so the only thing it can get wrong is the address,
/// and the address is built from a 64-bit push block: `int head_dim` followed
/// by TWO `uint64_t` strides, which means four bytes of padding the driver has
/// to insert or every stride is read shifted. The strides below are unequal
/// and the write position is not zero, so swapping them or dropping the
/// position lands somewhere else in a cache that is checked in full — every
/// slot the append did not name must still hold the sentinel it started with.
#[test]
fn kv_append_writes_one_position_and_leaves_the_rest() {
    let gpu = gpu!();

    let (head_dim, heads, max_seq) = (64usize, 3usize, 8usize);
    let pos = 5usize;
    let k_head_stride = max_seq * head_dim;
    let k_seq_stride = head_dim;

    let k_new: Vec<f32> = (0..heads * head_dim)
        .map(|i| ((i % 29) as f32 - 14.0) / 7.0)
        .collect();
    let v_new: Vec<f32> = (0..heads * head_dim)
        .map(|i| ((i % 31) as f32 - 15.0) / 9.0)
        .collect();

    // A sentinel rather than zero: zero is also what an unwritten buffer
    // reads as, so it could not tell "left alone" from "written with zero".
    let sentinel = -99.0f32;
    let cache = vec![sentinel; heads * max_seq * head_dim];

    let operands = vec![
        bf16_bytes(&k_new),
        bf16_bytes(&v_new),
        bf16_bytes(&cache),
        bf16_bytes(&cache),
        (pos as i32).to_le_bytes().to_vec(),
    ];

    let mut push = (head_dim as i32).to_le_bytes().to_vec();
    push.extend_from_slice(&[0u8; 4]); // uint64_t is 8-byte aligned
    push.extend_from_slice(&(k_head_stride as u64).to_le_bytes());
    push.extend_from_slice(&(k_seq_stride as u64).to_le_bytes());

    let out = gpu.dispatch(
        "kv_append_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [head_dim.div_ceil(256) as u32, heads as u32, 1],
    );

    let kq = bf16_read(&operands[0]);
    let vq = bf16_read(&operands[1]);
    let sq = bf16_to_f32(f32_to_bf16(sentinel));
    for (which, got) in [("k", bf16_read(&out[2])), ("v", bf16_read(&out[3]))] {
        let src = if which == "k" { &kq } else { &vq };
        for h in 0..heads {
            for s in 0..max_seq {
                for d in 0..head_dim {
                    let at = h * k_head_stride + s * k_seq_stride + d;
                    let want = if s == pos { src[h * head_dim + d] } else { sq };
                    assert_eq!(got[at], want, "{which} cache head {h} seq {s} dim {d}");
                }
            }
        }
    }
}

/// `kv_append_paged_bfloat16` — the same scatter through a page table.
///
/// The comment at the top of `kv_write.slang` records that these two bindings
/// were off by one until a SPIR-V audit caught it, which is precisely the
/// failure a test should be able to reproduce: the paged row keeps Metal's
/// placeholder ring operands, so `w_page` and `w_off` sit at bindings 10 and
/// 11 with six unused descriptors before them. Reading one slot over would
/// take the offset for the page. The pages below are therefore NOT in order
/// and the offsets are not zero, so page-major and offset-major addressing
/// give different answers.
#[test]
fn kv_append_paged_follows_the_write_page_table() {
    let gpu = gpu!();

    let (head_dim, heads, page_size, pages) = (32usize, 2usize, 4usize, 3usize);
    let w_page: Vec<u32> = vec![2, 0, 2, 1];
    let w_off: Vec<u32> = vec![1, 3, 2, 0];
    let tokens = w_page.len();

    let row_stride = heads * head_dim;
    let k_new: Vec<f32> = (0..tokens * row_stride)
        .map(|i| ((i % 37) as f32 - 18.0) / 6.0)
        .collect();
    let v_new: Vec<f32> = (0..tokens * row_stride)
        .map(|i| ((i % 23) as f32 - 11.0) / 4.0)
        .collect();

    let sentinel = -99.0f32;
    let cache = vec![sentinel; pages * page_size * row_stride];

    // Bindings 4..9 and 12 are the ring placeholders the row still names; the
    // shader declares none of them, but the descriptor set has to cover the
    // layout, so they are bound as one dead byte each.
    let mut operands = vec![
        bf16_bytes(&k_new),
        bf16_bytes(&v_new),
        bf16_bytes(&cache),
        bf16_bytes(&cache),
    ];
    operands.resize(10, vec![0u8; 4]);
    operands.push(w_page.iter().flat_map(|p| p.to_le_bytes()).collect());
    operands.push(w_off.iter().flat_map(|o| o.to_le_bytes()).collect());
    operands.push(vec![0u8; 4]);

    let mut push = Vec::new();
    for v in [head_dim as i32, page_size as i32, heads as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let out = gpu.dispatch(
        "kv_append_paged_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [head_dim.div_ceil(256) as u32, heads as u32, tokens as u32],
    );

    let kq = bf16_read(&operands[0]);
    let vq = bf16_read(&operands[1]);
    let sq = bf16_to_f32(f32_to_bf16(sentinel));

    let mut want_k = vec![sq; pages * page_size * row_stride];
    let mut want_v = want_k.clone();
    for i in 0..tokens {
        let slot = w_page[i] as usize * page_size + w_off[i] as usize;
        for h in 0..heads {
            for d in 0..head_dim {
                let src = i * row_stride + h * head_dim + d;
                let dst = slot * row_stride + h * head_dim + d;
                want_k[dst] = kq[src];
                want_v[dst] = vq[src];
            }
        }
    }
    assert_eq!(bf16_read(&out[2]), want_k, "paged k cache");
    assert_eq!(bf16_read(&out[3]), want_v, "paged v cache");
}

/// A scalar SDPA reference: plain softmax over the selected keys.
///
/// The shader computes the same thing by an ONLINE recurrence, rescaling the
/// running sum every time a new maximum arrives. The two agree only if the
/// rescale is exact, so the reference deliberately does it the other way — one
/// pass for the max, one for the sum — rather than mirroring the shader.
#[allow(clippy::too_many_arguments)]
fn sdpa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_base: usize,
    kv_head: usize,
    range: std::ops::Range<usize>,
    d: usize,
    k_head_stride: usize,
    k_seq_stride: usize,
    v_head_stride: usize,
    v_seq_stride: usize,
    scale: f32,
    sink: Option<f32>,
) -> Vec<f32> {
    let scores: Vec<f32> = range
        .clone()
        .map(|i| {
            let kb = kv_head * k_head_stride + i * k_seq_stride;
            (0..d).map(|j| scale * q[q_base + j] * k[kb + j]).sum()
        })
        .collect();
    let mut max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if let Some(s) = sink {
        max = max.max(s);
    }
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let mut denom: f32 = exps.iter().sum();
    if let Some(s) = sink {
        denom += (s - max).exp();
    }
    (0..d)
        .map(|c| {
            let num: f32 = range
                .clone()
                .zip(&exps)
                .map(|(i, e)| e * v[kv_head * v_head_stride + i * v_seq_stride + c])
                .sum();
            if denom == 0.0 { 0.0 } else { num / denom }
        })
        .collect()
}

/// `sdpa_vector_decode_bfloat16_d_64` — the dense, unpaged decode.
///
/// Four 64-bit strides in a push block that starts with two ints, so the
/// alignment padding has to be right or key and value addressing both walk.
/// The strides are chosen unequal and the K and V caches are given DIFFERENT
/// layouts, which is the only way to catch a body that reused `k_seq_stride`
/// for the value read — in production the two are usually identical.
///
/// GQA is set so several query heads share one KV head, since a shader that
/// used `q_head` where it meant `kv_head` agrees with the reference whenever
/// the factor is one.
#[test]
fn sdpa_vector_decode_matches_a_softmax_reference() {
    let gpu = gpu!();

    let d = 64usize;
    let (q_heads, gqa, n, rows) = (4usize, 2usize, 7usize, 2usize);
    let kv_heads = q_heads / gqa;

    // K is [head][seq][dim]; V is [seq][head][dim]. Nothing requires them to
    // agree, and making them disagree is the point.
    let (k_head_stride, k_seq_stride) = (n * d, d);
    let (v_head_stride, v_seq_stride) = (d, kv_heads * d);
    let scale = 0.125f32;

    let q: Vec<f32> = (0..q_heads * rows * d)
        .map(|i| ((i % 29) as f32 - 14.0) / 11.0)
        .collect();
    let k: Vec<f32> = (0..kv_heads * n * d)
        .map(|i| ((i % 37) as f32 - 18.0) / 13.0)
        .collect();
    let v: Vec<f32> = (0..kv_heads * n * d)
        .map(|i| ((i % 23) as f32 - 11.0) / 7.0)
        .collect();

    let operands = vec![
        bf16_bytes(&q),
        bf16_bytes(&k),
        bf16_bytes(&v),
        vec![0u8; q_heads * rows * d * 2],
    ];

    let mut push = (gqa as i32).to_le_bytes().to_vec();
    push.extend_from_slice(&(n as i32).to_le_bytes());
    for s in [k_head_stride, k_seq_stride, v_head_stride, v_seq_stride] {
        push.extend_from_slice(&(s as u64).to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());

    let got = gpu.dispatch(
        "sdpa_vector_decode_bfloat16_d_64",
        Capability::Baseline,
        &operands,
        &push,
        [q_heads as u32, rows as u32, 1],
    );

    let (qq, kq, vq) = (
        bf16_read(&operands[0]),
        bf16_read(&operands[1]),
        bf16_read(&operands[2]),
    );
    let got = bf16_read(&got[3]);
    for h in 0..q_heads {
        for r in 0..rows {
            let base = (h * rows + r) * d;
            let want = sdpa_reference(
                &qq,
                &kq,
                &vq,
                base,
                h / gqa,
                0..n,
                d,
                k_head_stride,
                k_seq_stride,
                v_head_stride,
                v_seq_stride,
                scale,
                None,
            );
            assert_close(
                &got[base..base + d],
                &want,
                &format!("sdpa_vector_decode head {h} row {r}"),
            );
        }
    }
}

/// `sdpa_vector_decode_swa_bfloat16_d_256` — the sliding window.
///
/// The window is where this differs, and the interesting part is that the
/// causal end is PER ROW: `n - (rows - 1 - row)`, so the last row sees `n`
/// keys and each earlier one sees fewer. Three rows with a window shorter than
/// the sequence gives three different key ranges in a single dispatch, and a
/// body that computed the end from the wrong row would still produce a valid
/// softmax over the wrong set.
///
/// The row strides are passed explicitly and set LARGER than the packed
/// spacing, because zero selects a fallback path and would leave the two push
/// fields unread.
#[test]
fn sdpa_swa_gives_each_row_its_own_window() {
    let gpu = gpu!();

    let d = 256usize;
    let (q_heads, gqa, n, rows, window) = (2usize, 2usize, 9usize, 3usize, 4usize);
    let kv_heads = q_heads / gqa;
    let (k_head_stride, k_seq_stride) = (n * d, d);
    let (v_head_stride, v_seq_stride) = (d, kv_heads * d);
    let scale = 0.0625f32;
    let row_stride = q_heads * d + 32;

    let q: Vec<f32> = (0..rows * row_stride)
        .map(|i| ((i % 31) as f32 - 15.0) / 12.0)
        .collect();
    let k: Vec<f32> = (0..kv_heads * n * d)
        .map(|i| ((i % 41) as f32 - 20.0) / 15.0)
        .collect();
    let v: Vec<f32> = (0..kv_heads * n * d)
        .map(|i| ((i % 19) as f32 - 9.0) / 6.0)
        .collect();

    let operands = vec![
        bf16_bytes(&q),
        bf16_bytes(&k),
        bf16_bytes(&v),
        vec![0u8; rows * row_stride * 2],
    ];

    let mut push = (gqa as i32).to_le_bytes().to_vec();
    push.extend_from_slice(&(n as i32).to_le_bytes());
    for s in [k_head_stride, k_seq_stride, v_head_stride, v_seq_stride] {
        push.extend_from_slice(&(s as u64).to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
    for s in [window, row_stride, row_stride] {
        push.extend_from_slice(&(s as i32).to_le_bytes());
    }

    let got = gpu.dispatch(
        "sdpa_vector_decode_swa_bfloat16_d_256",
        Capability::Baseline,
        &operands,
        &push,
        [q_heads as u32, rows as u32, 1],
    );

    let (qq, kq, vq) = (
        bf16_read(&operands[0]),
        bf16_read(&operands[1]),
        bf16_read(&operands[2]),
    );
    let got = bf16_read(&got[3]);
    let mut seen = std::collections::BTreeSet::new();
    for r in 0..rows {
        let end = n - (rows - 1 - r);
        let start = end.saturating_sub(window);
        seen.insert((start, end));
        for h in 0..q_heads {
            let base = r * row_stride + h * d;
            let want = sdpa_reference(
                &qq,
                &kq,
                &vq,
                base,
                h / gqa,
                start..end,
                d,
                k_head_stride,
                k_seq_stride,
                v_head_stride,
                v_seq_stride,
                scale,
                None,
            );
            assert_close(
                &got[base..base + d],
                &want,
                &format!("sdpa_swa head {h} row {r} keys {start}..{end}"),
            );
        }
    }
    // If the ranges were not all distinct the test would be checking one case
    // three times and the per-row end would go unexercised.
    assert_eq!(seen.len(), rows, "the rows did not get distinct windows");
}

/// The MoE routing pipeline: `route_sort`, then `route_gather`, then
/// `combine_sorted`, checked as a composition.
///
/// `route_sort` fills its spans with `atomicAdd`, so the ORDER within one
/// expert's run is a race and there is no reference for it. What is defined is
/// the structure, and that is what the first half checks: each expert's rows
/// land inside that expert's span and nowhere else, the span is padded up to a
/// whole number of tiles, `tile_expert` names the owner of every tile, and
/// `inv` is a genuine inverse of `perm`.
///
/// The second half is the part a race cannot hide behind. Gathering `x` and
/// then combining it must return each row scaled by the sum of its own expert
/// weights, WHATEVER order the sort chose — that identity holds only if `perm`
/// and `inv` are inverse in the same run, which is exactly the coupling that
/// would break if either kernel indexed by token where it meant slot.
#[test]
fn the_routing_pipeline_gathers_and_combines_back_to_where_it_started() {
    let gpu = gpu!();

    let (experts, tokens, k, tile, width) = (4usize, 5usize, 2usize, 2usize, 8usize);
    let n = tokens * k;
    // Uneven counts, and one expert (3) chosen only once, so at least one span
    // is padded and at least one tile is half empty.
    let expert_ids: Vec<i32> = vec![2, 0, 1, 2, 0, 1, 2, 3, 1, 0];
    assert_eq!(expert_ids.len(), n);

    let mut counts = vec![0usize; experts];
    for &e in &expert_ids {
        counts[e as usize] += 1;
    }
    let spans: Vec<usize> = counts.iter().map(|c| c.div_ceil(tile) * tile).collect();
    let padded: usize = spans.iter().sum();
    let bases: Vec<usize> = spans
        .iter()
        .scan(0, |a, s| {
            let b = *a;
            *a += s;
            Some(b)
        })
        .collect();

    let x_pitch = width + 3;
    let out_pitch = width + 5;

    let route_params =
        |p: [usize; 7]| -> Vec<u8> { p.iter().flat_map(|v| (*v as u32).to_le_bytes()).collect() };
    let params = route_params([n, experts, k, tile, padded, width, x_pitch]);

    let sort = vec![
        expert_ids.iter().flat_map(|e| e.to_le_bytes()).collect(),
        vec![0u8; padded * 4],
        vec![0u8; padded * 4],
        vec![0u8; (padded / tile) * 4],
        params.clone(),
        vec![0u8; n * 4],
    ];
    let sorted = gpu.dispatch("route_sort", Capability::Baseline, &sort, &[], [1, 1, 1]);

    let read_i32 = |b: &[u8]| -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let perm = read_i32(&sorted[1]);
    let row_expert = read_i32(&sorted[2]);
    let tile_expert = read_i32(&sorted[3]);
    let inv = read_i32(&sorted[5]);

    for e in 0..experts {
        let mut landed: Vec<i32> = (bases[e]..bases[e] + counts[e])
            .map(|at| {
                assert_eq!(row_expert[at], e as i32, "row_expert at {at}");
                perm[at]
            })
            .collect();
        landed.sort_unstable();
        let mut expect: Vec<i32> = expert_ids
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == e as i32)
            .map(|(i, _)| i as i32)
            .collect();
        expect.sort_unstable();
        assert_eq!(
            landed, expect,
            "expert {e} did not get exactly its own rows"
        );

        let pad = bases[e] + counts[e];
        for (i, &slot) in perm[pad..bases[e] + spans[e]].iter().enumerate() {
            assert_eq!(slot, -1, "padding slot {} was written", pad + i);
        }
        let first_tile = bases[e] / tile;
        for (i, &owner) in tile_expert[first_tile..(bases[e] + spans[e]) / tile]
            .iter()
            .enumerate()
        {
            assert_eq!(owner, e as i32, "tile {}", first_tile + i);
        }
    }
    for (i, &at) in inv.iter().enumerate() {
        assert!(at >= 0, "row {i} was not placed");
        assert_eq!(perm[at as usize], i as i32, "inv is not the inverse at {i}");
    }

    let x: Vec<f32> = (0..tokens * x_pitch)
        .map(|i| ((i % 29) as f32 - 14.0) / 9.0)
        .collect();
    let gather = vec![
        bf16_bytes(&x),
        vec![0u8; padded * width * 2],
        sorted[1].clone(),
        params,
    ];
    let gathered = gpu.dispatch(
        "route_gather",
        Capability::Baseline,
        &gather,
        &[],
        [width.div_ceil(16) as u32, padded.div_ceil(16) as u32, 1],
    );

    let xq = bf16_read(&gather[0]);
    let got = bf16_read(&gathered[1]);
    for r in 0..padded {
        for c in 0..width {
            let want = if perm[r] < 0 {
                0.0
            } else {
                xq[(perm[r] as usize / k) * x_pitch + c]
            };
            assert_eq!(got[r * width + c], want, "route_gather row {r} col {c}");
        }
    }

    let weights: Vec<f32> = (0..n).map(|i| 0.1 + (i % 7) as f32 / 20.0).collect();
    let combine = vec![
        gathered[1].clone(),
        bf16_bytes(&weights),
        vec![0u8; tokens * out_pitch * 2],
        [width, k, out_pitch]
            .iter()
            .flat_map(|v| (*v as u32).to_le_bytes())
            .collect(),
        sorted[5].clone(),
    ];
    let combined = gpu.dispatch(
        "combine_sorted",
        Capability::Baseline,
        &combine,
        &[],
        [width.div_ceil(16) as u32, tokens.div_ceil(16) as u32, 1],
    );

    let wq = bf16_read(&combine[1]);
    let got = bf16_read(&combined[2]);
    for t in 0..tokens {
        let total: f32 = (0..k).map(|e| wq[t * k + e]).sum();
        for c in 0..width {
            assert_close(
                &[got[t * out_pitch + c]],
                &[xq[t * x_pitch + c] * total],
                &format!("gather-then-combine token {t} col {c}"),
            );
        }
    }
}

/// `shared_expert_combine` and its strided sibling.
///
/// The gate is one value per row, applied across the whole row, and the two
/// variants do NOT index it the same way -- which is the whole reason to run
/// them on a device.
///
/// Non-strided reads `gate[row]`, because the gate is a `[rows, 1]` tensor.
/// Strided reads `gate[row * pitch]`, because there the gate projection's
/// output is written a full pitch apart like every other projection's;
/// `route.metal` states both halves.
///
/// This test used to allocate one `rows * pitch` gate for both and index the
/// reference by `row * row_stride`, which made it blind to exactly the bug it
/// was written to catch: the shader collapsed the row INDEX into the row's
/// DATA base and read `gate[row * width]`, and the reference did the same
/// thing. So the non-strided case now gets a gate buffer of exactly `rows`
/// elements, which makes the wrong index unrepresentable rather than merely
/// unchecked -- it runs off the end of the allocation.
#[test]
fn shared_expert_combine_gates_by_row_not_by_element() {
    let gpu = gpu!();

    let (rows, width) = (6usize, 32usize);
    let pitch = width + 9;

    let routed: Vec<f32> = (0..rows * pitch)
        .map(|i| ((i % 23) as f32 - 11.0) / 8.0)
        .collect();
    let shared: Vec<f32> = (0..rows * pitch)
        .map(|i| ((i % 31) as f32 - 15.0) / 6.0)
        .collect();
    // Spread across the sigmoid so no two rows share a gate.
    let gate_of = |i: usize| ((i % 37) as f32 - 18.0) / 3.0;

    // `gate_at` is the index the variant is contracted to read, and the buffer
    // is only as long as those indices reach.
    let check = |entrypoint: &str,
                 push: Vec<u8>,
                 row_stride: usize,
                 gate: Vec<f32>,
                 gate_at: &dyn Fn(usize) -> usize| {
        let operands = vec![
            bf16_bytes(&routed),
            bf16_bytes(&shared),
            bf16_bytes(&gate),
            vec![0u8; rows * pitch * 2],
        ];
        let out = gpu.dispatch(
            entrypoint,
            Capability::Baseline,
            &operands,
            &push,
            [width.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
        );
        let (r, s, g) = (
            bf16_read(&operands[0]),
            bf16_read(&operands[1]),
            bf16_read(&operands[2]),
        );
        let got = bf16_read(&out[3]);
        for row in 0..rows {
            let base = row * row_stride;
            let gate = 1.0 / (1.0 + (-g[gate_at(row)]).exp());
            for c in 0..width {
                assert_close(
                    &[got[base + c]],
                    &[r[base + c] + gate * s[base + c]],
                    &format!("{entrypoint} row {row} col {c}"),
                );
            }
        }
    };

    check(
        "shared_expert_combine",
        (width as u32).to_le_bytes().to_vec(),
        width,
        (0..rows).map(gate_of).collect(),
        &|row| row,
    );

    let mut push = (width as u32).to_le_bytes().to_vec();
    push.extend_from_slice(&(pitch as i32).to_le_bytes());
    check(
        "shared_expert_combine_strided",
        push,
        pitch,
        (0..rows * pitch).map(gate_of).collect(),
        &|row| row * pitch,
    );
}

/// The four small elementwise kernels, and the four different ways they are
/// handed the one number each of them needs.
///
/// These have almost no arithmetic, which is the reason to run them: what can
/// go wrong is where the scalar came from, and a number read from the wrong
/// place is a plausible number rather than a fault.
///
/// - `residual_add` needs none at all.
/// - `layer_scalar` takes its multiplier from element zero of a BUFFER, because
///   which layer is running is the fire's business and not the statement's.
/// - `logit_softcap` states its cap as a `Const<f32>` mark, so the cap is the
///   one field of a four-byte PUSH block.
/// - `ple_combine` states `inv_sqrt2` as a `Const<f32>` mark, so its scale is a
///   four-byte push range too.
///
/// Both of the last two used to carry a second word — `PleCombineParams`' count
/// and `SoftcapParams`' — that no shader read, and this test deliberately set
/// each one WRONG, to `0`, because they were documented as ABI parity fields
/// that must not act as bounds and a shader using one as a bound would write
/// nothing at all.
///
/// Neither dead word exists now: the marks replaced the structs, and a block of
/// one live word has nothing left to poison. What the poisoned fields guarded —
/// that no lane is bounded by a stated count — is still guarded by `n` being
/// ragged, which is the property that would catch it either way.
#[test]
fn the_small_elementwise_kernels_read_their_scalars_from_the_right_place() {
    let gpu = gpu!();

    // Ragged on purpose: at an exact multiple of 256 no lane in this
    // dispatch is ever past the end, so a missing bound reads as correct.
    let n = 460usize;
    let x: Vec<f32> = (0..n).map(|i| ((i % 53) as f32 - 26.0) / 5.0).collect();
    let y: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) / 3.0).collect();
    let groups = [n.div_ceil(256) as u32, 1, 1];

    let out = gpu.dispatch(
        "residual_add_bfloat16",
        Capability::Baseline,
        &[bf16_bytes(&x), bf16_bytes(&y), vec![0u8; n * 2]],
        &[],
        groups,
    );
    let (xq, yq) = (bf16_read(&bf16_bytes(&x)), bf16_read(&bf16_bytes(&y)));
    let want: Vec<f32> = xq.iter().zip(&yq).map(|(a, b)| a + b).collect();
    assert_close(&bf16_read(&out[2]), &want, "residual_add_bfloat16");

    // Three buffers and no block: `LayerScalarParams` held one dead word --
    // the hidden width, which the grid already is -- and is deleted on all
    // three planes.
    let scalar = -0.375f32;
    let out = gpu.dispatch(
        "layer_scalar_mul_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&x),
            bf16_bytes(&[scalar]),
            vec![0u8; n * 2],
        ],
        &[],
        groups,
    );
    let sq = bf16_to_f32(f32_to_bf16(scalar));
    let want: Vec<f32> = xq.iter().map(|a| a * sq).collect();
    assert_close(&bf16_read(&out[2]), &want, "layer_scalar_mul_bfloat16");

    // The cap is small enough that the inputs above run well past it in both
    // directions, so the tanh is saturated and a missing cap would show. It
    // rides the PUSH block now rather than a storage struct -- four bytes,
    // packed from the routine's `cap: Const<f32>` mark.
    let cap = 3.0f32;
    let out = gpu.dispatch(
        "logit_softcap_bfloat16",
        Capability::Baseline,
        &[bf16_bytes(&x), vec![0u8; n * 2]],
        &cap.to_le_bytes(),
        groups,
    );
    let want: Vec<f32> = xq.iter().map(|a| cap * (a / cap).tanh()).collect();
    assert_close(&bf16_read(&out[1]), &want, "logit_softcap_bfloat16");

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let out = gpu.dispatch(
        "ple_combine_bfloat16",
        Capability::Baseline,
        &[bf16_bytes(&x), bf16_bytes(&y), vec![0u8; n * 2]],
        &inv_sqrt2.to_le_bytes(),
        groups,
    );
    let want: Vec<f32> = xq
        .iter()
        .zip(&yq)
        .map(|(a, b)| (a + b) * inv_sqrt2)
        .collect();
    assert_close(&bf16_read(&out[2]), &want, "ple_combine_bfloat16");
}

/// `vnorm_single_row_bfloat16` — RMSNorm with no weight.
///
/// One workgroup per row and an `N_READS=4` strided loop, so the reduction
/// crosses subgroup boundaries and every lane contributes from four places.
/// The axis below is not a multiple of the workgroup span, which puts the tail
/// guard on both the accumulate and the store; rows are given very different
/// magnitudes so a reduction that leaked across rows would be obvious.
#[test]
fn vnorm_normalizes_each_row_by_its_own_rms() {
    let gpu = gpu!();

    let (rows, axis) = (3usize, 1000usize);
    let eps = 1e-5f32;
    let mut x = Vec::with_capacity(rows * axis);
    for r in 0..rows {
        let magnitude = [0.01f32, 1.0, 40.0][r];
        for i in 0..axis {
            x.push(((i % 47) as f32 - 23.0) * magnitude / 23.0);
        }
    }

    // Both scalars are STATED now -- `eps: Const<f32>`, `axis_size: Const<i32>`
    // -- so they ride an eight-byte push range where they were a `VNormParams`
    // storage struct the routine forwarded whole and could not name.
    let mut push = eps.to_le_bytes().to_vec();
    push.extend_from_slice(&(axis as u32).to_le_bytes());
    let operands = vec![bf16_bytes(&x), vec![0u8; rows * axis * 2]];
    let out = gpu.dispatch(
        "vnorm_single_row_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [rows as u32, 1, 1],
    );

    let xq = bf16_read(&operands[0]);
    let got = bf16_read(&out[1]);
    for r in 0..rows {
        let row = &xq[r * axis..(r + 1) * axis];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / axis as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        let want: Vec<f32> = row.iter().map(|v| v * inv).collect();
        assert_close(
            &got[r * axis..(r + 1) * axis],
            &want,
            &format!("vnorm row {r}"),
        );
    }
}

/// `split_qkv_bf16` — one packed row cut into three.
///
/// The three destinations have DIFFERENT widths and the two boundaries are
/// computed from the same pair of scalars, so reading them swapped, or
/// forgetting to subtract the preceding width, lands inside a neighbour rather
/// than out of bounds. The widths below are unequal and not multiples of the
/// 256-wide workgroup, which also puts the rounded x tail one row away from
/// falling through into the next — the case the shader's own comment names.
///
/// The pair arrives as a PUSH BLOCK. It was a fifth storage operand holding
/// `SplitQkvParams` until the two widths became `Const<u32>` marks; the bytes
/// are the same two words in the same order, so the only change here is which
/// argument of `dispatch` they are handed to.
#[test]
fn split_qkv_cuts_at_both_boundaries() {
    let gpu = gpu!();

    let (q_width, kv_width, rows) = (300usize, 100usize, 4usize);
    let packed_width = q_width + 2 * kv_width;
    let packed: Vec<f32> = (0..rows * packed_width)
        .map(|i| ((i % 61) as f32 - 30.0) / 7.0)
        .collect();

    // `q_width` at word 0 and `kv_width` at word 1, which is the order the
    // marks are declared in and therefore the order `Push` is laid out in.
    let mut widths = (q_width as u32).to_le_bytes().to_vec();
    widths.extend_from_slice(&(kv_width as u32).to_le_bytes());
    let operands = vec![
        bf16_bytes(&packed),
        vec![0u8; rows * q_width * 2],
        vec![0u8; rows * kv_width * 2],
        vec![0u8; rows * kv_width * 2],
    ];
    let out = gpu.dispatch(
        "split_qkv_bf16",
        Capability::Baseline,
        &operands,
        &widths,
        [packed_width.div_ceil(256) as u32, rows as u32, 1],
    );

    let pq = bf16_read(&operands[0]);
    let (q, k, v) = (bf16_read(&out[1]), bf16_read(&out[2]), bf16_read(&out[3]));
    for row in 0..rows {
        let base = row * packed_width;
        for c in 0..q_width {
            assert_eq!(q[row * q_width + c], pq[base + c], "q row {row} col {c}");
        }
        for c in 0..kv_width {
            assert_eq!(
                k[row * kv_width + c],
                pq[base + q_width + c],
                "k row {row} col {c}"
            );
            assert_eq!(
                v[row * kv_width + c],
                pq[base + q_width + kv_width + c],
                "v row {row} col {c}"
            );
        }
    }
}

/// `rms_residual` and `rms_residual_scaled`.
///
/// The residual is added AFTER the gain and the post-scale is applied to the
/// sum, so `(gain * normed + r) * post` and `gain * normed + r * post` differ
/// everywhere the residual is not zero — which is the whole tensor here. Both
/// variants run over the same inputs so the only difference is the extra
/// buffer, and `w_stride` is set to 2 over a doubled weight array to prove the
/// gain walk reads its own stride rather than assuming one.
#[test]
fn rms_residual_adds_after_the_gain_and_scales_after_the_add() {
    let gpu = gpu!();

    let (rows, axis) = (2usize, 600usize);
    let (eps, gain) = (1e-5f32, 1.5f32);
    let w_stride = 2usize;

    let x: Vec<f32> = (0..rows * axis)
        .map(|i| ((i % 53) as f32 - 26.0) / 11.0)
        .collect();
    let r: Vec<f32> = (0..rows * axis)
        .map(|i| ((i % 29) as f32 - 14.0) / 4.0)
        .collect();
    // Every second entry is poison: if the walk ignored `w_stride` it would
    // read these and every output would be wrong.
    let w: Vec<f32> = (0..axis * w_stride)
        .map(|i| {
            if i % w_stride == 0 {
                ((i / w_stride) % 17) as f32 / 40.0 - 0.2
            } else {
                -1000.0
            }
        })
        .collect();

    let mut params = eps.to_le_bytes().to_vec();
    for v in [axis as u32, w_stride as u32, 1u32] {
        params.extend_from_slice(&v.to_le_bytes());
    }
    params.extend_from_slice(&gain.to_le_bytes());

    let post = 0.625f32;
    let mut operands = vec![
        bf16_bytes(&x),
        bf16_bytes(&w),
        vec![0u8; rows * axis * 2],
        params,
        bf16_bytes(&r),
        bf16_bytes(&[post]),
    ];

    let (xq, wq, rq) = (
        bf16_read(&operands[0]),
        bf16_read(&operands[1]),
        bf16_read(&operands[4]),
    );
    let postq = bf16_to_f32(f32_to_bf16(post));

    for (entrypoint, post) in [
        ("rms_residual_bfloat16", 1.0),
        ("rms_residual_scaled_bfloat16", postq),
    ] {
        operands[2] = vec![0u8; rows * axis * 2];
        let out = gpu.dispatch(
            entrypoint,
            Capability::Baseline,
            &operands,
            &[],
            [rows as u32, 1, 1],
        );
        let got = bf16_read(&out[2]);
        for row in 0..rows {
            let slice = &xq[row * axis..(row + 1) * axis];
            let mean_sq: f32 = slice.iter().map(|v| v * v).sum::<f32>() / axis as f32;
            let inv = (mean_sq + eps).sqrt().recip();
            let want: Vec<f32> = (0..axis)
                .map(|i| {
                    // `plus_one` is set, the gemma convention: the stored
                    // weight is the deviation from one.
                    let g = gain * (1.0 + wq[w_stride * i]);
                    (g * (slice[i] * inv) + rq[row * axis + i]) * post
                })
                .collect();
            assert_close(
                &got[row * axis..(row + 1) * axis],
                &want,
                &format!("{entrypoint} row {row}"),
            );
        }
    }
}

/// `router_topk_scaled_bfloat16` — top-k with a per-expert rescale.
///
/// The scale is indexed by EXPERT, not by rank, so a shader that used the loop
/// counter would agree only where the chosen experts happen to come out in
/// order. The logits below are arranged so the top experts are NOT the first
/// ones, and the scale table is monotone, which makes rank-indexing produce a
/// visibly different set of weights.
///
/// `softmax_over_all` is exercised in both settings, since it changes which
/// denominator the weights are divided by and is a plain flag in a struct.
#[test]
fn router_topk_scaled_indexes_its_table_by_expert() {
    let gpu = gpu!();

    let (experts, rows, k) = (8usize, 3usize, 3usize);
    let pitch = experts + 5;

    let mut logits = vec![-30.0f32; rows * pitch];
    // Distinct winners per row, deliberately at high expert indices.
    let winners = [[7usize, 2, 5], [4, 6, 1], [3, 7, 0]];
    for (row, w) in winners.iter().enumerate() {
        for (rank, &e) in w.iter().enumerate() {
            logits[row * pitch + e] = 4.0 - rank as f32;
        }
    }
    let scale: Vec<f32> = (0..experts).map(|e| 0.5 + e as f32 / 8.0).collect();

    for softmax_over_all in [0u32, 1] {
        let mut params = (experts as u32).to_le_bytes().to_vec();
        for v in [k as u32, softmax_over_all, pitch as u32] {
            params.extend_from_slice(&v.to_le_bytes());
        }
        let operands = vec![
            bf16_bytes(&logits),
            vec![0u8; rows * k * 4],
            vec![0u8; rows * k * 2],
            params,
            bf16_bytes(&scale),
        ];
        let out = gpu.dispatch(
            "router_topk_scaled_bfloat16",
            Capability::Baseline,
            &operands,
            &[],
            [1, rows as u32, 1],
        );

        let lq = bf16_read(&operands[0]);
        let sq = bf16_read(&operands[4]);
        let ids: Vec<i32> = out[1]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let weights = bf16_read(&out[2]);

        for (row, expect) in winners.iter().enumerate() {
            let chosen: Vec<usize> = (0..k).map(|r| ids[row * k + r] as usize).collect();
            assert_eq!(
                chosen,
                expect.to_vec(),
                "row {row} picked the wrong experts"
            );

            let all = &lq[row * pitch..row * pitch + experts];
            let (max, denom) = if softmax_over_all == 1 {
                let m = all.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (m, all.iter().map(|v| (v - m).exp()).sum::<f32>())
            } else {
                let picked: Vec<f32> = chosen.iter().map(|&e| all[e]).collect();
                let m = picked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (m, picked.iter().map(|v| (v - m).exp()).sum::<f32>())
            };
            let want: Vec<f32> = chosen
                .iter()
                .map(|&e| (all[e] - max).exp() / denom * sq[e])
                .collect();
            assert_close(
                &weights[row * k..(row + 1) * k],
                &want,
                &format!("row {row} softmax_over_all={softmax_over_all}"),
            );
        }
    }
}

/// The three RoPE variants that are not the plain decode.
///
/// `neox_mb` is the same rotation over MANY rows, each with its own position,
/// so a body that read `position[0]` for all of them — which is what the
/// decode variant legitimately does — would be wrong from the second row on.
///
/// `neox_freqs_decode` replaces the `exp2(-d * base)` geometric series with a
/// table lookup and multiplies the result by `mscale`, so its push block drops
/// `base` and gains a float in a different slot. The table below is NOT
/// geometric, which is what stops it agreeing with the default body by
/// accident.
///
/// `neox_prop_decode` is the subtle one: its `d = 2i/head_dim` and its pair
/// partner `head_dim/2` are IDENTICAL to the default body whenever the grid
/// covers exactly half the head. Only a partial rotary — fewer pairs than
/// `head_dim/2` — separates them, so that is what this dispatches.
#[test]
fn the_rope_variants_differ_where_they_are_supposed_to() {
    let gpu = gpu!();

    let (head_dim, heads) = (16usize, 2usize);
    let (scale, base) = (1.0f32, 4.0f32);
    let pair_half = head_dim / 2;

    let rotate = |x: &mut [f32], i1: usize, i2: usize, theta: f32, out_scale: f32| {
        let (c, s) = (theta.cos(), theta.sin());
        let (x1, x2) = (x[i1], x[i2]);
        x[i1] = out_scale * (x1 * c - x2 * s);
        x[i2] = out_scale * (x1 * s + x2 * c);
    };

    // `neox_mb`: three rows, three positions.
    let rows = 3usize;
    let positions: Vec<i32> = vec![0, 5, 11];
    let x: Vec<f32> = (0..rows * heads * head_dim)
        .map(|i| ((i % 37) as f32 - 18.0) / 9.0)
        .collect();
    let mut operands = vec![
        bf16_bytes(&x),
        positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
    ];
    let mut push = scale.to_le_bytes().to_vec();
    push.extend_from_slice(&base.to_le_bytes());
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    let out = gpu.dispatch(
        "neox_mb_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [pair_half as u32, heads as u32, rows as u32],
    );

    let mut want = bf16_read(&operands[0]);
    for row in 0..rows {
        for h in 0..heads {
            for i in 0..pair_half {
                let i1 = (row * heads + h) * head_dim + i;
                let theta =
                    scale * positions[row] as f32 * (-(i as f32 / pair_half as f32) * base).exp2();
                rotate(&mut want, i1, i1 + pair_half, theta, 1.0);
            }
        }
    }
    assert_close(&bf16_read(&out[0]), &want, "neox_mb_bfloat16");

    // `neox_freqs_decode`: one row, a non-geometric table, and an mscale.
    let inv_freq: Vec<f32> = (0..pair_half)
        .map(|i| 1.0 / (1.0 + 3.0 * i as f32 + (i % 3) as f32))
        .collect();
    let mscale = 1.25f32;
    let x1: Vec<f32> = (0..heads * head_dim)
        .map(|i| ((i % 23) as f32 - 11.0) / 5.0)
        .collect();
    operands = vec![
        bf16_bytes(&x1),
        7i32.to_le_bytes().to_vec(),
        inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect(),
    ];
    let mut push = scale.to_le_bytes().to_vec();
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    push.extend_from_slice(&mscale.to_le_bytes());
    let out = gpu.dispatch(
        "neox_freqs_decode_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [pair_half as u32, heads as u32, 1],
    );

    let mut want = bf16_read(&operands[0]);
    for h in 0..heads {
        for i in 0..pair_half {
            let i1 = h * head_dim + i;
            rotate(
                &mut want,
                i1,
                i1 + pair_half,
                scale * 7.0 * inv_freq[i],
                mscale,
            );
        }
    }
    assert_close(&bf16_read(&out[0]), &want, "neox_freqs_decode_bfloat16");

    // The same table over many rows. `_decode` pins the row to zero, so this
    // is the variant where the position array is actually indexed, and running
    // it against the same frequencies proves the table lookup and the row walk
    // are independent of each other.
    let xm: Vec<f32> = (0..rows * heads * head_dim)
        .map(|i| ((i % 43) as f32 - 21.0) / 7.0)
        .collect();
    operands = vec![
        bf16_bytes(&xm),
        positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
        inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect(),
    ];
    let out = gpu.dispatch(
        "neox_freqs_mb_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [pair_half as u32, heads as u32, rows as u32],
    );

    let mut want = bf16_read(&operands[0]);
    for row in 0..rows {
        for h in 0..heads {
            for i in 0..pair_half {
                let i1 = (row * heads + h) * head_dim + i;
                let theta = scale * positions[row] as f32 * inv_freq[i];
                rotate(&mut want, i1, i1 + pair_half, theta, mscale);
            }
        }
    }
    assert_close(&bf16_read(&out[0]), &want, "neox_freqs_mb_bfloat16");

    // `neox_prop_decode`: a PARTIAL rotary, which is the only shape where the
    // proportional body and the default body disagree.
    let pairs = 3usize;
    assert!(pairs < pair_half, "a full rotary would not tell them apart");
    operands = vec![bf16_bytes(&x1), 7i32.to_le_bytes().to_vec()];
    let mut push = scale.to_le_bytes().to_vec();
    push.extend_from_slice(&base.to_le_bytes());
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    let out = gpu.dispatch(
        "neox_prop_decode_bfloat16",
        Capability::Baseline,
        &operands,
        &push,
        [pairs as u32, heads as u32, 1],
    );

    let mut want = bf16_read(&operands[0]);
    let mut as_default = want.clone();
    for h in 0..heads {
        for i in 0..pairs {
            let i1 = h * head_dim + i;
            let theta = scale * 7.0 * (-(2.0 * i as f32 / head_dim as f32) * base).exp2();
            rotate(&mut want, i1, i1 + pair_half, theta, 1.0);
            let theta = scale * 7.0 * (-(i as f32 / pairs as f32) * base).exp2();
            rotate(&mut as_default, i1, i1 + pairs, theta, 1.0);
        }
    }
    assert_close(&bf16_read(&out[0]), &want, "neox_prop_decode_bfloat16");
    assert!(
        as_default
            .iter()
            .zip(&want)
            .any(|(a, b)| (a - b).abs() > BF16_TOLERANCE),
        "the partial rotary did not separate prop from the default body"
    );
}

/// The two quantized matmuls that fold a residual, and the `_mb` / `_scaled`
/// embedding gathers — the last of the stated families.
///
/// The residual variants add their extra buffer AFTER the push block in the
/// row, which is the arrangement the binding audit exists for: `residual` is
/// operand six but descriptor FIVE, because the two scalars in between move to
/// push constants. Testing them is what shows the compaction was applied to
/// the shader and not only to the table.
#[test]
fn the_residual_matmuls_add_after_the_product() {
    let gpu = gpu!();

    let (group, bits) = (128usize, 4usize);
    let k = 256usize;
    // Ragged against the 8 rows a workgroup covers, so the epilogue's bound is
    // on the path here too -- and the residual it folds is indexed by row.
    let rows = 13usize;
    let (w, scales, biases, dense) = affine_weights(rows, k, group, bits);
    let x: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) / 12.0).collect();
    // Large enough that dropping it, or adding it before the reduction rather
    // than after, is far outside the tolerance.
    let residual: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32 - 3.0) * 2.0).collect();

    let operands = vec![
        w.clone(),
        scales.clone(),
        biases.clone(),
        bf16_bytes(&x),
        vec![0u8; rows * 2],
        bf16_bytes(&residual),
    ];
    let mut push = (k as i32).to_le_bytes().to_vec();
    push.extend_from_slice(&(rows as i32).to_le_bytes());

    let out = gpu.dispatch(
        "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
        Capability::Baseline,
        &operands,
        &push,
        [1, rows.div_ceil(8) as u32, 1],
    );

    let xq = bf16_read(&operands[3]);
    let rq = bf16_read(&operands[5]);
    let want: Vec<f32> = (0..rows)
        .map(|r| (0..k).map(|i| xq[i] * dense[r * k + i]).sum::<f32>() + rq[r])
        .collect();
    assert_close(&bf16_read(&out[4]), &want, "affine_qmv_fast_residual");

    // The matmul form, one query row, so the same reference serves.
    let (bm, bn) = (16usize, 16usize);
    let operands = vec![
        w,
        scales,
        biases,
        bf16_bytes(&x),
        vec![0u8; rows * 2],
        bf16_bytes(&residual),
    ];
    let out = gpu.dispatch(
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_16",
        Capability::Baseline,
        &operands,
        &push,
        [rows.div_ceil(bn) as u32, 1usize.div_ceil(bm) as u32, 1],
    );
    assert_close(&bf16_read(&out[4]), &want, "affine_qmm_t_residual");
}

/// `embed_gather_mb_4bit` and `embed_gather_scaled_4bit` — the other two
/// corners of the gather's two-by-two.
///
/// The family is `{single, mb} x {plain, scaled}` and the earlier test ran the
/// two diagonal corners, which leaves open the possibility that the multi-row
/// grid and the scale field are wired to each other. Running the other
/// diagonal closes it: `_mb` here has no scale field at all, and `_scaled`
/// here is single-row, so neither can be borrowing the other's behaviour.
#[test]
fn the_other_two_embed_gather_corners() {
    let gpu = gpu!();

    let (group, bits) = (64usize, 4usize);
    let (vocab, hidden) = (12usize, 128usize);
    let (w, scales, biases, dense) = affine_weights(vocab, hidden, group, bits);

    let ids: Vec<i32> = vec![4, 11, 1, 8];
    let mb = vec![
        w.clone(),
        scales.clone(),
        biases.clone(),
        ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
        vec![0u8; ids.len() * hidden * 2],
    ];
    let got = gpu.dispatch(
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &mb,
        &(hidden as i32).to_le_bytes(),
        [hidden.div_ceil(16) as u32, ids.len().div_ceil(16) as u32, 1],
    );
    let mut want = Vec::with_capacity(ids.len() * hidden);
    for &id in &ids {
        for c in 0..hidden {
            want.push(dense[id as usize * hidden + c]);
        }
    }
    assert_close(&bf16_read(&got[4]), &want, "embed_gather_mb");

    let embed_scale = 0.375f32;
    let single = vec![
        w,
        scales,
        biases,
        3i32.to_le_bytes().to_vec(),
        vec![0u8; hidden * 2],
    ];
    let mut push = (hidden as i32).to_le_bytes().to_vec();
    push.extend_from_slice(&embed_scale.to_le_bytes());
    let got = gpu.dispatch(
        "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
        Capability::Baseline,
        &single,
        &push,
        [hidden.div_ceil(256) as u32, 1, 1],
    );
    let want: Vec<f32> = (0..hidden)
        .map(|c| dense[3 * hidden + c] * embed_scale)
        .collect();
    assert_close(&bf16_read(&got[4]), &want, "embed_gather_scaled single");
}

/// `sdpa_paged_mma` — the prefill attention, both tiers.
///
/// This is the shader the wiki called the least trustworthy thing in the
/// crate. Its `@coopmat` body carried the same fp32-operand defect that a GPU
/// caught in `affine_qmm_t`, and it went unverified because its row is
/// UNSTATED. So this test does what `rms_strided_row` does and takes its
/// layout from the shader: it is a test of the BODY, not of the ABI. That is
/// worth doing here precisely because the body is the part with a known
/// history of being wrong.
///
/// Three things are checked at once. The scalar tier against a plain softmax
/// reference. The coopmat tier against the SAME reference, not merely against
/// the scalar tier — a shared defect would agree with itself. And the mask,
/// which is the one binding pair the decode test leaves switched off: one row
/// has `attention_mask_enabled` set and a mask that drops keys the causal
/// bound would otherwise keep, so a body that ignored the mask buffer returns
/// a different answer rather than the same one.
///
/// The rows are a real PREFILL: two requests, several tokens each, positions
/// increasing within a request. The decode tests all use one token per
/// request, which is the case where `req_of_token` and the row index are the
/// same number and therefore interchangeable.
#[test]
fn sdpa_paged_mma_agrees_with_a_softmax_reference_on_both_tiers() {
    let gpu = gpu!();

    let head_dim = 64usize;
    let page_size = 8usize;
    let n_kv_heads = 2usize;
    let gqa = 2usize;
    let n_q_heads = n_kv_heads * gqa;
    let scale = 0.125f32;

    // Two requests of different length; a token's position is its index
    // within its own request, which is what makes this prefill rather than a
    // batch of decodes.
    let req_of_token: Vec<i32> = vec![0, 0, 0, 0, 0, 1, 1, 1];
    let positions: Vec<i32> = vec![0, 1, 2, 3, 4, 0, 1, 2];
    let n_rows = req_of_token.len();
    let lengths = [5usize, 3];

    let pages_per: Vec<usize> = lengths.iter().map(|l| l.div_ceil(page_size)).collect();
    let total_pages: usize = pages_per.iter().sum();
    // Reversed, so identity addressing is caught.
    let physical: Vec<u32> = (0..total_pages as u32).rev().collect();
    let mut indptr = vec![0u32];
    for p in &pages_per {
        indptr.push(indptr.last().unwrap() + *p as u32);
    }

    let slots = total_pages * page_size;
    let kv_elems = slots * n_kv_heads * head_dim;
    let kf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 31) as f32 - 15.0) / 40.0)
        .collect();
    let vf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 23) as f32 - 11.0) / 30.0)
        .collect();
    let qf: Vec<f32> = (0..n_rows * n_q_heads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) / 20.0)
        .collect();

    // Row 3 masks out its own key 1. It has q_pos 3, so the causal bound keeps
    // keys 0..=3 and the mask must remove exactly one of them.
    let mask_stride = 8usize;
    let masked_row = 3usize;
    let dropped_key = 1usize;
    let mut mask = vec![1u8; n_rows * mask_stride];
    mask[masked_row * mask_stride + dropped_key] = 0;
    let mut mask_enabled = vec![0u8; n_rows];
    mask_enabled[masked_row] = 1;

    let sinks: Vec<f32> = (0..n_q_heads).map(|h| -0.25 + h as f32 * 0.5).collect();

    let operands = vec![
        bf16_bytes(&qf),
        bf16_bytes(&kf),
        bf16_bytes(&vf),
        vec![0u8; n_rows * n_q_heads * head_dim * 2],
        positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
        req_of_token.iter().flat_map(|r| r.to_le_bytes()).collect(),
        physical.iter().flat_map(|p| p.to_le_bytes()).collect(),
        indptr.iter().flat_map(|p| p.to_le_bytes()).collect(),
        mask,
        mask_enabled,
        bf16_bytes(&sinks),
    ];

    let mut push = Vec::new();
    for v in [gqa as i32, page_size as i32, n_kv_heads as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
    push.extend_from_slice(&(mask_stride as u32).to_le_bytes());
    push.extend_from_slice(&0i32.to_le_bytes()); // window: off
    push.extend_from_slice(&(n_rows as i32).to_le_bytes());

    let q = bf16_read(&operands[0]);
    let k = bf16_read(&operands[1]);
    let v = bf16_read(&operands[2]);
    let sq = bf16_read(&operands[10]);
    let slot_of = |req: usize, kp: usize| {
        let phys = physical[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };

    let reference = |with_sink: bool| -> Vec<f32> {
        let mut want = vec![0.0f32; n_rows * n_q_heads * head_dim];
        for row in 0..n_rows {
            let req = req_of_token[row] as usize;
            let q_pos = positions[row] as usize;
            let keys: Vec<usize> = (0..=q_pos)
                .filter(|kp| !(row == masked_row && *kp == dropped_key))
                .collect();
            for h in 0..n_q_heads {
                let kv_head = h / gqa;
                let q_base = (row * n_q_heads + h) * head_dim;
                let scores: Vec<f32> = keys
                    .iter()
                    .map(|&kp| {
                        let kb = (slot_of(req, kp) * n_kv_heads + kv_head) * head_dim;
                        (0..head_dim)
                            .map(|d| scale * q[q_base + d] * k[kb + d])
                            .sum::<f32>()
                    })
                    .collect();
                let mut hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if with_sink {
                    hi = hi.max(sq[h]);
                }
                let exps: Vec<f32> = scores.iter().map(|s| (s - hi).exp()).collect();
                let mut denom: f32 = exps.iter().sum();
                if with_sink {
                    denom += (sq[h] - hi).exp();
                }
                for d in 0..head_dim {
                    let acc: f32 = keys
                        .iter()
                        .zip(&exps)
                        .map(|(&kp, e)| {
                            e * v[(slot_of(req, kp) * n_kv_heads + kv_head) * head_dim + d]
                        })
                        .sum();
                    want[q_base + d] = acc / denom;
                }
            }
        }
        want
    };

    let groups = [n_q_heads as u32, n_rows.div_ceil(32) as u32, 1];
    let mut tiers = vec![Capability::Baseline];
    if gpu.tiers.contains(&Capability::Coopmat) {
        tiers.push(Capability::Coopmat);
    } else {
        eprintln!("SKIP coopmat tier: {} does not offer it", gpu.name);
    }

    for tier in tiers {
        for (entrypoint, with_sink) in [
            ("sdpa_paged_mma_bfloat16_d_64", false),
            ("sdpa_paged_mma_sink_bfloat16_d_64", true),
        ] {
            let out = gpu.dispatch(entrypoint, tier, &operands, &push, groups);
            assert_close(
                &bf16_read(&out[3]),
                &reference(with_sink),
                &format!("{entrypoint} @{}", tier.tag()),
            );
        }
    }
}

/// `sdpa_paged_mma` past one K tile and past one row block.
///
/// # What its sibling could not see
///
/// The test above is the shape check -- paging, GQA, masks, sinks -- and its
/// longest request is five tokens. `PIE_KT` is sixteen, so the tile loop it
/// runs executes exactly ONCE, and its eight rows fit in the first block of
/// thirty-two. Both loops that carry state ACROSS iterations were therefore
/// covered only in the case where there is no second iteration to carry it
/// to.
///
/// That is the whole risk surface of an online softmax. The running maximum
/// and the running denominator exist precisely to let tile `n + 1` correct
/// what tile `n` already accumulated, and a body that rescales wrongly, or
/// re-initialises per tile, or reduces a stale score, agrees with any
/// reference whose sequences are shorter than a tile.
///
/// So this one runs a forty-token request -- three tiles, with a ragged last
/// one -- beside a short request that still has to be addressed through its
/// own page list, and forty-three rows so that `gl_WorkGroupID.y` reaches its
/// second block and the `row_lo + rr` arithmetic is exercised rather than
/// assumed. The masked row is chosen at key 20 of row 37: both past the first
/// tile, so the mask has to survive being applied in a later iteration, and
/// in the second row block.
#[test]
fn sdpa_paged_mma_carries_its_softmax_across_tiles_and_row_blocks() {
    let gpu = gpu!();

    let head_dim = 64usize;
    let page_size = 8usize;
    let n_kv_heads = 2usize;
    let gqa = 2usize;
    let n_q_heads = n_kv_heads * gqa;
    let scale = 0.125f32;

    let lengths = [40usize, 3];
    let mut req_of_token: Vec<i32> = Vec::new();
    let mut positions: Vec<i32> = Vec::new();
    for (r, l) in lengths.iter().enumerate() {
        for p in 0..*l {
            req_of_token.push(r as i32);
            positions.push(p as i32);
        }
    }
    let n_rows = req_of_token.len();

    let pages_per: Vec<usize> = lengths.iter().map(|l| l.div_ceil(page_size)).collect();
    let total_pages: usize = pages_per.iter().sum();
    let physical: Vec<u32> = (0..total_pages as u32).rev().collect();
    let mut indptr = vec![0u32];
    for p in &pages_per {
        indptr.push(indptr.last().unwrap() + *p as u32);
    }

    let slots = total_pages * page_size;
    let kv_elems = slots * n_kv_heads * head_dim;
    let kf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 31) as f32 - 15.0) / 40.0)
        .collect();
    let vf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 23) as f32 - 11.0) / 30.0)
        .collect();
    let qf: Vec<f32> = (0..n_rows * n_q_heads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) / 20.0)
        .collect();

    let mask_stride = 48usize;
    let masked_row = 37usize;
    let dropped_key = 20usize;
    let mut mask = vec![1u8; n_rows * mask_stride];
    mask[masked_row * mask_stride + dropped_key] = 0;
    let mut mask_enabled = vec![0u8; n_rows];
    mask_enabled[masked_row] = 1;

    let sinks: Vec<f32> = (0..n_q_heads).map(|h| -0.25 + h as f32 * 0.5).collect();

    let operands = vec![
        bf16_bytes(&qf),
        bf16_bytes(&kf),
        bf16_bytes(&vf),
        vec![0u8; n_rows * n_q_heads * head_dim * 2],
        positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
        req_of_token.iter().flat_map(|r| r.to_le_bytes()).collect(),
        physical.iter().flat_map(|p| p.to_le_bytes()).collect(),
        indptr.iter().flat_map(|p| p.to_le_bytes()).collect(),
        mask,
        mask_enabled,
        bf16_bytes(&sinks),
    ];

    let mut push = Vec::new();
    for v in [gqa as i32, page_size as i32, n_kv_heads as i32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
    push.extend_from_slice(&(mask_stride as u32).to_le_bytes());
    push.extend_from_slice(&0i32.to_le_bytes()); // window: off
    push.extend_from_slice(&(n_rows as i32).to_le_bytes());

    let q = bf16_read(&operands[0]);
    let k = bf16_read(&operands[1]);
    let v = bf16_read(&operands[2]);
    let sq = bf16_read(&operands[10]);
    let slot_of = |req: usize, kp: usize| {
        let phys = physical[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };

    let reference = |with_sink: bool| -> Vec<f32> {
        let mut want = vec![0.0f32; n_rows * n_q_heads * head_dim];
        for row in 0..n_rows {
            let req = req_of_token[row] as usize;
            let q_pos = positions[row] as usize;
            let keys: Vec<usize> = (0..=q_pos)
                .filter(|kp| !(row == masked_row && *kp == dropped_key))
                .collect();
            for h in 0..n_q_heads {
                let kv_head = h / gqa;
                let q_base = (row * n_q_heads + h) * head_dim;
                let scores: Vec<f32> = keys
                    .iter()
                    .map(|&kp| {
                        let kb = (slot_of(req, kp) * n_kv_heads + kv_head) * head_dim;
                        (0..head_dim)
                            .map(|d| scale * q[q_base + d] * k[kb + d])
                            .sum::<f32>()
                    })
                    .collect();
                let mut hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if with_sink {
                    hi = hi.max(sq[h]);
                }
                let exps: Vec<f32> = scores.iter().map(|s| (s - hi).exp()).collect();
                let mut denom: f32 = exps.iter().sum();
                if with_sink {
                    denom += (sq[h] - hi).exp();
                }
                for d in 0..head_dim {
                    let acc: f32 = keys
                        .iter()
                        .zip(&exps)
                        .map(|(&kp, e)| {
                            e * v[(slot_of(req, kp) * n_kv_heads + kv_head) * head_dim + d]
                        })
                        .sum();
                    want[q_base + d] = acc / denom;
                }
            }
        }
        want
    };

    assert!(
        n_rows > 32 && lengths[0] > 16,
        "this test is only worth more than its sibling if it reaches a second \
         row block and a second K tile"
    );
    let groups = [n_q_heads as u32, n_rows.div_ceil(32) as u32, 1];
    let mut tiers = vec![Capability::Baseline];
    if gpu.tiers.contains(&Capability::Coopmat) {
        tiers.push(Capability::Coopmat);
    }
    for tier in tiers {
        for (entrypoint, with_sink) in [
            ("sdpa_paged_mma_bfloat16_d_64", false),
            ("sdpa_paged_mma_sink_bfloat16_d_64", true),
        ] {
            let out = gpu.dispatch(entrypoint, tier, &operands, &push, groups);
            assert_close(
                &bf16_read(&out[3]),
                &reference(with_sink),
                &format!("{entrypoint} @{}", tier.tag()),
            );
        }
    }
}

/// `affine_qmm_t` across its whole axis grid, on every tier the device offers.
///
/// This is the crate's largest kernel — 54 stated entrypoints — and until now
/// exactly one tile shape had produced a number, at one quantization point,
/// on the coopmat tier. That is the thinnest possible evidence for the shape
/// that matters most, because the `@coopmat` body reduces with a FIXED
/// `16x16x16` cooperative matrix and tiles at `bm x bn` around it: a tile that
/// does not divide cleanly is exactly where the fixed shape and the variable
/// one come apart, and it comes apart quietly, since a coopmat that reads
/// outside its tile still returns numbers.
///
/// So this sweeps `{32, 64, 128} x {4, 8} x {16, 32, 64}^2` and holds every
/// combination to the same dense reference. `M` and `N` are deliberately NOT
/// multiples of the tile, so every shape has a ragged edge to guard, and the
/// reference is built once from the codes rather than per shape — the answer
/// cannot depend on how the work was divided.
#[test]
fn affine_qmm_t_is_right_at_every_tile_shape_and_quantization_point() {
    let gpu = gpu!();

    let (m, n) = (33usize, 47usize); // prime-ish: every tiling has a tail
    let mut checked = 0usize;

    for group in [32usize, 64, 128] {
        for bits in [4usize, 8] {
            // K has to be a whole number of groups and of packing words.
            let k = group * 3;
            let (w, scales, biases, dense) = affine_weights(n, k, group, bits);
            let xf: Vec<f32> = (0..m * k)
                .map(|i| ((i % 29) as f32 - 14.0) / 24.0)
                .collect();
            let x = bf16_bytes(&xf);
            let xq = bf16_read(&x);

            let want: Vec<f32> = (0..m)
                .flat_map(|r| {
                    let (xq, dense) = (&xq, &dense);
                    (0..n).map(move |c| {
                        (0..k)
                            .map(|i| xq[r * k + i] * dense[c * k + i])
                            .sum::<f32>()
                    })
                })
                .collect();

            for bm in [16usize, 32, 64] {
                for bn in [16usize, 32, 64] {
                    let entrypoint =
                        format!("affine_qmm_t_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}");
                    for tier in [Capability::Baseline, Capability::Coopmat] {
                        if tier == Capability::Coopmat && !gpu.tiers.contains(&Capability::Coopmat)
                        {
                            continue;
                        }
                        // The coopmat tier is not instantiated for every tile
                        // shape; a missing module is a fact about the table,
                        // not a failure, so it is skipped rather than asserted.
                        let path = std::path::Path::new(SPV_DIR.expect("checked by gpu!()"))
                            .join(tier.module(&entrypoint));
                        if !path.exists() {
                            continue;
                        }
                        let got = qmm_t(
                            gpu,
                            tier,
                            &entrypoint,
                            m,
                            n,
                            k,
                            bm,
                            bn,
                            &w,
                            &scales,
                            &biases,
                            &x,
                        );
                        assert_close(&got, &want, &format!("{entrypoint} @{}", tier.tag()));
                        checked += 1;
                    }
                }
            }
        }
    }

    // A guard against the sweep quietly skipping everything: 3 groups x 2 bit
    // widths x 9 tile shapes is 54 baseline modules, and the coopmat tier adds
    // however many of them it was instantiated for.
    assert!(checked >= 54, "the sweep only ran {checked} modules");
    eprintln!("affine_qmm_t: {checked} modules checked");
}

/// The pointwise kernels at a width that is NOT a multiple of the workgroup.
///
/// Every other test in this file used a round width, which cannot see the one
/// thing that changes when these bodies are ported. That finding then spread:
/// several of those widths are ragged now (`n = 460` for the pointwise
/// activations, 13 and 5 output rows for the quantized GEMVs), and a round
/// shape is worth treating as a defect in a test of a ported kernel.
/// Metal launches them with
/// `dispatchThreads`, an EXACT thread count, so there is no tail to guard and
/// `GegluParams::unused` is genuinely unused. `vkCmdDispatch` takes
/// WORKGROUPS, so the extent is `256 * ceil(n / 256)` and a ragged width
/// leaves up to 255 threads past the end of the tensor — storing, not merely
/// reading.
///
/// The bound those bodies use is the output's own `length()`, which is the
/// descriptor range. That makes the guard EXACT whenever the buffer is the
/// tensor, which is what this test arranges: `n = 300` elements allocated as
/// 300, dispatched as two full groups of 256.
///
/// What this proves is that the ragged width COMPUTES correctly — every width
/// in this file used to be a multiple of 256, so the last partial group had
/// never run at all, and a body that over-trimmed would leave the tail of the
/// real data untouched. What it cannot prove is the store staying inside the
/// descriptor, and the reason is worth stating: this harness enables
/// `robustBufferAccess` (the tiled GEMM needs it), which makes an out-of-range
/// store defined and DISCARDED. Deleting the guard therefore still passes
/// here. The guard is not thereby pointless — it is what makes these bodies
/// correct on a driver that does not enable robustness, where the same store
/// is undefined — but this test is not the thing that holds it in place.
#[test]
fn the_pointwise_kernels_do_not_write_past_a_ragged_width() {
    let gpu = gpu!();

    let n = 300usize; // 256 < 300 < 512: one full group and a 44-element tail
    let groups = [n.div_ceil(256) as u32, 1, 1];

    let a: Vec<f32> = (0..n).map(|i| ((i % 53) as f32 - 26.0) / 5.0).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) / 3.0).collect();
    let (aq, bq) = (bf16_read(&bf16_bytes(&a)), bf16_read(&bf16_bytes(&b)));

    let (limit, alpha) = (7.0f32, 1.702f32);
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let gelu = |x: f32| {
        let k = 0.797_884_5f32;
        0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
    };

    let cases: Vec<(&str, Vec<u8>, Vec<f32>)> = vec![
        (
            "silu_mul_bfloat16",
            Vec::new(),
            aq.iter().zip(&bq).map(|(g, u)| silu(*g) * u).collect(),
        ),
        (
            "geglu_tanh_bfloat16",
            vec![0u8; 4],
            aq.iter().zip(&bq).map(|(g, u)| gelu(*g) * u).collect(),
        ),
        (
            "gptoss_swiglu_bfloat16",
            {
                let mut p = vec![0u8; 4];
                p.extend_from_slice(&limit.to_le_bytes());
                p.extend_from_slice(&alpha.to_le_bytes());
                p
            },
            aq.iter()
                .zip(&bq)
                .map(|(g, u)| {
                    let g = g.min(limit);
                    let u = u.clamp(-limit, limit);
                    (g / (1.0 + (-alpha * g).exp())) * (u + 1.0)
                })
                .collect(),
        ),
    ];

    for (entrypoint, params, want) in cases {
        let mut operands = vec![bf16_bytes(&a), bf16_bytes(&b), vec![0u8; n * 2]];
        if !params.is_empty() {
            operands.push(params);
        }
        let out = gpu.dispatch(entrypoint, Capability::Baseline, &operands, &[], groups);
        assert_close(&bf16_read(&out[2]), &want, entrypoint);
    }
}

/// `gated_rms_bfloat16` — `w * rmsnorm(x) * silu(z)`, at an axis wider than the
/// workgroup.
///
/// `V_d` is 128 on every GDN checkpoint the tree has seen, and the port read
/// one channel per lane under a fixed `[numthreads(256, 1, 1)]` because Metal
/// launches `tg = (V_d, 1, 1)` and genuinely has one. That is fine at 128 and
/// silently wrong above 256: channels past the workgroup were left out of the
/// sum of squares AND never written, so the mean was taken over a prefix and
/// the tail of the head kept whatever the buffer was born with.
///
/// So the case that matters here is `v_d = 300` — wider than the workgroup and
/// not a multiple of it, which also exercises the ragged last stride. The
/// 128-wide case runs beside it to show the narrow direction still holds; it
/// alone would pass against the broken shader.
#[test]
fn gated_rms_normalizes_an_axis_wider_than_its_workgroup() {
    let gpu = gpu!();

    for (heads, v_d) in [(2usize, 128usize), (3, 300), (2, 512)] {
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..heads * v_d)
            .map(|i| ((i % 53) as f32 - 26.0) / 11.0)
            .collect();
        let z: Vec<f32> = (0..heads * v_d)
            .map(|i| ((i % 37) as f32 - 18.0) / 9.0)
            .collect();
        let w: Vec<f32> = (0..v_d).map(|i| 0.5 + (i % 13) as f32 / 20.0).collect();

        // `eps` and `vd` are marks now, so the pair is an eight-byte push
        // range rather than a `GatedRmsParams` storage struct at binding 4.
        let mut push = eps.to_le_bytes().to_vec();
        push.extend_from_slice(&(v_d as u32).to_le_bytes());
        let operands = vec![
            bf16_bytes(&x),
            bf16_bytes(&z),
            bf16_bytes(&w),
            vec![0u8; heads * v_d * 2],
        ];
        // `gated_rms`'s grid is `(1, V_h, 1)` with the row base built from
        // `gl_WorkGroupID.z * gl_NumWorkGroups.y + gl_WorkGroupID.y`.
        let out = gpu.dispatch(
            "gated_rms_bfloat16",
            Capability::Baseline,
            &operands,
            &push,
            [1, heads as u32, 1],
        );

        let xq = bf16_read(&operands[0]);
        let zq = bf16_read(&operands[1]);
        let wq = bf16_read(&operands[2]);
        let got = bf16_read(&out[3]);
        for h in 0..heads {
            let row = &xq[h * v_d..(h + 1) * v_d];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / v_d as f32;
            let inv = (mean_sq + eps).sqrt().recip();
            let want: Vec<f32> = (0..v_d)
                .map(|i| {
                    let zr = zq[h * v_d + i];
                    (row[i] * inv * wq[i]) * (zr / (1.0 + (-zr).exp()))
                })
                .collect();
            assert_close(
                &got[h * v_d..(h + 1) * v_d],
                &want,
                &format!("gated_rms v_d={v_d} head {h}"),
            );
        }
    }
}

/// Every compiled module builds a PIPELINE on this device, under the
/// descriptor layout and push range the table implies for it.
///
/// The 37 tests above prove behaviour, one kernel at a time, for the kernels
/// somebody wrote a reference for. This proves the much weaker thing about ALL
/// of them -- that a shell which reads the table, builds the layout the table
/// describes and hands the driver the `.spv` gets a pipeline back rather than
/// an error. That is the first thing a shell does with every module it will
/// ever run, and until now nothing did it for more than a handful.
///
/// It is not redundant with the checks that already exist, because each of
/// those stops short of the driver:
///
/// * `--compile` proves `slangc` accepts the Slang. A module can compile and
///   still be unloadable -- it may want a capability this device lacks, or
///   exceed a limit `slangc` knows nothing about.
/// * `--bindings` read `OpDecorate Binding` out of the SPIR-V and compared it
///   to the row. That was a static comparison between two descriptions; it
///   could not tell whether the DRIVER agrees that a layout with that many
///   descriptors, and a push range of that size, is one it will accept. It is
///   also gone -- the mode read the row's half by running
///   `examples/dump_layout.rs`, which is deleted -- so the static half of this
///   list is now unheld and what follows is the only half left.
/// * `check_push_against_the_row` compares a hand-packed push block to the
///   row's derived layout, and only for the dispatches a test performs.
///
/// The tier loop matters as much as the module loop: a tier is a separate
/// body, compiled with different extensions, so `@coopmat` failing to load on
/// a device that reports `cooperativeMatrix` is exactly the kind of thing that
/// is invisible until a specific GPU meets a specific model. Tiers the device
/// does not claim are skipped rather than failed -- that is the backward
/// compatibility guarantee working, not a gap.
///
/// **What it can and cannot report.** With no validation layer installed --
/// and there is none on this box -- a driver answers a genuinely malformed
/// request by crashing or hanging rather than returning an error. Both were
/// observed while writing this: an unstated row's empty layout segfaults
/// inside `vkCreateComputePipelines`, and a truncated `.spv` hangs. So the
/// `Err` arm below catches the polite failures (a capability the device lacks,
/// a limit exceeded) and the impolite ones take the test process down instead.
/// That is still a signal, and a loud one; it is just not a tidy assertion.
/// Installing `VK_LAYER_KHRONOS_validation` would turn the second class into
/// the first, and is the obvious next thing to do to this harness.
#[test]
fn every_module_this_device_claims_it_can_load_builds_a_pipeline() {
    let gpu = gpu!();
    let dir = std::path::Path::new(SPV_DIR.expect("checked by gpu!()"));

    let mut built = 0usize;
    let mut skipped = 0usize;
    let mut unstated = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for name in kernels_vulkan::entrypoints() {
        // A RETIRED entrypoint has no row, and takes the same from-module
        // path an unstated one does -- which is the honest one for it: its
        // layout is now stated by a routine's signature, and reflecting the
        // module is exactly what the driver does for it at launch time.
        // `kernels::sig_in(KERNELS, &name)` STOOD HERE, and every entrypoint
        // took the `None` branch: `KERNELS` is empty, so `buffers` was 0 and
        // `push` was 0 for all 481 and the `unstated` fallback below ran every
        // time. The lookup is deleted rather than left to answer `None`
        // forever, and the branch it fed is now unconditional -- which is the
        // path a crossed entrypoint takes in production too, where a driver
        // reflects the module and takes the maximum of that and what the
        // routine states.
        // `retired().contains(&name)` STOOD HERE, asserting that a name with no
        // row was one of the names whose row had gone. Both sides came to be
        // read off the same shader tree -- every family has crossed, so "the
        // retired entrypoints" and "the entrypoints" were one set under two
        // names -- and `retired()` is deleted with the hand-written list it
        // flattened. The assert could only pass, and it did so for all 496.
        let (buffers, push) = (0u32, 0usize);

        // An UNSTATED row cannot supply the layout, and the first version of
        // this test skipped those 292 entrypoints. Finding out why cost a
        // SIGSEGV: `buffer_count` answers 0 for them, honestly -- the row
        // really does describe nothing -- but the shader behind the name still
        // declares its bindings, and handing the driver a module that reads
        // `binding = 1` under a layout with no descriptors is not an error it
        // reports. It is a segmentation fault inside
        // `vkCreateComputePipelines`, taking the test process with it.
        //
        // Skipping them left the majority of the table untested, and that was
        // the wrong conclusion. An unstated row is not unlaunchable: it is
        // launchable from somewhere OTHER than the row.
        // `driver-metal/src/lowering/dispatch.rs` shows where -- `if
        // sig.operands.is_empty()` falls back to the lowered plan's own
        // argument order, so the row's operand list is a reordering and
        // verification layer over the plan rather than the only description of
        // it. A Vulkan shell has the same fallback available; it needs a
        // descriptor COUNT at layout time, and the plan has one.
        //
        // So this test does what such a shell would do, from the only source
        // available offline: the module's own `OpDecorate Binding` set. That is
        // weaker evidence than a stated row -- it checks the module against
        // itself, not against the table -- but it is the difference between
        // 292 entrypoints proven to load and 292 never tried. Everything a
        // pipeline creation checks (capabilities against enabled features, the
        // workgroup against `maxComputeWorkGroupSize`, the push block against
        // the range) applies to them exactly as it does to the rest.
        let (buffers, push, _from_module) = if buffers == 0 && push == 0 {
            unstated += 1;
            (
                declared_binding_count(&spv_words(&dir.join(Capability::Baseline.module(&name)))),
                // The device's maximum, which any block fits inside. The
                // row-derived size is the interesting check and a row that
                // says nothing cannot make it; `--bindings` used to own that
                // comparison for the rows that can, and nothing owns it now.
                gpu.max_push as usize,
                true,
            )
        } else {
            (buffers, push, false)
        };

        for tier in Capability::PREFERENCE {
            let path = dir.join(tier.module(&name));
            // A tier is OPTIONAL per entrypoint -- only some rows have one --
            // so a missing file is the table saying this variant does not
            // exist, and a missing baseline is a different failure that
            // `every_tier_has_a_baseline_beneath_it` owns.
            if !path.exists() {
                continue;
            }
            if !gpu.tiers.contains(&tier) {
                skipped += 1;
                continue;
            }
            match gpu.build_pipeline(&path, buffers, push) {
                Ok(()) => built += 1,
                Err(e) => failures.push(format!("  {name} @{tier:?}: {e}")),
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} modules will not build a pipeline on this device under the \
         layout their row describes:\n{}",
        failures.len(),
        built + failures.len(),
        failures.join("\n")
    );
    // A loop that silently found nothing would pass, so say what it did. Every
    // entrypoint has a baseline module and every device loads that tier, so
    // the floor is now the whole table -- the unstated ones included, since
    // they are built against their own declared bindings rather than skipped.
    let all = kernels_vulkan::entrypoints().len();
    assert!(
        built >= all,
        "only {built} pipelines built for {all} entrypoints; the module \
         directory must be incomplete"
    );
    eprintln!(
        "{built} pipelines built ({unstated} of them against the module's own \
         declared bindings, their row stating nothing), {skipped} skipped as \
         unsupported tiers"
    );
}

/// What the three page-shape tails of `sdpa_paged_decode` actually are.
///
/// The row's axis carries seven points: four plain widths and three tails,
/// `_d_64_p32`, `_d_128_p32` and `_d_64_p32_sg8`. `kernels-metal` states the
/// same seven, and on THAT backend all three diverge from the plain points in
/// ways a caller has to know -- `FAST_FULL` deletes the window and mask
/// operands, and `_sg8` is a genuinely different threadgroup (`BN = 8`, 256
/// threads) whose shared arrays are sized for eight simdgroups, so launching
/// it at the row's 1024 walks off them.
///
/// A reader who checks the Metal comment and then looks at the Vulkan point of
/// the same name will conclude this tree has the same three shapes. It has
/// two-and-a-bit, and the difference is measured here rather than asserted:
///
/// * `_p32` really does differ from the plain point. `PIE_FAST_FULL` pins
///   `start = 0`, so a module built for a tail serves FULL attention and
///   ignores `window` -- the bytes differ, and that is the check.
/// * `_p32_sg8` is BYTE-IDENTICAL to `_p32`, the name included: the variant
///   name lives in the `.spv` FILENAME and every module's entrypoint is
///   spelled `main`, so the two are one file written twice.
///   `PIE_SHORT_GROUP=8` is set by the instantiate line and read by nothing,
///   so on Vulkan this point is a name and not a shape. It is allow-listed in
///   `scripts/vulkan-kernel-audit.py`'s `INERT_DEFINES`, which is how it stays
///   allowed rather than merely unnoticed.
///
/// None of the three is reachable. A text names
/// `sdpa_paged_decode_bfloat16_d_<width>` from its row's head dim and
/// `deployment::ATTN_HEAD_DIMS` is the four plain widths, so no width spells a
/// tail; `driver-vulkan`'s reachability census lists no `_p32` symbol either.
/// That is what makes the inert one survivable, and it is also why this is a
/// module comparison and not a dispatch: there is nothing to fire.
///
/// The reason to write it down at all is that the two halves fail in opposite
/// directions. If someone wires a tail on Vulkan believing the Metal
/// description, they get full attention where they asked for a window and the
/// same 64-wide workgroup where they asked for a short group -- silently, on
/// both counts.
#[test]
fn the_page_shape_tails_are_one_real_variant_and_one_bare_name() {
    let Some(dir) = SPV_DIR.map(std::path::Path::new) else {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    };
    let read = |name: &str| {
        std::fs::read(dir.join(format!("{name}.spv")))
            .unwrap_or_else(|e| panic!("`{name}.spv` reads: {e}"))
    };

    let plain = read("sdpa_paged_decode_bfloat16_d_64");
    let tail = read("sdpa_paged_decode_bfloat16_d_64_p32");
    let short = read("sdpa_paged_decode_bfloat16_d_64_p32_sg8");

    assert_ne!(
        plain, tail,
        "`_p32` compiles to the same module as the plain point, so \
         `PIE_FAST_FULL` and `PIE_PAGE_SIZE` are both inert and the tail is \
         claiming a shape it does not have"
    );
    assert_eq!(
        tail, short,
        "`_p32_sg8` differs from `_p32`, so `PIE_SHORT_GROUP` has grown a \
         body -- rewrite this test and take the pair out of the audit's \
         `INERT_DEFINES`"
    );
}

/// The nine routed `_fp16` modules are copies of their bf16 siblings.
///
/// `moe/qmm_t_routed.slang` takes `PIE_FP16` on nine of its instantiate lines
/// and reads it on none. The name comes from `quant/qmm_t.slang`, which really
/// does have an fp16 activation path -- a separate pre-cast buffer at binding
/// 7, and a `load_x` that reads it instead of `x` at 3. The routed file never
/// grew one, so the define has been carried along by the name alone.
///
/// It was allow-listed in the audit's `INERT_DEFINES` with that reasoning
/// written out, which is an argument and not a measurement: nine table rows
/// and nine modules of build time ride on it, and the argument would go on
/// reading true for exactly as long as it took someone to add the body and
/// forget the comment. Compiled bytes settle it. The pairs are IDENTICAL, the
/// entrypoint included -- every module's entry is spelled `main` and the
/// variant lives in the filename, so an inert define makes one file twice.
///
/// Harmless only because `affine_qmm_t_routed_fp16`'s row is UNSTATED:
/// `geometry::lanes` refuses an unstated rule before it computes a grid, so
/// the row cannot be dispatched at all. If it is ever stated, the body has to
/// exist FIRST -- otherwise a driver hands fp16 activations to a shader
/// reading bf16 and the name is the only thing that says otherwise. This test
/// failing is that day arriving, and the right response is to check the new
/// body is really the fp16 path rather than to delete the assertion.
#[test]
fn the_routed_fp16_modules_are_their_bf16_siblings_under_another_name() {
    let Some(dir) = SPV_DIR.map(std::path::Path::new) else {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    };

    let mut pairs = 0usize;
    for entry in std::fs::read_dir(dir).expect("the module dir reads") {
        let name = entry
            .expect("a dir entry reads")
            .file_name()
            .to_string_lossy()
            .into_owned();
        if !name.starts_with("affine_qmm_t_routed_fp16_") || !name.ends_with(".spv") {
            continue;
        }
        let sibling = name.replace("_fp16", "");
        let sibling_path = dir.join(&sibling);
        assert!(
            sibling_path.exists(),
            "`{name}` has no bf16 sibling `{sibling}`, so the pairing this \
             test rests on has changed shape"
        );
        assert_eq!(
            std::fs::read(dir.join(&name)).expect("the fp16 module reads"),
            std::fs::read(&sibling_path).expect("the bf16 module reads"),
            "`{name}` now differs from `{sibling}`: `PIE_FP16` has a body in \
             `moe/qmm_t_routed.slang`. Check it is the pre-cast path \
             `quant/qmm_t.slang` has -- a separate activation buffer at \
             binding 7 -- then state the row and drop this test"
        );
        pairs += 1;
    }
    assert_eq!(
        pairs, 9,
        "the routed fp16 axis is {pairs} points, not the nine the audit's \
         `INERT_DEFINES` entry and this test both describe"
    );
}

/// The GDN fixture: shapes chosen so the parts that can be wrong are exercised.
///
/// `Hk = 2, Hv = 4` makes `rep = 2`, which is the only setting under which
/// indexing q and k by the VALUE head differs from indexing by the key head.
/// Every checkpoint anyone runs locally has `rep == 1`, where the two
/// expressions are the same and a wrong one is invisible; `kernels-metal`'s
/// header records that this family shipped with exactly that bug. This tree
/// has the fixed form (`hk_idx = hv_idx / rep`), and the fixture is what makes
/// the fix load-bearing rather than incidental.
///
/// `Dk = 64` gives `n_per_t = 2`, so every per-lane loop runs twice -- at
/// `Dk = 32` the loops execute once and a body that ignored `i` would pass.
/// `Dv = 8` is two groups of the shader's four y-threads, so `dv_idx` has to
/// come from `SV_DispatchThreadID` and not `SV_GroupThreadID`.
struct GdnShape {
    b: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    kc: usize,
}

impl GdnShape {
    const FIXTURE: Self = Self {
        b: 2,
        hk: 2,
        hv: 4,
        dk: 64,
        dv: 8,
        kc: 4,
    };

    fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
    fn q_off(&self) -> usize {
        0
    }
    fn k_off(&self) -> usize {
        self.hk * self.dk
    }
    fn v_off(&self) -> usize {
        2 * self.hk * self.dk
    }
    /// The 44-byte block `PARAM_BLOCKS` records at binding 11.
    fn params(&self, eps: f32) -> Vec<u8> {
        let ints = [
            self.dk as i32,
            self.dv as i32,
            self.hk as i32,
            self.hv as i32,
            self.conv_dim() as i32,
            self.kc as i32,
            self.q_off() as i32,
            self.k_off() as i32,
            self.v_off() as i32,
        ];
        let mut out: Vec<u8> = ints.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.extend(eps.to_le_bytes());
        out.extend((1.0f32 / (self.dk as f32).sqrt()).to_le_bytes());
        out
    }
}

/// Deterministic values in a range where bf16 rounding is the dominant error.
fn gdn_spread(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503));
            let u = ((h >> 8) & 0xffff) as f32 / 65535.0;
            (u - 0.5) * 2.0 * scale
        })
        .collect()
}

/// The buffers a GDN core dispatch reads, in binding order 0..=10.
struct GdnInputs {
    mixed: Vec<f32>,
    conv_state: Vec<f32>,
    rstate: Vec<f32>,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    a_gate: Vec<f32>,
    b_gate: Vec<f32>,
}

impl GdnInputs {
    /// `slots` is how many recurrent slots the state slab holds; the unslotted
    /// form uses `b_idx` directly, so it needs at least `b` of them.
    fn build(s: &GdnShape, slots: usize) -> Self {
        let cd = s.conv_dim();
        Self {
            mixed: gdn_spread(s.b * cd, 1, 1.0),
            conv_state: gdn_spread(slots * s.kc * cd, 2, 1.0),
            rstate: gdn_spread(slots * s.hv * s.dv * s.dk, 3, 0.5),
            conv_w: gdn_spread(cd * s.kc, 4, 0.5),
            conv_b: gdn_spread(cd, 5, 0.25),
            // `exp(A_log)` is the decay's inner exponent, so keeping it small
            // and negative keeps `decay` inside (0, 1) the way a trained
            // model's does rather than saturating it to zero.
            a_log: gdn_spread(s.hv, 6, 1.0).iter().map(|v| v - 1.0).collect(),
            dt_bias: gdn_spread(s.hv, 7, 0.5),
            a_gate: gdn_spread(s.b * s.hv, 8, 1.0),
            b_gate: gdn_spread(s.b * s.hv, 9, 1.0),
        }
    }

    /// Binding order, with `mixed` and the gates rounded through bf16 first so
    /// the reference reads the same numbers the shader does. Only the ROUNDING
    /// is shared; every arithmetic step below is written out independently.
    fn operands(&self, s: &GdnShape, eps: f32, slots: usize) -> Vec<Vec<u8>> {
        let cd = s.conv_dim();
        vec![
            bf16_bytes(&self.mixed),
            self.conv_state
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            self.rstate.iter().flat_map(|v| v.to_le_bytes()).collect(),
            vec![0u8; s.b * s.hv * s.dv * 2],
            bf16_bytes(&self.conv_w),
            bf16_bytes(&self.conv_b),
            self.a_log.iter().flat_map(|v| v.to_le_bytes()).collect(),
            bf16_bytes(&self.dt_bias),
            bf16_bytes(&self.a_gate),
            bf16_bytes(&self.b_gate),
            vec![0u8; slots * s.kc * cd * 4],
            s.params(eps),
        ]
    }
}

/// What the fused core is supposed to compute, written from the algorithm.
///
/// Returns `(core_out, rstate_after, new_conv_state)`. Deliberately a plain
/// sequential walk in f32: the shader reduces across 32 lanes through shared
/// memory and this sums in order, so agreement is a statement about the
/// arithmetic and not about the reduction tree.
fn gdn_reference(
    s: &GdnShape,
    inp: &GdnInputs,
    eps: f32,
    slot_of: &dyn Fn(usize) -> usize,
    slots: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let cd = s.conv_dim();
    let rep = s.hv / s.hk;

    // The shader reads these through `PIE_LOAD`, so the reference must see the
    // bf16-rounded values and not the f32 ones it was handed.
    let r = |v: &[f32]| bf16_read(&bf16_bytes(v));
    let (mixed, conv_w, conv_b) = (r(&inp.mixed), r(&inp.conv_w), r(&inp.conv_b));
    let (dt_bias, a_gate, b_gate) = (r(&inp.dt_bias), r(&inp.a_gate), r(&inp.b_gate));

    let convsilu = |slot: usize, b: usize, c: usize| -> f32 {
        let mut acc = conv_b[c];
        for j in 0..s.kc - 1 {
            acc += inp.conv_state[(slot * s.kc + (j + 1)) * cd + c] * conv_w[c * s.kc + j];
        }
        acc += mixed[b * cd + c] * conv_w[c * s.kc + (s.kc - 1)];
        acc / (1.0 + (-acc).exp())
    };

    let mut core_out = vec![0.0f32; s.b * s.hv * s.dv];
    let mut rstate = inp.rstate.clone();
    let mut new_conv = vec![0.0f32; slots * s.kc * cd];

    let roll = |slot: usize, b: usize, c: usize, new_conv: &mut Vec<f32>| {
        for j in 0..s.kc - 1 {
            new_conv[(slot * s.kc + j) * cd + c] = inp.conv_state[(slot * s.kc + (j + 1)) * cd + c];
        }
        new_conv[(slot * s.kc + (s.kc - 1)) * cd + c] = mixed[b * cd + c];
    };

    for n in 0..s.b * s.hv {
        let (b, hv) = (n / s.hv, n % s.hv);
        let hk = hv / rep;
        let slot = slot_of(b);

        let q: Vec<f32> = (0..s.dk)
            .map(|d| convsilu(slot, b, s.q_off() + hk * s.dk + d))
            .collect();
        let k: Vec<f32> = (0..s.dk)
            .map(|d| convsilu(slot, b, s.k_off() + hk * s.dk + d))
            .collect();
        let qinv =
            (1.0 / (s.dk as f32).sqrt()) / (q.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        let kinv = 1.0 / (k.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();

        let ad = a_gate[b * s.hv + hv] + dt_bias[hv];
        let sp = ad.max(0.0) + (1.0 + (-ad.abs()).exp()).ln();
        let decay = (-inp.a_log[hv].exp() * sp).exp();
        let beta = 1.0 / (1.0 + (-b_gate[b * s.hv + hv]).exp());

        for dv in 0..s.dv {
            let vval = convsilu(slot, b, s.v_off() + hv * s.dv + dv);
            let base = ((slot * s.hv + hv) * s.dv + dv) * s.dk;
            let mut st: Vec<f32> = (0..s.dk).map(|d| inp.rstate[base + d] * decay).collect();
            let kv: f32 = (0..s.dk).map(|d| st[d] * (k[d] * kinv)).sum();
            let delta = (vval - kv) * beta;
            let mut outv = 0.0f32;
            for d in 0..s.dk {
                st[d] += (k[d] * kinv) * delta;
                outv += st[d] * (q[d] * qinv);
                rstate[base + d] = st[d];
            }
            core_out[(b * s.hv + hv) * s.dv + dv] = outv;
        }

        // q and k roll once per KEY head -- the shader guards it with
        // `hk_first`, so a second v-head of the same group must not repeat it.
        if hv % rep == 0 {
            for d in 0..s.dk {
                roll(slot, b, s.q_off() + hk * s.dk + d, &mut new_conv);
                roll(slot, b, s.k_off() + hk * s.dk + d, &mut new_conv);
            }
        }
        for dv in 0..s.dv {
            roll(slot, b, s.v_off() + hv * s.dv + dv, &mut new_conv);
        }
    }
    (core_out, rstate, new_conv)
}

/// Read the f32 slab a GDN operand comes back as.
fn f32_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// What the output slab may be off by, as a fraction of the slab's own scale.
///
/// MEASURED, not reasoned. This device delivers 2.9e-3 over the fixture, and
/// the bound is a shade over twice that. `BF16_TOLERANCE` is 0.02, which is
/// seven times the truth here -- see `GDN_STATE_TOL` for what lived in the gap.
const GDN_OUT_TOL: f32 = 6e-3;

/// What the recurrent state may be off by, as a fraction of its own scale.
///
/// Four orders tighter than the output's, and the reason is structural rather
/// than lucky: `core_out` is stored through bf16 while `rstate` is f32 in and
/// f32 out, so the state's only error is the order of a 64-term reduction.
/// This device delivers 1.1e-7 -- f32 epsilon -- against the 2.9e-3 the output
/// carries.
///
/// Checking it at `BF16_TOLERANCE` was therefore not slightly loose but
/// 170,000 times loose, and that is not an abstract complaint. Scaling the
/// reference's `decay` by 1.01 PASSED: a one-percent error in the central gate
/// of the whole kernel, invisible. `kernels-metal` records the same injection
/// getting through in the same place. This bound fails it by four orders.
const GDN_STATE_TOL: f32 = 1e-6;

/// Check a slab against its bound and return how much of the bound it used.
///
/// Returning the ratio is the point. A pass says the kernel is within the
/// bound; the ratio says whether the bound still describes the DEVICE, and a
/// bound that has drifted far above what the hardware does is the thing that
/// silently stops testing.
fn gdn_check(got: &[f32], want: &[f32], tol: f32, what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want
        .iter()
        .fold(0.0f32, |m, w| m.max(w.abs()))
        .max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let off = (g - w).abs() / scale;
        worst = worst.max(off);
        assert!(
            off <= tol,
            "{what}: element {i} is {g}, reference says {w} -- off by {off} of \
             the slab scale {scale}, and the bound is {tol}"
        );
    }
    worst
}

/// The measured ratio has to sit in a band, not merely under the bound.
///
/// Too high and the test is one element from flaking; too LOW and the bound
/// has stopped describing the device. `kernels-metal` had a guard here whose
/// stated job was this and which asserted that a perturbation of twice the
/// bound exceeds the bound -- `2b > b`, true of every bound ever written, and
/// read as assurance for as long as it stood.
fn gdn_band(worst: f32, tol: f32, what: &str) {
    let used = worst / tol;
    assert!(
        (0.0625..=1.0).contains(&used),
        "{what} used {used} of its {tol} bound (worst {worst}). Outside a \
         sixteenth-to-one band the bound no longer describes this device: \
         re-measure it rather than widening it"
    );
}

/// `gdn_core_bfloat16` computes the gated delta rule, on the GPU.
///
/// Sixteen GDN entry points are compiled into every native build of this crate
/// and none had ever been asked for a number. `ssm.rs` records why they cannot
/// be reached -- no `Source` for the recurrent state, and a tracer that emits
/// three ops where this is one fused dispatch -- and that is a statement about
/// LOWERING. It says nothing about whether the four hundred lines of
/// arithmetic underneath are right, and the lowering work would land on top of
/// them. So this builds the dispatch by hand, the way the paged-attention
/// proofs do, and reachability stops being a prerequisite for correctness.
///
/// Every term here is a place a port can be quietly wrong in a way that does
/// not crash: a softplus that overflows at large `|a|` (the shader spells the
/// stable `max(a,0) + log1p(exp(-|a|))` form and the reference spells it
/// again), an `eps` inside or outside the square root, a decay applied after
/// the delta instead of before, or a `1/sqrt(Dk)` folded into `k` rather than
/// `q`. The reference is a sequential f32 walk while the shader reduces
/// through shared memory across 32 lanes, so agreement is a claim about the
/// arithmetic rather than about the reduction order.
///
/// The three outputs are all checked. `core_out` alone would miss the whole
/// recurrent half: `rstate` is written IN PLACE, and a kernel that returned
/// the right value while leaving the state at its pre-decay contents would
/// pass a one-slab test and diverge on the second token. `new_conv_state` is
/// the shift-and-append, whose `hk_first` guard means the q and k channels are
/// rolled once per KEY head -- a body that rolled them per value head would
/// write the same bytes twice here and be wrong only in what it cost.
/// # What was injected
///
/// This test passed the first time it was run, which is not a thing to be
/// pleased about, so six faults were put through it and two got past:
///
/// * q and k indexed by the VALUE head -- caught.
/// * `decay` scaled by 1.01 -- PASSED, and the reason is the whole of
///   `GDN_STATE_TOL`'s note.
/// * `decay` applied after the delta instead of before -- caught.
/// * `1/sqrt(Dk)` folded into k rather than q -- caught.
/// * `eps` moved outside the square root -- PASSED, and the fix is the loud
///   pass at the end.
/// * the conv roll done per value head instead of per key head -- passed, and
///   correctly. `roll` writes a function of read-only inputs, so doing it
///   twice writes the same bytes; the `hk_first` guard saves work and cannot
///   change an answer. That one is not a fault this test should catch, and
///   noting it is how it stays out of the list of things believed tested.
#[test]
fn gdn_core_computes_the_gated_delta_rule() {
    let gpu = gpu!();
    let s = GdnShape::FIXTURE;
    let eps = 1e-6f32;
    let inp = GdnInputs::build(&s, s.b);

    let operands = inp.operands(&s, eps, s.b);
    let out = gpu.dispatch(
        "gdn_core_bfloat16",
        Capability::Baseline,
        &operands,
        &[],
        [1, (s.dv / 4) as u32, (s.b * s.hv) as u32],
    );

    let (want_out, want_state, want_conv) = gdn_reference(&s, &inp, eps, &|b| b, s.b);

    let out_used = gdn_check(
        &bf16_read(&out[3]),
        &want_out,
        GDN_OUT_TOL,
        "gdn_core core_out",
    );
    let state_used = gdn_check(
        &f32_read(&out[2]),
        &want_state,
        GDN_STATE_TOL,
        "gdn_core rstate",
    );
    // The conv roll is a copy, not arithmetic: every element is either an f32
    // moved unchanged or a bf16-rounded `mixed`, so it is exact and an
    // approximate check here would hide an off-by-one in the shift.
    assert_eq!(f32_read(&out[10]), want_conv, "gdn_core new_conv_state");

    // Both slabs, because the two are four orders of magnitude apart and one
    // number cannot describe both.
    gdn_band(out_used, GDN_OUT_TOL, "gdn_core core_out");
    gdn_band(state_used, GDN_STATE_TOL, "gdn_core rstate");

    // Once more at an eps large enough to observe, which the realistic value
    // is not. `eps = 1e-6` sits under a q norm of about four, so moving it
    // outside the square root -- `1/(sqrt(s) + eps)` instead of
    // `1/sqrt(s + eps)` -- changes the answer by about 1e-7 and this test
    // passed the injection. That is a fixture limit and not a kernel
    // property: the two forms differ by a whole percent when eps is
    // comparable to the sum, and the only reason a model never sees it is
    // that a trained norm is never near zero. Firing the same fixture at
    // `eps = 4.0` puts the placement back inside what the test can see, and
    // it is still ordinary arithmetic -- the reference spells the same
    // formula, so agreement pins WHERE the term goes.
    let loud = 4.0f32;
    let out = gpu.dispatch(
        "gdn_core_bfloat16",
        Capability::Baseline,
        &inp.operands(&s, loud, s.b),
        &[],
        [1, (s.dv / 4) as u32, (s.b * s.hv) as u32],
    );
    let (want_out, want_state, _) = gdn_reference(&s, &inp, loud, &|b| b, s.b);
    gdn_check(
        &bf16_read(&out[3]),
        &want_out,
        GDN_OUT_TOL,
        "gdn_core core_out at a loud eps",
    );
    gdn_check(
        &f32_read(&out[2]),
        &want_state,
        GDN_STATE_TOL,
        "gdn_core rstate at a loud eps",
    );
}

/// `gdn_core_slotted_bfloat16` is the same kernel, and its slot map is real.
///
/// The two entry points are one template at `PIE_SLOTTED = 0` and `1`, and the
/// file's claim is that at identity slots they are the same kernel. This tree
/// has been bitten by believing that shape of claim before -- `sdpa_paged_mma`
/// had three bodies answering one softmax that did not share one contract --
/// so it is fired rather than read.
///
/// At identity slots the two must agree to the LAST BIT, not to a tolerance:
/// they execute the same instructions over the same bytes, and any difference
/// at all means the indirection changed something it had no business
/// changing. Then the same map is PERMUTED over a permuted state slab, where
/// the answer must come back unchanged. That second half is the only check
/// that `slot_ids` is an indirection rather than a second spelling of
/// `b_idx`: under identity every wrong implementation that ignores the map
/// still passes, because ignoring it and following it are the same thing.
#[test]
fn the_slotted_gdn_core_follows_its_slot_map() {
    let gpu = gpu!();
    let s = GdnShape::FIXTURE;
    let eps = 1e-6f32;
    let inp = GdnInputs::build(&s, s.b);
    let groups = [1u32, (s.dv / 4) as u32, (s.b * s.hv) as u32];

    let plain = gpu.dispatch(
        "gdn_core_bfloat16",
        Capability::Baseline,
        &inp.operands(&s, eps, s.b),
        &[],
        groups,
    );

    let mut identity = inp.operands(&s, eps, s.b);
    identity.push((0..s.b as u32).flat_map(|i| i.to_le_bytes()).collect());
    let same = gpu.dispatch(
        "gdn_core_slotted_bfloat16",
        Capability::Baseline,
        &identity,
        &[],
        groups,
    );
    for (i, what) in [(3usize, "core_out"), (2, "rstate"), (10, "new_conv_state")] {
        assert_eq!(
            plain[i], same[i],
            "at identity slots the slotted form's {what} differs from the \
             unslotted form's. They are one template at `PIE_SLOTTED`, so a \
             difference here is the indirection changing the arithmetic"
        );
    }

    // Now the same problem wearing a permutation: slot `perm[b]` holds what
    // slot `b` held, and the map says so. Every slab the kernel indexes BY
    // SLOT moves -- `conv_state`, `rstate` and `new_conv_state` -- while
    // `mixed` and the gates stay indexed by `b_idx` and do not.
    let perm: Vec<usize> = (0..s.b).map(|b| (b + 1) % s.b).collect();
    let cd = s.conv_dim();
    let mut moved = GdnInputs::build(&s, s.b);
    for b in 0..s.b {
        let (from, to) = (b, perm[b]);
        for j in 0..s.kc {
            for c in 0..cd {
                moved.conv_state[(to * s.kc + j) * cd + c] =
                    inp.conv_state[(from * s.kc + j) * cd + c];
            }
        }
        let width = s.hv * s.dv * s.dk;
        moved.rstate[to * width..(to + 1) * width]
            .copy_from_slice(&inp.rstate[from * width..(from + 1) * width]);
    }
    let mut permuted = moved.operands(&s, eps, s.b);
    permuted.push(
        perm.iter()
            .flat_map(|p| (*p as u32).to_le_bytes())
            .collect(),
    );
    let after = gpu.dispatch(
        "gdn_core_slotted_bfloat16",
        Capability::Baseline,
        &permuted,
        &[],
        groups,
    );

    assert_eq!(
        after[3], plain[3],
        "moving every slot one place and saying so changed the OUTPUT, so \
         `slot_ids` is not being followed the whole way through"
    );
    // The state and the conv history come back permuted, since they are the
    // slabs the map addresses -- unpermuting them is what says the kernel
    // wrote through the map rather than past it.
    let (got_state, got_conv) = (f32_read(&after[2]), f32_read(&after[10]));
    let (want_state, want_conv) = (f32_read(&plain[2]), f32_read(&plain[10]));
    let width = s.hv * s.dv * s.dk;
    for b in 0..s.b {
        assert_eq!(
            got_state[perm[b] * width..(perm[b] + 1) * width],
            want_state[b * width..(b + 1) * width],
            "slot {} of the permuted rstate is not what slot {b} of the plain \
             run holds",
            perm[b]
        );
        for j in 0..s.kc {
            let (g, w) = ((perm[b] * s.kc + j) * cd, (b * s.kc + j) * cd);
            assert_eq!(
                got_conv[g..g + cd],
                want_conv[w..w + cd],
                "tap {j} of slot {} of the permuted conv history is not what \
                 slot {b} of the plain run holds",
                perm[b]
            );
        }
    }
}

impl GdnInputs {
    /// `gdn_prep`'s binding order, which is not the fused kernel's.
    ///
    /// The split pair rearranges the whole set: the prep drops `rstate` and
    /// `core_out` entirely and gains the three f32 scratch slabs, so a test
    /// that reused the fused operand vector would bind `conv_w` where the
    /// shader reads `rstate` and still run.
    fn prep_operands(&self, s: &GdnShape, eps: f32, slots: usize) -> Vec<Vec<u8>> {
        let cd = s.conv_dim();
        let n = s.b * s.hv;
        vec![
            bf16_bytes(&self.mixed),
            self.conv_state
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            bf16_bytes(&self.conv_w),
            bf16_bytes(&self.conv_b),
            self.a_log.iter().flat_map(|v| v.to_le_bytes()).collect(),
            bf16_bytes(&self.dt_bias),
            bf16_bytes(&self.a_gate),
            bf16_bytes(&self.b_gate),
            vec![0u8; n * s.dk * 4],
            vec![0u8; n * s.dk * 4],
            // The DECODE prep's gate slab is `2 * Hv` per row and nothing
            // more. `kernels-metal`'s header claimed `2*Hv + Hv*Dv` here for
            // as long as the file existed; that is the PREFILL layout, where
            // the scan really does read V back at `+ 2*Hv`. The decode
            // recurrent recomputes its own `vval`, because a v channel is
            // unique per `dv` and there is no redundancy to remove.
            vec![0u8; 2 * n * 4],
            vec![0u8; slots * s.kc * cd * 4],
            s.params(eps),
        ]
    }
}

/// The split GDN pair is the fused kernel, to the bit.
///
/// `gdn_prep` and `gdn_core_recurrent` exist to kill redundant q/k work:
/// every value channel of a head recomputes the same convolution, the same
/// pair of L2 norms and the same gates, so the split stages them once in f32
/// scratch and the recurrent half reads them back. The file's claim is that
/// this is the same arithmetic, and it has never been run.
///
/// `assert_eq!` and not a tolerance, deliberately. The failure a split like
/// this actually has is a reduction that associates differently -- prep
/// reduces under `[numthreads(32,1,1)]` and the fused kernel under
/// `[numthreads(32,4,1)]` -- and that lands a few ulps out, which is inside
/// any bound wide enough for a bf16 store. A tolerance would not have tested
/// the claim it was written to test.
///
/// The two halves also split the convolution writeback between them: prep
/// rolls the q and k channels under `hk_first` and the recurrent rolls the v
/// channels, so `new_conv_state` is only whole if BOTH ran and neither wrote
/// the other's. It is threaded from the first dispatch into the second rather
/// than allocated twice, which is what makes that checkable -- and each
/// dispatch here is its own submit, so the ordering edge is real rather than
/// a barrier someone remembered to encode.
#[test]
fn the_split_gdn_pair_is_the_fused_kernel_to_the_bit() {
    let gpu = gpu!();
    let s = GdnShape::FIXTURE;
    let eps = 1e-6f32;
    let inp = GdnInputs::build(&s, s.b);
    let n = s.b * s.hv;

    let fused = gpu.dispatch(
        "gdn_core_bfloat16",
        Capability::Baseline,
        &inp.operands(&s, eps, s.b),
        &[],
        [1, (s.dv / 4) as u32, n as u32],
    );

    let prepped = gpu.dispatch(
        "gdn_prep_bfloat16",
        Capability::Baseline,
        &inp.prep_operands(&s, eps, s.b),
        &[],
        [1, 1, n as u32],
    );

    // The recurrent half, reading the scratch the prep just wrote and
    // CONTINUING its convolution history rather than starting a fresh one.
    let recurrent = gpu.dispatch(
        "gdn_core_recurrent_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&inp.mixed),
            inp.conv_state
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            inp.rstate.iter().flat_map(|v| v.to_le_bytes()).collect(),
            vec![0u8; n * s.dv * 2],
            bf16_bytes(&inp.conv_w),
            bf16_bytes(&inp.conv_b),
            prepped[8].clone(),
            prepped[9].clone(),
            prepped[10].clone(),
            prepped[11].clone(),
            s.params(eps),
        ],
        &[],
        [1, (s.dv / 4) as u32, n as u32],
    );

    assert_eq!(
        recurrent[3], fused[3],
        "the split pair's core_out is not the fused kernel's, bit for bit"
    );
    assert_eq!(
        recurrent[2], fused[2],
        "the split pair's rstate is not the fused kernel's, bit for bit"
    );
    assert_eq!(
        recurrent[9], fused[10],
        "the split pair's new_conv_state is not the fused kernel's. The q and \
         k channels come from the prep and the v channels from the recurrent, \
         so a mismatch is one half writing the other's or neither writing some"
    );

    // The prep's own scratch, checked against the reference rather than only
    // against the fused kernel -- otherwise two halves that agreed with each
    // other and with nothing else would pass. `pre_q` carries the
    // `1/sqrt(Dk)` prescale and `pre_k` does not, which is the one asymmetry
    // in the pair and the one a port most easily mirrors.
    let (pre_q, pre_k) = (f32_read(&prepped[8]), f32_read(&prepped[9]));
    let gate = f32_read(&prepped[10]);
    let r = |v: &[f32]| bf16_read(&bf16_bytes(v));
    let (mixed, conv_w, conv_b) = (r(&inp.mixed), r(&inp.conv_w), r(&inp.conv_b));
    let cd = s.conv_dim();
    let rep = s.hv / s.hk;
    let convsilu = |b: usize, c: usize| -> f32 {
        let mut acc = conv_b[c];
        for j in 0..s.kc - 1 {
            acc += inp.conv_state[(b * s.kc + (j + 1)) * cd + c] * conv_w[c * s.kc + j];
        }
        acc += mixed[b * cd + c] * conv_w[c * s.kc + (s.kc - 1)];
        acc / (1.0 + (-acc).exp())
    };
    let mut want_q = Vec::new();
    let mut want_k = Vec::new();
    for i in 0..n {
        let (b, hv) = (i / s.hv, i % s.hv);
        let hk = hv / rep;
        let q: Vec<f32> = (0..s.dk)
            .map(|d| convsilu(b, s.q_off() + hk * s.dk + d))
            .collect();
        let k: Vec<f32> = (0..s.dk)
            .map(|d| convsilu(b, s.k_off() + hk * s.dk + d))
            .collect();
        let qinv =
            (1.0 / (s.dk as f32).sqrt()) / (q.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        let kinv = 1.0 / (k.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        want_q.extend(q.iter().map(|v| v * qinv));
        want_k.extend(k.iter().map(|v| v * kinv));
    }
    gdn_check(&pre_q, &want_q, GDN_STATE_TOL, "gdn_prep pre_q");
    gdn_check(&pre_k, &want_k, GDN_STATE_TOL, "gdn_prep pre_k");

    // And the gate slab is two floats a head, in that order. A prep that
    // wrote beta first would still produce a `decay` and a `beta` of the
    // right values and hand them to the recurrent the wrong way round.
    let (dt_bias, a_gate, b_gate) = (r(&inp.dt_bias), r(&inp.a_gate), r(&inp.b_gate));
    assert_eq!(
        gate.len(),
        2 * n,
        "the decode gate slab is two floats a head"
    );
    for i in 0..n {
        let (b, hv) = (i / s.hv, i % s.hv);
        let ad = a_gate[b * s.hv + hv] + dt_bias[hv];
        let sp = ad.max(0.0) + (1.0 + (-ad.abs()).exp()).ln();
        let decay = (-inp.a_log[hv].exp() * sp).exp();
        let beta = 1.0 / (1.0 + (-b_gate[b * s.hv + hv]).exp());
        for (at, want, what) in [(2 * i, decay, "decay"), (2 * i + 1, beta, "beta")] {
            assert!(
                (gate[at] - want).abs() <= 1e-6 * want.abs().max(1.0),
                "gate slot {at} holds {} and the {what} for head {i} is {want}",
                gate[at]
            );
        }
    }
}

/// The nine `(LANES, VROWS)` tilings the prefill scan is compiled at.
const GDN_SCAN_TILINGS: &[(&str, u32, u32)] = &[
    ("gdn_core_recurrent_prefill_bfloat16_l_4_v_1", 4, 1),
    ("gdn_core_recurrent_prefill_bfloat16_l_8_v_1", 8, 1),
    ("gdn_core_recurrent_prefill_bfloat16_l_8_v_2", 8, 2),
    ("gdn_core_recurrent_prefill_bfloat16_l_16_v_1", 16, 1),
    ("gdn_core_recurrent_prefill_bfloat16_l_16_v_2", 16, 2),
    ("gdn_core_recurrent_prefill_bfloat16_l_16_v_4", 16, 4),
    ("gdn_core_recurrent_prefill_bfloat16_l_32_v_2", 32, 2),
    ("gdn_core_recurrent_prefill_bfloat16_l_32_v_4", 32, 4),
    ("gdn_core_recurrent_prefill_bfloat16_l_32_v_8", 32, 8),
];

/// The prompt-prefill scan answers the decode, walked token by token.
///
/// Ten of this family's sixteen modules are the prefill path -- one prep and
/// nine `(LANES, VROWS)` tilings of the scan -- and none had ever run. There
/// is no reference worth writing for them, because the thing they have to
/// agree with is not an equation but the DECODE path: a prompt pushed through
/// the fused kernel one token at a time, with the convolution history
/// ping-ponged forward, is what the model would have computed had it decoded
/// the prompt instead of prefilling it. That walk is the oracle.
///
/// Five tokens against a four-tap window, so the FIR really does walk off
/// `conv_state` and onto the prompt: at `t = 0` three of its four taps come
/// from the carried history and at `t = 3` none do. A prompt shorter than
/// `Kc` would never leave the history and a much longer one would only repeat
/// the interior case.
///
/// # The pitch, which is not the one you would guess
///
/// `row_pitch` counts elements of the bf16 prompt, and the f32 scratch shares
/// its BYTE pitch -- so the scratch's row stride is `row_pitch / 2` floats,
/// which is the shader's `pitch_f`. That makes the pitch a real constraint
/// rather than bookkeeping: it has to cover `Hv * Dk` floats for `pre_q` AND
/// `2 * Hv + Hv * Dv` for the gate slab, whose tail is the precomputed V the
/// decode prep does not write. The fixture's natural `conv_dim` of 288 gives
/// a `pitch_f` of 144 against a `pre_q` row of 256, so the prompt rows are
/// PADDED to 512 and the shader reads only the meaningful head of each. A
/// pitch chosen by shape rather than by constraint would have run off the end
/// of one row into the next and returned plausible numbers.
#[test]
fn the_prefill_scan_answers_the_decode_walked_token_by_token() {
    let gpu = gpu!();
    let s = GdnShape {
        b: 1,
        ..GdnShape::FIXTURE
    };
    let eps = 1e-6f32;
    let tokens = 5usize;
    let cd = s.conv_dim();
    let pitch = 512usize;
    assert!(
        pitch / 2 >= (s.hv * s.dk).max(2 * s.hv + s.hv * s.dv) && pitch >= cd,
        "the fixture's pitch has to cover the widest scratch row and the prompt"
    );

    let inp = GdnInputs::build(&s, 1);
    let prompt: Vec<f32> = gdn_spread(tokens * pitch, 11, 1.0);
    let prompt_bytes = bf16_bytes(&prompt);
    // The prefill reads its gates per TOKEN, at `t * row_pitch + hv`, where
    // the decode reads them per request at `b * Hv + hv`. So the walk has to
    // be handed token `t`'s slice and not the same four values five times --
    // getting that wrong is what the first run of this test did, and it
    // agreed at token 0 and diverged at token 1, which is exactly what a
    // stale gate looks like.
    let pa = gdn_spread(tokens * pitch, 8, 1.0);
    let pb = gdn_spread(tokens * pitch, 9, 1.0);

    // The oracle: `gdn_core` once per token, carrying both slabs forward.
    let mut conv = inp.conv_state.clone();
    let mut state = inp.rstate.clone();
    let mut walked: Vec<Vec<u8>> = Vec::new();
    for t in 0..tokens {
        let mut step = GdnInputs::build(&s, 1);
        step.mixed = prompt[t * pitch..t * pitch + cd].to_vec();
        step.conv_state = conv.clone();
        step.rstate = state.clone();
        let (a_log, dt_bias) = (inp.a_log.clone(), inp.dt_bias.clone());
        let a_gate = pa[t * pitch..t * pitch + s.hv].to_vec();
        let b_gate = pb[t * pitch..t * pitch + s.hv].to_vec();
        let (conv_w, conv_b) = (inp.conv_w.clone(), inp.conv_b.clone());
        step.a_log = a_log;
        step.dt_bias = dt_bias;
        step.a_gate = a_gate;
        step.b_gate = b_gate;
        step.conv_w = conv_w;
        step.conv_b = conv_b;
        let out = gpu.dispatch(
            "gdn_core_bfloat16",
            Capability::Baseline,
            &step.operands(&s, eps, 1),
            &[],
            [1, (s.dv / 4) as u32, (s.hv) as u32],
        );
        walked.push(out[3].clone());
        state = f32_read(&out[2]);
        conv = f32_read(&out[10]);
    }

    // The prefill prep, over the whole prompt at once.
    let push: Vec<u8> = [pitch as i32, tokens as i32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let scratch = tokens * (pitch / 2) * 4;
    let prep = gpu.dispatch(
        "gdn_prep_prefill_bfloat16",
        Capability::Baseline,
        &[
            prompt_bytes.clone(),
            inp.conv_state
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            bf16_bytes(&inp.conv_w),
            bf16_bytes(&inp.conv_b),
            inp.a_log.iter().flat_map(|v| v.to_le_bytes()).collect(),
            bf16_bytes(&inp.dt_bias),
            // The prefill reads its gates per TOKEN at `t * row_pitch + hv`,
            // not per request, so these are prompt-shaped too.
            bf16_bytes(&pa),
            bf16_bytes(&pb),
            vec![0u8; scratch],
            vec![0u8; scratch],
            vec![0u8; scratch],
            vec![0u8; s.kc * cd * 4],
            s.params(eps),
            vec![0u8; 4],
        ],
        &push,
        [1, 1, (tokens * s.hv) as u32],
    );

    // Every tiling answers the same walk. They differ in how many lanes carry
    // a reduction and how many value rows a lane group holds, which is a
    // statement about scheduling and not about arithmetic -- but `LANES`
    // changes the WIDTH of the reduction and so its association, which is why
    // this is checked and not assumed.
    let mut states: Vec<(&str, u32, u32, Vec<u8>)> = Vec::new();
    for &(name, lanes, vrows) in GDN_SCAN_TILINGS {
        let rows = 32 / lanes;
        let scan = gpu.dispatch(
            name,
            Capability::Baseline,
            &[
                inp.rstate.iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![0u8; tokens * pitch * 2],
                prep[8].clone(),
                prep[9].clone(),
                prep[10].clone(),
                s.params(eps),
                vec![0u8; 4],
            ],
            &push,
            [1, (s.dv as u32).div_ceil(rows * vrows), s.hv as u32],
        );

        let got = bf16_read(&scan[1]);
        for t in 0..tokens {
            let want = bf16_read(&walked[t]);
            let row = &got[t * pitch..t * pitch + s.hv * s.dv];
            assert_eq!(
                row,
                &want[..],
                "{name} disagrees with the token-by-token walk at token {t}. \
                 `core_out` is bf16 and no reduction width in this family \
                 moves a value that far, so this is arithmetic and not \
                 association"
            );
        }
        states.push((name, lanes, vrows, scan[0].clone()));
    }
    // What `core_out` rounds away, `rstate` keeps. The output is bf16 and
    // agrees with the walk exactly; the state is f32 and sits 1.4e-7 from it,
    // which is where the difference between the tilings actually lives.
    //
    // And it lives there in a shape worth stating: two tilings hold the same
    // state BIT for bit exactly when they share `LANES`. That is each
    // parameter's own claim, measured rather than assumed -- `VROWS` changes
    // how many independent value rows a lane group carries and nothing that
    // is summed, while `LANES` changes the width of the lane reduction and so
    // its association. Written as an equivalence in both directions, so a
    // `VROWS` that started perturbing a sum fails here just as loudly as a
    // `LANES` that stopped.
    for (name, lanes, _, st) in &states {
        for (other, olanes, _, ost) in &states {
            assert_eq!(
                st == ost,
                lanes == olanes,
                "{name} and {other} hold {} recurrent states and their LANES \
                 are {}. Bit-equality here follows LANES and only LANES",
                if st == ost { "identical" } else { "different" },
                if lanes == olanes {
                    "the same"
                } else {
                    "different"
                },
            );
        }
    }

    let worst = states
        .iter()
        .map(|(n, _, _, st)| gdn_check(&f32_read(st), &state, GDN_STATE_TOL, n))
        .fold(0.0f32, f32::max);
    gdn_band(worst, GDN_STATE_TOL, "the prefill scan's rstate");
}

/// The slotted split pair follows the same map as the slotted fused kernel.
///
/// `gdn_prep_slotted` and `gdn_core_recurrent_slotted` are the last two of the
/// family's sixteen modules with no number to their name. They are the split
/// pair and the slot indirection at once, which is exactly the combination
/// where a port drops one of the two: the prep reads `slot_ids[b_idx]` for its
/// convolution history while the recurrent reads it again for the recurrent
/// state, and a half that quietly fell back to `b_idx` would agree with
/// everything under an identity map.
///
/// So the map is a permutation from the start, and the oracle is the slotted
/// FUSED kernel over the same permuted slabs -- which
/// `the_slotted_gdn_core_follows_its_slot_map` has already tied back to the
/// unslotted form. Bit for bit, for the reason the unslotted split has it:
/// the two halves reduce under different workgroup shapes, and a tolerance
/// wide enough for a bf16 store would not see the only failure this pair
/// actually has.
#[test]
fn the_slotted_split_pair_follows_the_same_map() {
    let gpu = gpu!();
    let s = GdnShape::FIXTURE;
    let eps = 1e-6f32;
    let n = s.b * s.hv;
    let cd = s.conv_dim();
    let inp = GdnInputs::build(&s, s.b);

    // Slot `perm[b]` holds what request `b` carries, so every slab addressed
    // BY SLOT moves and the per-request inputs stay where they are.
    let perm: Vec<usize> = (0..s.b).map(|b| (b + 1) % s.b).collect();
    let mut moved = GdnInputs::build(&s, s.b);
    for b in 0..s.b {
        for j in 0..s.kc {
            for c in 0..cd {
                moved.conv_state[(perm[b] * s.kc + j) * cd + c] =
                    inp.conv_state[(b * s.kc + j) * cd + c];
            }
        }
        let width = s.hv * s.dv * s.dk;
        moved.rstate[perm[b] * width..(perm[b] + 1) * width]
            .copy_from_slice(&inp.rstate[b * width..(b + 1) * width]);
    }
    let slots: Vec<u8> = perm
        .iter()
        .flat_map(|p| (*p as u32).to_le_bytes())
        .collect();

    let mut fused_ops = moved.operands(&s, eps, s.b);
    fused_ops.push(slots.clone());
    let fused = gpu.dispatch(
        "gdn_core_slotted_bfloat16",
        Capability::Baseline,
        &fused_ops,
        &[],
        [1, (s.dv / 4) as u32, n as u32],
    );

    let mut prep_ops = moved.prep_operands(&s, eps, s.b);
    prep_ops.push(slots.clone());
    let prepped = gpu.dispatch(
        "gdn_prep_slotted_bfloat16",
        Capability::Baseline,
        &prep_ops,
        &[],
        [1, 1, n as u32],
    );

    let recurrent = gpu.dispatch(
        "gdn_core_recurrent_slotted_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&moved.mixed),
            moved
                .conv_state
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            moved.rstate.iter().flat_map(|v| v.to_le_bytes()).collect(),
            vec![0u8; n * s.dv * 2],
            bf16_bytes(&moved.conv_w),
            bf16_bytes(&moved.conv_b),
            prepped[8].clone(),
            prepped[9].clone(),
            prepped[10].clone(),
            prepped[11].clone(),
            s.params(eps),
            slots,
        ],
        &[],
        [1, (s.dv / 4) as u32, n as u32],
    );

    for (split, whole, what) in [
        (3usize, 3usize, "core_out"),
        (2, 2, "rstate"),
        (9, 10, "new_conv_state"),
    ] {
        assert_eq!(
            recurrent[split], fused[whole],
            "under a permuted slot map the split pair's {what} is not the \
             fused kernel's. One of the two halves is reading a slot the \
             other is not"
        );
    }
}

/// The flash decode agrees with the single-pass decode, split every way.
///
/// `sdpa_paged_decode_split` + `sdpa_paged_decode_combine` compute the same
/// attention as `sdpa_paged_decode`, over a key range cut into `S` contiguous
/// chunks whose partial `(max, sum_exp, weighted V)` the fold merges. The
/// merge is algebraically exact and floating-point ASSOCIATIVE-ly different,
/// which is why this checks against the same scalar softmax the single-pass
/// test does rather than against the single-pass output: agreeing with the
/// other shader to the last bit is not what is being claimed.
///
/// The split counts are chosen for their degenerate cases and not for their
/// speed. `S = 64` against histories of 17, 5 and 32 keys hands MOST splits
/// nothing at all: their `max` stays at `PIE_SDPA_NEG_INF`, their `sum_exp`
/// stays zero, and the fold's weight for them is `exp(-3e38 - merged_max)`,
/// which must be a clean zero and not a NaN. A row whose mask drops every key
/// makes EVERY split empty, and then the merged sum is zero and the guarded
/// divide has to answer zero rather than `0/0`.
#[test]
fn the_flash_decode_agrees_with_the_scalar_reference_at_every_split_count() {
    let gpu = gpu!();

    let head_dim = 64usize;
    let page_size = 16usize;
    let n_kv_heads = 2usize;
    let gqa = 2usize;
    let n_q_heads = n_kv_heads * gqa;
    let rows = 3usize;
    let scale = 0.125f32;

    let lengths = [17usize, 5, 32];
    let pages_per: Vec<usize> = lengths.iter().map(|l| l.div_ceil(page_size)).collect();
    let total_pages: usize = pages_per.iter().sum();
    let physical: Vec<u32> = {
        let mut v: Vec<u32> = (0..total_pages as u32).collect();
        v.reverse();
        v
    };
    let mut indptr = vec![0u32];
    for p in &pages_per {
        indptr.push(indptr.last().unwrap() + *p as u32);
    }

    let slots = total_pages * page_size;
    let kv_elems = slots * n_kv_heads * head_dim;
    let kf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 31) as f32 - 15.0) / 40.0)
        .collect();
    let vf: Vec<f32> = (0..kv_elems)
        .map(|i| ((i % 23) as f32 - 11.0) / 30.0)
        .collect();
    let qf: Vec<f32> = (0..rows * n_q_heads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) / 20.0)
        .collect();
    let positions: Vec<i32> = lengths.iter().map(|l| *l as i32 - 1).collect();
    let req_of_token: Vec<i32> = (0..rows as i32).collect();
    let sinks: Vec<f32> = (0..n_q_heads).map(|h| -0.5 + h as f32 * 0.75).collect();

    // Row 2 masks everything off: the mask is enabled and every byte is zero.
    // `attention_mask_stride` must then be wide enough for the widest row, so
    // `keeps` reads the mask rather than falling out of bounds.
    let mask_stride = 32usize;
    let mut mask = vec![1u8; rows * mask_stride];
    for kp in 0..mask_stride {
        mask[2 * mask_stride + kp] = 0;
    }
    let enabled = vec![0u8, 0, 1];

    let mut push = Vec::new();
    push.extend_from_slice(&(gqa as i32).to_le_bytes());
    push.extend_from_slice(&(page_size as i32).to_le_bytes());
    push.extend_from_slice(&(n_kv_heads as i32).to_le_bytes());
    push.extend_from_slice(&scale.to_le_bytes());
    push.extend_from_slice(&(mask_stride as u32).to_le_bytes());
    push.extend_from_slice(&0i32.to_le_bytes());

    let q = bf16_read(&bf16_bytes(&qf));
    let k = bf16_read(&bf16_bytes(&kf));
    let v = bf16_read(&bf16_bytes(&vf));
    let slot_of = |req: usize, kp: usize| {
        let phys = physical[indptr[req] as usize + kp / page_size] as usize;
        phys * page_size + kp % page_size
    };

    // The reference, once: with a sink and without, for every row and head.
    let reference = |with_sink: bool| -> Vec<f32> {
        let mut want = vec![0.0f32; rows * n_q_heads * head_dim];
        for (row, &position) in positions.iter().enumerate() {
            let q_pos = position as usize;
            let kept: Vec<usize> = (0..=q_pos)
                .filter(|kp| enabled[row] == 0 || mask[row * mask_stride + kp] != 0)
                .collect();
            for h in 0..n_q_heads {
                let kv_head = h / gqa;
                let q_base = (row * n_q_heads + h) * head_dim;
                let scores: Vec<f32> = kept
                    .iter()
                    .map(|&kp| {
                        let k_base = (slot_of(row, kp) * n_kv_heads + kv_head) * head_dim;
                        (0..head_dim)
                            .map(|d| scale * q[q_base + d] * k[k_base + d])
                            .sum::<f32>()
                    })
                    .collect();
                let mut hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if with_sink {
                    hi = hi.max(sinks[h]);
                }
                if !hi.is_finite() {
                    continue; // every key masked and no sink: the shader answers zero
                }
                let exps: Vec<f32> = scores.iter().map(|s| (s - hi).exp()).collect();
                let mut denom: f32 = exps.iter().sum();
                if with_sink {
                    denom += (sinks[h] - hi).exp();
                }
                for d in 0..head_dim {
                    let acc: f32 = kept
                        .iter()
                        .zip(&exps)
                        .map(|(&kp, e)| {
                            let v_at = (slot_of(row, kp) * n_kv_heads + kv_head) * head_dim + d;
                            e * v[v_at]
                        })
                        .sum();
                    want[q_base + d] = if denom == 0.0 { 0.0 } else { acc / denom };
                }
            }
        }
        want
    };

    let plain_want = reference(false);
    let sink_want = reference(true);

    for splits in [2usize, 3, 8, 64] {
        // Binding 3 is `out_` and binding 10 is `sinks`: a split writes
        // neither, slangc drops both, and the descriptor set carries holes
        // there. The buffers are still handed over so the indices line up.
        let entries = splits * rows * n_q_heads;
        let partial_floats = entries * (head_dim + 2);
        let split_operands = vec![
            bf16_bytes(&qf),
            bf16_bytes(&kf),
            bf16_bytes(&vf),
            vec![0u8; 4],
            positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
            req_of_token.iter().flat_map(|r| r.to_le_bytes()).collect(),
            physical.iter().flat_map(|p| p.to_le_bytes()).collect(),
            indptr.iter().flat_map(|p| p.to_le_bytes()).collect(),
            mask.clone(),
            enabled.clone(),
            vec![0u8; 4],
            // Poisoned, not zeroed. The driver never clears this buffer, so a
            // split that skipped its write would be read as last token's
            // numbers -- here, as a NaN that cannot hide in a tolerance.
            std::iter::repeat_n(f32::NAN.to_le_bytes(), partial_floats)
                .flatten()
                .collect(),
        ];
        let after = gpu.dispatch(
            &format!("sdpa_paged_decode_split_bfloat16_d_{head_dim}"),
            Capability::Baseline,
            &split_operands,
            &push,
            [n_q_heads as u32, rows as u32, splits as u32],
        );

        let fold_push: Vec<u8> = (splits as i32).to_le_bytes().to_vec();
        let out_bytes = vec![0u8; rows * n_q_heads * head_dim * 2];
        let folded = gpu.dispatch(
            &format!("sdpa_paged_decode_combine_bfloat16_d_{head_dim}"),
            Capability::Baseline,
            &[out_bytes.clone(), vec![0u8; 4], after[11].clone()],
            &fold_push,
            [n_q_heads as u32, rows as u32, 1],
        );
        assert_close(
            &bf16_read(&folded[0]),
            &plain_want,
            &format!("flash decode at {splits} splits"),
        );

        let folded = gpu.dispatch(
            "sdpa_paged_decode_combine_sink_bfloat16_d_64",
            Capability::Baseline,
            &[out_bytes.clone(), bf16_bytes(&sinks), after[11].clone()],
            &fold_push,
            [n_q_heads as u32, rows as u32, 1],
        );
        assert_close(
            &bf16_read(&folded[0]),
            &sink_want,
            &format!("flash decode with a sink at {splits} splits"),
        );
    }
}

/// The fused norm+rope answers what the two kernels it replaces answer.
///
/// This is the ONLY thing that makes the fusion shippable, and it is worth
/// being precise about why the comparison is built the way it is. The two
/// stages are run against the same input with the same push words and the
/// same grids they get in production -- `rms_strided_head_row` at one
/// workgroup per (head, token), then `neox_mb` at one thread per rotary pair
/// -- and the fused kernel is run against a SECOND COPY of that input. Both
/// sides therefore round through bf16 in the same places, which is what lets
/// this be an equality-shaped assertion with a tolerance rather than a
/// hand-waved "close enough": the fused form's only arithmetic difference is
/// that it never stores the normed value before rotating it, so its `x1` and
/// `x2` are one bf16 round LESS quantised than the reference's.
///
/// That difference is real and it is in the fused form's favour, so the
/// tolerance is one-sided in principle and symmetric in practice; `axis` is
/// 64 and the values are deliberately not smooth, because a ramp would hide a
/// pair-indexing error -- `(i, i + half)` and `(2i, 2i+1)` agree on a ramp
/// and on nothing else.
#[test]
fn rms_rope_answers_what_the_norm_and_the_rotation_answer() {
    let gpu = gpu!();

    let heads = 3usize;
    let head_dim = 64usize;
    let rows = 2usize;
    let pitch = heads * head_dim;
    let eps = 1e-5f32;
    let scale = 1.0f32;
    // `exp2(-d * base)` is how both kernels spell the geometric ladder, so
    // this is log2 of the rope theta and not the theta.
    let base = (10000.0f32).log2();

    let x: Vec<f32> = (0..rows * pitch)
        .map(|i| ((i * 37 % 71) as f32 - 35.0) / 16.0)
        .collect();
    let w: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 13) as f32 / 32.0).collect();
    let positions: Vec<u8> = [7i32, 11]
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(head_dim as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes()); // w_stride
    params.extend_from_slice(&0u32.to_le_bytes()); // plus_one
    params.extend_from_slice(&1.0f32.to_le_bytes()); // gain

    // Stage one: the per-head norm, out of place, exactly as the text states
    // it today.
    let normed = gpu.dispatch(
        "rms_strided_head_row_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&x),
            bf16_bytes(&w),
            vec![0u8; rows * pitch * 2],
            params.clone(),
        ],
        &(pitch as i32).to_le_bytes(),
        [1, heads as u32, rows as u32],
    );

    // Stage two: the rotation, in place on what stage one wrote.
    let mut rope_push = Vec::new();
    rope_push.extend_from_slice(&scale.to_le_bytes());
    rope_push.extend_from_slice(&base.to_le_bytes());
    rope_push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    let two_stage = gpu.dispatch(
        "neox_mb_bfloat16",
        Capability::Baseline,
        &[normed[2].clone(), positions.clone()],
        &rope_push,
        [(head_dim / 2) as u32, heads as u32, rows as u32],
    );

    // The fused form, from the same input the first stage was given.
    // `RmsRopeParams` is `RmsParams` with four fields appended, and the
    // driver mints it as the statement's whole params run -- so the test
    // builds it the same way, by extending the five the norm already states.
    let mut fused_params = params.clone();
    fused_params.extend_from_slice(&(pitch as u32).to_le_bytes());
    fused_params.extend_from_slice(&(head_dim as u32).to_le_bytes()); // rotary
    fused_params.extend_from_slice(&scale.to_le_bytes());
    fused_params.extend_from_slice(&base.to_le_bytes());
    let fused = gpu.dispatch(
        "rms_rope_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&x),
            bf16_bytes(&w),
            fused_params,
            positions.clone(),
        ],
        &[],
        [1, heads as u32, rows as u32],
    );

    let want = bf16_read(&two_stage[0]);
    let got = bf16_read(&fused[0]);
    // Not `assert_close`: the fused form legitimately differs by the one bf16
    // round it does not do, which is up to half an ULP of a bf16 -- 0.4% --
    // and a tolerance that tight on a rotation that mixes two channels is a
    // flake waiting to happen. The bound is absolute against the input's own
    // scale, and the argmax-style claim beside it is that no element moved by
    // more than a rounding's worth.
    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 0.05,
        "the fused norm+rope and the two kernels it replaces disagree by {worst}"
    );
    // And it is not trivially right by writing nothing: a rotation at
    // position 7 moves every channel, so an untouched buffer would match the
    // INPUT rather than the reference.
    let xq = bf16_read(&bf16_bytes(&x));
    let moved = xq
        .iter()
        .zip(&got)
        .filter(|(a, b)| (*a - *b).abs() > 0.05)
        .count();
    assert!(
        moved > rows * pitch / 2,
        "only {moved} of {} elements moved, so the fused kernel is not doing the work",
        rows * pitch
    );
}

/// The partial-rotary arm, which no shipped path exercises yet.
///
/// gemma-4 rotates a quarter of each full-attention head and leaves the rest
/// alone, so the fused kernel has a branch that the qwen-shaped test above
/// can never reach: channels at or above `rotary` must come out NORMED AND
/// UNROTATED, not zero and not copied through unnormed. Both failure modes
/// are live -- an early `return` on the tail would leave the norm's output in
/// place (which happens to be right) while an early return placed before the
/// store would leave the INPUT in place (which is wrong and would pass a
/// sloppier assertion), so this checks the tail explicitly against the norm
/// rather than only checking the whole buffer against the reference.
///
/// The reference gets partial rotary for free: `neox` reads its half-width
/// off `gl_NumWorkGroups.x`, so dispatching a narrower grid rotates a prefix
/// and leaves the tail untouched. That is the same definition the fused
/// kernel is asked to honour, arrived at by a different mechanism.
#[test]
fn rms_rope_leaves_the_unrotated_tail_normed() {
    let gpu = gpu!();

    let heads = 2usize;
    let head_dim = 64usize;
    let rotary = 16usize;
    let rows = 2usize;
    let pitch = heads * head_dim;
    let eps = 1e-5f32;
    let scale = 1.0f32;
    let base = (10000.0f32).log2();

    let x: Vec<f32> = (0..rows * pitch)
        .map(|i| ((i * 29 % 53) as f32 - 26.0) / 12.0)
        .collect();
    let w: Vec<f32> = (0..head_dim).map(|i| 0.75 + (i % 7) as f32 / 16.0).collect();
    let positions: Vec<u8> = [3i32, 5].iter().flat_map(|p| p.to_le_bytes()).collect();

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(head_dim as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());

    let normed = gpu.dispatch(
        "rms_strided_head_row_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&x),
            bf16_bytes(&w),
            vec![0u8; rows * pitch * 2],
            params.clone(),
        ],
        &(pitch as i32).to_le_bytes(),
        [1, heads as u32, rows as u32],
    );
    let just_normed = bf16_read(&normed[2]);

    let mut rope_push = Vec::new();
    rope_push.extend_from_slice(&scale.to_le_bytes());
    rope_push.extend_from_slice(&base.to_le_bytes());
    rope_push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    let two_stage = gpu.dispatch(
        "neox_mb_bfloat16",
        Capability::Baseline,
        &[normed[2].clone(), positions.clone()],
        &rope_push,
        [(rotary / 2) as u32, heads as u32, rows as u32],
    );

    let mut fused_params = params.clone();
    fused_params.extend_from_slice(&(pitch as u32).to_le_bytes());
    fused_params.extend_from_slice(&(rotary as u32).to_le_bytes());
    fused_params.extend_from_slice(&scale.to_le_bytes());
    fused_params.extend_from_slice(&base.to_le_bytes());
    let fused = gpu.dispatch(
        "rms_rope_bfloat16",
        Capability::Baseline,
        &[
            bf16_bytes(&x),
            bf16_bytes(&w),
            fused_params,
            positions.clone(),
        ],
        &[],
        [1, heads as u32, rows as u32],
    );

    let want = bf16_read(&two_stage[0]);
    let got = bf16_read(&fused[0]);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 0.05, "partial rotary disagrees by {worst}");

    // The tail, stated separately and against the norm's own output, so that
    // "the whole buffer matched" cannot be satisfied by two kernels making
    // the same mistake past `rotary`.
    for row in 0..rows {
        for head in 0..heads {
            for c in rotary..head_dim {
                let at = row * pitch + head * head_dim + c;
                assert!(
                    (got[at] - just_normed[at]).abs() < 0.05,
                    "channel {c} of head {head} is {} where the norm alone says {}",
                    got[at],
                    just_normed[at]
                );
            }
        }
    }
    // And the head of the head really was rotated, or the tail check above is
    // passing for the trivial reason that nothing happened at all.
    let xq = bf16_read(&bf16_bytes(&x));
    let moved = (0..rotary)
        .filter(|c| (got[*c] - xq[*c]).abs() > 0.05)
        .count();
    assert!(moved > rotary / 2, "only {moved} rotary channels moved");
}
