//! What the device's driver states of itself for the spec rates
//! (`engine::spec`): its multiprocessor count, peak compute clock, peak
//! memory clock and memory bus width.

/// The bound device's stated figures, under the architecture token its
/// stored key names (`sm_120`, `gfx942`).
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
pub fn stated(ordinal: i32, arch: &str) -> Result<engine::spec::Stated, String> {
    use crate::device::rt;
    let attribute = |which: rt::cudaDeviceAttr, name: &str| -> Result<u32, String> {
        let mut value = 0i32;
        // SAFETY: `value` is a live out-parameter.
        let status = unsafe { rt::cudaDeviceGetAttribute(&raw mut value, which, ordinal) };
        if status != rt::cudaError::cudaSuccess {
            return Err(format!(
                "cudaDeviceGetAttribute({name}) answered {status:?}"
            ));
        }
        u32::try_from(value).map_err(|_| format!("{name} is {value}"))
    };
    Ok(engine::spec::Stated {
        arch: arch.to_string(),
        units: attribute(
            rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
            "cudaDevAttrMultiProcessorCount",
        )?,
        clock_khz: attribute(
            rt::cudaDeviceAttr::cudaDevAttrClockRate,
            "cudaDevAttrClockRate",
        )?,
        memory_clock_khz: attribute(
            rt::cudaDeviceAttr::cudaDevAttrMemoryClockRate,
            "cudaDevAttrMemoryClockRate",
        )?,
        bus_bits: attribute(
            rt::cudaDeviceAttr::cudaDevAttrGlobalMemoryBusWidth,
            "cudaDevAttrGlobalMemoryBusWidth",
        )?,
    })
}

/// The bound device's stated figures through HIP's attributes
/// (`hipDeviceAttributeClockRate`, `MemoryClockRate`, `MemoryBusWidth`,
/// `MultiprocessorCount`).
#[cfg(feature = "rocm")]
#[allow(unsafe_code)]
pub fn stated(ordinal: i32, arch: &str) -> Result<engine::spec::Stated, String> {
    use core::ffi::c_int;
    const CLOCK_RATE: c_int = 5;
    const MEMORY_BUS_WIDTH: c_int = 59;
    const MEMORY_CLOCK_RATE: c_int = 60;
    const MULTIPROCESSOR_COUNT: c_int = 63;
    let attribute = |which: c_int, name: &str| -> Result<u32, String> {
        let mut value: c_int = 0;
        // SAFETY: `value` is a live out-parameter.
        let status =
            unsafe { kernels_cuda::hip::rt::hipDeviceGetAttribute(&raw mut value, which, ordinal) };
        if status != kernels_cuda::hip::sys::HIP_SUCCESS {
            return Err(format!("hipDeviceGetAttribute({name}) answered {status}"));
        }
        u32::try_from(value).map_err(|_| format!("{name} is {value}"))
    };
    Ok(engine::spec::Stated {
        arch: arch.to_string(),
        units: attribute(MULTIPROCESSOR_COUNT, "MultiprocessorCount")?,
        clock_khz: attribute(CLOCK_RATE, "ClockRate")?,
        memory_clock_khz: attribute(MEMORY_CLOCK_RATE, "MemoryClockRate")?,
        bus_bits: attribute(MEMORY_BUS_WIDTH, "MemoryBusWidth")?,
    })
}

/// A runtimeless build reads no device.
#[cfg(not(feature = "device"))]
pub fn stated(_ordinal: i32, _arch: &str) -> Result<engine::spec::Stated, String> {
    Err("a runtimeless build reads no device".to_string())
}

#[cfg(all(test, feature = "device"))]
mod tests {
    /// On a card here: what its driver states for the spec rates.
    #[test]
    #[ignore = "reads the bound device"]
    fn the_bound_device_states_its_spec() {
        let stated = super::stated(0, "sm_120").expect("the driver states");
        eprintln!(
            "{stated:?}: {:?}",
            engine::spec::rates(&stated).map(|(_, line)| line)
        );
        assert!(stated.units > 0 && stated.clock_khz > 0);
    }
}
