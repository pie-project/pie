//! The host link a device states: its PCIe generation's per-lane transfer
//! rate and its width, as the host's PCI tree reports the most the device
//! and its port train to (`/sys/bus/pci/devices/<bus id>/max_link_speed`
//! and `max_link_width`, each end's). Its copy ceiling is those times the generation's
//! line coding (8b/10b to 5 GT/s, 128b/130b from 8 GT/s): the standard's
//! figures, nothing measured or chosen.

/// A device's current host link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    /// Transfers a second a lane (GT/s).
    pub gts: f64,
    pub width: u32,
}

impl Link {
    /// The bytes a second the link carries at most, both ways' one
    /// direction: lanes x transfers x the coding's payload bits, over 8.
    #[must_use]
    pub fn copy_ceiling(&self) -> f64 {
        let coding = if self.gts <= 5.0 {
            8.0 / 10.0
        } else {
            128.0 / 130.0
        };
        f64::from(self.width) * self.gts * 1e9 * coding / 8.0
    }
}

/// `max_link_speed`'s text (`"16.0 GT/s PCIe"`) and `max_link_width`'s
/// (`"16"`), as one [`Link`]; none where either does
/// not read as a rate.
#[must_use]
pub fn parse(speed: &str, width: &str) -> Option<Link> {
    let gts: f64 = speed.split_whitespace().next()?.parse().ok()?;
    let width: u32 = width.trim().parse().ok()?;
    (gts > 0.0 && width > 0).then_some(Link { gts, width })
}

/// The link the bound device and its upstream port train to, where the
/// host's PCI tree states it: the lesser of each end's `max_link_speed` and
/// `max_link_width`. Not `current_link_*`: an idle link trains down (Gen1
/// at rest) and back up under load, so its current state says nothing of
/// what a load's copies cross at.
#[cfg(feature = "device")]
pub fn stated(ordinal: i32) -> Result<Link, String> {
    let bus = bus_id(ordinal)?.to_lowercase();
    let at = std::path::Path::new("/sys/bus/pci/devices").join(&bus);
    let end = |dir: &std::path::Path| -> Result<Link, String> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .map_err(|why| format!("{}/{name}: {why}", dir.display()))
        };
        let (speed, width) = (read("max_link_speed")?, read("max_link_width")?);
        parse(&speed, &width).ok_or_else(|| {
            format!(
                "{}: max_link_speed `{}` and max_link_width `{}` state no rate",
                dir.display(),
                speed.trim(),
                width.trim()
            )
        })
    };
    let device = end(&at)?;
    // The port above it, where the tree states one (a virtual function's
    // parent may state none: the device's own ends then bound it).
    let port = std::fs::canonicalize(&at)
        .ok()
        .and_then(|path| path.parent().map(std::path::Path::to_path_buf))
        .and_then(|parent| end(&parent).ok());
    Ok(match port {
        Some(port) => Link {
            gts: device.gts.min(port.gts),
            width: device.width.min(port.width),
        },
        None => device,
    })
}

/// A runtimeless build reads no device.
#[cfg(not(feature = "device"))]
pub fn stated(_ordinal: i32) -> Result<Link, String> {
    Err("a runtimeless build reads no device".to_string())
}

#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn bus_id(ordinal: i32) -> Result<String, String> {
    use core::ffi::{c_char, c_int};
    let mut bus = [0 as c_char; 32];
    // SAFETY: `bus` is written within its length and NUL-terminated by the
    // runtime.
    let asked = unsafe {
        crate::device::rt::cudaDeviceGetPCIBusId(bus.as_mut_ptr(), bus.len() as c_int, ordinal)
    };
    if asked != crate::device::rt::cudaError::cudaSuccess {
        return Err(format!("cudaDeviceGetPCIBusId answered {asked:?}"));
    }
    // SAFETY: NUL-terminated by the runtime.
    Ok(unsafe { core::ffi::CStr::from_ptr(bus.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

#[cfg(feature = "rocm")]
fn bus_id(_ordinal: i32) -> Result<String, String> {
    kernels_cuda::hip::kfd::pci_bus_id()
}

#[cfg(test)]
mod tests {
    use super::{Link, parse};

    #[test]
    fn a_stated_link_carries_its_lanes_at_its_coding() {
        let gen4 = parse("16.0 GT/s PCIe\n", "16\n").expect("reads");
        assert_eq!(
            gen4,
            Link {
                gts: 16.0,
                width: 16
            }
        );
        assert!(
            (gen4.copy_ceiling() / 1e9 - 31.5).abs() < 0.1,
            "{}",
            gen4.copy_ceiling()
        );
        let gen1 = parse("2.5 GT/s PCIe", "1").expect("reads");
        assert!((gen1.copy_ceiling() / 1e9 - 0.25).abs() < 1e-9);
        assert_eq!(parse("Unknown", "16"), None);
        assert_eq!(parse("8.0 GT/s PCIe", "0"), None);
    }

    /// On a card here, the link its ends train to (offline: needs a device).
    #[cfg(feature = "device")]
    #[test]
    #[ignore = "reads the bound device's PCI tree"]
    fn the_bound_device_states_its_link() {
        let link = super::stated(0).expect("the host states the link");
        eprintln!("{link:?}: {:.1} GB/s", link.copy_ceiling() / 1e9);
        assert!(link.width >= 1 && link.gts >= 2.5);
    }
}
