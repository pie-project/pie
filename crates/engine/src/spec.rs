//! The roofline rates a device's published specification gives, where no
//! profile states them: what a load that measures nothing (`pie tune`
//! measures) weighs its fire top by on a device with no committed profile
//! of its own or of its family, named "computed from spec" in its line.
//!
//! Memory bandwidth is the driver's stated peak memory clock times its
//! stated bus width, twice a clock (as CUDA's `deviceQuery` sample computes
//! it), only on an architecture where that is checked against a datasheet
//! ([`Bandwidth::Driver`], the check cited); elsewhere (the HBM parts whose
//! stated bus figures are not the memory's: an MI300X states 1300 MHz over
//! 2048 bits, 665.6 GB/s by the formula, of its 5.3 TB/s) the datasheet's
//! figure, a part at a time by the device's name ([`Bandwidth::Parts`]).
//! Compute is the driver's stated multiprocessor (CU) count times its
//! stated peak clock times the architecture's published dense operations a
//! clock a multiprocessor ([`ARCHS`], one row an architecture, each citing
//! its source). No row holds a tuned figure: each is the vendor's, divided
//! back to one multiprocessor and one clock.
//!
//! A spec states peaks, never curves: the matmul and attention curves are
//! flat at the peak, and the per-row ops run at the memory bandwidth. What
//! a spec does not state (a recurrent scan's rate, the launch cost) stays
//! unstated, and the estimate refuses by it as before.

/// One architecture's published dense operations a clock a multiprocessor
/// (CU on AMD), two a multiply-add, by the dtypes the estimate reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arch {
    /// The token a device's stored key names it by (`sm_89`, `gfx942`).
    pub name: &'static str,
    /// bf16 matrix (tensor core / matrix core) with fp32 accumulate.
    pub bf16: f64,
    /// fp16 matrix with fp32 accumulate.
    pub fp16: f64,
    /// fp8 matrix, where the architecture has it.
    pub fp8: Option<f64>,
    /// fp32 vector (non-matrix) arithmetic.
    pub fp32: f64,
    /// Where the figures come from, and the shipped part they are divided
    /// back from.
    pub source: &'static str,
    /// Where the memory bandwidth comes from on this architecture.
    pub bandwidth: Bandwidth,
}

/// Where an architecture's memory bandwidth comes from: the driver's stated
/// memory clock x bus width x 2, only where that has been checked against a
/// datasheet on a part of the architecture (and the check is cited); else
/// the datasheet's own figure, a part at a time, matched by the device name
/// the driver states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bandwidth {
    /// The driver's formula, and the check that verifies it.
    Driver { verified: &'static str },
    /// Each part's datasheet bandwidth: a part with none listed states no
    /// bandwidth.
    Parts(&'static [Part]),
}

/// One part's datasheet memory bandwidth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Part {
    /// A piece of the device name the driver states (`"MI300X"`).
    pub name: &'static str,
    pub gbps: f64,
    pub source: &'static str,
}

/// One row an architecture. Each figure is a shipped part's published dense
/// peak divided by its multiprocessor count and its boost clock, rounded to
/// the power of two the hardware issues.
pub const ARCHS: &[Arch] = &[
    // NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020):
    // A100 SXM4, 108 SMs at 1410 MHz, 312 TFLOPS dense BF16 / FP16 tensor,
    // 19.5 TFLOPS FP32; no FP8.
    Arch {
        name: "sm_80",
        bf16: 2048.0,
        fp16: 2048.0,
        fp8: None,
        fp32: 128.0,
        source: "NVIDIA A100 Tensor Core GPU Architecture whitepaper (A100 SXM4)",
        bandwidth: Bandwidth::Driver {
            verified: "A100 SXM4 40GB: deviceQuery states a 1215 MHz memory clock over 5120 bits, x 2 = 1555 GB/s, the A100 datasheet's 1555 GB/s",
        },
    },
    // NVIDIA Ampere GA102 GPU Architecture whitepaper:
    // RTX A6000, 84 SMs at 1800 MHz, 154.8 TFLOPS dense FP16 / BF16 tensor
    // with FP32 accumulate (GeForce parts run that accumulate at half
    // rate), 38.7 TFLOPS FP32; no FP8.
    Arch {
        name: "sm_86",
        bf16: 1024.0,
        fp16: 1024.0,
        fp8: None,
        fp32: 256.0,
        source: "NVIDIA Ampere GA102 GPU Architecture whitepaper (RTX A6000)",
        bandwidth: Bandwidth::Driver {
            verified: "RTX A6000: deviceQuery states 8001 MHz over 384 bits, x 2 = 768 GB/s, the RTX A6000 datasheet's 768 GB/s",
        },
    },
    // NVIDIA Ada GPU Architecture whitepaper / L40S
    // datasheet: L40S, 142 SMs at 2520 MHz, 362 TFLOPS dense FP16 / BF16
    // tensor with FP32 accumulate, 733 TFLOPS dense FP8, 91.6 TFLOPS FP32.
    Arch {
        name: "sm_89",
        bf16: 1024.0,
        fp16: 1024.0,
        fp8: Some(2048.0),
        fp32: 256.0,
        source: "NVIDIA Ada GPU Architecture whitepaper; L40S datasheet",
        bandwidth: Bandwidth::Driver {
            verified: "L40S: deviceQuery states 9001 MHz over 384 bits, x 2 = 864 GB/s, the L40S datasheet's 864 GB/s",
        },
    },
    // NVIDIA H100 Tensor Core GPU Architecture whitepaper: H100
    // SXM5, 132 SMs at 1830 MHz, 989 TFLOPS dense BF16 / FP16 tensor, 1979
    // TFLOPS dense FP8, 67 TFLOPS FP32.
    Arch {
        name: "sm_90",
        bf16: 4096.0,
        fp16: 4096.0,
        fp8: Some(8192.0),
        fp32: 256.0,
        source: "NVIDIA H100 Tensor Core GPU Architecture whitepaper (H100 SXM5)",
        bandwidth: Bandwidth::Driver {
            verified: "H100 SXM5: deviceQuery states 2619 MHz over 5120 bits, x 2 = 3352 GB/s, the H100 datasheet's 3.35 TB/s",
        },
    },
    // NVIDIA Blackwell Architecture Technical Brief / DGX B200 datasheet:
    // B200, 148 SMs at 1965 MHz, 2.25 PFLOPS dense BF16 / FP16 tensor, 4.5
    // PFLOPS dense FP8, 80 TFLOPS FP32.
    Arch {
        name: "sm_100",
        bf16: 8192.0,
        fp16: 8192.0,
        fp8: Some(16384.0),
        fp32: 256.0,
        source: "NVIDIA Blackwell Architecture Technical Brief (B200)",
        bandwidth: Bandwidth::Parts(&[Part {
            name: "B200",
            gbps: 8000.0,
            source: "NVIDIA DGX B200 datasheet: 8 TB/s HBM3e a GPU",
        }]),
    },
    // NVIDIA RTX Blackwell GPU Architecture whitepaper: RTX PRO
    // 6000 Blackwell, 188 SMs at 2617 MHz, 503.8 TFLOPS dense FP16 / BF16
    // tensor with FP32 accumulate (GeForce parts at half rate), 1007.6
    // TFLOPS dense FP8, 126 TFLOPS FP32.
    Arch {
        name: "sm_120",
        bf16: 1024.0,
        fp16: 1024.0,
        fp8: Some(2048.0),
        fp32: 256.0,
        source: "NVIDIA RTX Blackwell GPU Architecture whitepaper (RTX PRO 6000)",
        bandwidth: Bandwidth::Driver {
            verified: "RTX PRO 4500 Blackwell (on this project's box): the driver states 14001 MHz over 256 bits, x 2 = 896 GB/s, the RTX PRO 4500 Blackwell datasheet's 896 GB/s",
        },
    },
    // AMD CDNA 2 Architecture whitepaper / MI250X datasheet: MI250X, 220 CUs
    // at 1700 MHz, 383 TFLOPS dense BF16 / FP16 matrix, 47.9 TFLOPS FP32
    // vector; no FP8.
    Arch {
        name: "gfx90a",
        bf16: 1024.0,
        fp16: 1024.0,
        fp8: None,
        fp32: 128.0,
        source: "AMD CDNA 2 Architecture whitepaper; Instinct MI250X datasheet",
        bandwidth: Bandwidth::Parts(&[
            Part {
                name: "MI250X",
                gbps: 1638.4,
                source: "AMD Instinct MI250X datasheet: 3.2768 TB/s over its two GCDs, each a device",
            },
            Part {
                name: "MI250",
                gbps: 1638.4,
                source: "AMD Instinct MI250 datasheet: 3.2768 TB/s over its two GCDs, each a device",
            },
            Part {
                name: "MI210",
                gbps: 1638.4,
                source: "AMD Instinct MI210 datasheet: 1.6384 TB/s",
            },
        ]),
    },
    // AMD CDNA 3 Architecture whitepaper: MI300X, 304 CUs at 2100
    // MHz, 1307.4 TFLOPS dense BF16 / FP16 matrix, 2614.9 TFLOPS dense FP8,
    // 163.4 TFLOPS FP32 vector.
    Arch {
        name: "gfx942",
        bf16: 2048.0,
        fp16: 2048.0,
        fp8: Some(4096.0),
        fp32: 256.0,
        source: "AMD CDNA 3 Architecture whitepaper (Instinct MI300X)",
        bandwidth: Bandwidth::Parts(&[
            Part {
                name: "MI300X",
                gbps: 5300.0,
                source: "AMD Instinct MI300X datasheet: 5.3 TB/s (8 HBM3 stacks); HIP states 1300 MHz over 2048 bits, which x 2 is 665.6 GB/s",
            },
            Part {
                name: "MI300A",
                gbps: 5300.0,
                source: "AMD Instinct MI300A datasheet: 5.3 TB/s",
            },
            Part {
                name: "MI325X",
                gbps: 6000.0,
                source: "AMD Instinct MI325X datasheet: 6 TB/s (HBM3E)",
            },
        ]),
    },
    // AMD CDNA 4 Architecture whitepaper: MI355X, 256 CUs at 2400 MHz, 2.5
    // PFLOPS dense BF16 / FP16 matrix, 5.0 PFLOPS dense FP8, 157.3 TFLOPS
    // FP32 vector.
    Arch {
        name: "gfx950",
        bf16: 4096.0,
        fp16: 4096.0,
        fp8: Some(8192.0),
        fp32: 256.0,
        source: "AMD CDNA 4 Architecture whitepaper (Instinct MI355X)",
        bandwidth: Bandwidth::Parts(&[
            Part {
                name: "MI355X",
                gbps: 8000.0,
                source: "AMD Instinct MI355X datasheet: 8 TB/s (HBM3E)",
            },
            Part {
                name: "MI350X",
                gbps: 8000.0,
                source: "AMD Instinct MI350X datasheet: 8 TB/s (HBM3E)",
            },
        ]),
    },
];

/// What a device's driver states of itself, as the spec rates read it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stated {
    /// The device's name, as the driver states it.
    pub device: String,
    /// The architecture's token (`sm_89`, `gfx942`).
    pub arch: String,
    /// Multiprocessors (CUs).
    pub units: u32,
    /// Peak compute clock, kHz.
    pub clock_khz: u32,
    /// Peak memory clock, kHz.
    pub memory_clock_khz: u32,
    /// Memory bus width, bits.
    pub bus_bits: u32,
}

/// The row of [`ARCHS`] `arch` names.
#[must_use]
pub fn arch(arch: &str) -> Option<&'static Arch> {
    ARCHS.iter().find(|row| row.name == arch)
}

/// The roofline rates `stated` gives through its architecture's row, and the
/// line that names them; why not, where the architecture has no row or the
/// driver stated a zero.
pub fn rates(stated: &Stated) -> Result<(profile::rates::Rates, String), String> {
    let row = arch(&stated.arch).ok_or_else(|| {
        format!(
            "architecture `{}` has no row in the spec table (engine::spec::ARCHS)",
            stated.arch
        )
    })?;
    if stated.units == 0 || stated.clock_khz == 0 {
        return Err(format!(
            "the driver stated {} multiprocessors at {} kHz",
            stated.units, stated.clock_khz
        ));
    }
    let (gbps, bandwidth) = match row.bandwidth {
        Bandwidth::Driver { verified } => {
            if stated.memory_clock_khz == 0 || stated.bus_bits == 0 {
                return Err(format!(
                    "the driver stated a {} kHz memory clock over a {}-bit bus",
                    stated.memory_clock_khz, stated.bus_bits
                ));
            }
            let gbps = 2.0 * f64::from(stated.memory_clock_khz) * 1e3 * f64::from(stated.bus_bits)
                / 8.0
                / 1e9;
            (
                gbps,
                format!(
                    "a {}-bit bus at {:.0} MHz, twice a clock = {gbps:.0} GB/s (the driver's \
                     figures, verified on {verified})",
                    stated.bus_bits,
                    f64::from(stated.memory_clock_khz) / 1e3
                ),
            )
        }
        Bandwidth::Parts(parts) => {
            let part = parts
                .iter()
                .find(|part| stated.device.contains(part.name))
                .ok_or_else(|| {
                    format!(
                        "`{}`'s bus figures are not the datasheet's on {}, and no part of the \
                         table's matches the device's name",
                        stated.device, stated.arch
                    )
                })?;
            (
                part.gbps,
                format!("{:.0} GB/s, the datasheet's ({})", part.gbps, part.source),
            )
        }
    };
    let clock = f64::from(stated.clock_khz) * 1e3;
    let tflops = f64::from(stated.units) * clock * row.bf16 / 1e12;
    let flat = |t: f64| vec![(256, t), (8192, t)];
    let rates = profile::rates::Rates {
        gemm_tflops_by_rows: Some(vec![(1024, flat(tflops)), (4096, flat(tflops))]),
        attn_tflops_by_keys: Some(vec![(512, tflops), (8192, tflops)]),
        hbm_gbps: Some(gbps),
        elementwise_gbps: Some(gbps),
        few_row_tflops: Some(tflops),
        ..profile::rates::Rates::default()
    };
    let line = format!(
        "computed from spec: {} {} at {:.0} MHz x {} bf16 dense ops a clock = {tflops:.1} \
         TFLOP/s ({}); {bandwidth}",
        stated.units,
        stated.arch,
        clock / 1e6,
        row.bf16,
        row.source
    );
    Ok((rates, line))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row names one architecture once, cites its source, and states
    /// positive figures, fp8 at or past bf16 where it has fp8.
    #[test]
    fn every_row_reads_and_cites_its_source() {
        for (at, row) in ARCHS.iter().enumerate() {
            assert!(!row.source.is_empty(), "{}", row.name);
            assert!(
                row.bf16 > 0.0 && row.fp16 > 0.0 && row.fp32 > 0.0,
                "{}",
                row.name
            );
            assert!(row.fp8.is_none_or(|fp8| fp8 >= row.bf16), "{}", row.name);
            match row.bandwidth {
                Bandwidth::Driver { verified } => assert!(!verified.is_empty(), "{}", row.name),
                Bandwidth::Parts(parts) => assert!(
                    !parts.is_empty()
                        && parts
                            .iter()
                            .all(|p| !p.name.is_empty() && p.gbps > 0.0 && !p.source.is_empty()),
                    "{}",
                    row.name
                ),
            }
            assert!(
                ARCHS[..at].iter().all(|other| other.name != row.name),
                "{} twice",
                row.name
            );
        }
    }

    /// A device stating only what its driver does (an L40S's 142 SMs at
    /// 2520 MHz, 18000 MHz over 384 bits) gets the spec's rates, enough for
    /// the roofline, named.
    #[test]
    fn a_device_with_only_driver_stated_values_gets_spec_rates_named() {
        let stated = Stated {
            device: "NVIDIA L40S".into(),
            arch: "sm_89".into(),
            units: 142,
            clock_khz: 2_520_000,
            memory_clock_khz: 9_001_000,
            bus_bits: 384,
        };
        let (spec, line) = rates(&stated).expect("sm_89 has a row");
        let tflops = spec.gemm_tflops_by_rows.as_ref().unwrap()[0].1[0].1;
        assert!((tflops - 366.4).abs() < 0.5, "{tflops}");
        assert!(
            (spec.hbm_gbps.unwrap() - 864.0).abs() < 1.0,
            "{:?}",
            spec.hbm_gbps
        );
        assert!(line.starts_with("computed from spec"), "{line}");
        assert!(crate::roofline::Constants::of(&spec).is_ok());
        assert!(
            rates(&Stated {
                arch: "sm_1".into(),
                ..stated.clone()
            })
            .is_err()
        );
        assert!(
            rates(&Stated {
                bus_bits: 0,
                ..stated
            })
            .is_err()
        );
    }

    /// An MI300X stating what HIP states (1300 MHz over 2048 bits, 665.6
    /// GB/s by the driver's formula) gets its datasheet's 5.3 TB/s, named;
    /// a gfx942 part the table does not list gets no bandwidth, said why.
    #[test]
    fn an_mi300x_gets_its_datasheet_bandwidth_not_its_bus_figures() {
        let stated = Stated {
            device: "AMD Instinct MI300X VF".into(),
            arch: "gfx942".into(),
            units: 304,
            clock_khz: 2_100_000,
            memory_clock_khz: 1_300_000,
            bus_bits: 2048,
        };
        let (spec, line) = rates(&stated).expect("the MI300X is listed");
        assert_eq!(spec.hbm_gbps, Some(5300.0), "{line}");
        assert!(line.contains("the datasheet's"), "{line}");
        let other = Stated {
            device: "AMD Instinct MI3XX".into(),
            ..stated
        };
        let why = rates(&other).expect_err("no part matches");
        assert!(why.contains("no part of the table's matches"), "{why}");
    }
}
