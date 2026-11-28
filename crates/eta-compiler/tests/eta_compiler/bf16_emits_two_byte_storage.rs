//! bf16 stores on the SIMT backends (CUDA, HIP: two bytes an element) and
//! on WGSL (two to a word), each rounded once to nearest even as the IR
//! rounds it.

use eta_compiler::codegen::simt::runtime::runtime_source;
use eta_compiler::codegen::wgsl::RUNTIME;

#[test]
fn the_runtimes_round_as_the_ir_does() {
    assert!(RUNTIME.contains("return (b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u;"));
    assert!(RUNTIME.contains("return ((b >> 16u) | 0x40u) & 0xFFFFu;"));
    assert!(RUNTIME.contains("const DT_BF16 : u32 = 5u;"));
    let runtime = runtime_source(eta_compiler::codegen::cuda::DIALECT);
    // The one bf16 store is a 16-bit store of the element's own two bytes,
    // not a read-modify-write of the word around it.
    assert!(
        runtime.contains("reinterpret_cast<m1_u16*>(data)[index] = (m1_u16)m1_bf16_bits(value);")
    );
    assert!(runtime.contains("#define M1_DT_BF16 5u"));
    assert_eq!(eta_ir::types::WIRE_BF16, 5);
    // The rounding, transcribed: it must be eta_ir's.
    assert!(
        runtime
            .contains("if ((b & 0x7fffffffu) > 0x7f800000u) return ((b >> 16) | 0x40u) & 0xffffu;")
    );
    assert!(runtime.contains("return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;"));
    fn simt_round(x: f32) -> u32 {
        let b = x.to_bits();
        if (b & 0x7fff_ffff) > 0x7f80_0000 {
            return ((b >> 16) | 0x40) & 0xffff;
        }
        b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16
    }
    let mut x = 0x0001_2345u32;
    for _ in 0..100_000 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let f = f32::from_bits(x);
        assert_eq!(
            simt_round(f),
            u32::from(eta_ir::types::bf16_bits(f)),
            "{x:#010x}"
        );
    }
}
