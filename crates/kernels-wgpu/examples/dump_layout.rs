//! The launch ABI, printed — what a shader author asks instead of counting.
//!
//! A `@binding(N)` in a `.wgsl` here is not free to choose. It is a function of
//! the row's operand order, computed by [`kernels_wgpu::bindings`], and the two
//! runs — storage buffers in `@group(0)`, scalars as fields of the one
//! `@group(1) @binding(0)` uniform block — are numbered INDEPENDENTLY. The
//! numbers therefore do not match Metal's, which numbers scalars alongside
//! buffers, and they do not match Vulkan's wherever a row's `InPacked` sits.
//! `kernels-vulkan`'s notes record sixty entrypoints that read a descriptor the
//! shell never wrote, every one of them a transcription across backends, and
//! its own `attn/kv_write.comp` declared the paged write page/offset at 9/10
//! where the row puts them at 10/11.
//!
//! Nothing at runtime reports that. `wgpu` checks a bind group against its
//! LAYOUT, and a layout derived from the same wrong reading agrees with it. So
//! the check has to be mechanical, and this is the tool:
//!
//! ```text
//! cargo run -p kernels-wgpu --example dump_layout                          # every entrypoint, terse
//! cargo run -p kernels-wgpu --example dump_layout -- kv_append_paged       # one row, per operand
//! cargo run -p kernels-wgpu --example dump_layout -- sdpa_paged_decode_bfloat16_d_64
//! ```
//!
//! The terse form is one line per entrypoint: `entrypoint <TAB> storage count
//! <TAB> uniform fields`, each field `name:offset:size`. The fields are NAMED
//! rather than counted because the dangerous mistake is order, not arity: a
//! block whose second field is `n_kv_heads` where the row's second scalar is
//! `page_size` reads a plausible number out of the wrong slot. The offset rides
//! along because a `vec2<u32>` aligns to 8, so a shader that declares `i32`
//! where the row says `Usize` keeps every name in order and still shifts every
//! field after it four bytes.

fn main() {
    let want: Vec<String> = std::env::args().skip(1).collect();

    if want.is_empty() {
        for entrypoint in kernels_wgpu::entrypoints() {
            let Some(row) = kernels_wgpu::sig(&entrypoint) else {
                // Not `expect`: the tree and the table are compared by
                // `tests/entrypoints.rs`, and a dump that panicked on a
                // disagreement would be useless for diagnosing it.
                println!("{entrypoint}\t(no row)");
                continue;
            };
            let storages = kernels_wgpu::storage_count(row);
            let uniforms: Vec<String> = kernels_wgpu::uniform_layout(row)
                .iter()
                .map(|f| format!("{}:{}:{}", f.name, f.offset, f.size))
                .collect();
            println!("{entrypoint}\t{storages}\t{}", uniforms.join(","));
        }
        return;
    }

    // An argument may be an ENTRYPOINT (`sdpa_paged_decode_bfloat16_d_64`, what
    // a `// pie:instantiate` line names) or a row SYMBOL (`sdpa_paged_decode`,
    // what the table names). Entrypoint first, because that is the name a
    // shader author has in front of them.
    for name in &want {
        let row = kernels_wgpu::sig(name)
            .or_else(|| kernels_wgpu::KERNELS.iter().find(|r| r.symbol == *name));
        let Some(row) = row else {
            println!("{name}\n   (no such entrypoint or row)");
            continue;
        };
        println!("{name}  [row `{}`]", row.symbol);
        if row.operands.is_empty() {
            // UNSTATED, which is not nullary: the row has not said what it
            // takes, so no layout can be derived from it and a shader for it
            // follows the lowered plan's own argument order instead. See
            // `.wiki/new-driver/vulkan.md` §13.
            println!("   (unstated -- the row names no operands)");
            continue;
        }
        let uniforms = kernels_wgpu::uniform_layout(row);
        for (op, binding) in row.operands.iter().zip(kernels_wgpu::bindings(row)) {
            let place = match binding {
                kernels_wgpu::Binding::Uniform(n) => {
                    let f = &uniforms[n as usize];
                    let wgsl = if f.split { "vec2<u32>" } else { "scalar" };
                    format!(
                        "Uniform field {n} at byte {} ({} wide, {wgsl})",
                        f.offset, f.size
                    )
                }
                kernels_wgpu::Binding::Storage(n) => format!("@group(0) @binding({n})"),
                kernels_wgpu::Binding::Packed => {
                    "Packed (a field of an earlier buffer's struct; no slot)".to_owned()
                }
            };
            println!("   {:<24} {:<10} {place}", op.name, format!("{:?}", op.ty));
        }
        println!(
            "   {:<24} {:<10} {} storage buffers, {} bytes of uniform block",
            "(totals)",
            "",
            kernels_wgpu::storage_count(row),
            kernels_wgpu::uniform_size(row),
        );
    }
}
