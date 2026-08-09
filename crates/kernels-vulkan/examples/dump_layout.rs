//! The launch ABI, printed.
//!
//! Exists because the binding numbers in a `.comp` are not free to choose: they
//! are a function of the row's operand order, computed by
//! [`kernels_vulkan::bindings`]. A shader author guessing them is how sixty
//! entrypoints came to read a descriptor the shell never writes -- see the
//! comment in `kernels/quant/qmv.comp`. Ask this instead of counting by hand.
//!
//! ```text
//! cargo run -p kernels-vulkan --example dump_layout                     # every entrypoint, terse
//! cargo run -p kernels-vulkan --example dump_layout -- kv_append_paged  # one row, per operand
//! ```
//!
//! `scripts/vulkan-kernel-audit.py --bindings` reads the terse form and checks
//! it against what the shaders actually declare, which is the half a human
//! reading two files cannot be trusted to do 480 times.

fn main() {
    let want: Vec<String> = std::env::args().skip(1).collect();

    if want.is_empty() {
        // `entrypoint <TAB> buffer count <TAB> push fields`, one line per
        // entrypoint, sorted -- a format the audit script reads without needing
        // a parser. Each push field is `name:offset:size`. The push fields are
        // NAMED rather than counted because the dangerous mistake is order, not
        // arity: a shader whose push block puts `row_stride` where the row puts
        // `out_vec_size` reads a real number from the wrong slot, which no
        // counting check would see.
        //
        // The offset and the width ride along because the name check cannot see
        // the OTHER silent mistake. A push block is std430, so a field's byte
        // offset depends on every width before it; a shader that declares `int`
        // where the row says `Usize` keeps every name in the right order and
        // still shifts all of them four bytes. `kernels_vulkan::push_layout`
        // computes the row's side, and the audit compares it against the types
        // the shader actually declares.
        for entrypoint in kernels_vulkan::entrypoints() {
            let row = kernels::sig_in(kernels_vulkan::KERNELS, &entrypoint)
                .expect("every entrypoint resolves; tests/entrypoints.rs pins that");
            let buffers = kernels_vulkan::buffer_count(row);
            let pushes: Vec<String> = kernels_vulkan::push_layout(row)
                .iter()
                .map(|f| format!("{}:{}:{}", f.name, f.offset, f.size))
                .collect();
            println!("{entrypoint}\t{buffers}\t{}", pushes.join(","));
        }
        return;
    }

    // An argument may be either an ENTRYPOINT (`sdpa_paged_decode_bfloat16_d_64`,
    // what a directive and the audit script name) or a row SYMBOL
    // (`sdpa_paged_decode`, what the table names). Entrypoint is tried first,
    // because that is the name a shader author has in front of them; matching
    // only the symbol printed nothing at all for every name they would think
    // to type.
    for name in &want {
        let by_entrypoint = kernels::sig_in(kernels_vulkan::KERNELS, name);
        let row = match by_entrypoint {
            Some(row) => Some(row),
            None => kernels_vulkan::KERNELS.iter().find(|r| r.symbol == *name),
        };
        let Some(row) = row else {
            println!("{name}\n   (no such entrypoint or row)");
            continue;
        };
        println!("{name}");
        let pushes = kernels_vulkan::push_layout(row);
        if row.operands.is_empty() {
            // UNSTATED, which is not nullary: the row has not said what it
            // takes, so no layout can be derived from it.
            println!("   (unstated -- the row names no operands)");
            continue;
        }
        for (op, binding) in row.operands.iter().zip(kernels_vulkan::bindings(row)) {
            // A `Push(n)` is a field index, and what a driver needs is a byte
            // offset -- which is not `n * 4`, because the block is std430 and
            // an eight-byte scalar is aligned to eight. Print the offset beside
            // the index so nobody derives it again by hand.
            let place = match binding {
                kernels_vulkan::Binding::Push(n) => {
                    let f = &pushes[n as usize];
                    format!("Push({n}) at byte {} ({} wide)", f.offset, f.size)
                }
                other => format!("{other:?}"),
            };
            println!("   {:<18} {place}", op.name);
        }
        println!(
            "   {:<18} {} bytes of push constants",
            "(block)",
            kernels_vulkan::push_size(row)
        );
    }
}
