//! Differential parity for the recurrent state cache.
//!
//! Mirrors `tests/oracle/recurrent/oracle.cpp` exactly: same grid, same
//! transcript, same order. The golden is the **C++'s** FNV-1a 64 over its own
//! output, so this test cannot pass by agreeing with itself.
//!
//! Regenerate with `tests/oracle/recurrent/run.sh`; check the mutation
//! sensitivity of the pin with `tests/oracle/recurrent/mutate.sh`.
//!
//! Set `RS_RUST_OUT=/tmp/rust.txt` here and `RS_ORACLE_OUT=/tmp/cpp.txt` there
//! to diff the two transcripts line by line.

#![cfg(feature = "_cuda")]

use driver_cuda::gpu::pools::recurrent_state_cache::{
    Buffer, PoolDims, RecurrentStateCache, StateOp, recurrent_state_bf16_default,
    stash_tokens_cap,
};

const GOLDEN_FNV1A64: u64 = 0x96718f7ba4005865;
const GOLDEN_ROWS: usize = 188;

/// The unit separator the oracle uses between fields of one row.
const US: char = '\u{1f}';

/// Buffers are named by allocation ordinal, not by role, because the order is
/// part of the contract: a stack with no linear layers skips the conv and
/// recurrent slabs, so its MTP tier is `buf0`.
struct Naming {
    conv: Option<usize>,
    rec: Option<usize>,
    mtp: Option<usize>,
}

impl Naming {
    fn of(c: &RecurrentStateCache) -> Self {
        let mut next = 0;
        let mut take = |present: bool| {
            present.then(|| {
                let n = next;
                next += 1;
                n
            })
        };
        let linear = c.layout().num_linear_layers() > 0;
        Self {
            conv: take(linear && c.layout().conv_slot_stride_bytes() > 0),
            rec: take(linear && c.layout().recurrent_slot_stride_bytes() > 0),
            mtp: take(c.has_mtp_hidden()),
        }
    }

    fn name(&self, buffer: Buffer) -> Option<usize> {
        match buffer {
            Buffer::Conv => self.conv,
            Buffer::Recurrent => self.rec,
            Buffer::MtpHidden => self.mtp,
        }
    }

    /// Render an offset the way the recorder's `where()` does. A buffer that
    /// was never allocated has no base, so every address in it is `null` --
    /// which is exactly what the C++ hands out, since a zero-element
    /// `DeviceBuffer` never allocates.
    fn at(&self, buffer: Buffer, offset: u64) -> String {
        self.name(buffer)
            .map_or_else(|| "null".to_owned(), |n| format!("buf{n}+{offset}"))
    }
}

fn op_row(n: &Naming, op: &StateOp) -> String {
    match *op {
        StateOp::Memset {
            buffer,
            offset,
            len,
        } => format!("memset {} val=0 len={len}", n.at(buffer, offset)),
        StateOp::Memset2D {
            buffer,
            offset,
            pitch,
            width,
            rows,
        } => format!(
            "memset2d {} val=0 pitch={pitch} width={width} rows={rows}",
            n.at(buffer, offset)
        ),
        StateOp::Memcpy {
            buffer,
            dst,
            src,
            len,
        } => format!(
            "memcpy dst={} src={} len={len} kind=3",
            n.at(buffer, dst),
            n.at(buffer, src)
        ),
        StateOp::Memcpy2D {
            buffer,
            dst,
            src,
            pitch,
            width,
            rows,
        } => format!(
            "memcpy2d dst={} src={} dpitch={pitch} spitch={pitch} width={width} rows={rows} kind=3",
            n.at(buffer, dst),
            n.at(buffer, src)
        ),
        StateOp::ZeroSlotsIfFresh {
            buffer,
            slot_bytes,
            row_pitch,
            rows,
            request_count,
        } => format!(
            "zerofresh {} slot={slot_bytes} pitch={row_pitch} rows={rows} reqs={request_count}",
            n.at(buffer, 0)
        ),
    }
}

fn ops_field(n: &Naming, ops: &[StateOp]) -> String {
    ops.iter()
        .map(|o| op_row(n, o))
        .collect::<Vec<_>>()
        .join(&US.to_string())
}

/// Render a `Result<Option<u64>>` accessor the way the oracle does.
///
/// Includes its own leading separator: the oracle writes `conv=<addr>` on
/// success but `conv!<message>` on an exception, so the `=` is part of the
/// success case rather than part of the label.
fn addr(n: &Naming, buffer: Buffer, r: driver_cuda::Result<Option<u64>>) -> String {
    match r {
        Ok(Some(off)) => format!("={}", n.at(buffer, off)),
        Ok(None) => "=null".to_owned(),
        Err(e) => format!("!{e}"),
    }
}

struct Sweep {
    rows: Vec<String>,
}

impl Sweep {
    fn emit(&mut self, id: &str, body: &str) {
        self.rows.push(format!("{id}|{body}"));
    }

    fn fields(&mut self, id: &str, fields: &[String]) {
        self.rows.push(format!("{id}|{}", fields.join(&US.to_string())));
    }
}

fn layers_label(linear: &[bool]) -> String {
    if linear.is_empty() {
        return "-".to_owned();
    }
    linear.iter().map(|&b| if b { 'L' } else { '.' }).collect()
}

/// The oracle's `scalar_dims` field: every scalar accessor the generated
/// bodies read off the live cache, through the cache's own forwarding
/// accessors rather than `layout()`, because those forwards are the surface
/// under test.
fn scalar_dims(c: &RecurrentStateCache) -> String {
    format!(
        "dims cd={} ck={} vh={} kd={} vd={} hs={} nl={} ms={} bf16={} \
         css={} rsf={} rsb={} vst={} vsh={}",
        c.conv_dim(),
        c.conv_kernel(),
        c.v_heads(),
        c.head_k_dim(),
        c.head_v_dim(),
        c.hidden_size(),
        c.num_layers(),
        c.max_slots(),
        i32::from(c.recurrent_state_bf16()),
        c.conv_slot_stride_bytes(),
        c.recurrent_slot_stride_floats(),
        c.recurrent_slot_stride_bytes(),
        c.verify_stash_max_tokens(),
        c.verify_stash_hidden(),
    )
}

fn shape_report(c: &RecurrentStateCache) -> String {
    let l = c.layout();
    format!(
        "layers={} slots={} convdim={} convk={} vh={} kd={} vd={} hidden={} bf16={} \
         convstride={} recfloats={} recstride={} frozen={}",
        l.num_layers(),
        l.max_slots(),
        l.conv_dim(),
        l.conv_kernel(),
        l.v_heads(),
        l.head_k_dim(),
        l.head_v_dim(),
        l.hidden_size(),
        i32::from(l.recurrent_is_bf16()),
        l.conv_slot_stride_bytes(),
        l.recurrent_slot_stride_elems(),
        l.recurrent_slot_stride_bytes(),
        i32::from(c.verify_frozen()),
    )
}

/// The construction transcript: the allocations, then the `reset()` the
/// constructor issues before returning.
fn ctor_field(c: &RecurrentStateCache, n: &Naming) -> String {
    let mut rows = Vec::new();
    let layers = u64::from(c.layout().num_linear_layers());
    let slots = u64::from(c.layout().max_slots());
    let mut alloc = |buffer: Buffer, bytes: u64| {
        if let Some(name) = n.name(buffer) {
            rows.push(format!("alloc buf{name} bytes={bytes}"));
        }
    };
    alloc(
        Buffer::Conv,
        c.layout().conv_slot_stride_bytes() * slots * layers,
    );
    alloc(
        Buffer::Recurrent,
        c.layout().recurrent_slot_stride_bytes() * slots * layers,
    );
    alloc(
        Buffer::MtpHidden,
        u64::from(c.layout().hidden_size()) * slots * 2,
    );
    for op in &c.reset() {
        rows.push(op_row(n, op));
    }
    rows.join(&US.to_string())
}

fn accessors(c: &RecurrentStateCache, n: &Naming) -> String {
    let mut rows = Vec::new();
    let nl = i32::try_from(c.layout().num_layers()).unwrap();
    let slots = i32::try_from(c.layout().max_slots()).unwrap();
    for layer in -1..=nl {
        for slot in -1..=slots {
            let mut row = format!("L{layer}/S{slot} ");
            row += &format!("conv{}", addr(n, Buffer::Conv, c.conv_state(layer, slot)));
            row += &format!(
                " rec{}",
                addr(n, Buffer::Recurrent, c.recurrent_state_raw(layer, slot))
            );
            row += &format!(
                " recf{}",
                addr(n, Buffer::Recurrent, c.recurrent_state_f32(layer, slot))
            );
            if layer == 0 {
                row += &format!(
                    " mtp{}",
                    addr(n, Buffer::MtpHidden, c.mtp_pending_hidden(slot))
                );
            }
            rows.push(row);
        }
    }
    rows.join(&US.to_string())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    sw: &mut Sweep,
    id: &str,
    linear: &[bool],
    conv_dim: i32,
    conv_kernel: i32,
    v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    hidden_size: i32,
    max_slots: i32,
    force_bf16: bool,
) {
    // `hidden_size` and `max_slots` are passed RAW, negatives included: their
    // clamps are the library's job, and doing them here would prove the test
    // clamps rather than that the port does.
    let cd = conv_dim.unsigned_abs();
    let ck = conv_kernel.unsigned_abs();
    let vh = v_heads.unsigned_abs();
    let kd = head_k_dim.unsigned_abs();
    let vd = head_v_dim.unsigned_abs();

    let mut c = if force_bf16 {
        RecurrentStateCache::allocate_bf16_recurrent(linear, cd, ck, vh, kd, vd, max_slots)
    } else {
        RecurrentStateCache::allocate(linear, cd, ck, vh, kd, vd, hidden_size, max_slots)
    };
    let n = Naming::of(&c);

    let mut fields = vec![shape_report(&c), ctor_field(&c, &n)];
    fields.push(ops_field(&n, &c.reset()));

    let slots = i32::try_from(c.layout().max_slots()).unwrap();
    for slot in [-1, 0, 1, slots - 1, slots] {
        // The message comes from the library, never from here: reciting it in
        // the test would leave the real one unchecked.
        let f = match c.reset_slot(slot) {
            Ok(ops) => ops_field(&n, &ops),
            Err(e) => format!("!{e}"),
        };
        fields.push(format!("slot{slot} {f}"));
    }

    for (src, dst) in [
        (0, 0),
        (0, 1),
        (1, 0),
        (-1, 0),
        (0, -1),
        (0, slots),
        (slots - 1, 0),
    ] {
        let full = match c.copy_slot_d2d(src, dst) {
            Ok(ops) => ops_field(&n, &ops),
            Err(e) => format!("!{e}"),
        };
        let lin = match c.copy_linear_state_slot_d2d(src, dst) {
            Ok(ops) => ops_field(&n, &ops),
            Err(e) => format!("!{e}"),
        };
        fields.push(format!("cp{src}->{dst} {full} lin {lin}"));
    }

    let ids = [0i32, 1, -1, 2];
    let fresh = [1u8, 0, 1, 1];
    for count in [0i32, 1, 4, -1] {
        let ops = c.reset_slots_if_fresh(Some(&ids), Some(&fresh), count);
        fields.push(format!("fresh{count} {}", ops_field(&n, &ops)));
    }
    // Null device arrays. The guard is the library's, not the test's.
    fields.push(format!(
        "freshnull {}",
        ops_field(&n, &c.reset_slots_if_fresh(None, Some(&fresh), 4))
    ));
    fields.push(format!(
        "freshnullf {}",
        ops_field(&n, &c.reset_slots_if_fresh(Some(&ids), None, 4))
    ));

    fields.push(accessors(&c, &n));

    c.set_verify_frozen(true);
    fields.push(format!("frozen={}", i32::from(c.verify_frozen())));
    c.set_verify_frozen(false);

    fields.push(scalar_dims(&c));

    sw.fields(id, &fields);
}

#[allow(clippy::too_many_arguments)]
fn run_tiers(
    sw: &mut Sweep,
    id: &str,
    linear: &[bool],
    max_slots: i32,
    stash_tokens: i32,
    stash_hidden: i32,
    pool_tokens: i32,
    pool_hidden: i32,
    pool_slots: i32,
) {
    let mut c = RecurrentStateCache::allocate(linear, 64, 4, 2, 8, 16, 32, max_slots);
    let base = Naming::of(&c);
    // The stash and the pool are allocated after the three state buffers, so
    // their ordinals continue the same sequence.
    let mut next = [base.conv, base.rec, base.mtp].iter().flatten().count();

    let mut fields = Vec::new();

    c.configure_verify_hidden_stash(
        stash_tokens.max(0).unsigned_abs(),
        stash_hidden.max(0).unsigned_abs(),
        None,
    );
    let stash_buf = c.verify_hidden_stash_enabled().then(|| {
        let n = next;
        next += 1;
        n
    });
    fields.push(format!(
        "stash {}",
        stash_buf.map_or_else(String::new, |n| format!(
            "alloc buf{n} bytes={}",
            c.verify_hidden_stash_bytes()
        ))
    ));
    let stash = c.verify_stash();
    fields.push(format!(
        "on={} tok={} hid={}",
        i32::from(c.verify_hidden_stash_enabled()),
        stash.map_or(0, |d| d.max_tokens),
        stash.map_or(0, |d| d.hidden),
    ));
    let nl = i32::try_from(c.layout().num_layers()).unwrap();
    fields.push(
        (-1..=nl)
            .map(|i| {
                let off = u32::try_from(i)
                    .ok()
                    .and_then(|i| c.verify_hidden_stash_layer(i));
                format!(
                    "s{i}={}",
                    off.and_then(|o| stash_buf.map(|n| format!("buf{n}+{o}")))
                        .unwrap_or_else(|| "null".to_owned())
                )
            })
            .collect::<Vec<_>>()
            .join(&US.to_string()),
    );

    c.configure_rs_buffer_pool(
        pool_tokens.max(0).unsigned_abs(),
        pool_hidden.max(0).unsigned_abs(),
        pool_slots.max(0).unsigned_abs(),
    );
    let pool_buf = c.rs_buffer_pool_enabled().then(|| {
        let n = next;
        next += 1;
        n
    });
    fields.push(format!(
        "pool {}",
        pool_buf.map_or_else(String::new, |n| format!(
            "alloc buf{n} bytes={}",
            c.rs_buffer_pool_bytes()
        ))
    ));
    let pool = c.rs_buffer_pool().unwrap_or(PoolDims {
        page_tokens: 0,
        hidden: 0,
        num_slots: 0,
    });
    fields.push(format!(
        "on={} tok={} hid={} slots={}",
        i32::from(c.rs_buffer_pool_enabled()),
        pool.page_tokens,
        pool.hidden,
        pool.num_slots,
    ));
    let mut rows = Vec::new();
    for i in -1..=nl {
        for s in -1..=pool_slots {
            let off = u32::try_from(i).ok().and_then(|i| {
                u32::try_from(s)
                    .ok()
                    .and_then(|s| c.rs_buffer_slab(i, s))
            });
            rows.push(format!(
                "p{i}/{s}={}",
                off.and_then(|o| pool_buf.map(|n| format!("buf{n}+{o}")))
                    .unwrap_or_else(|| "null".to_owned())
            ));
        }
    }
    fields.push(rows.join(&US.to_string()));

    fields.push(scalar_dims(&c));

    sw.fields(id, &fields);
}

/// The `PIE_RS_STASH_TOKENS` cap, swept separately.
///
/// The C++ calls `getenv` inside `configure_verify_hidden_stash` on every
/// call, so the value can change between cases in one process. Here the read
/// is hoisted into `stash_tokens_cap()` and injected, which keeps the parsing
/// under test without making the cache reach into the environment.
fn run_stash_cap(sw: &mut Sweep, id: &str, value: Option<&str>, max_tokens: u32) {
    // SAFETY-adjacent: this sweep is single-threaded and is the only reader of
    // the variable in this process.
    match value {
        Some(v) => unsafe { std::env::set_var("PIE_RS_STASH_TOKENS", v) },
        None => unsafe { std::env::remove_var("PIE_RS_STASH_TOKENS") },
    }
    let mut c = RecurrentStateCache::allocate(&[true, false, true], 64, 4, 2, 8, 16, 32, 2);
    let base = Naming::of(&c);
    let next = [base.conv, base.rec, base.mtp].iter().flatten().count();
    c.configure_verify_hidden_stash(max_tokens, 5, stash_tokens_cap());
    let alloc = if c.verify_hidden_stash_enabled() {
        format!("alloc buf{next} bytes={}", c.verify_hidden_stash_bytes())
    } else {
        String::new()
    };
    let d = c.verify_stash();
    sw.fields(
        id,
        &[
            alloc,
            format!(
                "tok={} hid={} on={}",
                d.map_or(0, |d| d.max_tokens),
                d.map_or(0, |d| d.hidden),
                i32::from(c.verify_hidden_stash_enabled()),
            ),
        ],
    );
    unsafe { std::env::remove_var("PIE_RS_STASH_TOKENS") };
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_recurrent_state_cache_matches_the_cpp_byte_for_byte() {
    let mut sw = Sweep { rows: Vec::new() };

    sw.emit(
        "bf16default",
        &i32::from(recurrent_state_bf16_default()).to_string(),
    );

    let patterns: Vec<Vec<bool>> = vec![
        vec![],
        vec![false],
        vec![true],
        vec![true, true, true, true],
        vec![false, false, false],
        vec![true, false, true, false, true, false],
        vec![false, true, false, true, false, true],
        vec![false, false, true, true, false, false, true],
        vec![true, false, false, false, false, false, false, true],
    ];
    for p in &patterns {
        for slots in [0, 1, 4] {
            let id = format!("pat/{}/{slots}", layers_label(p));
            run_case(&mut sw, &id, p, 128, 4, 2, 8, 16, 64, slots, false);
        }
    }

    let mixed = [true, false, true, false, true];
    for conv_dim in [0, 1, 96, 4096] {
        for conv_kernel in [0, 1, 4] {
            let id = format!("geo/c{conv_dim}x{conv_kernel}");
            run_case(
                &mut sw, &id, &mixed, conv_dim, conv_kernel, 3, 8, 16, 32, 3, false,
            );
        }
    }
    for v_heads in [0, 1, 5] {
        for kd in [0, 8, 128] {
            for vd in [0, 16, 64] {
                let id = format!("geo/v{v_heads}/{kd}/{vd}");
                run_case(&mut sw, &id, &mixed, 96, 4, v_heads, kd, vd, 32, 2, false);
            }
        }
    }

    for hidden in [-8, 0, 1, 2048] {
        for slots in [1, 3] {
            let id = format!("mtp/{hidden}/{slots}");
            run_case(&mut sw, &id, &mixed, 96, 4, 2, 8, 16, hidden, slots, false);
        }
    }

    for slots in [-4, -1, 0, 1, 2] {
        let id = format!("slots/{slots}");
        run_case(&mut sw, &id, &mixed, 96, 4, 2, 8, 16, 32, slots, false);
    }

    for p in &patterns {
        let id = format!("bf16/{}", layers_label(p));
        run_case(&mut sw, &id, p, 96, 4, 2, 8, 16, 2048, 3, true);
    }

    for p in [&vec![][..], &[false, false][..], &mixed[..]] {
        for st in [0, 1, 7] {
            for sh in [0, 5] {
                let id = format!("tier/{}/{st}x{sh}", layers_label(p));
                run_tiers(&mut sw, &id, p, 2, st, sh, 3, 4, 2);
            }
        }
        for pt in [0, 3] {
            for ph in [0, 4] {
                for ps in [0, 1, 5] {
                    let id = format!("tier/{}/pool{pt}x{ph}x{ps}", layers_label(p));
                    run_tiers(&mut sw, &id, p, 2, 6, 8, pt, ph, ps);
                }
            }
        }
    }

    let caps = [
        None,
        Some(""),
        Some("0"),
        Some("-1"),
        Some("1"),
        Some("7"),
        Some("256"),
        Some("8192"),
        Some("99999"),
        Some("abc"),
        Some("12x"),
        Some(" 7"),
        Some("+9"),
        Some("0007"),
        Some("2147483648"),
    ];
    for cap in caps {
        for mt in [0u32, 8, 8192] {
            let label = match cap {
                None => "<unset>",
                Some("") => "<empty>",
                Some(v) => v,
            };
            let id = format!("cap/{label}/{mt}");
            run_stash_cap(&mut sw, &id, cap, mt);
        }
    }

    let mut text = sw.rows.join("\n");
    text.push('\n');
    if let Ok(path) = std::env::var("RS_RUST_OUT") {
        std::fs::write(&path, &text).unwrap();
        eprintln!("transcript written to {path}");
    }

    assert_eq!(sw.rows.len(), GOLDEN_ROWS, "row count drifted");
    assert_eq!(
        fnv1a64(text.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript diverged from the C++; \
         set RS_RUST_OUT and RS_ORACLE_OUT and diff them"
    );
}
