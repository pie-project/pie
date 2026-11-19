//! A plan's int tables (`attn::plan::upload`) reach the device as they were
//! when staged, though the host buffer they came from is written over (or
//! dropped) the moment the call returns and the stream is still busy with
//! earlier work: the copy reads a pinned slot of its own, not the caller's
//! bytes.
#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::linear::gemm;

#[test]
fn a_plans_tables_land_as_staged_though_the_host_buffer_is_reused() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    const PLANS: usize = 200;
    const BYTES: usize = 64 << 10;
    let dst = gpu.upload(
        &vec![0u8; PLANS * BYTES],
        1,
        (PLANS * BYTES) as u32,
        Dtype::U8,
    );
    // Work ahead of the copies on the same stream, so they queue behind it.
    let (m, n, k) = (2048u32, 4096u32, 4096u32);
    let a = gpu.tensor(
        &values((m * k) as usize, 7, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect::<Vec<_>>(),
        m,
        k,
        Dtype::Bf16,
    );
    let w = gpu.tensor(
        &values((n * k) as usize, 8, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect::<Vec<_>>(),
        n,
        k,
        Dtype::Bf16,
    );
    let mut y = gpu.tensor(&vec![0.0; (m * n) as usize], m, n, Dtype::Bf16);
    let mut host = vec![0u8; BYTES];
    for at in 0..PLANS {
        gpu.laid(&ctx, |ctx| gemm::lm_head(ctx, a, w, &mut y))
            .unwrap();
        host.fill(at as u8 + 1);
        kernels_rocm::attn::plan::upload(
            &ctx,
            "test.plan_upload",
            &host,
            dst.ptr + (at * BYTES) as u64,
        )
        .unwrap();
        // The caller's buffer is someone else's now.
        host.fill(0xee);
    }
    let landed: Vec<u8> = gpu.read_as(&dst);
    for at in 0..PLANS {
        let region = &landed[at * BYTES..(at + 1) * BYTES];
        assert!(
            region.iter().all(|&byte| byte == at as u8 + 1),
            "plan {at}'s tables landed as {:?}…, not as staged",
            &region[..8]
        );
    }
}
