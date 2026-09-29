//! The PJRT layer compiles what the StableHLO builder prints and runs it on
//! whatever plugin `PIE_XLA_PLUGIN` / `TPU_LIBRARY_PATH` names; skipped when
//! none is present.

use std::sync::Arc;

use engine_xla::pjrt::{Api, Arg, Client, ElementType};
use kernels_xla::hlo::{Elem, Fold, Func, Ty, bf16_bits};

fn client() -> Option<Client> {
    let api: Arc<Api> = match Api::discover(None) {
        Ok(api) => api,
        Err(e) => {
            eprintln!("skipped: {e}");
            return None;
        }
    };
    Some(Client::create(api).expect("client"))
}

fn bf16_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn a_matmul_and_a_reduce_answer_the_host() {
    let Some(client) = client() else { return };
    let dev = client.devices()[0];
    eprintln!("{:?} on {}", client, client.device_kind(dev).unwrap());

    let (m, n, k) = (4i64, 3i64, 8i64);
    let mut f = Func::new("main");
    let x = f.param(Ty::new(Elem::Bf16, &[m, k]), None);
    let w = f.param(Ty::new(Elem::Bf16, &[n, k]), None);
    let y = f.matmul_nt(x, w, Elem::F32).unwrap();
    let s = f.reduce(y, &[1], Fold::Sum).unwrap();
    let (arg, _) = f.argmax(y, 1, Elem::I32).unwrap();
    let text = f.module("smoke", &[y, s, arg]);
    let exe = client.compile(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(exe.outputs(), 3);

    let xs: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
    let ws: Vec<f32> = (0..n * k).map(|i| (i % 3) as f32 - 1.0).collect();
    let xb = client.upload(dev, &bf16_bytes(&xs), ElementType::Bf16, &[m, k]).unwrap();
    let wb = client.upload(dev, &bf16_bytes(&ws), ElementType::Bf16, &[n, k]).unwrap();
    let (outs, done) = exe.execute(dev, vec![Arg::Keep(&xb), Arg::Keep(&wb)]).unwrap();
    done.wait().unwrap();

    let got = f32s(&outs[0].download().unwrap());
    let mut want = vec![0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            want[(i * n + j) as usize] =
                (0..k).map(|t| xs[(i * k + t) as usize] * ws[(j * k + t) as usize]).sum();
        }
    }
    assert_eq!(got, want);
    let sums = f32s(&outs[1].download().unwrap());
    for i in 0..m as usize {
        assert_eq!(sums[i], want[i * n as usize..(i + 1) * n as usize].iter().sum::<f32>());
    }
    let args: Vec<i32> = outs[2]
        .download()
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for i in 0..m as usize {
        let row = &want[i * n as usize..(i + 1) * n as usize];
        let best = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(args[i], row.iter().position(|&v| v == best).unwrap() as i32);
    }
}

#[test]
fn a_donated_buffer_is_updated_in_place() {
    let Some(client) = client() else { return };
    let dev = client.devices()[0];
    let mut f = Func::new("main");
    let pool = f.param(Ty::new(Elem::F32, &[8, 4]), Some(0));
    let row = f.param(Ty::new(Elem::F32, &[1, 4]), None);
    let at = f.param(Ty::scalar(Elem::I32), None);
    let zero = f.const_i(Elem::I32, 0, &[]);
    let out = f.dynamic_update_slice(pool, row, &[at, zero]).unwrap();
    let exe = client.compile(&f.module("dus", &[out])).unwrap();

    let pool_b = client
        .upload(dev, &vec![0u8; 8 * 4 * 4], ElementType::F32, &[8, 4])
        .unwrap();
    let row_b = client
        .upload(dev, &[1f32, 2., 3., 4.].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(), ElementType::F32, &[1, 4])
        .unwrap();
    let at_b = client.upload(dev, &5i32.to_le_bytes(), ElementType::S32, &[]).unwrap();
    let (outs, done) = exe
        .execute(dev, vec![Arg::Donate(pool_b), Arg::Keep(&row_b), Arg::Keep(&at_b)])
        .unwrap();
    done.wait().unwrap();
    let got = f32s(&outs[0].download().unwrap());
    assert_eq!(&got[20..24], &[1., 2., 3., 4.]);
    assert!(got[..20].iter().chain(&got[24..]).all(|&v| v == 0.0));
}
