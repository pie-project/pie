//! A bank in pinned host memory, read inside a program by indices the
//! program holds: each selected block reaches the device, bit for bit.
//! Skipped without a PJRT plugin.

use engine_xla::pjrt::{Api, Arg, Client, ElementType};

const E: i64 = 16;
const R: i64 = 64;
const C: i64 = 128;
const K: usize = 4;

fn program() -> String {
    let mut body = String::new();
    let mut parts = Vec::new();
    for k in 0..K {
        body += &format!(
            "    %i{k} = stablehlo.slice %arg1 [{k}:{n}] : (tensor<{K}xi32>) -> tensor<1xi32>\n\
             \x20   %s{k} = stablehlo.reshape %i{k} : (tensor<1xi32>) -> tensor<i32>\n\
             \x20   %d{k} = stablehlo.dynamic_slice %arg0, %s{k}, %z, %z, sizes = [1, {R}, {C}] : \
             (tensor<{E}x{R}x{C}xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x{R}x{C}xbf16>\n\
             \x20   %p{k} = stablehlo.custom_call @annotate_device_placement(%d{k}) {{has_side_effect = true, \
             mhlo.frontend_attributes = {{_xla_buffer_placement = \"device\"}}}} : \
             (tensor<1x{R}x{C}xbf16>) -> tensor<1x{R}x{C}xbf16>\n",
            n = k + 1
        );
        parts.push(format!("%p{k}"));
    }
    format!(
        "module @host_bank {{\n  func.func public @main(%arg0: tensor<{E}x{R}x{C}xbf16> \
         {{mhlo.memory_kind = \"pinned_host\"}}, %arg1: tensor<{K}xi32>) -> tensor<{K}x{R}x{C}xbf16> {{\n\
         \x20   %z = stablehlo.constant dense<0> : tensor<i32>\n{body}\
         \x20   %out = stablehlo.concatenate {}, dim = 0 : ({}) -> tensor<{K}x{R}x{C}xbf16>\n\
         \x20   return %out : tensor<{K}x{R}x{C}xbf16>\n  }}\n}}\n",
        parts.join(", "),
        vec![format!("tensor<1x{R}x{C}xbf16>"); K].join(", ")
    )
}

#[test]
fn a_pinned_host_bank_lands_its_selected_blocks() {
    let Some(_lock) = crate::common_dit::device() else {
        return;
    };
    let client = Client::create(Api::discover(None).expect("plugin")).expect("client");
    let device = client.device(0).expect("a device");
    let Some(host) = client.memory(device, "pinned_host").expect("memories") else {
        eprintln!("no pinned_host memory: skipped");
        return;
    };
    let bank: Vec<u8> = (0..(E * R * C * 2) as usize)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let on_host = client
        .upload_to(device, Some(host), &bank, ElementType::Bf16, &[E, R, C])
        .expect("the bank lands in host memory");
    let picks: [i32; K] = [3, 15, 0, 7];
    let idx: Vec<u8> = picks.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx = client
        .upload(device, &idx, ElementType::S32, &[K as i64])
        .expect("indices");
    let exe = client.compile(&program()).expect("the program compiles");
    let (outs, done) = exe
        .execute(device, vec![Arg::Keep(&on_host), Arg::Keep(&idx)])
        .expect("runs");
    done.wait().expect("done");
    let got = outs[0].download().expect("downloads");
    let block = (R * C * 2) as usize;
    let want: Vec<u8> = picks
        .iter()
        .flat_map(|&e| bank[e as usize * block..(e as usize + 1) * block].to_vec())
        .collect();
    assert_eq!(got, want, "the selected blocks, bit for bit");
    eprintln!("pinned_host bank: {K} of {E} blocks landed bit for bit");
}
