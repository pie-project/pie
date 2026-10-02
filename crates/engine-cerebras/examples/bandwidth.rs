//! Host-to-device bandwidth on the simulator by rectangle shape and memcpy
//! channels: compiles a program holding one array per PE and times one
//! `memcpy_h2d` and one `memcpy_d2h` over the rectangle.
//!
//! ```text
//! cargo run --release -p engine-cerebras --example bandwidth -- <w> <h> <channels> [words_per_pe]
//! ```

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use engine_cerebras::sdk::{
    Arch, Compile, Cslc, DataType, Order, Platform, Rect, Runtime, Sdk, Simulator, Target,
};

fn main() {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("a number"))
        .collect();
    let w = args.first().copied().unwrap_or(8);
    let h = args.get(1).copied().unwrap_or(1);
    let channels = args.get(2).copied().unwrap_or(1);
    let per_pe = args.get(3).copied().unwrap_or(4096);
    let dir = PathBuf::from(format!("/tmp/pie-cerebras-bw-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let mut pe = String::new();
    pe.push_str("param memcpy_params;\nparam pe_id: i16;\nconst sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
    let _ = writeln!(pe, "var buf: [{per_pe}]f32;\nvar buf_ptr: [*]f32 = &buf;");
    // Each PE stamps its row-major index on its first word: which block a
    // PE receives from a row-major copy shows on the way back.
    pe.push_str(
        "fn run() void {\n  buf[0] = @as(f32, pe_id);\n  sys_mod.unblock_cmd_stream();\n}\n",
    );
    pe.push_str("comptime {\n  @export_symbol(buf_ptr, \"buf\");\n  @export_symbol(run);\n}\n");
    let mut layout = String::new();
    let _ = writeln!(
        layout,
        "const memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {w}, .height = {h} }});\n"
    );
    let _ = writeln!(layout, "layout {{\n  @set_rectangle({w}, {h});");
    let _ = writeln!(
        layout,
        "  for (@range(i16, {h})) |y| {{\n    for (@range(i16, {w})) |x| {{\n      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = y * {w} + x }});\n    }}\n  }}"
    );
    layout.push_str(
        "  @export_name(\"buf\", [*]f32, true);\n  @export_name(\"run\", fn()void);\n}\n",
    );
    std::fs::write(dir.join("pe.csl"), pe).expect("pe.csl");
    std::fs::write(dir.join("layout.csl"), layout).expect("layout.csl");
    let out = dir.join("out");
    let mut compile = Compile::memcpy(Arch::Wse3, dir.join("layout.csl"), w, h, &out);
    compile.channels = channels;
    let t = Instant::now();
    Cslc::find()
        .expect("cslc")
        .compile(&compile)
        .expect("compiles");
    eprintln!(
        "{w}x{h} channels {channels}: compiled in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    std::env::set_current_dir(&dir).expect("cd");
    let sdk = Sdk::open().expect("sdk");
    let t = Instant::now();
    let mut rt = Runtime::new(
        sdk,
        &out,
        Platform::Simulator(Simulator::new(Target::Wse3)),
        "WARNING",
    )
    .expect("runtime");
    rt.load();
    rt.run();
    eprintln!("  started in {:.1}s", t.elapsed().as_secs_f64());
    let pes = (w * h) as usize;
    let words: Vec<u32> = (0..pes * per_pe as usize).map(|i| i as u32).collect();
    let rect = Rect {
        x: 0,
        y: 0,
        w: w as i32,
        h: h as i32,
    };
    let t = Instant::now();
    rt.memcpy_h2d(
        "buf",
        &words,
        rect,
        per_pe as usize,
        Order::RowMajor,
        DataType::Bits32,
    )
    .expect("h2d");
    let h2d = t.elapsed().as_secs_f64();
    rt.launch("run", &[]).expect("launch");
    let mut back = vec![0u32; words.len()];
    let t = Instant::now();
    rt.memcpy_d2h(
        "buf",
        &mut back,
        rect,
        per_pe as usize,
        Order::RowMajor,
        DataType::Bits32,
    )
    .expect("d2h");
    let d2h = t.elapsed().as_secs_f64();
    let mut expect = words.clone();
    for b in 0..pes {
        expect[b * per_pe as usize] = (b as f32).to_bits();
    }
    let stamps: Vec<u32> = (0..pes)
        .map(|b| f32::from_bits(back[b * per_pe as usize]) as u32)
        .collect();
    println!("blocks carry PE indices {stamps:?} (row-major expects 0..{pes})");
    assert_eq!(
        back, expect,
        "the words round-trip block by block in row-major PE order"
    );
    println!(
        "{w}x{h} channels {channels}: {} words; h2d {h2d:.1}s ({:.0} words/s), d2h {d2h:.1}s ({:.0} words/s)",
        words.len(),
        words.len() as f64 / h2d,
        words.len() as f64 / d2h
    );
    rt.stop();
    let _ = std::fs::remove_dir_all(&dir);
}
