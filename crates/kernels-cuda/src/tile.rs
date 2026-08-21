pub mod tile_alternatives {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("tile/alternatives.cuh");
}

pub mod argmax_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("sample/argmax_tile.cuh");
}

pub mod gather_rows_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("layout/gather_rows_tile.cuh");
}

pub mod dequant_wna16_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("quant/dequant_wna16_tile.cuh");
}

pub mod wna16_gemv_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("quant/wna16_gemv_tile.cuh");
}
