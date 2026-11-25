//! `Encode`'s contract: `groups` counts workgroups, never tiles times
//! threads. Regression: the register GEMM took its tile from `SV_GroupID`
//! but launched 256 times the tiles it has (a5fe3b70f).

use std::cell::RefCell;

use dtype::Dtype;

use crate::encode::{ABSENT, ArgValue, Encode, Fire, Target, Tier};
use crate::error::Error;
use crate::linear::gemm;
use crate::tensor::Tensor;

/// A sink that fires nothing and keeps every fire.
struct Recorder(RefCell<Vec<Fire>>);

impl Encode for Recorder {
    fn fire(&self, fire: Fire, _args: &[ArgValue]) -> Result<(), Error> {
        self.0.borrow_mut().push(fire);
        Ok(())
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(ABSENT))
    }

    fn target(&self) -> Target {
        Target::Spirv
    }
}

#[test]
fn the_register_gemm_launches_a_workgroup_a_tile() {
    // (rows, columns, contraction, columns a tile): the 64- and 32-column
    // forms; either tile lands 256 rows.
    for (m, n, k, cols) in [(2048, 1024, 1024, 64), (2048, 96, 64, 32)] {
        let rec = Recorder(RefCell::new(Vec::new()));
        let t = |buf, rows, width| Tensor::new(buf, rows, width, Dtype::Bf16);
        let tactic = gemm::Tactic {
            tier: Tier::Baseline,
            vector: 0,
            wide: true,
        };
        let act = t(0, m, k);
        gemm::matmul(
            &rec,
            tactic,
            act,
            t(1, n, k),
            t(2, m, n),
            crate::ssm::RowSpan::whole(act),
        )
        .unwrap();
        let fires = rec.0.take();
        assert_eq!(fires.len(), 1);
        let fire = &fires[0];
        assert!(fire.entrypoint.starts_with("linear_gemm_register"));
        assert_eq!(
            fire.groups,
            [n.div_ceil(cols), m.div_ceil(256), 1],
            "{m} x {n} x {k}: `{}`",
            fire.entrypoint
        );
    }
}
