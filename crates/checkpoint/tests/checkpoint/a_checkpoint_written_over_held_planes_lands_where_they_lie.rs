use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::overwrite::{Held, Overwrite};
use checkpoint::plan::StorageTarget;
use checkpoint::types::{DType, Encoding};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn checkpoint() -> (tempfile::TempDir, ModelContract) {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = ztensor::Writer::create(dir.path().join("model.zt")).unwrap();
    let a = bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    writer.add("a", vec![2, 3], ztensor::Leaf::F32, &a).unwrap();
    writer
        .add(
            "b",
            vec![4],
            ztensor::Leaf::F32,
            &bytes(&[7.0, 8.0, 9.0, 10.0]),
        )
        .unwrap();
    writer.finish().unwrap();
    let raw = Encoding::Raw(DType::F32);
    let rows =
        |src: &str, shape: Vec<i64>| Expr::src(src).transmute(TensorType::raw(shape, DType::F32));
    let tensor = |name: &str, expr: Expr, shape: Vec<i64>| {
        TensorContract::new(name, expr, shape, raw.clone())
    };
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![
            tensor("w.a", Expr::src("a"), vec![2, 3]),
            tensor("w.b", Expr::src("b"), vec![4]),
            tensor(
                "w.bb",
                Expr::concat(0, vec![Expr::src("b"), Expr::src("b")]),
                vec![8],
            ),
            tensor(
                "w.ab",
                Expr::concat(0, vec![rows("a", vec![3, 2]), rows("b", vec![2, 2])]),
                vec![5, 2],
            ),
        ],
        groups: Vec::new(),
    };
    (dir, contract)
}

fn held<'a>(name: &'a str, shape: &'a [u64]) -> Option<Held<'a>> {
    Some(Held {
        name,
        shape,
        bytes: shape.iter().product::<u64>() * 4,
    })
}

#[test]
fn a_checkpoint_lands_each_plane_over_the_held_one() {
    let (dir, contract) = checkpoint();
    let planes = [
        held("w.b", &[4]),
        None,
        held("w.a", &[2, 3]),
        held("w.bb", &[8]),
        held("w.ab", &[5, 2]),
    ];
    let overwrite =
        Overwrite::plan(dir.path(), &contract, StorageTarget::default(), &planes).unwrap();
    let mut written = Vec::new();
    let landed = overwrite
        .run(None, |at, plane| {
            written.push((at, plane.to_vec()));
            Ok(())
        })
        .unwrap();
    written.sort();
    assert_eq!(landed, (4, 112));
    assert_eq!(
        written,
        [
            (0, bytes(&[7.0, 8.0, 9.0, 10.0])),
            (2, bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            (3, bytes(&[7.0, 8.0, 9.0, 10.0, 7.0, 8.0, 9.0, 10.0])),
            (
                4,
                bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            ),
        ]
    );
}

#[test]
fn a_checkpoint_that_does_not_fit_the_held_planes_is_refused() {
    let (dir, contract) = checkpoint();
    let rest = [held("w.b", &[4]), held("w.bb", &[8]), held("w.ab", &[5, 2])];
    for (planes, why) in [
        (vec![held("w.a", &[3, 2])], "`w.a`"),
        (vec![held("w.a", &[2, 4])], "`w.a`"),
        (vec![], "lands `w.a`"),
        (
            vec![held("w.a", &[2, 3]), held("w.c", &[4])],
            "lands no `w.c`",
        ),
    ] {
        let planes: Vec<_> = planes.into_iter().chain(rest).collect();
        let refused =
            Overwrite::plan(dir.path(), &contract, StorageTarget::default(), &planes).unwrap_err();
        assert!(refused.to_string().contains(why), "{refused}");
    }
}
