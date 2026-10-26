//! Generated row kernels, HIP's: a fused region (kernels-ir's [`Region`]),
//! scheduled ([`kernels_ir::schedule`]) and printed as HIP C++ ([`print`])
//! in the row skeleton's hooks (`elemwise/skeleton.hip`): the unit a generated
//! kernel compiles from. Its own copy of kernels-cuda's printer and skeleton.

pub mod print;

pub use kernels_ir::Region;

/// The struct every generated body defines, in the namespace its unit puts
/// it in.
pub const BODY: &str = "::pie::synth::Body";

/// The threads of a row skeleton's block.
pub const BLOCK: u32 = 256;

/// A region's body printed in the row skeleton's hook form ([`print`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Body {
    /// The struct's HIP C++ source.
    pub source: String,
    /// FNV-1a over `source`: the body's part of a generated kernel's key.
    pub hash: u64,
    /// The passes the body makes over the row.
    pub passes: u32,
}

/// Why the skeleton does not take a region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(pub String);

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

impl Body {
    /// `region` scheduled and printed, or why the row core or the printer
    /// does not take it.
    pub fn of(region: &Region) -> Result<Body, Refusal> {
        let kernel = kernels_ir::schedule(region)
            .map_err(|why| Refusal(format!("the row core does not schedule the region: {why}")))?;
        // HIP's skeleton holds a row's elements and reductions; a head's
        // values, a table and a shared row are kernels-cuda's alone so far.
        if !kernel.heads.is_empty() || !kernel.tables.is_empty() || kernel.shares {
            return Err(Refusal(
                "a kernel over heads, a table or a gather within the row, which HIP's skeleton \
                 does not hold"
                    .into(),
            ));
        }
        let source = print::body(&kernel).map_err(|why| match why {
            print::Unprinted::Rounding(dtype) => Refusal(format!(
                "a value rounds to {dtype:?}, which no hook rounds to"
            )),
            print::Unprinted::Vectors => Refusal(
                "a small vector one a row or a repeat over one, which HIP's skeleton does not \
                 hold yet"
                    .into(),
            ),
        })?;
        Ok(Body {
            hash: kernels_ir::fnv1a64(source.as_bytes()),
            source,
            passes: kernel.passes(),
        })
    }

    /// The unit the kernel compiles from: the skeleton's header and the body.
    #[must_use]
    pub fn unit(&self) -> String {
        format!(
            "// A generated row kernel: region body {:016x}.\n\
             #include \"elemwise/skeleton.hip\"\n\n\
             namespace pie::synth {{\n\n{}\n}}\n",
            self.hash, self.source
        )
    }
}

/// The skeleton's kernel around the body for `ins` parameters and `outs`
/// results, `per` vectors of 8 a thread, the parameters `ones` sets (by bit)
/// read as one element.
#[must_use]
pub fn entrypoint(per: u32, ins: usize, outs: usize, ones: u32) -> String {
    format!("::pie::elemwise::row_skeleton<{BODY}, {BLOCK}, {per}, {ins}, {outs}, {ones}u>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_ir::{Binary, Op, Shape};

    /// A row scaled by a value of a small vector one a row: HIP's skeleton
    /// holds no small vector yet, and its printer says so.
    #[test]
    fn a_small_vector_is_refused_by_name() {
        let region = Region {
            width: 64,
            params: vec![Shape::Row, Shape::Vector(4)],
            ops: vec![Op::Pick(1, 2), Op::Binary(Binary::Mul, 0, 2)],
            results: vec![3],
            rounds: vec![],
        };
        let why = Body::of(&region).expect_err("refused").to_string();
        assert!(why.contains("small vector"), "{why}");
    }
}
