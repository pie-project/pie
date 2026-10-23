use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::tensor::Tensor;

pub fn copy_words(ctx: &Ctx<'_>, src: Tensor, dst: Tensor, bytes: u64) -> Result<(), Error> {
    const OP: &str = "gather.rs_copy";
    if bytes == 0 {
        return Ok(());
    }
    if !bytes.is_multiple_of(4) {
        return Err(refuse(
            OP,
            format!("{bytes} bytes is not a whole number of 32-bit words"),
        ));
    }
    let words = u32::try_from(bytes / 4).map_err(|_| {
        refuse(
            OP,
            format!("{bytes} bytes is more than one launch addresses"),
        )
    })?;
    ctx.fire(
        Fire::at("runtime/blit.metal", "rs_copy_words")
            .apply(Grid::of([words, 1, 1], [words.min(256), 1, 1])),
        &[src.arg(), dst.arg_mut(), stated(OP, words)?.arg()],
    )
}

/// Up to four word copies in one launch, `(src, dst, bytes)` each; the
/// copies must not overlap one another, since they run side by side.
pub fn copy_words_batch(ctx: &Ctx<'_>, copies: &[(Tensor, Tensor, u64)]) -> Result<(), Error> {
    const OP: &str = "gather.rs_copy";
    let copies: Vec<&(Tensor, Tensor, u64)> = copies.iter().filter(|c| c.2 > 0).collect();
    match copies.as_slice() {
        [] => return Ok(()),
        [(src, dst, bytes)] => return copy_words(ctx, *src, *dst, *bytes),
        _ => {}
    }
    if copies.len() > 4 {
        return Err(refuse(
            OP,
            format!("{} copies is more than one launch takes", copies.len()),
        ));
    }
    let mut words = [0u32; 4];
    let mut args = Vec::with_capacity(12);
    for (at, word) in words.iter_mut().enumerate() {
        match copies.get(at) {
            Some((src, dst, bytes)) => {
                if !bytes.is_multiple_of(4) {
                    return Err(refuse(
                        OP,
                        format!("{bytes} bytes is not a whole number of 32-bit words"),
                    ));
                }
                *word = u32::try_from(bytes / 4).map_err(|_| {
                    refuse(
                        OP,
                        format!("{bytes} bytes is more than one launch addresses"),
                    )
                })?;
                args.push(src.arg());
                args.push(dst.arg_mut());
            }
            None => {
                args.push(ctx.absent()?);
                args.push(ctx.absent()?);
            }
        }
    }
    let widest = words.iter().copied().max().unwrap_or(1).max(1);
    args.extend(words.iter().map(|w| w.arg()));
    ctx.fire(
        Fire::at("runtime/blit.metal", "rs_copy_words4").apply(Grid::of(
            [widest, copies.len() as u32, 1],
            [widest.min(256), 1, 1],
        )),
        &args,
    )
}
