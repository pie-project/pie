//! A buffer's words laid out over a rectangle of PEs by its shard
//! ([`lay_out`]), and gathered back ([`gather_up`]).

/// A three-way grid shard's `(row parts, column parts, PE -> (row part, column part))`.
type GridParts = (usize, usize, Box<dyn Fn(usize) -> (usize, usize)>);

/// The per-PE layout of a buffer for a rectangle of `pes`: PE `x`'s words
/// back to back, as `memcpy_h2d` over the rectangle takes them.
pub(super) fn lay_out(
    words: &[u32],
    rows: usize,
    width: usize,
    shard: kernels_cerebras::phase::Shard,
    pes: usize,
) -> Vec<u32> {
    use kernels_cerebras::phase::Shard;
    match shard {
        Shard::Whole | Shard::Local(_) => words
            .iter()
            .copied()
            .cycle()
            .take(words.len() * pes)
            .collect(),
        Shard::Rows(_) => words.to_vec(),
        Shard::Cols(n) => {
            let n = n as usize;
            let per = width / n;
            let mut out = Vec::with_capacity(words.len());
            for x in 0..n {
                for r in 0..rows {
                    out.extend_from_slice(&words[r * width + x * per..r * width + (x + 1) * per]);
                }
            }
            out
        }
        Shard::RowsBy { parts, period } => {
            let (parts, period) = (parts as usize, period.max(1) as usize);
            let per = rows / parts.max(1);
            let mut out = Vec::with_capacity(per * width * pes);
            for x in 0..pes {
                let part = (x / period) % parts.max(1);
                let from = part * per * width;
                out.extend_from_slice(words.get(from..from + per * width).unwrap_or(&[]));
                out.resize((x + 1) * per * width, 0);
            }
            out
        }
        Shard::ColsBy { parts, period } | Shard::SumCols { parts, period } => {
            let (parts, period) = (parts.max(1) as usize, period.max(1) as usize);
            let by_period = matches!(shard, Shard::SumCols { .. });
            let pc = width / parts;
            let mut out = Vec::with_capacity(rows * pc * pes);
            for x in 0..pes {
                let part = if by_period {
                    x / period
                } else {
                    (x / period) % parts
                };
                for r in 0..rows {
                    let from = r * width + part * pc;
                    out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                }
                out.resize((x + 1) * rows * pc, 0);
            }
            out
        }
        Shard::RowsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::ColsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::SumGrid {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::Roots {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            // (row parts, column parts) of this buffer and the PE's parts.
            let (rparts, cparts, part_of): GridParts = match shard {
                Shard::RowsDepth { .. } => (rg, kg, Box::new(move |x| (x / (cg * kg), x % kg))),
                Shard::ColsDepth { .. } => (cg, kg, Box::new(move |x| ((x / kg) % cg, x % kg))),
                _ => (rg, cg, Box::new(move |x| (x / (cg * kg), (x / kg) % cg))),
            };
            let (pr, pc) = (rows / rparts, width / cparts);
            let mut out = Vec::with_capacity(pr * pc * pes);
            for x in 0..pes {
                let (rp, cp) = part_of(x);
                for r in rp * pr..(rp + 1) * pr {
                    let from = r * width + cp * pc;
                    out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                }
                out.resize((x + 1) * pr * pc, 0);
            }
            out
        }
        Shard::Tile {
            rows: rg,
            cols: cg,
            segments,
        } => {
            let (rg, cg, seg) = (
                rg.max(1) as usize,
                cg.max(1) as usize,
                segments.max(1) as usize,
            );
            let (pr, sw) = (rows / rg, width / seg);
            let pc = sw / cg;
            let mut out = Vec::with_capacity(pr * pc * seg * pes);
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                for r in rp * pr..(rp + 1) * pr {
                    for sg in 0..seg {
                        let from = r * width + sg * sw + cp * pc;
                        out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                    }
                }
                out.resize((x + 1) * pr * pc * seg, 0);
            }
            out
        }
        Shard::Grid { rows: rg, cols: cg } => {
            let (rg, cg) = (rg.max(1) as usize, cg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut out = Vec::with_capacity(pr * pc * pes);
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                for r in rp * pr..(rp + 1) * pr {
                    let from = r * width + cp * pc;
                    out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                }
                out.resize((x + 1) * pr * pc, 0);
            }
            out
        }
    }
}

/// The inverse of [`lay_out`]: a buffer's words from what the PEs hold.
pub(super) fn gather_up(
    laid: &[u32],
    rows: usize,
    width: usize,
    shard: kernels_cerebras::phase::Shard,
    pes: usize,
) -> Vec<u32> {
    use kernels_cerebras::phase::Shard;
    let n = rows * width;
    match shard {
        Shard::Whole | Shard::Local(_) => laid[..n.min(laid.len())].to_vec(),
        Shard::Rows(_) => laid.to_vec(),
        Shard::Cols(parts) => {
            let parts = parts as usize;
            let per = width / parts;
            let mut out = vec![0u32; n];
            for x in 0..parts {
                for r in 0..rows {
                    let from = (x * rows + r) * per;
                    out[r * width + x * per..r * width + (x + 1) * per]
                        .copy_from_slice(&laid[from..from + per]);
                }
            }
            out
        }
        Shard::RowsBy { parts, period } => {
            let (parts, period) = (parts as usize, period.max(1) as usize);
            let per = rows / parts.max(1);
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let part = (x / period) % parts.max(1);
                let (from, to) = (x * per * width, part * per * width);
                if let (Some(src), Some(dst)) = (
                    laid.get(from..from + per * width),
                    out.get_mut(to..to + per * width),
                ) {
                    dst.copy_from_slice(src);
                }
            }
            out
        }
        Shard::ColsBy { parts, period } => {
            let (parts, period) = (parts.max(1) as usize, period.max(1) as usize);
            let pc = width / parts;
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let part = (x / period) % parts;
                for r in 0..rows {
                    let from = x * rows * pc + r * pc;
                    let to = r * width + part * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
        Shard::SumCols { parts, period } => {
            let (parts, period) = (parts.max(1) as usize, period.max(1) as usize);
            let pc = width / parts;
            let mut sums = vec![0f32; n];
            for x in 0..pes {
                let part = x / period;
                for r in 0..rows {
                    for c in 0..pc {
                        if let (Some(v), Some(slot)) = (
                            laid.get(x * rows * pc + r * pc + c),
                            sums.get_mut(r * width + part * pc + c),
                        ) {
                            *slot += f32::from_bits(*v);
                        }
                    }
                }
            }
            sums.iter().map(|v| v.to_bits()).collect()
        }
        Shard::RowsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::ColsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            let (rparts, cparts, part_of): GridParts = match shard {
                Shard::RowsDepth { .. } => (rg, kg, Box::new(move |x| (x / (cg * kg), x % kg))),
                _ => (cg, kg, Box::new(move |x| ((x / kg) % cg, x % kg))),
            };
            let (pr, pc) = (rows / rparts, width / cparts);
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let (rp, cp) = part_of(x);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    let from = x * pr * pc + i * pc;
                    let to = r * width + cp * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
        Shard::SumGrid {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut sums = vec![0f32; n];
            for x in 0..pes {
                let (rp, cp) = (x / (cg * kg), (x / kg) % cg);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    for c in 0..pc {
                        if let (Some(v), Some(slot)) = (
                            laid.get(x * pr * pc + i * pc + c),
                            sums.get_mut(r * width + cp * pc + c),
                        ) {
                            *slot += f32::from_bits(*v);
                        }
                    }
                }
            }
            sums.iter().map(|v| v.to_bits()).collect()
        }
        Shard::Roots {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            // The fabric summed each block into its row's first PE.
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut out = vec![0u32; n];
            for x in (0..pes).step_by(kg) {
                let (rp, cp) = (x / (cg * kg), (x / kg) % cg);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    let from = x * pr * pc + i * pc;
                    let to = r * width + cp * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
        Shard::Tile {
            rows: rg,
            cols: cg,
            segments,
        } => {
            let (rg, cg, seg) = (
                rg.max(1) as usize,
                cg.max(1) as usize,
                segments.max(1) as usize,
            );
            let (pr, sw) = (rows / rg, width / seg);
            let pc = sw / cg;
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                let mut from = x * pr * pc * seg;
                for r in rp * pr..(rp + 1) * pr {
                    for sg in 0..seg {
                        let to = r * width + sg * sw + cp * pc;
                        if let (Some(src), Some(dst)) =
                            (laid.get(from..from + pc), out.get_mut(to..to + pc))
                        {
                            dst.copy_from_slice(src);
                        }
                        from += pc;
                    }
                }
            }
            out
        }
        Shard::Grid { rows: rg, cols: cg } => {
            let (rg, cg) = (rg.max(1) as usize, cg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    let from = x * pr * pc + i * pc;
                    let to = r * width + cp * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
    }
}
