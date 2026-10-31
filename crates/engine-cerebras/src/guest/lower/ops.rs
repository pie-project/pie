//! The lowering of each eta op: a value's CSL over the PE's block (a wide
//! value) or its whole (a local one), the reductions over the wide axis
//! combining across the row.

use super::*;

impl Gen<'_> {
    #[allow(clippy::too_many_lines)]
    pub(super) fn op(&mut self, op: &LaunchOp) -> Lowering<()> {
        if op.tag == tags::SINK_CALL {
            // A library sink (`lora`, `metal.discard`): the interpreter does
            // nothing for it, and neither does the device.
            return Ok(());
        }
        let result = op.result_id;
        if result as usize >= self.package.values.len() {
            return refuse(format!("result {result} is past the package's values"));
        }
        match op.tag {
            tags::EXP
            | tags::LOG
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
            | tags::RECIP => {
                let x = self.arg(op, 0)?;
                if x.ty != Ty::F32 {
                    return refuse("a float map over a non-float operand");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let one = flit(1.0);
                let f = |xi: String| match op.tag {
                    tags::EXP => format!("math.exp_f32({xi})"),
                    tags::LOG => format!("math.log_f32({xi})"),
                    tags::SIN => format!("math.sin_f32({xi})"),
                    tags::COS => format!("math.cos_f32({xi})"),
                    tags::SQRT => format!("math.sqrt_f32({xi})"),
                    tags::RSQRT => format!("{one} / math.sqrt_f32({xi})"),
                    _ => format!("{one} / {xi}"),
                };
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {};", r.name, f(Gen::at(&x, i))));
                });
                Ok(())
            }
            tags::NEG => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let e: fn(String) -> String = match x.ty {
                    Ty::F32 => {
                        |xi| format!("@bitcast(f32, @bitcast(u32, {xi}) ^ @as(u32, 0x80000000))")
                    }
                    Ty::I32 => |xi| format!("@as(i32, 0) - {xi}"),
                    Ty::U32 => |xi| format!("@as(u32, 0) - {xi}"),
                    Ty::Bool => return refuse("neg on bool"),
                };
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {};", r.name, e(Gen::at(&x, i))));
                });
                Ok(())
            }
            tags::ABS => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                self.each(r.n, |blk, i| {
                    let xi = Gen::at(&x, i);
                    match x.ty {
                        Ty::F32 => blk.line(format!("{}[{i}] = math.abs_f32({xi});", r.name)),
                        Ty::I32 => blk.line(format!(
                            "if ({xi} < 0) {{ {0}[{i}] = @as(i32, 0) - {xi}; }} else {{ {0}[{i}] = {xi}; }}",
                            r.name
                        )),
                        _ => blk.line(format!("{}[{i}] = {xi};", r.name)),
                    };
                });
                Ok(())
            }
            tags::SIGN => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let (one, m1, zero) = match x.ty {
                    Ty::F32 => (flit(1.0), flit(-1.0), flit(0.0)),
                    Ty::I32 => (ilit(1), ilit(-1), ilit(0)),
                    Ty::U32 => (ulit(1), ulit(0), ulit(0)),
                    Ty::Bool => return refuse("sign on bool"),
                };
                self.each(r.n, |blk, i| {
                    let xi = Gen::at(&x, i);
                    if x.ty == Ty::U32 {
                        blk.line(format!(
                            "if ({xi} != 0) {{ {0}[{i}] = {one}; }} else {{ {0}[{i}] = {zero}; }}",
                            r.name
                        ));
                    } else {
                        blk.line(format!(
                            "if ({xi} > {zero}) {{ {0}[{i}] = {one}; }} else if ({xi} < {zero}) {{ {0}[{i}] = {m1}; }} else {{ {0}[{i}] = {zero}; }}",
                            r.name
                        ));
                    }
                });
                Ok(())
            }
            tags::CAST => {
                let x = self.arg(op, 0)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                self.cast(&x, &r)
            }
            tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::REM
            | tags::MAX_ELEM
            | tags::MIN_ELEM => {
                let a = self.arg(op, 0)?;
                let b = self.arg(op, 1)?;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a, &b])?;
                self.arith(op.tag, &a, &b, &r)
            }
            tags::GT | tags::GE | tags::EQ | tags::NE | tags::LT | tags::LE => {
                let a = self.arg(op, 0)?;
                let b = self.arg(op, 1)?;
                if a.ty == Ty::Bool {
                    return refuse("an ordered comparison of bools");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a, &b])?;
                let sym = match op.tag {
                    tags::GT => ">",
                    tags::GE => ">=",
                    tags::EQ => "==",
                    tags::NE => "!=",
                    tags::LT => "<",
                    _ => "<=",
                };
                // A NaN operand compares false (true for `!=`), stated
                // outright: the compiler's float math may not keep IEEE
                // comparisons on NaN.
                let float = a.ty == Ty::F32;
                self.each(r.n, |blk, i| {
                    let (x, y) = (Gen::at(&a, i), Gen::at(&b, i));
                    let cond = if !float {
                        format!("{x} {sym} {y}")
                    } else if sym == "!=" {
                        format!("math.isNaN_f32({x}) or math.isNaN_f32({y}) or {x} != {y}")
                    } else {
                        format!("!math.isNaN_f32({x}) and !math.isNaN_f32({y}) and {x} {sym} {y}")
                    };
                    blk.line(format!(
                        "if ({cond}) {{ {0}[{i}] = 1; }} else {{ {0}[{i}] = 0; }}",
                        r.name
                    ));
                });
                Ok(())
            }
            tags::AND | tags::OR => {
                let a = self.arg(op, 0)?;
                let b = self.arg(op, 1)?;
                if a.ty != Ty::Bool || b.ty != Ty::Bool {
                    return refuse("and/or on non-bool");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a, &b])?;
                let sym = if op.tag == tags::AND { "and" } else { "or" };
                self.each(r.n, |blk, i| {
                    blk.line(format!(
                        "if ({1} != 0 {sym} {2} != 0) {{ {0}[{i}] = 1; }} else {{ {0}[{i}] = 0; }}",
                        r.name,
                        Gen::at(&a, i),
                        Gen::at(&b, i)
                    ));
                });
                Ok(())
            }
            tags::NOT => {
                let a = self.arg(op, 0)?;
                if a.ty != Ty::Bool {
                    return refuse("not on non-bool");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&a])?;
                self.each(r.n, |blk, i| {
                    blk.line(format!(
                        "if ({1} == 0) {{ {0}[{i}] = 1; }} else {{ {0}[{i}] = 0; }}",
                        r.name,
                        Gen::at(&a, i)
                    ));
                });
                Ok(())
            }
            tags::SELECT => {
                let c = self.arg(op, 0)?;
                let a = self.arg(op, 1)?;
                let b = self.arg(op, 2)?;
                if c.ty != Ty::Bool || a.ty != b.ty {
                    return refuse("select's condition or arms");
                }
                let r = self.result(op, 0)?;
                self.agree(&r, &[&c, &a, &b])?;
                self.each(r.n, |blk, i| {
                    blk.line(format!(
                        "if ({1} != 0) {{ {0}[{i}] = {2}; }} else {{ {0}[{i}] = {3}; }}",
                        r.name,
                        Gen::at(&c, i),
                        Gen::at(&a, i),
                        Gen::at(&b, i)
                    ));
                });
                Ok(())
            }
            tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN => self.reduce(op),
            tags::REDUCE_ARGMAX => self.argmax(op),
            tags::CUMSUM | tags::CUMPROD => self.scan(op),
            tags::BROADCAST => self.broadcast(op),
            tags::RESHAPE | tags::KERNEL_CALL => {
                if op.tag == tags::KERNEL_CALL && op.args.len() != 1 {
                    return refuse("a kernel call that is not the identity boundary");
                }
                let (id, x) = (op.args[0], self.arg(op, 0)?);
                let shape = self.shape(result);
                if self.dtype(id) != self.dtype(result) || numel(&self.shape(id)) != numel(&shape) {
                    return refuse("a reshape that changes the count or the dtype");
                }
                if x.wide != self.is_wide(result) || x.packed != self.packed.contains(&result) {
                    return refuse("a reshape across the wide axis");
                }
                // The same array under the result's id.
                self.bind(result, x);
                Ok(())
            }
            tags::TRANSPOSE => {
                let (id, x) = (op.args[0], self.arg(op, 0)?);
                let dims = self.shape(id);
                if dims.len() != 2 {
                    return refuse("a transpose of other than a matrix");
                }
                if x.wide {
                    return refuse("a transpose of a wide value");
                }
                let (m, n) = (dims[0] as usize, dims[1] as usize);
                let r = self.result(op, 0)?;
                if r.wide {
                    return refuse("a transpose into a wide value");
                }
                let xn = x.name.clone();
                self.each(r.n, |blk, o| {
                    blk.line(format!(
                        "{}[{o}] = {xn}[({o} % {m}) * {n} + {o} / {m}];",
                        r.name
                    ));
                });
                Ok(())
            }
            tags::SORT_DESC | tags::TOP_K => self.order(op),
            tags::MATMUL => {
                let (ia, a) = (op.args[0], self.arg(op, 0)?);
                let (ib, b) = (op.args[1], self.arg(op, 1)?);
                let (sa, sb) = (self.shape(ia), self.shape(ib));
                if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
                    return refuse("matmul of other than conformant matrices");
                }
                if a.ty != Ty::F32 || b.ty != Ty::F32 {
                    return refuse("matmul of non-floats");
                }
                if a.wide || b.wide {
                    return refuse("matmul over a wide value");
                }
                let (m, kk, n) = (sa[0] as usize, sa[1] as usize, sb[1] as usize);
                let r = self.result(op, 0)?;
                if r.wide {
                    return refuse("matmul into a wide value");
                }
                let zero = flit(0.0);
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {zero};", r.name));
                });
                let (an, bn, rn) = (a.name.clone(), b.name.clone(), r.name.clone());
                // The interpreter's order: rows, then the contraction, then
                // the columns, skipping a zero left operand.
                self.each(m, |blk, i| {
                    blk.nest(
                        &format!("{{ var l: i32 = 0; while (l < {kk}) : (l += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var xv: f32 = {an}[{i} * {kk} + l];"));
                            blk.line(format!("if (xv == {zero}) {{ continue; }}"));
                            blk.nest(
                                &format!("{{ var j: i32 = 0; while (j < {n}) : (j += 1) {{"),
                                "} }",
                                |blk| {
                                    blk.line(format!(
                                        "{rn}[{i} * {n} + j] = {rn}[{i} * {n} + j] + xv * {bn}[l * {n} + j];"
                                    ));
                                },
                            );
                        },
                    );
                });
                Ok(())
            }
            tags::PIVOT_THRESHOLD => self.pivot(op),
            tags::GATHER => self.gather(op),
            tags::GATHER_ROW => self.gather_row(op),
            tags::SCATTER_ADD | tags::SCATTER_SET => self.scatter(op),
            tags::IOTA => {
                let r = self.result(op, 0)?;
                if r.ty != Ty::U32 {
                    return refuse("an iota that is not u32");
                }
                let base = if r.wide {
                    format!("@as(u32, px * {})", self.block())
                } else {
                    "@as(u32, 0)".to_string()
                };
                self.each(r.n, |blk, i| {
                    blk.line(format!("{}[{i}] = {base} + @as(u32, {i});", r.name));
                });
                Ok(())
            }
            tags::MASK_APPLY_PACKED => {
                let (ix, x) = (op.args[0], self.arg(op, 0)?);
                let mask = self.arg(op, 1)?;
                if x.ty != Ty::F32 || mask.ty != Ty::U32 {
                    return refuse("mask_apply of other than f32 logits and u32 words");
                }
                let shape = self.shape(ix);
                let n = *shape.last().unwrap_or(&1) as usize;
                let r = self.result(op, 0)?;
                self.agree(&r, &[&x])?;
                let ninf = flit(f32::NEG_INFINITY);
                let (xn, mn) = (x.name.clone(), mask.name.clone());
                // Column `c` of the row reads bit `c` of the words; a wide
                // logits row is this PE's block of the columns, and a wide
                // mask this PE's block of the words.
                let (len, col0, w, w0) = if r.wide {
                    let block = self.block();
                    let w0 = if mask.wide {
                        format!("px * {}", block / 32)
                    } else {
                        "0".to_string()
                    };
                    (block, format!("px * {block}"), mask.n, w0)
                } else {
                    if mask.wide {
                        return refuse("a wide mask over a local row");
                    }
                    (n, "0".to_string(), mask.n, "0".to_string())
                };
                self.each(r.n, |blk, j| {
                    blk.line(format!(
                        "var c: i32 = {col0} + {j} % {len}; var wi: i32 = (c >> 5) - ({w0}); var word: u32 = 0;"
                    ));
                    blk.line(format!("if (wi >= 0 and wi < {w}) {{ word = {mn}[wi]; }}"));
                    blk.line(format!(
                        "if (((word >> @as(u32, c & 31)) & 1) != 0) {{ {0}[{j}] = {xn}[{j}]; }} else {{ {0}[{j}] = {ninf}; }}",
                        r.name
                    ));
                });
                Ok(())
            }
            tags::CAUSAL_MASK | tags::SLIDING_WINDOW_MASK | tags::SINK_WINDOW_MASK => {
                let pos = self.arg(op, 0)?;
                if pos.ty != Ty::U32 || pos.wide {
                    return refuse("structured mask positions that are not local u32");
                }
                let keys = op.imm as usize;
                let r = self.result(op, 0)?;
                let (local_keys, key0) = if r.wide {
                    (self.block(), format!("@as(u32, px * {})", self.block()))
                } else {
                    (keys, "@as(u32, 0)".to_string())
                };
                if r.n != pos.n * local_keys {
                    return refuse("a structured mask that is not positions x keys");
                }
                let window = if op.tag == tags::SLIDING_WINDOW_MASK {
                    op.imm2
                } else {
                    op.imm3
                };
                let sink = op.imm2;
                let tag = op.tag;
                let (pn, rn) = (pos.name.clone(), r.name.clone());
                self.each(pos.n, |blk, p| {
                    blk.line(format!("var position: u32 = {pn}[{p}];"));
                    blk.nest(
                        &format!("{{ var kl: i32 = 0; while (kl < {local_keys}) : (kl += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var key: u32 = {key0} + @as(u32, kl);"));
                            blk.line("var allowed: bool = key <= position;");
                            if tag != tags::CAUSAL_MASK {
                                // `key.saturating_add(window) > position`.
                                blk.line(format!(
                                    "var recent: bool = (key > @as(u32, 0xFFFFFFFF) - @as(u32, {window})) or (key + @as(u32, {window}) > position);"
                                ));
                                if tag == tags::SLIDING_WINDOW_MASK {
                                    blk.line("allowed = allowed and recent;");
                                } else {
                                    blk.line(format!(
                                        "allowed = allowed and (key < @as(u32, {sink}) or recent);"
                                    ));
                                }
                            }
                            blk.line(format!(
                                "if (allowed) {{ {rn}[{p} * {local_keys} + kl] = 1; }} else {{ {rn}[{p} * {local_keys} + kl] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            tags::RNG | tags::RNG_KEYED => {
                let r = self.result(op, 0)?;
                if r.ty != Ty::F32 {
                    return refuse("a draw that is not f32");
                }
                self.need("g_rng");
                let seed = if op.tag == tags::RNG {
                    u64lit(rng::seed_eff_stream(0, op.imm))
                } else {
                    let (is, state) = (op.args[0], self.arg(op, 0)?);
                    if state.ty != Ty::U32 || numel(&self.shape(is)) < 1 || state.wide {
                        return refuse("an rng state that is not local u32 words");
                    }
                    let ctr = if state.n > 1 {
                        format!("@as(u64, {}[1])", state.name)
                    } else {
                        "@as(u64, 0)".to_string()
                    };
                    format!(
                        "g_splitmix((@as(u64, {}[0]) << {}) | {ctr})",
                        state.name,
                        rng::RNG_FORMULA.keyed_word_bits
                    )
                };
                self.tmp += 1;
                let sv = format!("seed{}", self.tmp);
                self.b.line(format!("var {sv}: u64 = {seed};"));
                let seed = sv;
                let rn = r.name.clone();
                // Lane `j` of the draw is global: a wide draw's PE starts at
                // its block in each row of the value.
                let lane0 = if r.wide {
                    let block = self.block();
                    let v = self.wide.unwrap_or(1) as usize;
                    format!("({{j}} / {block}) * {v} + px * {block} + {{j}} % {block}")
                } else {
                    "{j}".to_string()
                };
                let lane = |j: &str| lane0.replace("{j}", j);
                match op.rng_kind {
                    RngKind::Uniform => self.each(r.n, |blk, j| {
                        blk.line(format!(
                            "{rn}[{j}] = g_uniform({seed}, @as(u32, {}));",
                            lane(j)
                        ));
                    }),
                    RngKind::Gumbel => self.each(r.n, |blk, j| {
                        blk.line(format!(
                            "{rn}[{j}] = @bitcast(f32, @bitcast(u32, math.log_f32(@bitcast(f32, @bitcast(u32, math.log_f32(g_uniform({seed}, @as(u32, {})))) ^ @as(u32, 0x80000000)))) ^ @as(u32, 0x80000000));",
                            lane(j)
                        ));
                    }),
                    RngKind::Normal => {
                        let stride = rng::NORMAL_PAIR_STRIDE;
                        let tau = flit(rng::NORMAL_TWO_PI);
                        let m2 = flit(-2.0);
                        self.each(r.n, |blk, j| {
                            blk.line(format!(
                                "var lane: u32 = @as(u32, {}) * @as(u32, {stride});",
                                lane(j)
                            ));
                            blk.line(format!("var un0: f32 = g_uniform({seed}, lane); var un1: f32 = g_uniform({seed}, lane + 1);"));
                            blk.line(format!(
                                "{rn}[{j}] = math.sqrt_f32({m2} * math.log_f32(un0)) * math.cos_f32({tau} * un1);"
                            ));
                        });
                    }
                }
                Ok(())
            }
            other => refuse(format!("`{}` has no CSL form here", op_name(other))),
        }
    }

    fn zero(ty: Ty) -> String {
        match ty {
            Ty::F32 => flit(0.0),
            Ty::I32 => ilit(0),
            Ty::U32 => ulit(0),
            Ty::Bool => "0".to_string(),
        }
    }

    /// Whether the index lane `ix` (of `idx`'s type) is inside `0..bound`.
    fn index_valid(idx: &Var, ix: &str, bound: usize) -> String {
        match idx.ty {
            Ty::U32 => format!("{ix} < @as(u32, {bound})"),
            Ty::I32 => format!("{ix} >= 0 and {ix} < {bound}"),
            Ty::Bool => format!("@as(i32, {ix}) < {bound}"),
            Ty::F32 => format!(
                "({ix} == {ix}) and {ix} >= {} and {ix} < {}",
                flit(0.0),
                flit(bound as f32)
            ),
        }
    }

    /// The row count and per-PE row length of a reduction over `x`'s last
    /// axis.
    fn rows_len(&self, id: u32, x: &Var) -> (usize, usize) {
        let rows = canonical_rows(&self.shape(id)).max(1);
        (rows, x.n / rows)
    }

    fn cast(&mut self, x: &Var, r: &Var) -> Lowering<()> {
        let (from, to) = (x.ty, r.ty);
        // A test and its two arms, or one expression.
        let (test, yes, no): (Option<&str>, String, String) = match (from, to) {
            (a, b) if a == b => (None, "{x}".into(), String::new()),
            (Ty::I32 | Ty::U32, Ty::F32) => (None, "@as(f32, {x})".into(), String::new()),
            (Ty::Bool, Ty::F32) => (Some("{x} != 0"), flit(1.0), flit(0.0)),
            (Ty::F32, Ty::I32) => {
                self.need("g_cast");
                (None, "g_f2i({x})".into(), String::new())
            }
            (Ty::F32, Ty::U32) => {
                self.need("g_cast");
                (None, "g_f2u({x})".into(), String::new())
            }
            (Ty::U32, Ty::I32) => (None, "@bitcast(i32, {x})".into(), String::new()),
            (Ty::Bool, Ty::I32) => (None, "@as(i32, {x})".into(), String::new()),
            (Ty::I32, Ty::U32) => (None, "@bitcast(u32, {x})".into(), String::new()),
            (Ty::Bool, Ty::U32) => (None, "@as(u32, {x})".into(), String::new()),
            // NaN != 0 holds: a NaN lane is true, as `e != 0.0` says.
            (Ty::F32, Ty::Bool) => (
                Some("math.isNaN_f32({x}) or {x} != ZERO"),
                "1".into(),
                "0".into(),
            ),
            (Ty::I32 | Ty::U32, Ty::Bool) => (Some("{x} != 0"), "1".into(), "0".into()),
            _ => return refuse(format!("a cast from {from:?} to {to:?}")),
        };
        let zero = flit(0.0);
        self.each(r.n, |blk, i| {
            let xi = Gen::at(x, i);
            match test {
                None => blk.line(format!("{}[{i}] = {};", r.name, yes.replace("{x}", &xi))),
                Some(t) => blk.line(format!(
                    "if ({1}) {{ {0}[{i}] = {yes}; }} else {{ {0}[{i}] = {no}; }}",
                    r.name,
                    t.replace("{x}", &xi).replace("ZERO", &zero)
                )),
            };
        });
        Ok(())
    }

    fn arith(&mut self, tag: u8, a: &Var, b: &Var, r: &Var) -> Lowering<()> {
        let d = a.ty;
        if d == Ty::Bool {
            return refuse("arithmetic on bools");
        }
        let float = d == Ty::F32;
        let e: Box<dyn Fn(String, String) -> String> = match tag {
            tags::ADD => Box::new(|x, y| format!("{x} + {y}")),
            tags::SUB => Box::new(|x, y| format!("{x} - {y}")),
            tags::MUL => Box::new(|x, y| format!("{x} * {y}")),
            tags::DIV if float => Box::new(|x, y| format!("{x} / {y}")),
            tags::REM if float => {
                self.need("g_fmod");
                Box::new(|x, y| format!("g_fmod({x}, {y})"))
            }
            tags::DIV | tags::REM => {
                // The interpreter divides in i64: by zero is 0, and
                // i32::MIN / -1 wraps (i32::MIN % -1 is 0).
                self.need("g_int");
                let name = match (tag, d) {
                    (tags::DIV, Ty::I32) => "g_idiv",
                    (tags::REM, Ty::I32) => "g_irem",
                    (tags::DIV, _) => "g_udiv",
                    _ => "g_urem",
                };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            tags::MAX_ELEM | tags::MIN_ELEM if float => {
                // `f32::max`/`min`: a NaN operand yields the other one.
                self.need("g_tree");
                let name = if tag == tags::MAX_ELEM {
                    "g_fmax"
                } else {
                    "g_fmin"
                };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            tags::MAX_ELEM => {
                self.need("g_int");
                let name = if d == Ty::I32 { "g_imax" } else { "g_umax" };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            tags::MIN_ELEM => {
                self.need("g_int");
                let name = if d == Ty::I32 { "g_imin" } else { "g_umin" };
                Box::new(move |x, y| format!("{name}({x}, {y})"))
            }
            _ => return refuse("an arithmetic tag"),
        };
        self.each(r.n, |blk, i| {
            blk.line(format!(
                "{}[{i}] = {};",
                r.name,
                e(Gen::at(a, i), Gen::at(b, i))
            ));
        });
        Ok(())
    }

    // ------------------------------------------------------------- reductions

    fn reduce(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        let (rows, len) = self.rows_len(id, &x);
        let r = self.result(op, 0)?;
        if r.wide {
            return refuse("a reduction into a wide value");
        }
        if r.n != rows {
            return refuse("a reduction that is not one value per row");
        }
        if x.ty == Ty::Bool {
            return refuse("a reduction over bools");
        }
        let xn = x.name.clone();
        // Each PE's partial over its block (the whole row when local).
        let part = if x.wide {
            self.temp(x.ty.elem(), rows)
        } else {
            r.name.clone()
        };
        if x.ty == Ty::F32 {
            self.need("g_tree");
            let lvl = self.temp("f32", len.div_ceil(32).max(1));
            let f = match op.tag {
                tags::REDUCE_SUM => "g_tsum",
                tags::REDUCE_MAX => "g_tmax",
                _ => "g_tmin",
            };
            let pn = part.clone();
            let held = Gen::held(&x, len);
            self.each(rows, |blk, rr| {
                blk.line(format!(
                    "{pn}[{rr}] = {f}(@ptrcast([*]f32, &{xn}), {rr} * {len}, {held}, @ptrcast([*]f32, &{lvl}));"
                ));
            });
        } else {
            let (ty, init) = match (op.tag, x.ty) {
                (tags::REDUCE_SUM, _) => (x.ty, "0".to_string()),
                (tags::REDUCE_MAX, Ty::I32) => (Ty::I32, ilit(i32::MIN)),
                (tags::REDUCE_MIN, Ty::I32) => (Ty::I32, ilit(i32::MAX)),
                (tags::REDUCE_MAX, _) => (Ty::U32, ulit(0)),
                (tags::REDUCE_MIN, _) => (Ty::U32, ulit(u32::MAX)),
                _ => unreachable!(),
            };
            let elem = ty.elem();
            let tag = op.tag;
            let pn = part.clone();
            let held = Gen::held(&x, len);
            self.each(rows, |blk, rr| {
                blk.line(format!("var acc: {elem} = {init};"));
                blk.nest(
                    &format!("{{ var j: i32 = 0; while (j < {held}) : (j += 1) {{"),
                    "} }",
                    |blk| {
                        let e = format!("{xn}[{rr} * {len} + j]");
                        match tag {
                            tags::REDUCE_SUM => blk.line(format!("acc = acc + {e};")),
                            tags::REDUCE_MAX => {
                                blk.line(format!("if ({e} > acc) {{ acc = {e}; }}"))
                            }
                            _ => blk.line(format!("if ({e} < acc) {{ acc = {e}; }}")),
                        };
                    },
                );
                blk.line(format!("{pn}[{rr}] = acc;"));
            });
        }
        if !x.wide {
            return Ok(());
        }
        // The partials combine across the row (f32 sums on the fabric).
        let cols = self.cols;
        let elem = x.ty.elem();
        let tag = op.tag;
        let float = x.ty == Ty::F32;
        if float {
            self.need("g_tree");
        }
        if float && tag == tags::REDUCE_SUM {
            let res = self.combine_sum(&part, rows);
            let rn = r.name.clone();
            self.each(rows, |blk, rr| {
                blk.line(format!("{rn}[{rr}] = @bitcast(f32, {res}[{rr}]);"));
            });
            return Ok(());
        }
        let res = self.combine(&part, rows, rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var acc: {elem} = @bitcast({elem}, {recv}[rr]);"));
                    blk.nest(
                        &format!("{{ var c: i32 = 1; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            let e = format!("@bitcast({elem}, {recv}[c * {rows} + rr])");
                            match (tag, float) {
                                (tags::REDUCE_SUM, _) => blk.line(format!("acc = acc + {e};")),
                                (tags::REDUCE_MAX, true) => {
                                    blk.line(format!("acc = g_cmax(acc, {e});"))
                                }
                                (tags::REDUCE_MIN, true) => {
                                    blk.line(format!("acc = g_cmin(acc, {e});"))
                                }
                                (tags::REDUCE_MAX, false) => {
                                    blk.line(format!("if ({e} > acc) {{ acc = {e}; }}"))
                                }
                                _ => blk.line(format!("if ({e} < acc) {{ acc = {e}; }}")),
                            };
                        },
                    );
                    blk.line("RES[rr] = @bitcast(u32, acc);");
                },
            );
        });
        let rn = r.name.clone();
        self.each(rows, |blk, rr| {
            blk.line(format!("{rn}[{rr}] = @bitcast({elem}, {res}[{rr}]);"));
        });
        Ok(())
    }

    fn argmax(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        let (rows, len) = self.rows_len(id, &x);
        let r = self.result(op, 0)?;
        if x.ty == Ty::Bool {
            return refuse("argmax over bools");
        }
        Gen::lanes_only(&x, "argmax")?;
        if r.wide || r.n != rows {
            return refuse("an argmax that is not one index per row");
        }
        self.argmax_rows(&x, rows, len, &r);
        Ok(())
    }

    /// `r[rr]` = the argmax of row `rr` (`len` lanes on this PE) of `x`:
    /// NaN never wins, ties go to the lowest index; a wide `x` combines the
    /// PEs' candidates across the row.
    fn argmax_rows(&mut self, x: &Var, rows: usize, len: usize, r: &Var) {
        debug_assert!(!x.packed, "argmax over a packed mask is refused before");
        let xn = x.name.clone();
        let elem = x.ty.elem();
        let float = x.ty == Ty::F32;
        let zero = Gen::zero(x.ty);
        // Per row: the best, its index (global when wide), whether any.
        let (best, bat, have) = (
            self.temp(elem, rows),
            self.temp("i32", rows),
            self.temp("u32", rows),
        );
        let base = if x.wide {
            format!("px * {}", self.block())
        } else {
            "0".to_string()
        };
        let held = Gen::held(x, len);
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "var best: {elem} = {zero}; var bat: i32 = 0; var have: bool = false;"
            ));
            blk.nest(
                &format!("{{ var j: i32 = 0; while (j < {held}) : (j += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var x: {elem} = {xn}[{rr} * {len} + j];"));
                    if float {
                        blk.line("if (math.isNaN_f32(x)) { continue; }");
                    }
                    blk.line("if (!have or x > best) { best = x; bat = j; have = true; }");
                },
            );
            blk.line(format!(
                "{best}[{rr}] = best; {bat}[{rr}] = {base} + bat; if (have) {{ {have}[{rr}] = 1; }} else {{ {have}[{rr}] = 0; }}"
            ));
        });
        if !x.wide {
            let rn = r.name.clone();
            self.each(rows, |blk, rr| {
                blk.line(format!("{rn}[{rr}] = {bat}[{rr}];"));
            });
            return;
        }
        // Two rounds of a word a row: every PE offers its best (its type's
        // bottom when it has none), the root takes the greatest; then the
        // PEs holding it offer their index (the axis's length for the
        // others), the root takes the lowest. 0 when no PE has any.
        let send = self.temp("u32", rows);
        let bottom_bits = match x.ty {
            Ty::F32 => flit(f32::NEG_INFINITY),
            Ty::I32 => ilit(i32::MIN),
            _ => ulit(0),
        };
        let bottom_word = x.ty.to_word(&bottom_bits);
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "if ({have}[{rr}] != 0) {{ {send}[{rr}] = {}; }} else {{ {send}[{rr}] = {bottom_word}; }}",
                x.ty.to_word(&format!("{best}[{rr}]"))
            ));
        });
        let cols = self.cols;
        let v = self.wide.unwrap_or(1);
        let xty = x.ty;
        let top = self.combine(&send, rows, rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!(
                        "var acc: {elem} = {}; var any: bool = false;",
                        xty.of_word(&format!("{recv}[rr]"))
                    ));
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "var x: {elem} = {};",
                                xty.of_word(&format!("{recv}[c * {rows} + rr]"))
                            ));
                            if float {
                                blk.line("if (math.isNaN_f32(x)) { continue; }");
                            }
                            blk.line("if (!any or x > acc) { acc = x; any = true; }");
                        },
                    );
                    blk.line(format!("RES[rr] = {};", xty.to_word("acc")));
                },
            );
        });
        // A PE holding the best (and any lane) offers its lowest index.
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "var tb: {elem} = {}; if ({have}[{rr}] != 0 and {best}[{rr}] == tb) {{ {send}[{rr}] = @bitcast(u32, {bat}[{rr}]); }} else {{ {send}[{rr}] = {v}; }}",
                xty.of_word(&format!("{top}[{rr}]"))
            ));
        });
        let res = self.combine(&send, rows, rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var at: i32 = {v};"));
                    blk.line(format!(
                        "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ var xi: i32 = @bitcast(i32, {recv}[c * {rows} + rr]); if (xi < at) {{ at = xi; }} }} }}"
                    ));
                    blk.line(format!("if (at >= {v}) {{ at = 0; }} RES[rr] = @bitcast(u32, at);"));
                },
            );
        });
        let rn = r.name.clone();
        self.each(rows, |blk, rr| {
            blk.line(format!("{rn}[{rr}] = @bitcast(i32, {res}[{rr}]);"));
        });
    }

    fn scan(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        if x.ty != Ty::F32 {
            // The interpreter scans in f32 and hands back f32 lanes
            // whatever the declared dtype: nothing to agree with.
            return refuse("a scan over integers");
        }
        let (rows, len) = self.rows_len(id, &x);
        let r = self.result(op, 0)?;
        self.agree(&r, &[&x])?;
        let sum = op.tag == tags::CUMSUM;
        let (init, sym) = if sum {
            (flit(0.0), "+")
        } else {
            (flit(1.0), "*")
        };
        let xn = x.name.clone();
        let rn = r.name.clone();
        // The scan of this PE's block; a wide scan then carries each
        // earlier block's total in.
        let total = x.wide.then(|| self.temp("f32", rows));
        let tn = total.clone();
        let held = Gen::held(&x, len);
        self.each(rows, |blk, rr| {
            blk.line(format!("var acc: f32 = {init};"));
            blk.nest(
                &format!("{{ var j: i32 = 0; while (j < {held}) : (j += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("acc = acc {sym} {xn}[{rr} * {len} + j];"));
                    blk.line(format!("{rn}[{rr} * {len} + j] = acc;"));
                },
            );
            if let Some(t) = &tn {
                blk.line(format!("{t}[{rr}] = acc;"));
            }
        });
        let Some(total) = total else {
            return Ok(());
        };
        // The root folds the blocks' totals into each PE's carry: the fold
        // of the blocks before it (the identity for the first).
        let cols = self.cols;
        let res = self.combine(&total, rows, rows * cols, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!("var carry: f32 = {init};"));
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("RES[c * {rows} + rr] = @bitcast(u32, carry);"));
                            blk.line(format!(
                                "carry = carry {sym} @bitcast(f32, {recv}[c * {rows} + rr]);"
                            ));
                        },
                    );
                },
            );
        });
        self.each(rows, |blk, rr| {
            blk.line(format!(
                "var carry: f32 = @bitcast(f32, {res}[px * {rows} + {rr}]);"
            ));
            blk.nest(
                &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                "} }",
                |blk| {
                    blk.line(format!(
                        "{rn}[{rr} * {len} + j] = carry {sym} {rn}[{rr} * {len} + j];"
                    ));
                },
            );
        });
        Ok(())
    }

    fn broadcast(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        let src = self.shape(id);
        let target = self.shape(op.result_id);
        let r = self.result(op, 0)?;
        let rank = target.len();
        if src.len() > rank {
            return refuse("a broadcast to a lower rank");
        }
        if src.iter().all(|&d| d == 1) {
            self.each(r.n, |blk, i| {
                blk.line(format!("{}[{i}] = {}[0];", r.name, x.name));
            });
            return Ok(());
        }
        // The interpreter aligns the source's axes to the target's leading
        // axes (`broadcast_value`). A wide result walks its PE's block of
        // the last axis; a wide source is read at its block.
        let sdim = |i: usize| -> u64 { if i < src.len() { u64::from(src[i]) } else { 1 } };
        let mut sstride = vec![1u64; rank.max(1)];
        for i in (0..rank.saturating_sub(1)).rev() {
            sstride[i] = sstride[i + 1] * sdim(i + 1);
        }
        let mut tstride = vec![1u64; rank.max(1)];
        for i in (0..rank.saturating_sub(1)).rev() {
            tstride[i] = tstride[i + 1] * u64::from(target[i + 1]);
        }
        if x.wide && (src.len() != rank || sdim(rank - 1) != u64::from(target[rank - 1])) {
            return refuse("a wide source broadcast along its wide axis");
        }
        Gen::lanes_only(&x, "a broadcast")?;
        Gen::lanes_only(&r, "a broadcast")?;
        let block = self.block();
        let v = self.wide.unwrap_or(1) as usize;
        let xn = x.name.clone();
        let (r_wide, x_wide) = (r.wide, x.wide);
        self.each(r.n, |blk, lin| {
            if r_wide {
                // The global index of this PE's element `lin`.
                blk.line(format!(
                    "var rem: i32 = ({lin} / {block}) * {v} + px * {block} + {lin} % {block}; var sidx: i32 = 0; var coord: i32 = 0;"
                ));
            } else {
                blk.line(format!(
                    "var rem: i32 = {lin}; var sidx: i32 = 0; var coord: i32 = 0;"
                ));
            }
            for i in 0..rank {
                let ts = tstride[i].max(1);
                blk.line(format!("coord = rem / {ts}; rem = rem % {ts};"));
                if sdim(i) != 1 {
                    blk.line(format!("sidx = sidx + coord * {};", sstride[i]));
                }
            }
            if x_wide {
                // The source's global index, as this PE's block index.
                blk.line(format!(
                    "sidx = (sidx / {v}) * {block} + sidx % {v} - px * {block};"
                ));
            }
            blk.line(format!("{}[{lin}] = {xn}[sidx];", r.name));
        });
        Ok(())
    }

    fn order(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (id, x) = (op.args[0], self.arg(op, 0)?);
        if x.ty != Ty::F32 {
            return refuse("an ordering of non-floats");
        }
        Gen::lanes_only(&x, "an ordering")?;
        let (rows, len) = if op.tag == tags::SORT_DESC {
            if x.wide {
                return refuse("a full ordering of a wide value");
            }
            (1, x.n)
        } else {
            self.rows_len(id, &x)
        };
        let k = if op.tag == tags::TOP_K {
            op.imm as usize
        } else {
            len
        };
        if k < 1 || k > len {
            return refuse(format!("top-{k} of {len}"));
        }
        let values = self.result(op, 0)?;
        let index = self.result(op, 1)?;
        if index.ty != Ty::U32 {
            return refuse("an order that is not u32");
        }
        if values.wide || index.wide {
            return refuse("an ordering into a wide value");
        }
        self.need("g_sort");
        let key = self.temp("i32", len);
        let idx = self.temp("u32", len);
        let tk = self.temp("i32", len);
        let ti = self.temp("u32", len);
        let xn = x.name.clone();
        let held = Gen::held(&x, len);
        if !x.wide {
            self.each(rows, |blk, rr| {
                blk.line(format!(
                    "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {len}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
                ));
                blk.line(format!(
                    "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{tk}), @ptrcast([*]u32, &{ti}));"
                ));
                blk.nest(
                    &format!("{{ var j: i32 = 0; while (j < {k}) : (j += 1) {{"),
                    "} }",
                    |blk| {
                        blk.line(format!("var src: u32 = {idx}[j];"));
                        blk.line(format!("{}[{rr} * {k} + j] = src;", index.name));
                        blk.line(format!(
                            "{}[{rr} * {k} + j] = {xn}[{rr} * {len} + @as(i32, src)];",
                            values.name
                        ));
                    },
                );
            });
            return Ok(());
        }
        self.need("g_copy");
        // Each PE sorts its block (every row's sorted keys and indices kept
        // in `skey`/`sidx`); then k rounds, each the row's best remaining
        // candidate: every PE offers its next unused (key, global index,
        // value), the root takes the greatest key (lowest index on a tie)
        // and the owner advances.
        let block = self.block();
        // Every row sorted, kept in `skey`/`sidx` (`tk`/`ti` are a row's
        // scratch in the local path; here the storage, scratch included).
        self.free_now(&tk);
        self.free_now(&ti);
        let skey = self.temp("i32", rows * len);
        let sidx = self.temp("u32", rows * len);
        self.sort_rows(&xn, rows, len, &held, &key, &idx, &skey, &sidx);
        let next = self.temp("i32", rows);
        let round = self.temp("u32", 1);
        self.each(rows, |blk, rr| {
            blk.line(format!("{next}[{rr}] = 0;"));
        });
        self.b.line(format!("{round}[0] = 0;"));
        let send = self.temp("u32", 4 * rows);
        let cols = self.cols;
        let lp = self.open_loop(k);
        self.each(rows, |blk, rr| {
            blk.line(format!("var at: i32 = 4 * {rr}; var nx: i32 = {next}[{rr}];"));
            blk.line(format!(
                "if (nx < {held}) {{ var src: i32 = @as(i32, {sidx}[{rr} * {len} + nx]); {send}[at] = @bitcast(u32, {skey}[{rr} * {len} + nx]); {send}[at + 1] = @as(u32, px * {block} + src); {send}[at + 2] = @bitcast(u32, {xn}[{rr} * {len} + src]); {send}[at + 3] = 1; }} else {{ {send}[at] = 0; {send}[at + 1] = 0; {send}[at + 2] = 0; {send}[at + 3] = 0; }}"
            ));
        });
        let per_pe = 4 * rows;
        let res = self.combine(&send, per_pe, 3 * rows, |blk, recv| {
            blk.nest(
                &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                "} }",
                |blk| {
                    blk.line("var bc: i32 = -1; var bkey: i32 = 0; var bidx: u32 = 0;");
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var at: i32 = c * {per_pe} + 4 * rr;"));
                            blk.line(format!("if ({recv}[at + 3] == 0) {{ continue; }}"));
                            blk.line(format!(
                                "var ck: i32 = @bitcast(i32, {recv}[at]); var ci: u32 = {recv}[at + 1];"
                            ));
                            blk.line("if (bc < 0 or ck > bkey or (ck == bkey and ci < bidx)) { bc = c; bkey = ck; bidx = ci; }");
                        },
                    );
                    blk.line(format!(
                        "var src: i32 = bc * {per_pe} + 4 * rr; RES[3 * rr] = @as(u32, bc); RES[3 * rr + 1] = {recv}[src + 1]; RES[3 * rr + 2] = {recv}[src + 2];"
                    ));
                },
            );
        });
        let (vn, inm) = (values.name.clone(), index.name.clone());
        self.each(rows, |blk, rr| {
            blk.line(format!("var j: i32 = @bitcast(i32, {round}[0]);"));
            blk.line(format!(
                "{inm}[{rr} * {k} + j] = {res}[3 * {rr} + 1]; {vn}[{rr} * {k} + j] = @bitcast(f32, {res}[3 * {rr} + 2]);"
            ));
            blk.line(format!(
                "if (@bitcast(i32, {res}[3 * {rr}]) == px) {{ {next}[{rr}] = {next}[{rr}] + 1; }}"
            ));
        });
        self.b.line(format!("{round}[0] = {round}[0] + 1;"));
        self.close_loop(lp, k);
        Ok(())
    }

    fn gather(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (is, src) = (op.args[0], self.arg(op, 0)?);
        let idx = self.arg(op, 1)?;
        let ss = self.shape(is);
        let n0 = *ss.first().unwrap_or(&1) as usize;
        let rest = ss
            .iter()
            .skip(1)
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        let r = self.result(op, 0)?;
        if idx.wide || r.wide {
            return refuse("a gather by wide indices");
        }
        let k = idx.n;
        let zero = Gen::zero(r.ty);
        let (sn, rn, idn) = (src.name.clone(), r.name.clone(), idx.name.clone());
        let ixe = idx.ty.elem();
        let valid = Gen::index_valid(&idx, "ix", n0);
        if !src.wide {
            self.each(k, |blk, kk| {
                blk.line(format!("var ix: {ixe} = {idn}[{kk}];"));
                blk.nest(
                    &format!("{{ var q: i32 = 0; while (q < {rest}) : (q += 1) {{"),
                    "} }",
                    |blk| {
                        blk.line(format!(
                            "if ({valid}) {{ {rn}[{kk} * {rest} + q] = {sn}[@as(i32, ix) * {rest} + q]; }} else {{ {rn}[{kk} * {rest} + q] = {zero}; }}"
                        ));
                    },
                );
            });
            return Ok(());
        }
        // A wide source: a vector, each PE holding a block of it. The PE
        // owning an index answers it; the root takes the owner's word (a
        // zero for an index outside the vector).
        if ss.len() != 1 {
            return refuse("a gather from a wide matrix");
        }
        Gen::lanes_only(&src, "a gather")?;
        let block = self.block();
        let send = self.temp("u32", 2 * k);
        self.each(k, |blk, kk| {
            blk.line(format!("var ix: {ixe} = {idn}[{kk}];"));
            blk.line(format!("{send}[2 * {kk}] = 0; {send}[2 * {kk} + 1] = 0;"));
            blk.line(format!(
                "if ({valid}) {{ var g: i32 = @as(i32, ix) - px * {block}; if (g >= 0 and g < bn) {{ {send}[2 * {kk}] = {}; {send}[2 * {kk} + 1] = 1; }} }}",
                src.ty.to_word(&format!("{sn}[g]"))
            ));
        });
        let cols = self.cols;
        let res = self.combine(&send, 2 * k, k, |blk, recv| {
            blk.nest(
                &format!("{{ var kk: i32 = 0; while (kk < {k}) : (kk += 1) {{"),
                "} }",
                |blk| {
                    blk.line("RES[kk] = 0;");
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if ({recv}[c * {} + 2 * kk + 1] != 0) {{ RES[kk] = {recv}[c * {} + 2 * kk]; }}",
                                2 * k,
                                2 * k
                            ));
                        },
                    );
                },
            );
        });
        let rty = r.ty;
        self.each(k, |blk, kk| {
            blk.line(format!(
                "{rn}[{kk}] = {};",
                rty.of_word(&format!("{res}[{kk}]"))
            ));
        });
        Ok(())
    }

    fn gather_row(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (is, src) = (op.args[0], self.arg(op, 0)?);
        let idx = self.arg(op, 1)?;
        let ss = self.shape(is);
        if ss.len() != 2 {
            return refuse("gather_row of other than a matrix");
        }
        let (m, n) = (ss[0] as usize, ss[1] as usize);
        if idx.n != m || idx.wide {
            return refuse("gather_row's index is not one local index per row");
        }
        Gen::lanes_only(&src, "gather_row")?;
        let r = self.result(op, 0)?;
        if r.wide {
            return refuse("gather_row into a wide value");
        }
        let zero = Gen::zero(r.ty);
        let valid = Gen::index_valid(&idx, "ix", n);
        let (sn, rn, idn) = (src.name.clone(), r.name.clone(), idx.name.clone());
        let ixe = idx.ty.elem();
        if !src.wide {
            self.each(m, |blk, i| {
                blk.line(format!("var ix: {ixe} = {idn}[{i}];"));
                blk.line(format!(
                    "if ({valid}) {{ {rn}[{i}] = {sn}[{i} * {n} + @as(i32, ix)]; }} else {{ {rn}[{i}] = {zero}; }}"
                ));
            });
            return Ok(());
        }
        let block = self.block();
        let send = self.temp("u32", 2 * m);
        self.each(m, |blk, i| {
            blk.line(format!("var ix: {ixe} = {idn}[{i}];"));
            blk.line(format!("{send}[2 * {i}] = 0; {send}[2 * {i} + 1] = 0;"));
            blk.line(format!(
                "if ({valid}) {{ var g: i32 = @as(i32, ix) - px * {block}; if (g >= 0 and g < bn) {{ {send}[2 * {i}] = {}; {send}[2 * {i} + 1] = 1; }} }}",
                src.ty.to_word(&format!("{sn}[{i} * {block} + g]"))
            ));
        });
        let cols = self.cols;
        let res = self.combine(&send, 2 * m, m, |blk, recv| {
            blk.nest(
                &format!("{{ var i: i32 = 0; while (i < {m}) : (i += 1) {{"),
                "} }",
                |blk| {
                    blk.line("RES[i] = 0;");
                    blk.nest(
                        &format!("{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if ({recv}[c * {} + 2 * i + 1] != 0) {{ RES[i] = {recv}[c * {} + 2 * i]; }}",
                                2 * m,
                                2 * m
                            ));
                        },
                    );
                },
            );
        });
        let rty = r.ty;
        self.each(m, |blk, i| {
            blk.line(format!(
                "{rn}[{i}] = {};",
                rty.of_word(&format!("{res}[{i}]"))
            ));
        });
        Ok(())
    }

    fn pivot(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (ix, x) = (op.args[0], self.arg(op, 0)?);
        if x.ty != Ty::F32 {
            return refuse("a pivot over non-floats");
        }
        Gen::lanes_only(&x, "a pivot")?;
        let (rows, len) = self.rows_len(ix, &x);
        let payload = self.get(op.pred_payload)?;
        let pn = payload.n;
        if payload.wide || (pn != 1 && pn != rows) {
            return refuse(format!("a pivot payload of {pn} for {rows} rows"));
        }
        let r = self.result(op, 0)?;
        self.agree(&r, &[&x])?;
        if x.wide {
            return self.pivot_wide(op, &x, &payload, &r, rows, len);
        }
        let xn = x.name.clone();
        let rn = r.name.clone();
        match op.pred_tag {
            0 => {
                // rank_le(k): a lane is kept when fewer than k non-NaN lanes
                // are greater, i.e. when its order key is at or above the
                // k-th largest key.
                let k_expr = match payload.ty {
                    Ty::I32 => format!("g_iclamp(PAYLOAD, 0, {len})"),
                    Ty::U32 => format!("g_uclamp(PAYLOAD, {len})"),
                    _ => return refuse("rank_le over a non-integer k"),
                };
                self.need("g_int");
                self.need("g_sort");
                let key = self.temp("i32", len);
                let idx = self.temp("u32", len);
                let tk = self.temp("i32", len);
                let ti = self.temp("u32", len);
                let bottom = ilit(i32::MIN);
                self.each(rows, |blk, rr| {
                    let k_expr = k_expr.replace("PAYLOAD", &Gen::at(&payload, rr));
                    blk.line(format!("var k: i32 = {k_expr};"));
                    blk.line(format!(
                        "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {len}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
                    ));
                    blk.line("var count: i32 = 0;");
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if ({key}[j] != {bottom}) {{ count = count + 1; }}"
                            ));
                        },
                    );
                    blk.line("var t: i32 = k; if (count < t) { t = count; }");
                    blk.line(format!(
                        "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{tk}), @ptrcast([*]u32, &{ti}));"
                    ));
                    // `key` holds the keys descending now; a lane's own key
                    // is recomputed from its value.
                    blk.line(format!(
                        "var pivot: i32 = {bottom}; if (t > 0) {{ pivot = {key}[t - 1]; }}"
                    ));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var kj: i32 = g_okey({xn}[{rr} * {len} + j]);"));
                            blk.line(format!(
                                "if (t > 0 and kj != {bottom} and kj >= pivot) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            1 => {
                // cummass_le(p): in descending order, a lane is kept while
                // the mass before it is below p.
                if payload.ty != Ty::F32 {
                    return refuse("cummass_le over a non-float p");
                }
                self.need("g_sort");
                let key = self.temp("i32", len);
                let idx = self.temp("u32", len);
                let tk = self.temp("i32", len);
                let ti = self.temp("u32", len);
                let zero = flit(0.0);
                self.each(rows, |blk, rr| {
                    blk.line(format!("var p: f32 = {};", Gen::at(&payload, rr)));
                    blk.line(format!(
                        "g_keys(@ptrcast([*]f32, &{xn}), {rr} * {len}, {len}, {len}, @ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}));"
                    ));
                    blk.line(format!(
                        "g_msort(@ptrcast([*]i32, &{key}), @ptrcast([*]u32, &{idx}), {len}, @ptrcast([*]i32, &{tk}), @ptrcast([*]u32, &{ti}));"
                    ));
                    blk.line(format!("var excl: f32 = {zero};"));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var src: i32 = @as(i32, {idx}[j]);"));
                            blk.line(format!(
                                "if (!math.isNaN_f32(excl) and !math.isNaN_f32(p) and excl < p) {{ {rn}[{rr} * {len} + src] = 1; }} else {{ {rn}[{rr} * {len} + src] = 0; }}"
                            ));
                            blk.line(format!("excl = excl + {xn}[{rr} * {len} + src];"));
                        },
                    );
                });
                Ok(())
            }
            _ => {
                if payload.ty != Ty::F32 {
                    return refuse("prob_ge over a non-float threshold");
                }
                self.each(rows, |blk, rr| {
                    blk.line(format!("var thr: f32 = {};", Gen::at(&payload, rr)));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if (!math.isNaN_f32({xn}[{rr} * {len} + j]) and !math.isNaN_f32(thr) and {xn}[{rr} * {len} + j] >= thr) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
        }
    }

    /// A pivot over a wide value: the threshold is found by bisection over
    /// the order keys, every step one count (or mass) combined across the
    /// row; every PE walks the same bisection, so only the totals travel.
    #[allow(clippy::too_many_lines)]
    fn pivot_wide(
        &mut self,
        op: &LaunchOp,
        x: &Var,
        payload: &Var,
        r: &Var,
        rows: usize,
        len: usize,
    ) -> Lowering<()> {
        self.need("g_copy");
        self.need("g_sort");
        self.need("g_int");
        let xn = x.name.clone();
        let rn = r.name.clone();
        let total = rows * len;
        let key = self.temp("i32", len);
        let idx = self.temp("u32", len);
        let skey = self.temp("i32", total);
        let sidx = self.temp("u32", total);
        let bottom = ilit(i32::MIN);
        let top = ilit(i32::MAX);
        // Every row's keys sorted descending, with their source indices
        // (the block's lanes past this PE's share key as NaN does: last).
        self.sort_rows(&xn, rows, len, "bn", &key, &idx, &skey, &sidx);
        let cols = self.cols;
        let lo = self.temp("i32", rows);
        let hi = self.temp("i32", rows);
        let mid = self.temp("i32", rows);
        let send = self.temp("u32", rows);
        match op.pred_tag {
            0 => {
                // rank_le(k): the k-th largest key (k capped by the non-NaN
                // count); kept: a real lane at or above it.
                let k_expr = match payload.ty {
                    Ty::I32 => format!("g_iclamp(PAYLOAD, 0, {})", len * cols),
                    Ty::U32 => format!("g_uclamp(PAYLOAD, {})", len * cols),
                    _ => return refuse("rank_le over a non-integer k"),
                };
                // The non-NaN count across the row.
                // Counts travel as f32 (exact below 2^24) and add on the fabric.
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{send}[{rr}] = @bitcast(u32, @as(f32, g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {bottom} + 1)));"
                    ));
                });
                let totals = self.combine_sum(&send, rows);
                let t = self.temp("i32", rows);
                self.each(rows, |blk, rr| {
                    let k_expr = k_expr.replace("PAYLOAD", &Gen::at(payload, rr));
                    blk.line(format!("var k: i32 = {k_expr};"));
                    blk.line(format!(
                        "var count: i32 = @as(i32, @bitcast(f32, {totals}[{rr}])); if (count < k) {{ k = count; }} {t}[{rr}] = k;"
                    ));
                    blk.line(format!("{lo}[{rr}] = {bottom} + 1; {hi}[{rr}] = {top};"));
                });
                // Shortcut: with k within the candidates each PE can offer,
                // the k-th largest key is among the PEs' top keys, found by
                // the root in one round; the bisection then runs one round.
                let kmax = len.min((1024 / (cols * rows)).max(1));
                let cand = self.temp("u32", kmax * rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{{ var q: i32 = 0; while (q < {kmax}) : (q += 1) {{ {cand}[{rr} * {kmax} + q] = @bitcast(u32, {skey}[{rr} * {len} + q]); }} }}"
                    ));
                });
                let per_pe = kmax * rows;
                let taken = self.temp("u16", cols);
                let pick = self.combine(&cand, per_pe, 2 * rows, |blk, recv| {
                    blk.nest(
                        &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "var tt: i32 = {t}[rr]; RES[2 * rr] = 0; RES[2 * rr + 1] = 0;"
                            ));
                            blk.nest(&format!("if (tt > 0 and tt <= {kmax}) {{"), "}", |blk| {
                                blk.line(format!(
                                    "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ {taken}[c] = 0; }} }}"
                                ));
                                blk.line("var pivot: i32 = 0;");
                                blk.nest(
                                    "{ var j: i32 = 0; while (j < tt) : (j += 1) {",
                                    "} }",
                                    |blk| {
                                        blk.line("var bc: i32 = -1; var bkey: i32 = 0;");
                                        blk.line(format!(
                                            "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ if (@as(i32, {taken}[c]) >= {kmax}) {{ continue; }} var ck: i32 = @bitcast(i32, {recv}[c * {per_pe} + rr * {kmax} + @as(i32, {taken}[c])]); if (bc < 0 or ck > bkey) {{ bc = c; bkey = ck; }} }} }}"
                                        ));
                                        blk.line(format!("pivot = bkey; {taken}[bc] = {taken}[bc] + 1;"));
                                    },
                                );
                                blk.line("RES[2 * rr] = @bitcast(u32, pivot); RES[2 * rr + 1] = 1;");
                            });
                        },
                    );
                });
                let settled = self.temp("u32", 1);
                self.b.line(format!("{settled}[0] = 1;"));
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "if ({pick}[2 * {rr} + 1] != 0) {{ {lo}[{rr}] = @bitcast(i32, {pick}[2 * {rr}]); {hi}[{rr}] = {lo}[{rr}]; }} else {{ {settled}[0] = 0; }}"
                    ));
                });
                // Bisection: the largest T with count(key >= T) >= t (one
                // round when the shortcut settled every row).
                let lp = self.open_loop(32);
                let counter = lp.1.clone();
                self.b
                    .line(format!("if ({settled}[0] != 0) {{ {counter}[0] = 31; }}"));
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "var m64: i64 = (@as(i64, {lo}[{rr}]) + @as(i64, {hi}[{rr}]) + 1) / 2; {mid}[{rr}] = @as(i32, m64);"
                    ));
                    blk.line(format!(
                        "{send}[{rr}] = @bitcast(u32, @as(f32, g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {mid}[{rr}])));"
                    ));
                });
                let counts = self.combine_sum(&send, rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "if ({lo}[{rr}] < {hi}[{rr}]) {{ if (@as(i32, @bitcast(f32, {counts}[{rr}])) >= {t}[{rr}]) {{ {lo}[{rr}] = {mid}[{rr}]; }} else {{ {hi}[{rr}] = {mid}[{rr}] - 1; }} }}"
                    ));
                });
                self.close_loop(lp, 32);
                self.each(rows, |blk, rr| {
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var kj: i32 = g_okey({xn}[{rr} * {len} + j]);"));
                            blk.line(format!(
                                "if ({t}[{rr}] > 0 and kj != {bottom} and kj >= {lo}[{rr}]) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            1 => {
                // cummass_le(p): in descending order, a lane is kept while
                // the mass before it is below p. The boundary key T* is the
                // lowest key whose mass above it is below p; keys above T*
                // are kept, ties at T* in index order while the mass stays
                // below p, lower keys not.
                if payload.ty != Ty::F32 {
                    return refuse("cummass_le over a non-float p");
                }
                // Prefix masses of every sorted row.
                let pre = self.temp("f32", rows * (len + 1));
                let zero = flit(0.0);
                let len1 = len + 1;
                self.each(rows, |blk, rr| {
                    blk.line(format!("var acc: f32 = {zero}; {pre}[{rr} * {len1}] = acc;"));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if (j < bn) {{ acc = acc + {xn}[{rr} * {len} + @as(i32, {sidx}[{rr} * {len} + j])]; }} {pre}[{rr} * {len1} + j + 1] = acc;"
                            ));
                        },
                    );
                    blk.line(format!("{lo}[{rr}] = {bottom} + 1; {hi}[{rr}] = {top};"));
                });
                // Shortcut: each PE offers the values of its top candidates
                // (the key is the value's); the root walks the merged order
                // adding the mass as the interpreter does. The walk settles
                // when it stops at a key strictly below the last kept one
                // (`Kb`) with no PE run dry above it: every PE then keeps its
                // lanes keyed at or above `Kb`. A stop inside a run of equal
                // keys, or past a PE's candidates, leaves the row to the
                // bisection.
                let m = len.min((512 / (cols * rows)).max(1));
                let cand = self.temp("u32", m * rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{{ var q: i32 = 0; while (q < {m}) : (q += 1) {{ {cand}[{rr} * {m} + q] = @bitcast(u32, {xn}[{rr} * {len} + @as(i32, {sidx}[{rr} * {len} + q])]); }} }}"
                    ));
                });
                let per_pe = m * rows;
                let taken = self.temp("u16", cols);
                let pv_root = Gen::at(payload, "rr");
                let walk = self.combine(&cand, per_pe, 2 * rows, |blk, recv| {
                    blk.nest(
                        &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "var p: f32 = {pv_root}; var excl: f32 = {zero}; var ok: bool = true; var more: bool = true; var kb: i32 = {top}; var any: bool = false;"
                            ));
                            blk.line(format!(
                                "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ {taken}[c] = 0; }} }}"
                            ));
                            blk.nest("while (more) {", "}", |blk| {
                                blk.line("var bc: i32 = -1; var bkey: i32 = 0; var exhausted: bool = false;");
                                blk.line(format!(
                                    "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ if (@as(i32, {taken}[c]) >= {m}) {{ if ({m} < {len}) {{ exhausted = true; }} continue; }} var ck: i32 = g_okey(@bitcast(f32, {recv}[c * {per_pe} + rr * {m} + @as(i32, {taken}[c])])); if (bc < 0 or ck > bkey) {{ bc = c; bkey = ck; }} }} }}"
                                ));
                                // The next in order is kept while the mass before it is
                                // below p; the walk stops at the first not kept, and is
                                // void when a PE ran out of candidates before that, or
                                // when it stops among equal keys.
                                blk.line("if (bc < 0) { more = false; if (exhausted) { ok = false; } }");
                                blk.nest("else {", "}", |blk| {
                                    blk.line("var keep: bool = !math.isNaN_f32(excl) and !math.isNaN_f32(p) and excl < p;");
                                    blk.line("if (!keep) { more = false; if (any and bkey == kb) { ok = false; } }");
                                    blk.line(format!(
                                        "else {{ if (exhausted and bkey <= g_exhausted_value(@ptrcast([*]u32, &{recv}), {per_pe}, {m}, rr, @ptrcast([*]u16, &{taken}), {cols})) {{ ok = false; more = false; }} else {{ excl = excl + @bitcast(f32, {recv}[bc * {per_pe} + rr * {m} + @as(i32, {taken}[bc])]); {taken}[bc] = {taken}[bc] + 1; kb = bkey; any = true; }} }}"
                                    ));
                                });
                            });
                            blk.line(format!(
                                "if (ok) {{ RES[2 * rr] = 1; }} else {{ RES[2 * rr] = 0; }} if (any) {{ RES[2 * rr + 1] = @bitcast(u32, kb); }} else {{ RES[2 * rr + 1] = @bitcast(u32, {top}); }}"
                            ));
                        },
                    );
                });
                let settled = self.temp("u32", 1);
                let okrow = self.temp("u32", rows);
                self.b.line(format!("{settled}[0] = 1;"));
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{okrow}[{rr}] = {walk}[2 * {rr}]; if ({okrow}[{rr}] == 0) {{ {settled}[0] = 0; }}"
                    ));
                });
                let pv = |rr: &str| Gen::at(payload, rr);
                // Bisection: the smallest T with mass(key > T) < p (one round
                // when the walk settled every row).
                let lp = self.open_loop(32);
                let counter = lp.1.clone();
                self.b
                    .line(format!("if ({settled}[0] != 0) {{ {counter}[0] = 31; }}"));
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "var m64: i64 = @as(i64, {lo}[{rr}]) + (@as(i64, {hi}[{rr}]) - @as(i64, {lo}[{rr}])) / 2; {mid}[{rr}] = @as(i32, m64);"
                    ));
                    blk.line(format!(
                        "var above: i32 = g_count_gt(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {mid}[{rr}]); {send}[{rr}] = @bitcast(u32, {pre}[{rr} * {len1} + above]);"
                    ));
                });
                let mass = self.combine_sum(&send, rows);
                self.each(rows, |blk, rr| {
                    let p = pv(rr);
                    blk.line(format!(
                        "if ({lo}[{rr}] < {hi}[{rr}]) {{ var mm: f32 = @bitcast(f32, {mass}[{rr}]); if (!math.isNaN_f32(mm) and !math.isNaN_f32({p}) and mm < {p}) {{ {hi}[{rr}] = {mid}[{rr}]; }} else {{ {lo}[{rr}] = {mid}[{rr}] + 1; }} }}"
                    ));
                });
                self.close_loop(lp, 32);
                // T*: the lowest key present at or above the bisection's T,
                // or none (nothing is kept, as when p is NaN or at most 0):
                // each PE offers its lowest such key (the top key for none),
                // the root takes the least.
                let send2 = self.temp("u32", rows);
                self.each(rows, |blk, rr| {
                    let p = pv(rr);
                    blk.line(format!(
                        "var ge: i32 = g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, {lo}[{rr}]);"
                    ));
                    blk.line(format!(
                        "var ok: bool = ge > 0 and !math.isNaN_f32({p}) and {zero} < {p};"
                    ));
                    blk.line(format!(
                        "if (ok) {{ {send2}[{rr}] = @bitcast(u32, {skey}[{rr} * {len} + ge - 1]); }} else {{ {send2}[{rr}] = @bitcast(u32, {top}); }}"
                    ));
                });
                let tstar = self.combine(&send2, rows, rows, |blk, recv| {
                    blk.nest(
                        &format!("{{ var rr: i32 = 0; while (rr < {rows}) : (rr += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var best: i32 = {top};"));
                            blk.line(format!(
                                "{{ var c: i32 = 0; while (c < {cols}) : (c += 1) {{ var kk: i32 = @bitcast(i32, {recv}[c * {rows} + rr]); if (kk < best) {{ best = kk; }} }} }}"
                            ));
                            blk.line("RES[rr] = @bitcast(u32, best);");
                        },
                    );
                });
                // The mass above T* and the ties at T* (a count), summed on
                // the fabric; the tie value is T*'s own. Every PE then knows
                // how many ties are kept (`m`): all, none, or the first m in
                // index order, found by bisecting the index space (a round
                // per bit; one round when no row has a partial tie).
                let send3 = self.temp("u32", 2 * rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!("var ts: i32 = @bitcast(i32, {tstar}[{rr}]);"));
                    blk.line(format!(
                        "var above: i32 = g_count_gt(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, ts); var ge: i32 = g_count_ge(@ptrcast([*]i32, &{skey}), {rr} * {len}, {len}, ts);"
                    ));
                    blk.line(format!(
                        "{send3}[2 * {rr}] = @bitcast(u32, {pre}[{rr} * {len1} + above]); {send3}[2 * {rr} + 1] = @bitcast(u32, @as(f32, ge - above));"
                    ));
                });
                let sums = self.combine_sum(&send3, 2 * rows);
                let kept_ties = self.temp("i32", rows);
                let all_ties = self.temp("i32", rows);
                let ilo = self.temp("i32", rows);
                let ihi = self.temp("i32", rows);
                let partial = self.temp("u32", 1);
                let v = self.wide.unwrap_or(1) as usize;
                self.b.line(format!("{partial}[0] = 0;"));
                self.each(rows, |blk, rr| {
                    let p = pv(rr);
                    blk.line(format!(
                        "var ts: i32 = @bitcast(i32, {tstar}[{rr}]); var tv: f32 = g_unkey(ts); var mass: f32 = @bitcast(f32, {sums}[2 * {rr}]); var ties: i32 = @as(i32, @bitcast(f32, {sums}[2 * {rr} + 1]));"
                    ));
                    blk.line(format!(
                        "var m: i32 = 0; var excl: f32 = mass; if (ts != {top}) {{ while (m < ties) {{ if (math.isNaN_f32(excl) or math.isNaN_f32({p}) or !(excl < {p})) {{ break; }} excl = excl + tv; m = m + 1; }} }}"
                    ));
                    blk.line(format!(
                        "{kept_ties}[{rr}] = m; {all_ties}[{rr}] = ties; {ilo}[{rr}] = 0; {ihi}[{rr}] = {v}; if (m > 0 and m < ties and {okrow}[{rr}] == 0) {{ {partial}[0] = 1; }}"
                    ));
                });
                // The smallest index bound I with m ties below it: ties with
                // an index under I are kept.
                let ibits = (v + 1).next_power_of_two().trailing_zeros() as usize + 1;
                let isend = self.temp("u32", rows);
                let imid = self.temp("i32", rows);
                let lp2 = self.open_loop(ibits);
                let counter2 = lp2.1.clone();
                self.b.line(format!(
                    "if ({partial}[0] == 0) {{ {counter2}[0] = {}; }}",
                    ibits - 1
                ));
                let block = self.block();
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "{imid}[{rr}] = {ilo}[{rr}] + ({ihi}[{rr}] - {ilo}[{rr}]) / 2; var ts: i32 = @bitcast(i32, {tstar}[{rr}]); var below: i32 = 0;"
                    ));
                    blk.line(format!(
                        "{{ var j: i32 = 0; while (j < bn) : (j += 1) {{ if (px * {block} + j < {imid}[{rr}] and g_okey({xn}[{rr} * {len} + j]) == ts) {{ below = below + 1; }} }} }}"
                    ));
                    blk.line(format!("{isend}[{rr}] = @bitcast(u32, @as(f32, below));"));
                });
                let belows = self.combine_sum(&isend, rows);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "if ({ilo}[{rr}] < {ihi}[{rr}]) {{ if (@as(i32, @bitcast(f32, {belows}[{rr}])) >= {kept_ties}[{rr}]) {{ {ihi}[{rr}] = {imid}[{rr}]; }} else {{ {ilo}[{rr}] = {imid}[{rr}] + 1; }} }}"
                    ));
                });
                self.close_loop(lp2, ibits);
                self.each(rows, |blk, rr| {
                    blk.line(format!(
                        "var ts: i32 = @bitcast(i32, {tstar}[{rr}]); var any: bool = ts != {top}; var m: i32 = {kept_ties}[{rr}]; var ties: i32 = {all_ties}[{rr}]; var ibound: i32 = {ilo}[{rr}];"
                    ));
                    blk.line(format!(
                        "var walked: bool = {okrow}[{rr}] != 0; var kb: i32 = @bitcast(i32, {walk}[2 * {rr} + 1]);"
                    ));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!("var kj: i32 = g_okey({xn}[{rr} * {len} + j]); var keep: bool = false;"));
                            // The walk's verdict: lanes keyed at or above the last kept.
                            blk.line(format!("if (walked) {{ keep = kb != {top} and kj != {bottom} and kj >= kb; }}"));
                            blk.line(format!(
                                "else if (any) {{ if (kj > ts) {{ keep = true; }} else if (kj == ts) {{ if (m >= ties) {{ keep = true; }} else if (m > 0) {{ keep = px * {block} + j < ibound; }} }} }}"
                            ));
                            blk.line(format!(
                                "if (keep) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
            _ => {
                if payload.ty != Ty::F32 {
                    return refuse("prob_ge over a non-float threshold");
                }
                self.each(rows, |blk, rr| {
                    blk.line(format!("var thr: f32 = {};", Gen::at(payload, rr)));
                    blk.nest(
                        &format!("{{ var j: i32 = 0; while (j < {len}) : (j += 1) {{"),
                        "} }",
                        |blk| {
                            blk.line(format!(
                                "if (!math.isNaN_f32({xn}[{rr} * {len} + j]) and !math.isNaN_f32(thr) and {xn}[{rr} * {len} + j] >= thr) {{ {rn}[{rr} * {len} + j] = 1; }} else {{ {rn}[{rr} * {len} + j] = 0; }}"
                            ));
                        },
                    );
                });
                Ok(())
            }
        }
    }

    fn scatter(&mut self, op: &LaunchOp) -> Lowering<()> {
        let (ib, base) = (op.args[0], self.arg(op, 0)?);
        let idx = self.arg(op, 1)?;
        let vals = self.arg(op, 2)?;
        let add = op.tag == tags::SCATTER_ADD;
        if add && base.ty == Ty::Bool {
            return refuse("scatter_add into bools");
        }
        if vals.ty != base.ty {
            return refuse("scatter values of another dtype than the base");
        }
        if idx.wide || vals.wide {
            return refuse("a scatter by wide indices or of wide values");
        }
        let bs = self.shape(ib);
        let n0 = *bs.first().unwrap_or(&1) as usize;
        let rest = bs
            .iter()
            .skip(1)
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        let k = idx.n;
        let scalar = vals.n == 1 && k * rest != 1;
        if !scalar && vals.n != k * rest {
            return refuse(format!("{} scatter values for {k} rows of {rest}", vals.n));
        }
        if base.wide && bs.len() != 1 {
            return refuse("a scatter into a wide matrix");
        }
        Gen::lanes_only(&base, "a scatter")?;
        let r = self.result(op, 0)?;
        self.agree(&r, &[&base])?;
        let (bn, rn, idn, vn) = (
            base.name.clone(),
            r.name.clone(),
            idx.name.clone(),
            vals.name.clone(),
        );
        self.copy_of(r.ty, &rn, "0", &bn, "0", r.n);
        let valid = Gen::index_valid(&idx, "ix", n0);
        let ixe = idx.ty.elem();
        let block = self.block();
        let wide = base.wide;
        // Sequential: a later write to a row wins. A wide base's PE applies
        // the writes landing in its block.
        self.each(k, |blk, kk| {
            blk.line(format!("var ix: {ixe} = {idn}[{kk}];"));
            blk.nest(&format!("if ({valid}) {{"), "}", |blk| {
                if wide {
                    blk.line(format!("var g: i32 = @as(i32, ix) - px * {block};"));
                    blk.line("if (g >= 0 and g < bn) {");
                } else {
                    blk.line("var g: i32 = @as(i32, ix); {");
                }
                blk.nest(
                    &format!("{{ var q: i32 = 0; while (q < {rest}) : (q += 1) {{"),
                    "} }",
                    |blk| {
                        let src = if scalar {
                            format!("{vn}[0]")
                        } else {
                            format!("{vn}[{kk} * {rest} + q]")
                        };
                        let dst = format!("{rn}[g * {rest} + q]");
                        if add {
                            blk.line(format!("{dst} = {dst} + {src};"));
                        } else {
                            blk.line(format!("{dst} = {src};"));
                        }
                    },
                );
                blk.line("}");
            });
        });
        Ok(())
    }
}
