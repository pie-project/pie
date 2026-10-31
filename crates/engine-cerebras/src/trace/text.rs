//! A rendered program as one text and back: the compile cache's key and
//! what crosses from the tracer to the device: the manifest as JSON, then
//! the layout and PE sources, each under a `===<name>` line.

use super::*;

/// One text for a rendered program (hashed and cached as one).
#[must_use]
pub fn render(r: &Rendered) -> String {
    use kernels_cerebras::phase::{LaneKind, PageSource, Shard};
    let manifest = serde_json::json!({
        "entry": r.manifest.entry,
        "rect": r.manifest.rect,
        "exports": r.manifest.exports.iter().map(|e| serde_json::json!({
            "buf": e.buf, "name": e.name, "rows": e.rows, "width": e.width,
            "dtype": format!("{:?}", e.dtype), "elem": e.elem,
            "role": format!("{:?}", e.role),
            "packed": e.packed,
            "keep": e.keep,
            "shard": match e.shard {
                Shard::Whole => "whole".to_string(),
                Shard::Rows(n) => format!("rows:{n}"),
                Shard::Cols(n) => format!("cols:{n}"),
                Shard::Local(n) => format!("local:{n}"),
                Shard::RowsBy { parts, period } => format!("rowsby:{parts}:{period}"),
                Shard::Grid { rows, cols } => format!("grid:{rows}:{cols}"),
                Shard::ColsBy { parts, period } => format!("colsby:{parts}:{period}"),
                Shard::SumCols { parts, period } => format!("sumcols:{parts}:{period}"),
                Shard::Tile { rows, cols, segments } => format!("tile:{rows}:{cols}:{segments}"),
                Shard::RowsDepth { rows, cols, depth } => format!("rowsdepth:{rows}:{cols}:{depth}"),
                Shard::ColsDepth { rows, cols, depth } => format!("colsdepth:{rows}:{cols}:{depth}"),
                Shard::SumGrid { rows, cols, depth } => format!("sumgrid:{rows}:{cols}:{depth}"),
                Shard::Roots { rows, cols, depth } => format!("roots:{rows}:{cols}:{depth}"),
            },
        })).collect::<Vec<_>>(),
        "views": r.manifest.views.iter().map(|v| serde_json::json!({
            "name": v.name, "root": v.root, "offset": v.offset, "len": v.len,
        })).collect::<Vec<_>>(),
        "host": r.manifest.host.as_ref().map(|h| serde_json::json!({
            "rounds": h.rounds,
            "op": match &h.op {
                HostOp::Embed { ids, table, y, rows, width, limit } => serde_json::json!({
                    "kind": "embed", "ids": ids, "table": table, "y": y, "rows": rows, "width": width, "limit": limit,
                }),
                HostOp::SplitRows { x, left, right, rows, width, cut } => serde_json::json!({
                    "kind": "split_rows", "x": x, "left": left, "right": right, "rows": rows, "width": width, "cut": cut,
                }),
                HostOp::SplitQGate { packed, q, gate, rows, heads, head_dim } => serde_json::json!({
                    "kind": "split_q_gate", "packed": packed, "q": q, "gate": gate, "rows": rows, "heads": heads, "head_dim": head_dim,
                }),
                HostOp::Matmul { act, w, y, m, k, n } => serde_json::json!({
                    "kind": "matmul", "act": act, "w": w, "y": y, "m": m, "k": k, "n": n,
                }),
                HostOp::Argmax { x, y, rows, width, column, y_width } => serde_json::json!({
                    "kind": "argmax", "x": x, "y": y, "rows": rows, "width": width, "column": column, "y_width": y_width,
                }),
                HostOp::ArgmaxMerge { parts, y, rows, cg, block, column, y_width } => serde_json::json!({
                    "kind": "argmax_merge", "parts": parts, "y": y, "rows": rows, "cg": cg, "block": block, "column": column, "y_width": y_width,
                }),
                HostOp::SplitQkv { packed, q, k, v, rows, q_width, kv_width } => serde_json::json!({
                    "kind": "split_qkv", "packed": packed, "q": q, "k": k, "v": v, "rows": rows, "q_width": q_width, "kv_width": kv_width,
                }),
                HostOp::EmbedWeighted { ids, weights, table, y, rows, taps, width, limit } => serde_json::json!({
                    "kind": "embed_weighted", "ids": ids, "weights": weights, "table": table, "y": y,
                    "rows": rows, "taps": taps, "width": width, "limit": limit,
                }),
                HostOp::TopkSoftmax { logits, routes, weights, rows, width, experts, top_k } => serde_json::json!({
                    "kind": "topk_softmax", "logits": logits, "routes": routes, "weights": weights,
                    "rows": rows, "width": width, "experts": experts, "top_k": top_k,
                }),
                HostOp::MatmulGrouped { x, w, routes, y, rows, groups, k, n, experts } => serde_json::json!({
                    "kind": "matmul_grouped", "x": x, "w": w, "routes": routes, "y": y,
                    "rows": rows, "groups": groups, "k": k, "n": n, "experts": experts,
                }),
                HostOp::TopK { x, values, indices, rows, width, k } => serde_json::json!({
                    "kind": "topk", "x": x, "values": values, "indices": indices,
                    "rows": rows, "width": width, "k": k,
                }),
                HostOp::SelectorWalk { cand, indptr, unary, hp, tokens, pred, succ, picks, rows, k, rank, vocab, first } => serde_json::json!({
                    "kind": "selector_walk", "cand": cand, "indptr": indptr, "unary": unary, "hp": hp,
                    "tokens": tokens, "pred": pred, "succ": succ, "picks": picks,
                    "rows": rows, "k": k, "rank": rank, "vocab": vocab, "first": first,
                }),
                HostOp::PleNgramIds { ids, indptr, slots, state, out, rows, eos, mults, primes, offsets, heads_per_ngram } => serde_json::json!({
                    "kind": "ple_ngram_ids", "ids": ids, "indptr": indptr, "slots": slots, "state": state, "out": out,
                    "rows": rows, "eos": eos, "mults": mults, "primes": primes, "offsets": offsets,
                    "heads_per_ngram": heads_per_ngram,
                }),
                HostOp::PageWrite { src, table, write_page, write_offset, rows, width, page_size, table_rows } => serde_json::json!({
                    "kind": "page_write", "src": src, "table": table, "write_page": write_page, "write_offset": write_offset,
                    "rows": rows, "width": width, "page_size": page_size, "table_rows": table_rows,
                }),
                HostOp::Boundary { positions, request_of_token, row_valid, boundary_pos, boundary_req, boundary_rope, rows, ratio } => serde_json::json!({
                    "kind": "boundary", "positions": positions, "request_of_token": request_of_token, "row_valid": row_valid,
                    "boundary_pos": boundary_pos, "boundary_req": boundary_req, "boundary_rope": boundary_rope,
                    "rows": rows, "ratio": ratio,
                }),
                HostOp::BlockMean { boundary_pos, boundary_req, keys, indices, indptr, entries, rows, head_dim, ratio, page_size } => serde_json::json!({
                    "kind": "block_mean", "boundary_pos": boundary_pos, "boundary_req": boundary_req, "keys": keys,
                    "indices": indices, "indptr": indptr, "entries": entries,
                    "rows": rows, "head_dim": head_dim, "ratio": ratio, "page_size": page_size,
                }),
                HostOp::PoolWrite { entries, boundary_pos, boundary_req, keys, indices, indptr, rows, width, page_size } => serde_json::json!({
                    "kind": "pool_write", "entries": entries, "boundary_pos": boundary_pos, "boundary_req": boundary_req,
                    "keys": keys, "indices": indices, "indptr": indptr, "rows": rows, "width": width, "page_size": page_size,
                }),
                HostOp::IndexTopk { q, weights, keys, indices, indptr, positions, request_of_token, selection, rows, heads, head_dim, top_k, ratio, page_size, max_pages } => serde_json::json!({
                    "kind": "index_topk", "q": q, "weights": weights, "keys": keys, "indices": indices, "indptr": indptr,
                    "positions": positions, "request_of_token": request_of_token, "selection": selection,
                    "rows": rows, "heads": heads, "head_dim": head_dim, "top_k": top_k, "ratio": ratio,
                    "page_size": page_size, "max_pages": max_pages,
                }),
                HostOp::GroupRoutes { routes, rows, groups } => serde_json::json!({
                    "kind": "group_routes", "routes": routes, "rows": rows, "groups": groups,
                }),
                HostOp::EmbedConcat { ids, table, y, rows, heads, width, limit } => serde_json::json!({
                    "kind": "embed_concat", "ids": ids, "table": table, "y": y,
                    "rows": rows, "heads": heads, "width": width, "limit": limit,
                }),
                HostOp::Cells { tgrid, positions, y, clips, rows, axis, width, base } => serde_json::json!({
                    "kind": "cells", "tgrid": tgrid, "positions": positions, "y": y,
                    "clips": clips, "rows": rows, "axis": axis, "width": width, "base": base,
                }),
                HostOp::ConcatCols { x, y, rows, width, stride, at } => serde_json::json!({
                    "kind": "concat_cols", "x": x, "y": y, "rows": rows, "width": width,
                    "stride": stride, "at": at,
                }),
                HostOp::GroupRows { csr, groups, lanes, y, csr_len, lane_count, rows } => serde_json::json!({
                    "kind": "group_rows", "csr": csr, "groups": groups, "lanes": lanes, "y": y,
                    "csr_len": csr_len, "lane_count": lane_count, "rows": rows,
                }),
                HostOp::GroupFirst { x, csr, groups, lanes, y, csr_len, lane_count, rows, width, x_rows } => serde_json::json!({
                    "kind": "group_first", "x": x, "csr": csr, "groups": groups, "lanes": lanes,
                    "y": y, "csr_len": csr_len, "lane_count": lane_count, "rows": rows,
                    "width": width, "x_rows": x_rows,
                }),
                HostOp::Patch { x, grid, y, o_grid, clips, rows, channels, p, fold, bases } => serde_json::json!({
                    "kind": "patch", "x": x, "grid": grid, "y": y, "o_grid": o_grid,
                    "clips": clips, "rows": rows, "channels": channels,
                    "p0": p[0], "p1": p[1], "p2": p[2], "fold": u32::from(*fold),
                    "x_base": bases[0], "y_base": bases[1],
                }),
            },
        })),
        "table": r.manifest.table.as_ref().map(table_json),
        "tables": r.manifest.tables.iter().map(table_json).collect::<Vec<_>>(),
        "lanes": r.manifest.lanes.iter().map(|l| serde_json::json!({
            "pes": l.pes,
            "kind": match &l.kind {
                LaneKind::PerRow { lanes } => serde_json::json!({ "per_row": lanes }),
                LaneKind::Ragged { indptr, lanes } => serde_json::json!({ "indptr": indptr, "lanes": lanes }),
                LaneKind::ByTable { lane_of_row, lanes } => serde_json::json!({ "lane_of_row": lane_of_row, "lanes": lanes }),
            },
            "pages": l.pages.as_ref().map(|p| serde_json::json!({
                "source": match &p.source {
                    PageSource::Csr { indptr, indices } => serde_json::json!({ "indptr": indptr, "indices": indices }),
                    PageSource::Rows { table } => serde_json::json!({ "table": table }),
                },
                "page_size": p.page_size, "max_pages": p.max_pages, "rewritten": p.rewritten, "page_groups": p.page_groups,
                "strided": p.strided, "strided_pages": p.strided_pages,
            })),
            "header": l.header, "slots": l.slots, "lanes_per_pe": l.lanes_per_pe, "a_first": l.a_first,
            "banks": l.banks.iter().map(|(n, s, split)| serde_json::json!([n, s, match split {
                BankSplit::Whole => "whole",
                BankSplit::Block => "block",
                BankSplit::ByB => "by_b",
            }])).collect::<Vec<_>>(),
            "row_outputs": l.row_outputs,
            "reduce": l.reduce.iter().map(|(n, r)| match r {
                Reduce::LogSumExp => serde_json::json!({ "name": n, "lse": serde_json::Value::Null }),
                Reduce::Weighted { lse } => serde_json::json!({ "name": n, "lse": lse }),
                Reduce::Sum => serde_json::json!({ "name": n, "sum": true }),
                Reduce::FabricSum { group } => serde_json::json!({ "name": n, "fabric": group }),
                Reduce::FabricRoot { group } => serde_json::json!({ "name": n, "root": group }),
            }).collect::<Vec<_>>(),
            "cols": l.cols.iter().map(|c| serde_json::json!({
                "name": c.name, "rows": c.rows, "width": c.width, "by_rows": c.by_rows, "row_block": c.row_block,
                "segments": c.segments.iter().map(|s| [s.base, s.a_group, s.a_stride, s.b_stride, s.span]).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "window": l.window.map(|(start, len)| [start, len]),
            "block": l.block.map(|b| [b.a, b.b, b.w, b.a_groups, b.b_groups]),
        })).collect::<Vec<_>>(),
    });
    format!(
        "===manifest.json\n{manifest}\n===layout.csl\n{}===pe.csl\n{}",
        r.layout, r.pe
    )
}

/// The rendered program back from its text.
#[must_use]
pub fn unrender(text: &str) -> Option<Rendered> {
    use crate::fabric::lanes::{BlockPlan, LanePlan, PagePlan};
    use crate::fabric::{Export, Manifest, Symbol, View};
    use kernels_cerebras::phase::{LaneKind, PageSource, Shard};
    let rest = text.strip_prefix("===manifest.json\n")?;
    let (manifest, rest) = rest.split_once("===layout.csl\n")?;
    let (layout, pe) = rest.split_once("===pe.csl\n")?;
    let m: serde_json::Value = serde_json::from_str(manifest.trim()).ok()?;
    let dtype_of = |s: &str| -> Option<Dtype> {
        Some(match s {
            "F32" => Dtype::F32,
            "Bf16" => Dtype::Bf16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "U8" => Dtype::U8,
            _ => return None,
        })
    };
    let mut exports = Vec::new();
    for e in m["exports"].as_array()? {
        exports.push(Export {
            buf: e["buf"].as_u64()? as u32,
            name: e["name"].as_str()?.to_string(),
            rows: e["rows"].as_u64()? as u32,
            width: e["width"].as_u64()? as u32,
            dtype: dtype_of(e["dtype"].as_str()?)?,
            packed: e["packed"].as_bool().unwrap_or(false),
            keep: e["keep"].as_bool().unwrap_or(false),
            elem: match e["elem"].as_str()? {
                "f32" => "f32",
                "i32" => "i32",
                "u32" => "u32",
                _ => return None,
            },
            role: match e["role"].as_str()? {
                "Input" => Symbol::Input,
                "Output" => Symbol::Output,
                "InOut" => Symbol::InOut,
                _ => return None,
            },
            shard: match e["shard"].as_str().unwrap_or("whole") {
                "whole" => Shard::Whole,
                other => {
                    let (kind, n) = other.split_once(':')?;
                    let nums: Vec<u32> = n
                        .split(':')
                        .map(|p| p.parse::<u32>().ok())
                        .collect::<Option<_>>()?;
                    let (n, second) = (*nums.first()?, nums.get(1).copied());
                    match (kind, second) {
                        ("tile", Some(cols)) => Shard::Tile {
                            rows: n,
                            cols,
                            segments: *nums.get(2)?,
                        },
                        ("rowsdepth", Some(cols)) => Shard::RowsDepth {
                            rows: n,
                            cols,
                            depth: *nums.get(2)?,
                        },
                        ("colsdepth", Some(cols)) => Shard::ColsDepth {
                            rows: n,
                            cols,
                            depth: *nums.get(2)?,
                        },
                        ("sumgrid", Some(cols)) => Shard::SumGrid {
                            rows: n,
                            cols,
                            depth: *nums.get(2)?,
                        },
                        ("roots", Some(cols)) => Shard::Roots {
                            rows: n,
                            cols,
                            depth: *nums.get(2)?,
                        },
                        ("rows", None) => Shard::Rows(n),
                        ("cols", None) => Shard::Cols(n),
                        ("local", None) => Shard::Local(u64::from(n)),
                        ("rowsby", Some(period)) => Shard::RowsBy { parts: n, period },
                        ("grid", Some(cols)) => Shard::Grid { rows: n, cols },
                        ("colsby", Some(period)) => Shard::ColsBy { parts: n, period },
                        ("sumcols", Some(period)) => Shard::SumCols { parts: n, period },
                        _ => return None,
                    }
                }
            },
        });
    }
    let mut views = Vec::new();
    for v in m["views"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        views.push(View {
            name: v["name"].as_str()?.to_string(),
            root: v["root"].as_str()?.to_string(),
            offset: v["offset"].as_u64()?,
            len: v["len"].as_u64()?,
        });
    }
    Some(Rendered {
        layout: layout.to_string(),
        pe: pe.to_string(),
        manifest: Manifest {
            exports,
            entry: m["entry"].as_str()?.to_string(),
            rect: (m["rect"][0].as_u64()? as u32, m["rect"][1].as_u64()? as u32),
            views,
            host: match &m["host"] {
                serde_json::Value::Null => None,
                h => {
                    let s = |k: &str| h["op"][k].as_str().map(str::to_string);
                    let u = |k: &str| h["op"][k].as_u64().map(|v| v as u32);
                    let op = match h["op"]["kind"].as_str()? {
                        "embed" => HostOp::Embed {
                            ids: s("ids")?,
                            table: s("table")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            limit: u("limit")?,
                        },
                        "split_rows" => HostOp::SplitRows {
                            x: s("x")?,
                            left: s("left")?,
                            right: s("right")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            cut: u("cut")?,
                        },
                        "split_q_gate" => HostOp::SplitQGate {
                            packed: s("packed")?,
                            q: s("q")?,
                            gate: s("gate")?,
                            rows: u("rows")?,
                            heads: u("heads")?,
                            head_dim: u("head_dim")?,
                        },
                        "topk_softmax" => HostOp::TopkSoftmax {
                            logits: s("logits")?,
                            routes: s("routes")?,
                            weights: s("weights")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            experts: u("experts")?,
                            top_k: u("top_k")?,
                        },
                        "split_qkv" => HostOp::SplitQkv {
                            packed: s("packed")?,
                            q: s("q")?,
                            k: s("k")?,
                            v: s("v")?,
                            rows: u("rows")?,
                            q_width: u("q_width")?,
                            kv_width: u("kv_width")?,
                        },
                        "embed_weighted" => HostOp::EmbedWeighted {
                            ids: s("ids")?,
                            weights: s("weights")?,
                            table: s("table")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            taps: u("taps")?,
                            width: u("width")?,
                            limit: u("limit")?,
                        },
                        "argmax" => HostOp::Argmax {
                            x: s("x")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            column: u("column")?,
                            y_width: u("y_width")?,
                        },
                        "argmax_merge" => HostOp::ArgmaxMerge {
                            parts: s("parts")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            cg: u("cg")?,
                            block: u("block")?,
                            column: u("column")?,
                            y_width: u("y_width")?,
                        },
                        "matmul" => HostOp::Matmul {
                            act: s("act")?,
                            w: s("w")?,
                            y: s("y")?,
                            m: u("m")?,
                            k: u("k")?,
                            n: u("n")?,
                        },
                        "matmul_grouped" => HostOp::MatmulGrouped {
                            x: s("x")?,
                            w: s("w")?,
                            routes: s("routes")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            groups: u("groups")?,
                            k: u("k")?,
                            n: u("n")?,
                            experts: u("experts")?,
                        },
                        "topk" => HostOp::TopK {
                            x: s("x")?,
                            values: s("values")?,
                            indices: s("indices")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            k: u("k")?,
                        },
                        "selector_walk" => HostOp::SelectorWalk {
                            cand: s("cand")?,
                            indptr: s("indptr")?,
                            unary: s("unary")?,
                            hp: s("hp"),
                            tokens: s("tokens")?,
                            pred: s("pred")?,
                            succ: s("succ")?,
                            picks: s("picks")?,
                            rows: u("rows")?,
                            k: u("k")?,
                            rank: u("rank")?,
                            vocab: u("vocab")?,
                            first: u("first")?,
                        },
                        "ple_ngram_ids" => {
                            let u64s = |k: &str| -> Option<Vec<u64>> {
                                h["op"][k]
                                    .as_array()?
                                    .iter()
                                    .map(serde_json::Value::as_u64)
                                    .collect()
                            };
                            HostOp::PleNgramIds {
                                ids: s("ids")?,
                                indptr: s("indptr"),
                                slots: s("slots")?,
                                state: s("state")?,
                                out: s("out")?,
                                rows: u("rows")?,
                                eos: u("eos")?,
                                mults: u64s("mults")?,
                                primes: u64s("primes")?,
                                offsets: u64s("offsets")?,
                                heads_per_ngram: u("heads_per_ngram")?,
                            }
                        }
                        "page_write" => HostOp::PageWrite {
                            src: s("src")?,
                            table: s("table")?,
                            write_page: s("write_page")?,
                            write_offset: s("write_offset")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            page_size: u("page_size")?,
                            table_rows: u("table_rows")?,
                        },
                        "boundary" => HostOp::Boundary {
                            positions: s("positions")?,
                            request_of_token: s("request_of_token")?,
                            row_valid: s("row_valid")?,
                            boundary_pos: s("boundary_pos")?,
                            boundary_req: s("boundary_req")?,
                            boundary_rope: s("boundary_rope")?,
                            rows: u("rows")?,
                            ratio: u("ratio")?,
                        },
                        "block_mean" => HostOp::BlockMean {
                            boundary_pos: s("boundary_pos")?,
                            boundary_req: s("boundary_req")?,
                            keys: s("keys")?,
                            indices: s("indices")?,
                            indptr: s("indptr")?,
                            entries: s("entries")?,
                            rows: u("rows")?,
                            head_dim: u("head_dim")?,
                            ratio: u("ratio")?,
                            page_size: u("page_size")?,
                        },
                        "pool_write" => HostOp::PoolWrite {
                            entries: s("entries")?,
                            boundary_pos: s("boundary_pos")?,
                            boundary_req: s("boundary_req")?,
                            keys: s("keys")?,
                            indices: s("indices")?,
                            indptr: s("indptr")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            page_size: u("page_size")?,
                        },
                        "index_topk" => HostOp::IndexTopk {
                            q: s("q")?,
                            weights: s("weights"),
                            keys: s("keys")?,
                            indices: s("indices")?,
                            indptr: s("indptr")?,
                            positions: s("positions")?,
                            request_of_token: s("request_of_token")?,
                            selection: s("selection")?,
                            rows: u("rows")?,
                            heads: u("heads")?,
                            head_dim: u("head_dim")?,
                            top_k: u("top_k")?,
                            ratio: u("ratio")?,
                            page_size: u("page_size")?,
                            max_pages: u("max_pages")?,
                        },
                        "group_routes" => HostOp::GroupRoutes {
                            routes: s("routes")?,
                            rows: u("rows")?,
                            groups: u("groups")?,
                        },
                        "embed_concat" => HostOp::EmbedConcat {
                            ids: s("ids")?,
                            table: s("table")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            heads: u("heads")?,
                            width: u("width")?,
                            limit: u("limit")?,
                        },
                        "cells" => HostOp::Cells {
                            tgrid: s("tgrid")?,
                            positions: s("positions")?,
                            y: s("y")?,
                            clips: u("clips")?,
                            rows: u("rows")?,
                            axis: u("axis")?,
                            width: u("width")?,
                            base: u("base")?,
                        },
                        "concat_cols" => HostOp::ConcatCols {
                            x: s("x")?,
                            y: s("y")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            stride: u("stride")?,
                            at: u("at")?,
                        },
                        "group_rows" => HostOp::GroupRows {
                            csr: s("csr")?,
                            groups: s("groups")?,
                            lanes: s("lanes")?,
                            y: s("y")?,
                            csr_len: u("csr_len")?,
                            lane_count: u("lane_count")?,
                            rows: u("rows")?,
                        },
                        "group_first" => HostOp::GroupFirst {
                            x: s("x")?,
                            csr: s("csr")?,
                            groups: s("groups")?,
                            lanes: s("lanes")?,
                            y: s("y")?,
                            csr_len: u("csr_len")?,
                            lane_count: u("lane_count")?,
                            rows: u("rows")?,
                            width: u("width")?,
                            x_rows: u("x_rows")?,
                        },
                        "patch" => HostOp::Patch {
                            x: s("x")?,
                            grid: s("grid")?,
                            y: s("y")?,
                            o_grid: s("o_grid")?,
                            clips: u("clips")?,
                            rows: u("rows")?,
                            channels: u("channels")?,
                            p: [u("p0")?, u("p1")?, u("p2")?],
                            fold: u("fold")? != 0,
                            bases: [u("x_base")?, u("y_base")?],
                        },
                        _ => return None,
                    };
                    Some(HostPhase {
                        op,
                        rounds: h["rounds"]
                            .as_array()?
                            .iter()
                            .map(|v| Some(v.as_str()?.to_string()))
                            .collect::<Option<Vec<_>>>()?,
                    })
                }
            },
            table: table_of(&m["table"]),
            tables: m["tables"]
                .as_array()
                .map(|a| a.iter().map(table_of).collect::<Option<Vec<_>>>())
                .unwrap_or_else(|| Some(Vec::new()))?,
            lanes: m["lanes"]
                .as_array()
                .map(|a| a.as_slice())
                .unwrap_or(&[])
                .iter()
                .map(|l| {
                    Some(LanePlan {
                        window: l["window"].as_array().and_then(|w| {
                            Some((w.first()?.as_u64()? as u32, w.get(1)?.as_u64()? as u32))
                        }),
                        pes: l["pes"].as_u64()? as u32,
                        kind: if let Some(lanes) = l["kind"]["per_row"].as_u64() {
                            LaneKind::PerRow {
                                lanes: lanes as u32,
                            }
                        } else if let Some(table) = l["kind"]["lane_of_row"].as_str() {
                            LaneKind::ByTable {
                                lane_of_row: table.to_string(),
                                lanes: l["kind"]["lanes"].as_u64()? as u32,
                            }
                        } else {
                            LaneKind::Ragged {
                                indptr: l["kind"]["indptr"].as_str()?.to_string(),
                                lanes: l["kind"]["lanes"].as_u64()? as u32,
                            }
                        },
                        pages: match &l["pages"] {
                            serde_json::Value::Null => None,
                            p => Some(PagePlan {
                                source: if let Some(table) = p["source"]["table"].as_str() {
                                    PageSource::Rows {
                                        table: table.to_string(),
                                    }
                                } else {
                                    PageSource::Csr {
                                        indptr: p["source"]["indptr"].as_str()?.to_string(),
                                        indices: p["source"]["indices"].as_str()?.to_string(),
                                    }
                                },
                                page_size: p["page_size"].as_u64()? as u32,
                                max_pages: p["max_pages"].as_u64()? as u32,
                                rewritten: p["rewritten"]
                                    .as_array()?
                                    .iter()
                                    .map(|v| Some(v.as_str()?.to_string()))
                                    .collect::<Option<Vec<_>>>()?,
                                page_groups: p["page_groups"].as_u64().unwrap_or(1) as u32,
                                strided: p["strided"].as_bool().unwrap_or(false),
                                strided_pages: p["strided_pages"].as_u64().unwrap_or(0) as u32,
                            }),
                        },
                        reduce: l["reduce"]
                            .as_array()
                            .map(|v| {
                                v.iter()
                                    .map(|r| {
                                        let name = r["name"].as_str()?.to_string();
                                        if let Some(group) = r["fabric"].as_u64() {
                                            return Some((
                                                name,
                                                Reduce::FabricSum {
                                                    group: group as u32,
                                                },
                                            ));
                                        }
                                        if let Some(group) = r["root"].as_u64() {
                                            return Some((
                                                name,
                                                Reduce::FabricRoot {
                                                    group: group as u32,
                                                },
                                            ));
                                        }
                                        Some(match (r["lse"].as_str(), r["sum"].as_bool()) {
                                            (_, Some(true)) => (name, Reduce::Sum),
                                            (Some(lse), _) => (
                                                name,
                                                Reduce::Weighted {
                                                    lse: lse.to_string(),
                                                },
                                            ),
                                            _ => (name, Reduce::LogSumExp),
                                        })
                                    })
                                    .collect::<Option<Vec<_>>>()
                            })
                            .unwrap_or_else(|| Some(Vec::new()))?,
                        header: l["header"].as_str()?.to_string(),
                        slots: l["slots"].as_str()?.to_string(),
                        banks: l["banks"]
                            .as_array()?
                            .iter()
                            .map(|b| {
                                Some((
                                    b[0].as_str()?.to_string(),
                                    b[1].as_u64()? as u32,
                                    match (b[2].as_str(), b[2].as_bool()) {
                                        (Some("whole"), _) | (_, Some(false)) => BankSplit::Whole,
                                        (Some("by_b"), _) => BankSplit::ByB,
                                        _ => BankSplit::Block,
                                    },
                                ))
                            })
                            .collect::<Option<Vec<_>>>()?,
                        lanes_per_pe: l["lanes_per_pe"].as_u64()? as u32,
                        a_first: l["a_first"].as_bool().unwrap_or(false),
                        row_outputs: l["row_outputs"]
                            .as_array()?
                            .iter()
                            .map(|b| {
                                Some((
                                    b[0].as_str()?.to_string(),
                                    b[1].as_u64()? as u32,
                                    b[2].as_u64()? as u32,
                                    b[3].as_u64()? as u32,
                                ))
                            })
                            .collect::<Option<Vec<_>>>()?,
                        cols: l["cols"]
                            .as_array()
                            .map(|v| {
                                v.iter()
                                    .map(|c| {
                                        Some(ColPlane {
                                            name: c["name"].as_str()?.to_string(),
                                            rows: c["rows"].as_u64()? as u32,
                                            width: c["width"].as_u64()? as u32,
                                            by_rows: c["by_rows"].as_bool().unwrap_or(false),
                                            row_block: c["row_block"].as_bool().unwrap_or(false),
                                            segments: c["segments"]
                                                .as_array()?
                                                .iter()
                                                .map(|s| {
                                                    Some(Segment {
                                                        base: s[0].as_u64()? as u32,
                                                        a_group: s[1].as_u64()? as u32,
                                                        a_stride: s[2].as_u64()? as u32,
                                                        b_stride: s[3].as_u64()? as u32,
                                                        span: s[4].as_u64()? as u32,
                                                    })
                                                })
                                                .collect::<Option<Vec<_>>>()?,
                                        })
                                    })
                                    .collect::<Option<Vec<_>>>()
                            })
                            .unwrap_or_else(|| Some(Vec::new()))?,
                        block: match l["block"].as_array() {
                            None => None,
                            Some(b) if b.len() == 5 => Some(BlockPlan {
                                a: b[0].as_u64()? as u32,
                                b: b[1].as_u64()? as u32,
                                w: b[2].as_u64()? as u32,
                                a_groups: b[3].as_u64()? as u32,
                                b_groups: b[4].as_u64()? as u32,
                            }),
                            Some(_) => return None,
                        },
                    })
                })
                .collect::<Option<Vec<_>>>()?,
        },
    })
}

/// A table-driven program's table as JSON (see `table_of`).
fn table_json(t: &crate::fabric::Table) -> serde_json::Value {
    use crate::fabric::TableOp;
    use kernels_cerebras::csl::Arg;
    use kernels_cerebras::phase::FabricOp;
    let arg = |a: &Arg| match a {
        Arg::Ptr(b) => {
            serde_json::json!({ "ptr": b.name, "rows": b.rows, "width": b.width, "elem": b.elem })
        }
        Arg::Scratch(name, elem) => serde_json::json!({ "scratch": name, "elem": elem }),
        Arg::Int(v) => serde_json::json!({ "int": v }),
        Arg::Bool(v) => serde_json::json!({ "bool": v }),
        Arg::Float(v) => serde_json::json!({ "float": v.to_bits() }),
        Arg::PeTimes(c) => serde_json::json!({ "pe_times": c }),
        Arg::Word(name, i) => serde_json::json!({ "word": name, "index": i }),
        Arg::Dummy(elem) => serde_json::json!({ "dummy": elem }),
        Arg::Expr(e) => serde_json::json!({ "expr": e }),
    };
    serde_json::json!({
        "kernels": t.kernels.iter().map(|k| k.name()).collect::<Vec<_>>(),
        "ops_words": t.ops_words,
        "arena_words": t.arena_words,
        "keep_chunks": t.keep_chunks,
        "rows": t.rows,
        "move_rows": t.move_rows,
        "phases": t.phases,
        "consts": t.consts.iter().map(|(n, w)| serde_json::json!([n, w])).collect::<Vec<_>>(),
        "arena": t.arena.iter().map(|(n, o, w)| serde_json::json!([n, o, w])).collect::<Vec<_>>(),
        "ops": t.ops.iter().map(|op| match op {
            TableOp::Call { root, on, kernel, args } => serde_json::json!({
                "root": root, "on": on, "kernel": kernel, "args": args.iter().map(arg).collect::<Vec<_>>(),
            }),
            TableOp::Collective(FabricOp::Reduce { send, recv, count }) => serde_json::json!({ "reduce": [send, recv], "count": count }),
            TableOp::Collective(FabricOp::Gather { send, recv, count }) => serde_json::json!({ "gather": [send, recv], "count": count }),
            TableOp::Collective(FabricOp::Broadcast { buf, count }) => serde_json::json!({ "broadcast": buf, "count": count }),
            TableOp::Collective(FabricOp::BroadcastFrom { root, buf, count }) => serde_json::json!({ "broadcast_from": [root, buf], "count": count }),
            TableOp::Collective(FabricOp::Handoff { root, buf, count }) => serde_json::json!({ "handoff": [root, buf], "count": count }),
            TableOp::Nop => serde_json::json!({ "nop": true }),
        }).collect::<Vec<_>>(),
    })
}

/// The table of a program's JSON manifest, `None` when it has none.
fn table_of(v: &serde_json::Value) -> Option<crate::fabric::Table> {
    use crate::fabric::{Table, TableOp};
    use kernels_cerebras::csl::Arg;
    use kernels_cerebras::phase::FabricOp;
    if v.is_null() {
        return None;
    }
    let elem_of = |e: &str| -> Option<&'static str> {
        match e {
            "f32" => Some("f32"),
            "i32" => Some("i32"),
            "u32" => Some("u32"),
            "u64" => Some("u64"),
            _ => None,
        }
    };
    let arg = |a: &serde_json::Value| -> Option<Arg> {
        Some(if let Some(name) = a["ptr"].as_str() {
            Arg::Ptr(Buf {
                name: name.to_string(),
                rows: a["rows"].as_u64()? as u32,
                width: a["width"].as_u64()? as u32,
                elem: elem_of(a["elem"].as_str()?)?,
            })
        } else if let Some(name) = a["scratch"].as_str() {
            Arg::Scratch(name.to_string(), elem_of(a["elem"].as_str()?)?)
        } else if let Some(v) = a["int"].as_i64() {
            Arg::Int(v)
        } else if let Some(v) = a["bool"].as_bool() {
            Arg::Bool(v)
        } else if let Some(bits) = a["float"].as_u64() {
            Arg::Float(f32::from_bits(bits as u32))
        } else if let Some(c) = a["pe_times"].as_i64() {
            Arg::PeTimes(c)
        } else if let Some(name) = a["word"].as_str() {
            Arg::Word(name.to_string(), a["index"].as_u64()? as u32)
        } else if let Some(e) = a["dummy"].as_str() {
            Arg::Dummy(elem_of(e)?)
        } else {
            Arg::Expr(a["expr"].as_str()?.to_string())
        })
    };
    let ops = v["ops"]
        .as_array()?
        .iter()
        .map(|op| -> Option<TableOp> {
            let count = op["count"].as_u64();
            if op["nop"].as_bool() == Some(true) {
                return Some(TableOp::Nop);
            }
            Some(if let Some(pair) = op["reduce"].as_array() {
                TableOp::Collective(FabricOp::Reduce {
                    send: pair.first()?.as_str()?.to_string(),
                    recv: pair.get(1)?.as_str()?.to_string(),
                    count: count?,
                })
            } else if let Some(pair) = op["gather"].as_array() {
                TableOp::Collective(FabricOp::Gather {
                    send: pair.first()?.as_str()?.to_string(),
                    recv: pair.get(1)?.as_str()?.to_string(),
                    count: count?,
                })
            } else if let Some(buf) = op["broadcast"].as_str() {
                TableOp::Collective(FabricOp::Broadcast {
                    buf: buf.to_string(),
                    count: count?,
                })
            } else if let Some(pair) = op["broadcast_from"].as_array() {
                TableOp::Collective(FabricOp::BroadcastFrom {
                    root: pair.first()?.as_u64()? as u32,
                    buf: pair.get(1)?.as_str()?.to_string(),
                    count: count?,
                })
            } else if let Some(pair) = op["handoff"].as_array() {
                TableOp::Collective(FabricOp::Handoff {
                    root: pair.first()?.as_u64()? as u32,
                    buf: pair.get(1)?.as_str()?.to_string(),
                    count: count?,
                })
            } else {
                TableOp::Call {
                    root: op["root"].as_bool()?,
                    on: op["on"].as_u64().map(|v| v as u32),
                    kernel: op["kernel"].as_u64()? as u32,
                    args: op["args"]
                        .as_array()?
                        .iter()
                        .map(arg)
                        .collect::<Option<Vec<_>>>()?,
                }
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Table {
        kernels: v["kernels"]
            .as_array()?
            .iter()
            .map(|k| kernels_cerebras::library::Kernel::of(k.as_str()?))
            .collect::<Option<Vec<_>>>()?,
        ops_words: v["ops_words"].as_u64()? as u32,
        ops,
        arena: v["arena"]
            .as_array()?
            .iter()
            .map(|e| Some((e[0].as_str()?.to_string(), e[1].as_u64()?, e[2].as_u64()?)))
            .collect::<Option<Vec<_>>>()?,
        arena_words: v["arena_words"].as_u64()?,
        keep_chunks: v["keep_chunks"].as_u64().unwrap_or(0) as u32,
        rows: v["rows"].as_u64().unwrap_or(0) as u32,
        move_rows: v["move_rows"].as_u64().unwrap_or(0) as u32,
        phases: v["phases"].as_u64().unwrap_or(1) as u32,
        consts: v["consts"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|e| {
                        Some((
                            e[0].as_str()?.to_string(),
                            e[1].as_array()?
                                .iter()
                                .map(|w| w.as_u64().map(|w| w as u32))
                                .collect::<Option<Vec<_>>>()?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .unwrap_or_else(|| Some(Vec::new()))?,
    })
}
