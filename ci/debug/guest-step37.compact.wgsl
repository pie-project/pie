struct PieStep {
    node: u32,
    rung: u32,
    selector: u32,
    area: u32,
    status_base: u32,
    descs_base: u32,
    params_base: u32,
    offs_base: u32,
    cfg_base: u32,
    heap_base: u32,
    pad1_: u32,
    pad2_: u32,
}

const PTIR_WG: u32 = 256u;
const DESC_WORDS: u32 = 9u;
const PARAM_WORDS: u32 = 16u;
const DT_F32_: u32 = 0u;
const DT_I32_: u32 = 1u;
const DT_U32_: u32 = 2u;
const DT_BOOL: u32 = 3u;
const DT_BF16_: u32 = 5u;
const CFG_HEAP_WORDS: u32 = 1u;
const OUT_OF_BOUNDS: u32 = 181u;
const REDUCE_LEVELS: u32 = 7u;
const RED_SUM: u32 = 0u;
const RED_MAX: u32 = 1u;
const RED_MIN: u32 = 2u;
const RED_ARGMAX: u32 = 3u;
const RED_GUMBEL: u32 = 4u;
const PTIR_REDUCE_LEVELS: u32 = 7u;
const SORT_ROUNDS: u32 = 28u;
const PIVOT_UNROLL: u32 = 32u;
const SEL_CAP: u32 = 1024u;
const SEL_GROUPS: u32 = 32u;
const SEL_WORDS: u32 = 10320u;
const SEL_MODE: u32 = 0u;
const SEL_ACTIVE: u32 = 1u;
const SEL_WANT: u32 = 2u;
const SEL_PREFIX: u32 = 3u;
const SEL_REMAINING: u32 = 4u;
const SEL_TOTAL_LT: u32 = 5u;
const SEL_KEEP: u32 = 6u;
const SEL_RUNS: u32 = 0u;
const SEL_DECIDED: u32 = 1u;
const SEL_LADDER: u32 = 2u;
const SEL_KEEP_CUT: u32 = 0u;
const SEL_KEEP_ALL: u32 = 1u;
const SEL_KEEP_NONE: u32 = 2u;

@group(0) @binding(0) 
var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(1) 
var<storage, read_write> tables: array<u32>;
@group(0) @binding(4) 
var<storage, read_write> heap: array<u32>;
var<private> tid: u32;
var<private> lanes: u32;
var<private> sel_lt: u32;
var<private> sel_group: u32;
var<private> sel_groups: u32;
var<private> lim_heap: u32;
var<private> b_status: u32;
var<private> b_descs: u32;
var<private> b_params: u32;
var<private> b_offs: u32;
var<private> b_heap: u32;
var<private> tmp_base: u32;
var<private> heap_bad: bool;
var<private> heap_at: u32;
@group(0) @binding(6) 
var<storage, read_write> pie_step: PieStep;

fn pie_fault(code: u32, at: u32, limit: u32) {
    let _e5 = b_status;
    let _e8 = atomicExchange((&status[_e5]), 3u);
    if (_e8 != 0u) {
        return;
    }
    let _e13 = b_status;
    atomicStore((&status[(_e13 + 1u)]), code);
    let _e19 = b_status;
    atomicStore((&status[(_e19 + 2u)]), at);
    let _e25 = b_status;
    atomicStore((&status[(_e25 + 3u)]), limit);
    return;
}

fn heap_refuse(w: u32, ok: bool) {
    var local: bool;
    var local_1: bool;

    let _e4 = heap_at;
    if !(ok) {
        let _e9 = heap_bad;
        local = !(_e9);
    } else {
        local = false;
    }
    let _e12 = local;
    heap_at = select(_e4, w, _e12);
    let _e16 = heap_bad;
    if !(_e16) {
        local_1 = !(ok);
    } else {
        local_1 = true;
    }
    let _e22 = local_1;
    heap_bad = _e22;
    return;
}

fn hld(w_1: u32) -> u32 {
    let _e2 = lim_heap;
    let ok_1 = (w_1 < _e2);
    heap_refuse(w_1, ok_1);
    let _e7 = b_heap;
    let _e9 = lim_heap;
    let _e15 = heap[(_e7 + min(w_1, (_e9 - 1u)))];
    return select(0u, _e15, ok_1);
}

fn hst(w_2: u32, x: u32) {
    let _e3 = lim_heap;
    let ok_2 = (w_2 < _e3);
    heap_refuse(w_2, ok_2);
    if ok_2 {
        let _e7 = b_heap;
        heap[(_e7 + w_2)] = x;
        return;
    } else {
        return;
    }
}

fn heap_flush() {
    let _e1 = heap_bad;
    if _e1 {
        let _e4 = heap_at;
        let _e6 = lim_heap;
        pie_fault(OUT_OF_BOUNDS, _e4, _e6);
        return;
    } else {
        return;
    }
}

fn d_word(v: u32, k: u32) -> u32 {
    let _e4 = b_descs;
    let _e10 = tables[((_e4 + (v * DESC_WORDS)) + k)];
    return _e10;
}

fn d_len(v_1: u32) -> u32 {
    let _e2 = d_word(v_1, 0u);
    return _e2;
}

fn d_rank(v_2: u32) -> u32 {
    let _e2 = d_word(v_2, 3u);
    return _e2;
}

fn d_dtype(v_3: u32) -> u32 {
    let _e2 = d_word(v_3, 4u);
    return _e2;
}

fn d_dim(v_4: u32, k_1: u32) -> u32 {
    let _e4 = d_word(v_4, (5u + k_1));
    return _e4;
}

fn base(v_5: u32) -> u32 {
    let _e3 = b_offs;
    let _e6 = tables[(_e3 + v_5)];
    return (_e6 >> 2u);
}

fn rows_of(v_6: u32) -> u32 {
    var n: u32 = 1u;
    var k_2: u32 = 0u;

    let _e1 = d_rank(v_6);
    if (_e1 < 2u) {
        return 1u;
    }
    loop {
        let _e9 = k_2;
        if ((_e9 + 1u) < _e1) {
        } else {
            break;
        }
        {
            let _e13 = n;
            let _e14 = k_2;
            let _e15 = d_dim(v_6, _e14);
            n = (_e13 * _e15);
        }
        continuing {
            let _e17 = k_2;
            k_2 = (_e17 + 1u);
        }
    }
    let _e20 = n;
    return _e20;
}

fn p_at(p: u32, k_3: u32) -> u32 {
    let _e4 = b_params;
    let _e10 = tables[((_e4 + (p * PARAM_WORDS)) + k_3)];
    return _e10;
}

fn p_a0_(p_1: u32) -> u32 {
    let _e2 = p_at(p_1, 1u);
    return _e2;
}

fn p_a1_(p_2: u32) -> u32 {
    let _e2 = p_at(p_2, 2u);
    return _e2;
}

fn p_o0_(p_3: u32) -> u32 {
    let _e2 = p_at(p_3, 4u);
    return _e2;
}

fn p_pred_tag(p_4: u32) -> u32 {
    let _e2 = p_at(p_4, 10u);
    return _e2;
}

fn p_pred_payload(p_5: u32) -> u32 {
    let _e1 = p_a1_(p_5);
    return _e1;
}

fn ld_raw(v_7: u32, i: u32) -> u32 {
    let _e2 = base(v_7);
    let _e4 = hld((_e2 + i));
    return _e4;
}

fn ld_byte(v_8: u32, i_1: u32) -> u32 {
    let _e2 = base(v_8);
    let _e6 = hld((_e2 + (i_1 >> 2u)));
    return ((_e6 >> ((i_1 & 3u) * 8u)) & 255u);
}

fn ld_h_bits(v_9: u32, i_2: u32) -> u32 {
    let _e2 = base(v_9);
    let _e6 = hld((_e2 + (i_2 >> 1u)));
    return ((_e6 >> ((i_2 & 1u) * 16u)) & 65535u);
}

fn ld_h(v_10: u32, i_3: u32) -> f32 {
    let _e2 = ld_h_bits(v_10, i_3);
    return bitcast<f32>((_e2 << 16u));
}

fn ld_f(v_11: u32, i_4: u32) -> f32 {
    let _e2 = d_dtype(v_11);
    if (_e2 == DT_F32_) {
        let _e5 = ld_raw(v_11, i_4);
        return bitcast<f32>(_e5);
    }
    if (_e2 == DT_BF16_) {
        let _e9 = ld_h(v_11, i_4);
        return _e9;
    }
    if (_e2 == DT_I32_) {
        let _e12 = ld_raw(v_11, i_4);
        return f32(bitcast<i32>(_e12));
    }
    if (_e2 == DT_U32_) {
        let _e17 = ld_raw(v_11, i_4);
        return f32(_e17);
    }
    let _e19 = ld_byte(v_11, i_4);
    return select(0f, 1f, (_e19 != 0u));
}

fn ld_i(v_12: u32, i_5: u32) -> i32 {
    let _e2 = d_dtype(v_12);
    if (_e2 == DT_F32_) {
        let _e5 = ld_raw(v_12, i_5);
        return i32(bitcast<f32>(_e5));
    }
    if (_e2 == DT_BF16_) {
        let _e10 = ld_h(v_12, i_5);
        return i32(_e10);
    }
    if (_e2 == DT_I32_) {
        let _e14 = ld_raw(v_12, i_5);
        return bitcast<i32>(_e14);
    }
    if (_e2 == DT_U32_) {
        let _e18 = ld_raw(v_12, i_5);
        return i32(_e18);
    }
    let _e20 = ld_byte(v_12, i_5);
    return select(0i, 1i, (_e20 != 0u));
}

fn st_b_word(v_13: u32, w_3: u32, n_1: u32, b0_: bool, b1_: bool, b2_: bool, b3_: bool) {
    var packed: u32 = 0u;
    var local_2: bool;
    var local_3: bool;
    var local_4: bool;
    var local_5: bool;

    if b0_ {
        local_2 = (((w_3 * 4u) + 0u) < n_1);
    } else {
        local_2 = false;
    }
    let _e17 = local_2;
    if _e17 {
        let _e18 = packed;
        packed = (_e18 | 255u);
    }
    if b1_ {
        local_3 = (((w_3 * 4u) + 1u) < n_1);
    } else {
        local_3 = false;
    }
    let _e29 = local_3;
    if _e29 {
        let _e30 = packed;
        packed = (_e30 | 65280u);
    }
    if b2_ {
        local_4 = (((w_3 * 4u) + 2u) < n_1);
    } else {
        local_4 = false;
    }
    let _e41 = local_4;
    if _e41 {
        let _e42 = packed;
        packed = (_e42 | 16711680u);
    }
    if b3_ {
        local_5 = (((w_3 * 4u) + 3u) < n_1);
    } else {
        local_5 = false;
    }
    let _e53 = local_5;
    if _e53 {
        let _e54 = packed;
        packed = (_e54 | 4278190080u);
    }
    let _e57 = base(v_13);
    let _e59 = packed;
    hst((_e57 + w_3), _e59);
    return;
}

fn pick(n_2: u32, i_6: u32) -> u32 {
    if (n_2 == 1u) {
        return 0u;
    }
    return i_6;
}

fn f32_is_nan(x_1: f32) -> bool {
    var local_6: bool;

    let b_1 = bitcast<u32>(x_1);
    if ((b_1 & 2139095040u) == 2139095040u) {
        local_6 = ((b_1 & 8388607u) != 0u);
    } else {
        local_6 = false;
    }
    let _e13 = local_6;
    return _e13;
}

fn sort_key(x_2: f32) -> u32 {
    var asc: u32;

    let _e1 = f32_is_nan(x_2);
    if _e1 {
        return 4294967295u;
    }
    let b_2 = select(bitcast<u32>(x_2), 0u, (x_2 == 0f));
    if ((b_2 & 2147483648u) != 0u) {
        asc = ~(b_2);
    } else {
        asc = (b_2 | 2147483648u);
    }
    let _e17 = asc;
    return (4294967295u - _e17);
}

fn sort_cell(buf: u32, n_3: u32, c: u32) -> u32 {
    let _e4 = tmp_base;
    return (_e4 + (((buf * n_3) + c) * 2u));
}

fn pivot_flag(n_4: u32, t: u32) -> u32 {
    let _e3 = tmp_base;
    return (_e3 + t);
}

fn pivot_meta(n_5: u32, row: u32) -> u32 {
    let _e3 = tmp_base;
    return ((_e3 + n_5) + row);
}

fn pivot_val(n_6: u32, t_1: u32) -> u32 {
    let _e3 = tmp_base;
    return ((_e3 + (2u * n_6)) + t_1);
}

fn count_key_below(n_7: u32, rowb: u32, len: u32, kh: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32;

    hi = len;
    loop {
        let _e7 = lo;
        let _e8 = hi;
        if (_e7 >= _e8) {
            break;
        }
        let _e10 = lo;
        let _e11 = hi;
        let _e12 = lo;
        let m = (_e10 + ((_e11 - _e12) / 2u));
        let _e19 = sort_cell(0u, n_7, (rowb + m));
        let _e20 = hld(_e19);
        if (_e20 < kh) {
            lo = (m + 1u);
        } else {
            hi = m;
        }
    }
    let _e24 = lo;
    return _e24;
}

fn op_pivot_finish(p_6: u32) {
    var row_1: u32;
    var local_7: bool;
    var excl: f32;
    var t_2: u32;
    var local_8: bool;
    var local_9: bool;
    var w_4: u32;
    var b: array<bool, 4>;
    var k_4: u32;
    var on: bool;

    let _e1 = p_a0_(p_6);
    let _e2 = p_o0_(p_6);
    let _e3 = p_pred_payload(p_6);
    let _e4 = p_pred_tag(p_6);
    let _e5 = d_len(_e1);
    let _e6 = rows_of(_e1);
    let len_1 = (_e5 / max(_e6, 1u));
    let _e10 = d_len(_e3);
    if (_e4 == 1u) {
        let _e14 = tid;
        row_1 = _e14;
        loop {
            let _e16 = row_1;
            if (_e16 < _e6) {
            } else {
                break;
            }
            {
                if (len_1 == 0u) {
                    continue;
                }
                let _e20 = row_1;
                let rowb_1 = (_e20 * len_1);
                let _e22 = row_1;
                let _e23 = pick(_e10, _e22);
                let _e24 = ld_f(_e3, _e23);
                let _e25 = pivot_val(_e5, rowb_1);
                let _e26 = pivot_flag(_e5, rowb_1);
                let _e30 = hld(((_e25 + len_1) - 1u));
                let last = bitcast<f32>(_e30);
                if !((last < 0f)) {
                    let _e37 = f32_is_nan(last);
                    local_7 = !(_e37);
                } else {
                    local_7 = false;
                }
                let monotone = local_7;
                excl = 0f;
                t_2 = 0u;
                loop {
                    let _e45 = t_2;
                    if ((_e45 + PIVOT_UNROLL) > len_1) {
                        break;
                    }
                    let _e49 = t_2;
                    let _e53 = hld(((_e25 + _e49) + 0u));
                    let v0_ = bitcast<f32>(_e53);
                    let _e55 = t_2;
                    let _e59 = hld(((_e25 + _e55) + 1u));
                    let v1_ = bitcast<f32>(_e59);
                    let _e61 = t_2;
                    let _e65 = hld(((_e25 + _e61) + 2u));
                    let v2_ = bitcast<f32>(_e65);
                    let _e67 = t_2;
                    let _e71 = hld(((_e25 + _e67) + 3u));
                    let v3_ = bitcast<f32>(_e71);
                    let _e73 = t_2;
                    let _e77 = hld(((_e25 + _e73) + 4u));
                    let v4_ = bitcast<f32>(_e77);
                    let _e79 = t_2;
                    let _e83 = hld(((_e25 + _e79) + 5u));
                    let v5_ = bitcast<f32>(_e83);
                    let _e85 = t_2;
                    let _e89 = hld(((_e25 + _e85) + 6u));
                    let v6_ = bitcast<f32>(_e89);
                    let _e91 = t_2;
                    let _e95 = hld(((_e25 + _e91) + 7u));
                    let v7_ = bitcast<f32>(_e95);
                    let _e97 = t_2;
                    let _e101 = hld(((_e25 + _e97) + 8u));
                    let v8_ = bitcast<f32>(_e101);
                    let _e103 = t_2;
                    let _e107 = hld(((_e25 + _e103) + 9u));
                    let v9_ = bitcast<f32>(_e107);
                    let _e109 = t_2;
                    let _e113 = hld(((_e25 + _e109) + 10u));
                    let v10_ = bitcast<f32>(_e113);
                    let _e115 = t_2;
                    let _e119 = hld(((_e25 + _e115) + 11u));
                    let v11_ = bitcast<f32>(_e119);
                    let _e121 = t_2;
                    let _e125 = hld(((_e25 + _e121) + 12u));
                    let v12_ = bitcast<f32>(_e125);
                    let _e127 = t_2;
                    let _e131 = hld(((_e25 + _e127) + 13u));
                    let v13_ = bitcast<f32>(_e131);
                    let _e133 = t_2;
                    let _e137 = hld(((_e25 + _e133) + 14u));
                    let v14_ = bitcast<f32>(_e137);
                    let _e139 = t_2;
                    let _e143 = hld(((_e25 + _e139) + 15u));
                    let v15_ = bitcast<f32>(_e143);
                    let _e145 = t_2;
                    let _e149 = hld(((_e25 + _e145) + 16u));
                    let v16_ = bitcast<f32>(_e149);
                    let _e151 = t_2;
                    let _e155 = hld(((_e25 + _e151) + 17u));
                    let v17_ = bitcast<f32>(_e155);
                    let _e157 = t_2;
                    let _e161 = hld(((_e25 + _e157) + 18u));
                    let v18_ = bitcast<f32>(_e161);
                    let _e163 = t_2;
                    let _e167 = hld(((_e25 + _e163) + 19u));
                    let v19_ = bitcast<f32>(_e167);
                    let _e169 = t_2;
                    let _e173 = hld(((_e25 + _e169) + 20u));
                    let v20_ = bitcast<f32>(_e173);
                    let _e175 = t_2;
                    let _e179 = hld(((_e25 + _e175) + 21u));
                    let v21_ = bitcast<f32>(_e179);
                    let _e181 = t_2;
                    let _e185 = hld(((_e25 + _e181) + 22u));
                    let v22_ = bitcast<f32>(_e185);
                    let _e187 = t_2;
                    let _e191 = hld(((_e25 + _e187) + 23u));
                    let v23_ = bitcast<f32>(_e191);
                    let _e193 = t_2;
                    let _e197 = hld(((_e25 + _e193) + 24u));
                    let v24_ = bitcast<f32>(_e197);
                    let _e199 = t_2;
                    let _e203 = hld(((_e25 + _e199) + 25u));
                    let v25_ = bitcast<f32>(_e203);
                    let _e205 = t_2;
                    let _e209 = hld(((_e25 + _e205) + 26u));
                    let v26_ = bitcast<f32>(_e209);
                    let _e211 = t_2;
                    let _e215 = hld(((_e25 + _e211) + 27u));
                    let v27_ = bitcast<f32>(_e215);
                    let _e217 = t_2;
                    let _e221 = hld(((_e25 + _e217) + 28u));
                    let v28_ = bitcast<f32>(_e221);
                    let _e223 = t_2;
                    let _e227 = hld(((_e25 + _e223) + 29u));
                    let v29_ = bitcast<f32>(_e227);
                    let _e229 = t_2;
                    let _e233 = hld(((_e25 + _e229) + 30u));
                    let v30_ = bitcast<f32>(_e233);
                    let _e235 = t_2;
                    let _e239 = hld(((_e25 + _e235) + 31u));
                    let v31_ = bitcast<f32>(_e239);
                    let _e241 = t_2;
                    let _e247 = excl;
                    hst(((_e26 + _e241) + 0u), select(0u, 1u, (_e247 < _e24)));
                    let _e250 = excl;
                    excl = (_e250 + v0_);
                    let _e252 = t_2;
                    let _e258 = excl;
                    hst(((_e26 + _e252) + 1u), select(0u, 1u, (_e258 < _e24)));
                    let _e261 = excl;
                    excl = (_e261 + v1_);
                    let _e263 = t_2;
                    let _e269 = excl;
                    hst(((_e26 + _e263) + 2u), select(0u, 1u, (_e269 < _e24)));
                    let _e272 = excl;
                    excl = (_e272 + v2_);
                    let _e274 = t_2;
                    let _e280 = excl;
                    hst(((_e26 + _e274) + 3u), select(0u, 1u, (_e280 < _e24)));
                    let _e283 = excl;
                    excl = (_e283 + v3_);
                    let _e285 = t_2;
                    let _e291 = excl;
                    hst(((_e26 + _e285) + 4u), select(0u, 1u, (_e291 < _e24)));
                    let _e294 = excl;
                    excl = (_e294 + v4_);
                    let _e296 = t_2;
                    let _e302 = excl;
                    hst(((_e26 + _e296) + 5u), select(0u, 1u, (_e302 < _e24)));
                    let _e305 = excl;
                    excl = (_e305 + v5_);
                    let _e307 = t_2;
                    let _e313 = excl;
                    hst(((_e26 + _e307) + 6u), select(0u, 1u, (_e313 < _e24)));
                    let _e316 = excl;
                    excl = (_e316 + v6_);
                    let _e318 = t_2;
                    let _e324 = excl;
                    hst(((_e26 + _e318) + 7u), select(0u, 1u, (_e324 < _e24)));
                    let _e327 = excl;
                    excl = (_e327 + v7_);
                    let _e329 = t_2;
                    let _e335 = excl;
                    hst(((_e26 + _e329) + 8u), select(0u, 1u, (_e335 < _e24)));
                    let _e338 = excl;
                    excl = (_e338 + v8_);
                    let _e340 = t_2;
                    let _e346 = excl;
                    hst(((_e26 + _e340) + 9u), select(0u, 1u, (_e346 < _e24)));
                    let _e349 = excl;
                    excl = (_e349 + v9_);
                    let _e351 = t_2;
                    let _e357 = excl;
                    hst(((_e26 + _e351) + 10u), select(0u, 1u, (_e357 < _e24)));
                    let _e360 = excl;
                    excl = (_e360 + v10_);
                    let _e362 = t_2;
                    let _e368 = excl;
                    hst(((_e26 + _e362) + 11u), select(0u, 1u, (_e368 < _e24)));
                    let _e371 = excl;
                    excl = (_e371 + v11_);
                    let _e373 = t_2;
                    let _e379 = excl;
                    hst(((_e26 + _e373) + 12u), select(0u, 1u, (_e379 < _e24)));
                    let _e382 = excl;
                    excl = (_e382 + v12_);
                    let _e384 = t_2;
                    let _e390 = excl;
                    hst(((_e26 + _e384) + 13u), select(0u, 1u, (_e390 < _e24)));
                    let _e393 = excl;
                    excl = (_e393 + v13_);
                    let _e395 = t_2;
                    let _e401 = excl;
                    hst(((_e26 + _e395) + 14u), select(0u, 1u, (_e401 < _e24)));
                    let _e404 = excl;
                    excl = (_e404 + v14_);
                    let _e406 = t_2;
                    let _e412 = excl;
                    hst(((_e26 + _e406) + 15u), select(0u, 1u, (_e412 < _e24)));
                    let _e415 = excl;
                    excl = (_e415 + v15_);
                    let _e417 = t_2;
                    let _e423 = excl;
                    hst(((_e26 + _e417) + 16u), select(0u, 1u, (_e423 < _e24)));
                    let _e426 = excl;
                    excl = (_e426 + v16_);
                    let _e428 = t_2;
                    let _e434 = excl;
                    hst(((_e26 + _e428) + 17u), select(0u, 1u, (_e434 < _e24)));
                    let _e437 = excl;
                    excl = (_e437 + v17_);
                    let _e439 = t_2;
                    let _e445 = excl;
                    hst(((_e26 + _e439) + 18u), select(0u, 1u, (_e445 < _e24)));
                    let _e448 = excl;
                    excl = (_e448 + v18_);
                    let _e450 = t_2;
                    let _e456 = excl;
                    hst(((_e26 + _e450) + 19u), select(0u, 1u, (_e456 < _e24)));
                    let _e459 = excl;
                    excl = (_e459 + v19_);
                    let _e461 = t_2;
                    let _e467 = excl;
                    hst(((_e26 + _e461) + 20u), select(0u, 1u, (_e467 < _e24)));
                    let _e470 = excl;
                    excl = (_e470 + v20_);
                    let _e472 = t_2;
                    let _e478 = excl;
                    hst(((_e26 + _e472) + 21u), select(0u, 1u, (_e478 < _e24)));
                    let _e481 = excl;
                    excl = (_e481 + v21_);
                    let _e483 = t_2;
                    let _e489 = excl;
                    hst(((_e26 + _e483) + 22u), select(0u, 1u, (_e489 < _e24)));
                    let _e492 = excl;
                    excl = (_e492 + v22_);
                    let _e494 = t_2;
                    let _e500 = excl;
                    hst(((_e26 + _e494) + 23u), select(0u, 1u, (_e500 < _e24)));
                    let _e503 = excl;
                    excl = (_e503 + v23_);
                    let _e505 = t_2;
                    let _e511 = excl;
                    hst(((_e26 + _e505) + 24u), select(0u, 1u, (_e511 < _e24)));
                    let _e514 = excl;
                    excl = (_e514 + v24_);
                    let _e516 = t_2;
                    let _e522 = excl;
                    hst(((_e26 + _e516) + 25u), select(0u, 1u, (_e522 < _e24)));
                    let _e525 = excl;
                    excl = (_e525 + v25_);
                    let _e527 = t_2;
                    let _e533 = excl;
                    hst(((_e26 + _e527) + 26u), select(0u, 1u, (_e533 < _e24)));
                    let _e536 = excl;
                    excl = (_e536 + v26_);
                    let _e538 = t_2;
                    let _e544 = excl;
                    hst(((_e26 + _e538) + 27u), select(0u, 1u, (_e544 < _e24)));
                    let _e547 = excl;
                    excl = (_e547 + v27_);
                    let _e549 = t_2;
                    let _e555 = excl;
                    hst(((_e26 + _e549) + 28u), select(0u, 1u, (_e555 < _e24)));
                    let _e558 = excl;
                    excl = (_e558 + v28_);
                    let _e560 = t_2;
                    let _e566 = excl;
                    hst(((_e26 + _e560) + 29u), select(0u, 1u, (_e566 < _e24)));
                    let _e569 = excl;
                    excl = (_e569 + v29_);
                    let _e571 = t_2;
                    let _e577 = excl;
                    hst(((_e26 + _e571) + 30u), select(0u, 1u, (_e577 < _e24)));
                    let _e580 = excl;
                    excl = (_e580 + v30_);
                    let _e582 = t_2;
                    let _e588 = excl;
                    hst(((_e26 + _e582) + 31u), select(0u, 1u, (_e588 < _e24)));
                    let _e591 = excl;
                    excl = (_e591 + v31_);
                    let _e593 = t_2;
                    t_2 = (_e593 + PIVOT_UNROLL);
                    let _e596 = excl;
                    if !((_e596 < _e24)) {
                        local_8 = monotone;
                    } else {
                        local_8 = false;
                    }
                    let _e602 = local_8;
                    if _e602 {
                        break;
                    }
                }
                loop {
                    let _e603 = t_2;
                    if (_e603 >= len_1) {
                        break;
                    }
                    let _e605 = t_2;
                    let _e609 = excl;
                    hst((_e26 + _e605), select(0u, 1u, (_e609 < _e24)));
                    let _e612 = excl;
                    let _e613 = t_2;
                    let _e615 = hld((_e25 + _e613));
                    excl = (_e612 + bitcast<f32>(_e615));
                    let _e618 = t_2;
                    t_2 = (_e618 + 1u);
                    let _e621 = excl;
                    if !((_e621 < _e24)) {
                        local_9 = monotone;
                    } else {
                        local_9 = false;
                    }
                    let _e627 = local_9;
                    if _e627 {
                        break;
                    }
                }
                let _e628 = row_1;
                let _e629 = pivot_meta(_e5, _e628);
                let _e630 = t_2;
                hst(_e629, _e630);
            }
            continuing {
                let _e631 = row_1;
                let _e633 = lanes;
                row_1 = (_e631 + _e633);
            }
        }
        return;
    }
    let words = ((_e5 + 3u) / 4u);
    let _e640 = tid;
    w_4 = _e640;
    loop {
        let _e642 = w_4;
        if (_e642 < words) {
        } else {
            break;
        }
        {
            b = array<bool, 4>();
            k_4 = 0u;
            loop {
                let _e648 = k_4;
                if (_e648 < 4u) {
                } else {
                    break;
                }
                {
                    let _e651 = w_4;
                    let _e654 = k_4;
                    let i_7 = ((_e651 * 4u) + _e654);
                    on = false;
                    if (i_7 < _e5) {
                        let row_2 = (i_7 / max(len_1, 1u));
                        let rowb_2 = (row_2 * len_1);
                        let _e663 = ld_f(_e1, i_7);
                        if (_e4 == 0u) {
                            let _e666 = pick(_e10, row_2);
                            let _e667 = ld_i(_e3, _e666);
                            let want = clamp(_e667, 0i, i32(len_1));
                            let _e671 = f32_is_nan(_e663);
                            if !(_e671) {
                                let _e673 = sort_key(_e663);
                                let _e674 = count_key_below(_e5, rowb_2, len_1, _e673);
                                on = (i32(_e674) < want);
                            }
                        } else {
                            let _e677 = pick(_e10, row_2);
                            let _e678 = ld_f(_e3, _e677);
                            on = (_e663 >= _e678);
                        }
                    }
                    let _e680 = k_4;
                    let _e682 = on;
                    b[_e680] = _e682;
                }
                continuing {
                    let _e683 = k_4;
                    k_4 = (_e683 + 1u);
                }
            }
            let _e686 = w_4;
            let _e688 = b[0];
            let _e690 = b[1];
            let _e692 = b[2];
            let _e694 = b[3];
            st_b_word(_e2, _e686, _e5, _e688, _e690, _e692, _e694);
        }
        continuing {
            let _e695 = w_4;
            let _e697 = lanes;
            w_4 = (_e695 + _e697);
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    tid = gid.x;
    lanes = (grid.x * 256u);
    sel_lt = lid.x;
    sel_group = wid.x;
    sel_groups = grid.x;
    let _e19 = pie_step.status_base;
    b_status = _e19;
    let _e23 = pie_step.descs_base;
    b_descs = _e23;
    let _e27 = pie_step.params_base;
    b_params = _e27;
    let _e31 = pie_step.offs_base;
    b_offs = _e31;
    let _e35 = pie_step.heap_base;
    b_heap = _e35;
    let _e40 = pie_step.cfg_base;
    let _e44 = tables[(_e40 + CFG_HEAP_WORDS)];
    lim_heap = _e44;
    let _e48 = pie_step.area;
    tmp_base = _e48;
    let _e51 = pie_step.node;
    op_pivot_finish(_e51);
    heap_flush();
    return;
}
