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

fn d_dtype(v_2: u32) -> u32 {
    let _e2 = d_word(v_2, 4u);
    return _e2;
}

fn base(v_3: u32) -> u32 {
    let _e3 = b_offs;
    let _e6 = tables[(_e3 + v_3)];
    return (_e6 >> 2u);
}

fn p_at(p: u32, k_1: u32) -> u32 {
    let _e4 = b_params;
    let _e10 = tables[((_e4 + (p * PARAM_WORDS)) + k_1)];
    return _e10;
}

fn p_a0_(p_1: u32) -> u32 {
    let _e2 = p_at(p_1, 1u);
    return _e2;
}

fn p_o0_(p_2: u32) -> u32 {
    let _e2 = p_at(p_2, 4u);
    return _e2;
}

fn ld_raw(v_4: u32, i: u32) -> u32 {
    let _e2 = base(v_4);
    let _e4 = hld((_e2 + i));
    return _e4;
}

fn ld_byte(v_5: u32, i_1: u32) -> u32 {
    let _e2 = base(v_5);
    let _e6 = hld((_e2 + (i_1 >> 2u)));
    return ((_e6 >> ((i_1 & 3u) * 8u)) & 255u);
}

fn ld_h_bits(v_6: u32, i_2: u32) -> u32 {
    let _e2 = base(v_6);
    let _e6 = hld((_e2 + (i_2 >> 1u)));
    return ((_e6 >> ((i_2 & 1u) * 16u)) & 65535u);
}

fn ld_h(v_7: u32, i_3: u32) -> f32 {
    let _e2 = ld_h_bits(v_7, i_3);
    return bitcast<f32>((_e2 << 16u));
}

fn ld_f(v_8: u32, i_4: u32) -> f32 {
    let _e2 = d_dtype(v_8);
    if (_e2 == DT_F32_) {
        let _e5 = ld_raw(v_8, i_4);
        return bitcast<f32>(_e5);
    }
    if (_e2 == DT_BF16_) {
        let _e9 = ld_h(v_8, i_4);
        return _e9;
    }
    if (_e2 == DT_I32_) {
        let _e12 = ld_raw(v_8, i_4);
        return f32(bitcast<i32>(_e12));
    }
    if (_e2 == DT_U32_) {
        let _e17 = ld_raw(v_8, i_4);
        return f32(_e17);
    }
    let _e19 = ld_byte(v_8, i_4);
    return select(0f, 1f, (_e19 != 0u));
}

fn ld_i(v_9: u32, i_5: u32) -> i32 {
    let _e2 = d_dtype(v_9);
    if (_e2 == DT_F32_) {
        let _e5 = ld_raw(v_9, i_5);
        return i32(bitcast<f32>(_e5));
    }
    if (_e2 == DT_BF16_) {
        let _e10 = ld_h(v_9, i_5);
        return i32(_e10);
    }
    if (_e2 == DT_I32_) {
        let _e14 = ld_raw(v_9, i_5);
        return bitcast<i32>(_e14);
    }
    if (_e2 == DT_U32_) {
        let _e18 = ld_raw(v_9, i_5);
        return i32(_e18);
    }
    let _e20 = ld_byte(v_9, i_5);
    return select(0i, 1i, (_e20 != 0u));
}

fn ld_u(v_10: u32, i_6: u32) -> u32 {
    let _e2 = d_dtype(v_10);
    if (_e2 == DT_F32_) {
        let _e5 = ld_raw(v_10, i_6);
        return u32(bitcast<f32>(_e5));
    }
    if (_e2 == DT_BF16_) {
        let _e10 = ld_h(v_10, i_6);
        return u32(_e10);
    }
    if (_e2 == DT_I32_) {
        let _e14 = ld_raw(v_10, i_6);
        return bitcast<u32>(bitcast<i32>(_e14));
    }
    if (_e2 == DT_U32_) {
        let _e19 = ld_raw(v_10, i_6);
        return _e19;
    }
    let _e22 = ld_byte(v_10, i_6);
    return select(0u, 1u, (_e22 != 0u));
}

fn st_raw(v_11: u32, i_7: u32, w_3: u32) {
    let _e3 = base(v_11);
    hst((_e3 + i_7), w_3);
    return;
}

fn st_f(v_12: u32, i_8: u32, x_1: f32) {
    st_raw(v_12, i_8, bitcast<u32>(x_1));
    return;
}

fn st_i(v_13: u32, i_9: u32, x_2: i32) {
    st_raw(v_13, i_9, bitcast<u32>(x_2));
    return;
}

fn st_u(v_14: u32, i_10: u32, x_3: u32) {
    st_raw(v_14, i_10, x_3);
    return;
}

fn st_b_word(v_15: u32, w_4: u32, n: u32, b0_: bool, b1_: bool, b2_: bool, b3_: bool) {
    var packed: u32 = 0u;
    var local_2: bool;
    var local_3: bool;
    var local_4: bool;
    var local_5: bool;

    if b0_ {
        local_2 = (((w_4 * 4u) + 0u) < n);
    } else {
        local_2 = false;
    }
    let _e17 = local_2;
    if _e17 {
        let _e18 = packed;
        packed = (_e18 | 255u);
    }
    if b1_ {
        local_3 = (((w_4 * 4u) + 1u) < n);
    } else {
        local_3 = false;
    }
    let _e29 = local_3;
    if _e29 {
        let _e30 = packed;
        packed = (_e30 | 65280u);
    }
    if b2_ {
        local_4 = (((w_4 * 4u) + 2u) < n);
    } else {
        local_4 = false;
    }
    let _e41 = local_4;
    if _e41 {
        let _e42 = packed;
        packed = (_e42 | 16711680u);
    }
    if b3_ {
        local_5 = (((w_4 * 4u) + 3u) < n);
    } else {
        local_5 = false;
    }
    let _e53 = local_5;
    if _e53 {
        let _e54 = packed;
        packed = (_e54 | 4278190080u);
    }
    let _e57 = base(v_15);
    let _e59 = packed;
    hst((_e57 + w_4), _e59);
    return;
}

fn bf16_round(x_4: f32) -> u32 {
    let b_1 = bitcast<u32>(x_4);
    if ((b_1 & 2147483647u) > 2139095040u) {
        return (((b_1 >> 16u) | 64u) & 65535u);
    }
    return (((b_1 + 32767u) + ((b_1 >> 16u) & 1u)) >> 16u);
}

fn st_h_word(v_16: u32, w_5: u32, n_1: u32, x0_: f32, x1_: f32) {
    var packed_1: u32;

    let _e5 = bf16_round(x0_);
    packed_1 = _e5;
    if (((w_5 * 2u) + 1u) < n_1) {
        let _e12 = packed_1;
        let _e13 = bf16_round(x1_);
        packed_1 = (_e12 | (_e13 << 16u));
    }
    let _e17 = base(v_16);
    let _e19 = packed_1;
    hst((_e17 + w_5), _e19);
    return;
}

fn h_words(n_2: u32) -> u32 {
    return ((n_2 + 1u) / 2u);
}

fn pick(n_3: u32, i_11: u32) -> u32 {
    if (n_3 == 1u) {
        return 0u;
    }
    return i_11;
}

fn unary_b(which: u32, a: u32, j: u32) -> bool {
    let _e3 = ld_f(a, j);
    return (_e3 != 0f);
}

fn unary_f(which_1: u32, a_1: u32, j_1: u32) -> f32 {
    let _e3 = ld_f(a_1, j_1);
    switch which_1 {
        case 0u: {
            return exp(_e3);
        }
        case 1u: {
            return log(_e3);
        }
        case 2u: {
            return -(_e3);
        }
        case 3u: {
            return (1f / _e3);
        }
        case 4u: {
            return abs(_e3);
        }
        case 5u: {
            if (_e3 > 0f) {
                return 1f;
            }
            if (_e3 < 0f) {
                return -1f;
            }
            return 0f;
        }
        case 7u: {
            return sin(_e3);
        }
        case 8u: {
            return cos(_e3);
        }
        case 9u: {
            return sqrt(_e3);
        }
        case 10u: {
            return (1f / sqrt(_e3));
        }
        default: {
        }
    }
    return _e3;
}

fn unary_i(which_2: u32, a_2: u32, j_2: u32) -> i32 {
    switch which_2 {
        case 0u: {
            let _e3 = ld_f(a_2, j_2);
            return i32(exp(_e3));
        }
        case 1u: {
            let _e6 = ld_f(a_2, j_2);
            return i32(log(_e6));
        }
        case 3u: {
            let _e9 = ld_f(a_2, j_2);
            return i32((1f / _e9));
        }
        case 2u: {
            let _e13 = ld_i(a_2, j_2);
            return -(_e13);
        }
        case 4u: {
            let _e15 = ld_i(a_2, j_2);
            if (_e15 == i32(-2147483648)) {
                return _e15;
            }
            return abs(_e15);
        }
        case 5u: {
            let _e19 = ld_i(a_2, j_2);
            return sign(_e19);
        }
        case 7u: {
            let _e21 = ld_f(a_2, j_2);
            return i32(sin(_e21));
        }
        case 8u: {
            let _e24 = ld_f(a_2, j_2);
            return i32(cos(_e24));
        }
        case 9u: {
            let _e27 = ld_f(a_2, j_2);
            return i32(sqrt(_e27));
        }
        case 10u: {
            let _e30 = ld_f(a_2, j_2);
            return i32((1f / sqrt(_e30)));
        }
        default: {
        }
    }
    let _e35 = ld_i(a_2, j_2);
    return _e35;
}

fn unary_u(which_3: u32, a_3: u32, j_3: u32) -> u32 {
    switch which_3 {
        case 0u: {
            let _e3 = ld_f(a_3, j_3);
            return u32(exp(_e3));
        }
        case 1u: {
            let _e6 = ld_f(a_3, j_3);
            return u32(log(_e6));
        }
        case 3u: {
            let _e9 = ld_f(a_3, j_3);
            return u32((1f / _e9));
        }
        case 2u: {
            let _e14 = ld_u(a_3, j_3);
            return (0u - _e14);
        }
        case 4u: {
            let _e16 = ld_u(a_3, j_3);
            return _e16;
        }
        case 5u: {
            let _e19 = ld_u(a_3, j_3);
            return select(0u, 1u, (_e19 != 0u));
        }
        case 7u: {
            let _e23 = ld_f(a_3, j_3);
            return u32(sin(_e23));
        }
        case 8u: {
            let _e26 = ld_f(a_3, j_3);
            return u32(cos(_e26));
        }
        case 9u: {
            let _e29 = ld_f(a_3, j_3);
            return u32(sqrt(_e29));
        }
        case 10u: {
            let _e32 = ld_f(a_3, j_3);
            return u32((1f / sqrt(_e32)));
        }
        default: {
        }
    }
    let _e37 = ld_u(a_3, j_3);
    return _e37;
}

fn map_unary(p_3: u32, which_4: u32) {
    var w_6: u32;
    var b: array<bool, 4>;
    var k_2: u32;
    var v_17: bool;
    var w_7: u32;
    var x1_1: f32;
    var i_12: u32;

    let _e2 = p_a0_(p_3);
    let _e3 = p_o0_(p_3);
    let _e4 = d_len(_e3);
    let _e5 = d_dtype(_e3);
    if (_e5 == DT_BOOL) {
        let words = ((_e4 + 3u) / 4u);
        let _e13 = tid;
        w_6 = _e13;
        loop {
            let _e15 = w_6;
            if (_e15 < words) {
            } else {
                break;
            }
            {
                b = array<bool, 4>();
                k_2 = 0u;
                loop {
                    let _e21 = k_2;
                    if (_e21 < 4u) {
                    } else {
                        break;
                    }
                    {
                        let _e24 = w_6;
                        let _e27 = k_2;
                        let i_13 = ((_e24 * 4u) + _e27);
                        v_17 = false;
                        if (i_13 < _e4) {
                            let _e32 = d_len(_e2);
                            let _e33 = pick(_e32, i_13);
                            let _e34 = unary_b(which_4, _e2, _e33);
                            v_17 = _e34;
                        }
                        let _e35 = k_2;
                        let _e37 = v_17;
                        b[_e35] = _e37;
                    }
                    continuing {
                        let _e38 = k_2;
                        k_2 = (_e38 + 1u);
                    }
                }
                let _e41 = w_6;
                let _e43 = b[0];
                let _e45 = b[1];
                let _e47 = b[2];
                let _e49 = b[3];
                st_b_word(_e3, _e41, _e4, _e43, _e45, _e47, _e49);
            }
            continuing {
                let _e50 = w_6;
                let _e52 = lanes;
                w_6 = (_e50 + _e52);
            }
        }
        return;
    }
    if (_e5 == DT_BF16_) {
        let _e56 = d_len(_e2);
        let _e58 = tid;
        w_7 = _e58;
        loop {
            let _e60 = w_7;
            let _e61 = h_words(_e4);
            if (_e60 < _e61) {
            } else {
                break;
            }
            {
                let _e63 = w_7;
                let i_14 = (_e63 * 2u);
                x1_1 = 0f;
                if ((i_14 + 1u) < _e4) {
                    let _e73 = pick(_e56, (i_14 + 1u));
                    let _e74 = unary_f(which_4, _e2, _e73);
                    x1_1 = _e74;
                }
                let _e75 = w_7;
                let _e76 = pick(_e56, i_14);
                let _e77 = unary_f(which_4, _e2, _e76);
                let _e78 = x1_1;
                st_h_word(_e3, _e75, _e4, _e77, _e78);
            }
            continuing {
                let _e79 = w_7;
                let _e81 = lanes;
                w_7 = (_e79 + _e81);
            }
        }
        return;
    }
    let _e84 = tid;
    i_12 = _e84;
    loop {
        let _e86 = i_12;
        if (_e86 < _e4) {
        } else {
            break;
        }
        {
            let _e88 = d_len(_e2);
            let _e89 = i_12;
            let _e90 = pick(_e88, _e89);
            if (_e5 == DT_F32_) {
                let _e93 = i_12;
                let _e94 = unary_f(which_4, _e2, _e90);
                st_f(_e3, _e93, _e94);
            } else {
                if (_e5 == DT_I32_) {
                    let _e97 = i_12;
                    let _e98 = unary_i(which_4, _e2, _e90);
                    st_i(_e3, _e97, _e98);
                } else {
                    let _e99 = i_12;
                    let _e100 = unary_u(which_4, _e2, _e90);
                    st_u(_e3, _e99, _e100);
                }
            }
        }
        continuing {
            let _e101 = i_12;
            let _e103 = lanes;
            i_12 = (_e101 + _e103);
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
    let _e54 = pie_step.selector;
    map_unary(_e51, _e54);
    heap_flush();
    return;
}
