"""Bit-level emulations (torch half: one rounding an op) of the wafer program CY's f16 SiLU,
act = silu(g) u, as the kernel computes it (kernels-cerebras programs/wafer/ctl_cy_qwen.csl's silu16).

- silu16: the f32-scaled form: s g log2e and s u ln2 in f32, rounded once to f16.
- silu16w (SLW 1): silu16 without the subnormal flush: t clamped at -24 (not -15),
  2^(n+9) from m's bits, e' = 2^r 2^(n+9), e = e' 2^-9 for 1 + e and a e = (e' a) 2^-9, so f16
  never goes subnormal on the path and sig(a) no longer flushes to 0 for g < -10.05.
Inputs gs, us: f32 tensors of the rms-scaled gate and up (s g, s u)."""
import torch
LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453
C = [0.9999283788785166, 0.6932608957863087, 0.24260644929856207, 0.05516966520599117]
QC = [2.1207294161729284, -1.4540415401729079, 0.323112507040746]
def h(v): return v.half().float()
def c16(x): return float(torch.tensor(x).half())
def silu16(gs, us):
    """Returns the f16-valued act as f32."""
    G = h(gs * LOG2E); U = h(us * LN2)
    T = torch.clamp(-G.abs(), min=-15.0)
    M = h(T + 1536.0); N = h(M - 1536.0); R = h(T - N)
    P = h(R * c16(C[3])); P = h(P + c16(C[2])); P = h(P * R); P = h(P + c16(C[1])); P = h(P * R); P = h(P + c16(C[0]))
    K = torch.where(N + 15 >= 1, torch.exp2(N), torch.zeros_like(N))
    E = h(P * K)
    W = h(E + 1.0)
    Q = h(W * c16(QC[2])); Q = h(Q + c16(QC[1])); Q = h(Q * W); Q = h(Q + c16(QC[0]))
    W = h(W * Q); W = h(2.0 - W); Q = h(Q * W)
    P = h(E * G); P = torch.maximum(P, G); O = h(P * U); O = h(O * Q)
    return O
def silu16w(gs, us, lo=-24.0, k=9):
    """SLW 1: as silu16, the tail kept (see the module docs)."""
    G = h(gs * LOG2E); U = h(us * LN2)
    T = torch.clamp(-G.abs(), min=lo)
    M = h(T + 1536.0); N = h(M - 1536.0); R = h(T - N)
    P = h(R * c16(C[3])); P = h(P + c16(C[2])); P = h(P * R); P = h(P + c16(C[1])); P = h(P * R); P = h(P + c16(C[0]))
    K = torch.where(N + k + 15 >= 1, torch.exp2(N + k), torch.zeros_like(N))
    Ep = h(P * K)                        # e 2^k
    E = h(Ep * 2.0 ** -k)
    W = h(E + 1.0)
    Q = h(W * c16(QC[2])); Q = h(Q + c16(QC[1])); Q = h(Q * W); Q = h(Q + c16(QC[0]))
    W = h(W * Q); W = h(2.0 - W); Q = h(Q * W)
    P = h(Ep * G); P = h(P * 2.0 ** -k); P = torch.maximum(P, G); O = h(P * U); O = h(O * Q)
    return O
