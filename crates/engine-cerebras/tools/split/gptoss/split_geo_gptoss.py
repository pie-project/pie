"""gpt-oss role-split geometry (routes, roles and rcfg words of every PE).

A unit is [AX attention columns | CX | MX MoE columns] by PY weight rows, with the CY row at local y = PY / 2:
the base Geo's unit with its control column at local x = AX instead of the middle. Units: UNX unit columns of
UNY, visited as a snake from the bottom of column 0 up, then down column 1, ... so unit 0 and the last unit
(UNX even) sit on the bottom unit row, next to the head band below them.
- Row lines: one two-level line a weight row into CX (both halves: the attention half's o_proj partials and
  the MoE half's down partials come at different times; CX takes each from its own level-2 colour).
- Column lines into CY. Attention rows: the scores' segment over each kv head's SWA columns. CY row: the
  router's segment over each SWR (= the expert count) MoE columns (router output c mod SWR on MoE column c).
- Handoffs: the base Geo's (vertical inside a unit column, horizontal at the snake's turn).
- Head band (rows HB0 ..): PY dims-rows (dims-row i = the last unit's CX row i, Hc dims) and a control row at
  its middle (HXC), by HVC vocabulary columns from x = 0. A band column's logits reduce along the column into
  its control PE (line2, the root's multicast HR3 = the loop's column word). The control row: the wafer
  loop's max-reduce (line2, root in the middle). x in: the last unit's CX row r -> along the row (HA) to its
  MoE column r (a relay) -> down (HV2) to band row r, whose PE there multicasts it along the row (HM).
  Embedding return: the winner column's PEs send along their dims-row west (HE bus) to column 0, which sends
  it east and north (V1) up unit 0's MoE column i to unit 0's row i, whose relay sends it west (W2) to CX."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_geo import Geo, RX, TX, OPP, XA, XB, MX, YA, YB, MY, LB, SA, SB, SM, CS, CS2, CM, HA, HB, HX, HV, X2, Y2, HM, HR3, WA, WB, WC2, WM, ALL

HV2, HE, V1, W2 = 9, 10, 8, 13


class GeoG(Geo):
    def __init__(self, AX, MXc, PY, UNX, UNY, Hc, SWA, SWR, HVC, G2=0, G2X=None, G2Y=None, G2H=None, WL=1, G2W=24):
        super().__init__(AX + MXc, PY, UNX, UNY, 0, PY, Hc, SWA, 1, 1, 1, 0, None, G2, G2X, G2Y, G2H, WL, G2W)
        self.AX, self.MXc, self.SWR, self.HVC = AX, MXc, SWR, HVC
        self.XC = AX
        self.UW, self.UH = AX + MXc + 1, PY + 1
        order = []
        for ux in range(UNX):
            ys = range(UNY - 1, -1, -1) if ux % 2 == 0 else range(UNY)
            order += [(ux, uy) for uy in ys]
        self.order = order
        assert UNX % 2 == 0, "the last unit sits on the bottom row next to the band"
        self.HB0 = UNY * self.UH
        self.HXC = PY // 2
        self.W = max(UNX * self.UW, HVC)
        self.H = self.HB0 + PY + 1
        self.HR = PY                     # (Geo's head rows: unused)
        self.cyseg, self.rout, self.hrow, self.rootq = {}, {}, {}, {}
        self.hturn = {}
        self.embrelay = set()

    def unit_origin(self, u):
        ux, uy = self.order[u]
        return ux * self.UW, uy * self.UH

    def by(self, i):                     # dims-row i's y
        return self.HB0 + self.wl(i, self.HXC)

    def kind(self, x, y):
        if y >= self.HB0:
            if x >= self.HVC:
                return "filler"
            return "hctl" if y - self.HB0 == self.HXC else "head"
        ux, uy = x // self.UW, y // self.UH
        if ux >= self.UNX or uy >= self.UNY:
            return "filler"
        lx, ly = x % self.UW, y % self.UH
        if lx == self.XC and ly == self.YC:
            return "filler"
        if lx == self.XC:
            return "cx"
        if ly == self.YC:
            return "cya" if lx < self.XC else "cym"
        return "wa" if lx < self.XC else "wm"

    # ------------------------------------------------------------ build
    def build(self):
        PX, PY, XC, YC, AX = self.PX, self.PY, self.XC, self.YC, self.AX
        assert AX % self.SWA == 0 and self.MXc % self.SWR == 0
        for u in range(len(self.order)):
            X0, Y0 = self.unit_origin(u)
            for r in range(PY):
                y = Y0 + self.wl(r, YC)
                o, h = self.line2([(X0 + lx, y) for lx in range(PX + 1)], XC, XA, XB, X2, MX, self.G2X)
                self.rootq[(X0 + XC, y)] = h
                for p, v in o.items():
                    self.wrole.setdefault(p, {})["x"] = v
                self.add(X0 + XC, y, LB, RX["R"] | TX["R"])
                wpts = [(X0 + c, y) for c in range(AX)]
                for s0 in range(0, AX, self.SWA):
                    seg = wpts[s0:s0 + self.SWA]
                    root = len(seg) // 2
                    o, (lo, hi) = self.line(seg, root, SA, SB, SM)
                    for p, (role, snd, rcv) in o.items():
                        self.wseg[p] = (role, [rcv, None, snd, SM])
                    two = lo is not None and hi is not None
                    self.wseg[seg[root]] = (4 if two else 6, [lo or hi, hi if two else None, SM, None])
            for c in range(PX):
                x = X0 + self.wl(c, XC)
                o, h = self.line2([(x, Y0 + ly) for ly in range(PY + 1)], YC, YA, YB, Y2, MY, self.G2Y)
                self.rootq[(x, Y0 + YC)] = h
                for p, v in o.items():
                    self.wrole.setdefault(p, {})["y"] = v
                self.add(x, Y0 + YC, LB, RX["R"] | TX["R"])
            y = Y0 + YC
            cpts = [(X0 + XC + 1 + c, y) for c in range(self.MXc)]
            for s0 in range(0, self.MXc, self.SWR):
                seg = cpts[s0:s0 + self.SWR]
                for i, p in enumerate(seg):
                    self.rout[p] = i
                root = len(seg) // 2
                o, (lo, hi) = self.line(seg, root, CS, CS2, CM, loop=False)
                for p, (role, snd, rcv) in o.items():
                    self.cyseg[p] = (role, rcv, snd, CM, None)
                two = lo is not None and hi is not None
                self.cyseg[seg[root]] = (4 if two else 6, lo or hi, CM, None, hi if two else None)
        for u in range(len(self.order) - 1):
            self.handoff(u, u + 1)
        self.head_in(len(self.order) - 1)
        for j in range(self.HVC):
            pts = [(j, self.HB0 + ly) for ly in range(PY + 1)]
            o, h = self.line2(pts, self.HXC, XA, XB, X2, HR3, self.G2H)
            self.hrow.update(o)
            self.rootq[(j, self.HB0 + self.HXC)] = h
            self.add(j, self.HB0 + self.HXC, LB, RX["R"] | TX["R"])
        self.wrow = {}
        if self.WL:
            self.wloop()

    def head_in(self, ul):
        X0, Y0 = self.unit_origin(ul)
        XC, YC, PY = self.XC, self.YC, self.PY
        assert Y0 + self.UH == self.HB0 and PY <= self.MXc
        self.hsrc = {}
        for r in range(PY):
            y = Y0 + self.wl(r, YC)
            xr = X0 + XC + 1 + r                    # MoE column r: the turn
            self.path([(x, y) for x in range(X0 + XC, xr + 1)], HA)     # (HA: the last unit hands off nothing)
            self.cxq.setdefault((X0 + XC, y), {})["out"] = HA
            yb = self.by(r)
            self.path([(xr, yy) for yy in range(y, yb + 1)], HV2)
            self.relay.setdefault((xr, y), {})["post"] = (HA, HV2)
            assert xr < self.HVC
            self.hsrc[(xr, yb)] = 1
            self.mcast([(j, yb) for j in range(self.HVC)], xr, HM)

    def wloop(self):
        HB0, HXC, PY = self.HB0, self.HXC, self.PY
        yc = HB0 + HXC
        o, h = self.line2([(j, yc) for j in range(self.HVC)], self.HVC // 2, WA, WB, WC2, WM, self.G2W)
        self.wrow.update(o)
        self.wroot = (self.HVC // 2, yc, h)
        U0 = self.unit_origin(0)
        assert U0[1] + self.UH == HB0
        for i in range(PY):
            yb = self.by(i)
            # the bus: every PE of dims-row i west, column 0 its end (the winner's route goes to its ramp)
            for j in range(1, self.HVC):
                self.add(j, yb, HE, RX["E"] | TX["W"] if j < self.HVC - 1 else RX["R"] | TX["W"])
            self.add(0, yb, HE, RX["E"] | TX["R"])
            self.hturn[(0, yb)] = 1
            ci = U0[0] + self.XC + 1 + i            # unit 0's MoE column i
            yu = U0[1] + self.wl(i, self.YC)
            assert ci < self.HVC
            pts = [(x, yb) for x in range(0, ci + 1)] + [(ci, yy) for yy in range(yb - 1, yu - 1, -1)]
            self.path(pts, V1)
            self.path([(x, yu) for x in range(ci, U0[0] + self.XC - 1, -1)], W2)
            self.relay.setdefault((ci, yu), {})["pre"] = (V1, W2)
            self.embrelay.add((ci, yu))
            self.cxq.setdefault((U0[0] + self.XC, yu), {})["emb"] = W2

    # ------------------------------------------------------------ rcfg words
    # (the head band's bus colour is a winner's own send colour at run time: never a park there)
    def parks(self, x, y, n):
        if y < self.HB0:
            return super().parks(x, y, n)
        r = self.routes.get((x, y), {})
        free = [c for c in ALL if c not in r and c != HE] + [c for c in ALL if c in r and c != HE and not (r[c] & TX["R"]) and (r[c] & 0xe0) != RX["R"]]
        assert len(free) >= n, ((x, y), r)
        return free[:n]

    def tmpc(self, x, y, finals, comptime):
        return super().tmpc(x, y, list(finals) + ([HE] if y >= self.HB0 else []), comptime)

    def rcfg_w(self, x, y, r, rwc=0, nrel=None):
        ro = self.wrole[(x, y)]
        rx, ry = ro["x"], ro["y"]
        need = sum(1 for v in (rx[2], rx[3], ry[2], ry[3]) if v is None)
        it = iter(self.parks(x, y, need))
        w = self.words(x, y, 16)
        q = [rx[2] if rx[2] is not None else next(it), rx[3] if rx[3] is not None else next(it), ry[2] if ry[2] is not None else next(it),
             ry[3] if ry[3] is not None else next(it), MX, MY, rx[1], ry[1]]
        role_s, sq = self.wseg.get((x, y), (5, [None, None, None, None]))
        sq = [v if v is not None else (q[5], q[4], q[6], q[4])[i] for i, v in enumerate(sq)]
        rel = self.relay.get((x, y), {})
        n = self.Hc if nrel is None else nrel
        pre = (n,) + rel["pre"] if "pre" in rel else (0, 0, 0)
        post = (n,) + rel["post"] if "post" in rel else (0, 0, 0)
        w += q + [rx[0], ry[0], role_s, rwc, q[4]] + sq + [r] + list(pre) + list(post)
        emb = 1 if (x, y) in self.embrelay else 0
        return w + [self.tmpc(x, y, q[:6], (13, 14, 17, 18, 19, 20)), emb]

    def rcfg_cx(self, x, y, first):
        pk = self.parks(x, y, 3)
        w = self.words(x, y, 12)
        h = self.cxq.get((x, y), {})
        lo, hi = self.rootq[(x, y)]
        q = [lo if lo is not None else pk[0], hi if hi is not None else pk[1], pk[2], h.get("inn", pk[0]), LB, MX, h.get("out", pk[1])]
        fl = (1 if first or "inn" not in h else 0) | (2 if "out" not in h else 0)
        em = h.get("emb")
        q6 = em if em is not None else self.parks(x, y, 4)[3]
        fl |= 4 if (self.WL and em is not None) else 0
        # iq4 is unused (no loop-back): any distinct park
        fin = [q[0], q[1], q[2], q[3], q6]
        assert len(set(fin)) == 5, ((x, y), fin)
        return w + q + [fl, self.tmpc(x, y, fin, (13, 14, 17, 18, 19)), q6]

    def rcfg_cy(self, x, y):
        pk = self.parks(x, y, 4)
        w = self.words(x, y, 12)
        role, si, so, sm, s2 = self.cyseg.get((x, y), (5, None, None, None, None))
        lo, hi = self.rootq[(x, y)]
        q = [lo, hi, LB, si if si is not None else pk[0], LB, MY, so if so is not None else pk[3]]
        tail = [role, sm if sm is not None else pk[1], s2 if s2 is not None else pk[2]]
        return w + q + tail + [self.tmpc(x, y, q[:4] + tail[1:], (13, 14, 17, 18, 19, 20)), self.rout.get((x, y), 0)]

    def rcfg_head(self, x, y):
        pk = self.parks(x, y, 6)
        w = self.words(x, y, 12)
        ro = self.hrow[(x, y)]
        src = self.hsrc.get((x, y), 0)
        bus = self.hturn.get((x, y), 0) if self.WL else 0
        q = [ro[2] if ro[2] is not None else pk[0], ro[3] if ro[3] is not None else pk[1], HV2 if src else pk[2],
             HE if bus else pk[3], pk[5] if src else HM, HR3 if self.WL else pk[4]]
        assert len(set(q)) == 6, ((x, y), q)
        fl = (1 if self.WL else 0) | (2 if bus else 0)
        return w + q + [ro[1], HM, 1 if ro[0] == 3 else ro[0], 2 if src else 0, self.tmpc(x, y, q, (13, 14, 17, 18, 19, 20)), fl, 0,
                        HE if self.WL else pk[5], V1 if self.WL else pk[5]]

    def rcfg_hctl(self, x, y, vr):
        w = self.words(x, y, 12)
        lo, hi = self.rootq[(x, y)]
        pk = self.parks(x, y, 4)
        if self.WL:
            if (x, y) == self.wroot[:2]:
                wl_, wh_ = self.wroot[2]
                role, snd, q5, q6, q7 = 4, WM, wl_ or pk[0], wh_ or pk[1], pk[2]
                role |= (8 if wl_ is not None else 0) | (16 if wh_ is not None else 0)
            else:
                role, snd, r1, r2 = self.wrow[(x, y)]
                q5, q6, q7 = r1 if r1 is not None else pk[0], r2 if r2 is not None else pk[1], WM
            oq3, oq4 = snd, HR3
        else:
            role, q5, q6, q7, oq3, oq4 = 0, pk[0], pk[1], pk[2], pk[3], pk[3]
        fin = [lo, hi, LB, q5, q6, q7]
        return w + [lo, hi, LB, LB, x * vr, self.tmpc(x, y, fin, (13, 14, 17, 18, 19, 20)), q5, q6, q7, oq3, oq4, role, 1 if self.WL else 0]


def tiny_geo(**kw):
    # hidden 32 (PY 4 x Hc 8), 2 kv heads x 2 columns (Dc 4), 8 experts, Ic 4 on 8 MoE columns, 2 units of K 2
    return GeoG(AX=4, MXc=8, PY=4, UNX=2, UNY=1, Hc=8, SWA=2, SWR=8, HVC=26, **kw)


def full_geo(**kw):
    # gpt-oss-20b: Hc 18 x 160 rows, 64 attention columns (8 a kv head, Dc 8), 288 MoE columns (Ic 10),
    # 2 x 6 units of K 2, head band 160 dims-rows x 748 vocabulary columns (VR 269)
    return GeoG(AX=64, MXc=288, PY=160, UNX=2, UNY=6, Hc=18, SWA=8, SWR=32, HVC=748, **kw)


GEOS = {"tiny": tiny_geo, "full": full_geo}
