"""Gemma 4 12B role-split geometry (spatial/split/README.md, Gemma 4 12B): split_geo.Geo's units, chains, segments,
handoffs, head and token loop, with
- explicit unit origins (unit 0 and the last unit on the same rows: the head's turns),
- the row-role split: a unit's weight rows r < RA are attention rows (Hc A dims, the KV, the scores segments), the
  rest plain rows (Hc W dims); the CX of a row holds its row's dims; handoffs carry 2 Hc(r) + 1 words a row,
- head groups: head column c takes x's dims of the CX rows in groups[c] (consecutive rows, never across the CY
  row): the group's members chain their words north to the root (its first row) on the last unit's CX column (CH
  colours, alternating a hop), the root sends them east (HX) to the head PE on its row, which multicasts them down
  the column (HM). The wafer loop's embedding goes back to unit 0's CX of the root row (HE) and down the group's
  chain on unit 0's CX column.
No PLE, no shared KV."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # split_geo.py (tools/split)
import split_geo as G
from split_geo import RX, TX

CH = (4, 5)          # the head-group chains (CX columns only)


class GeoG12(G.Geo):
    def __init__(self, PX, PY, HC, HR, HcA, HcW, RA, SWA, origins, groups, W=None, K=6, **kw):
        super().__init__(PX, PY, 1, 1, HC, HR, HcW, SWA, SWA, SWA, 1, 0, **kw)
        self.HcA, self.HcW, self.RA, self.K = HcA, HcW, RA, K
        self.origins = list(origins)
        self.order = list(range(len(origins)))
        self.groups = [list(g_) for g_ in groups]
        assert len(self.groups) == HC and sorted(r for g_ in self.groups for r in g_) == list(range(PY))
        for g_ in self.groups:
            assert g_ == list(range(g_[0], g_[0] + len(g_))) and not (g_[0] < self.YC <= g_[-1]), g_
        self.HX0 = W if W is not None else max(x for x, _ in origins) + self.UW
        self.W = self.HX0 + HC + 1
        self.H = max(max(y for _, y in origins) + self.UH, HR)
        self.cxch = {}   # CX (x, y) -> dict(ci=, co=, nfx=, nfe=, member=, emember=)

    def hc(self, r):
        return self.HcA if r < self.RA else self.HcW

    def nx(self, c):
        return sum(self.hc(r) for r in self.groups[c])

    def unit_origin(self, u):
        return self.origins[u]

    def unit_of(self, x, y):
        for u, (X0, Y0) in enumerate(self.origins):
            if X0 <= x < X0 + self.UW and Y0 <= y < Y0 + self.UH:
                return u
        return None

    def kind(self, x, y):
        if x >= self.HX0:
            if x - self.HX0 > self.HC or y >= self.HR:
                return "filler"
            return "hctl" if x - self.HX0 == self.HXC else "head"
        u = self.unit_of(x, y)
        if u is None:
            return "filler"
        X0, Y0 = self.origins[u]
        lx, ly = x - X0, y - Y0
        if lx == self.XC and ly == self.YC:
            return "filler"
        if lx == self.XC:
            return "cx"
        if ly == self.YC:
            return "cy"
        return "w"

    def build(self, has_att=True, has_gdn=False):
        PX, PY, XC, YC = self.PX, self.PY, self.XC, self.YC
        assert self.SWA > 0 and XC % self.SWA == 0, "a segment would span the control column"
        for u in range(len(self.order)):
            X0, Y0 = self.unit_origin(u)
            for r in range(PY):
                y = Y0 + self.wl(r, YC)
                o, h = self.line2([(X0 + lx, y) for lx in range(PX + 1)], XC, G.XA, G.XB, G.X2, G.MX, self.G2X)
                self.rootq[(X0 + XC, y)] = h
                for p, v in o.items():
                    self.wrole.setdefault(p, {})["x"] = v
                self.add(X0 + XC, y, G.LB, RX["R"] | TX["R"])
                if r >= self.RA:
                    continue
                wpts = [(X0 + self.wl(c, XC), y) for c in range(PX)]
                for s0 in range(0, PX, self.SWA):
                    seg = wpts[s0:s0 + self.SWA]
                    root = len(seg) // 2
                    o, (lo, hi) = self.line(seg, root, G.SA, G.SB, G.SM)
                    for p, (role, snd, rcv) in o.items():
                        self.wseg[p] = (role, [rcv, None, snd, G.SM])
                    two = lo is not None and hi is not None
                    self.wseg[seg[root]] = (4 if two else 6, [lo or hi, hi if two else None, G.SM, None])
            for c in range(PX):
                x = X0 + self.wl(c, XC)
                o, h = self.line2([(x, Y0 + ly) for ly in range(PY + 1)], YC, G.YA, G.YB, G.Y2, G.MY, self.G2Y)
                self.rootq[(x, Y0 + YC)] = h
                for p, v in o.items():
                    self.wrole.setdefault(p, {})["y"] = v
            y = Y0 + YC
            for x in [X0 + self.wl(c, XC) for c in range(PX)]:
                self.add(x, y, G.LB, RX["R"] | TX["R"])
        for u in range(len(self.order) - 1):
            self.handoff(u, u + 1)
        self.head_in(len(self.order) - 1)
        for y in range(self.HR):
            o, h = self.line2([(self.HX0 + i, y) for i in range(self.HC + 1)], self.HXC, G.XA, G.XB, G.X2, G.HR3, self.G2H)
            self.hrow.update(o)
            self.rootq[(self.HX0 + self.HXC, y)] = h
            self.add(self.HX0 + self.HXC, y, G.LB, RX["R"] | TX["R"])
        self.wrow = {}
        self.hturn = {}
        if self.WL:
            self.wloop()

    def handoff(self, ua, ub):
        (ax, ay), (bx, by) = self.origins[ua], self.origins[ub]
        XC, YC, PY = self.XC, self.YC, self.PY
        for r in range(PY):
            pr = self.wl(r, YC)
            xa, ya = ax + XC, ay + pr
            if ax == bx:
                hv = G.HV[ua % 2]
                pc = self.wl(r, XC)
                xc_ = ax + pc
                st = 1 if xc_ > xa else -1
                yb = by + pr
                vs = 1 if yb > ya else -1
                self.path([(x, ya) for x in range(xa, xc_ + st, st)], G.HA)
                self.path([(xc_, y) for y in range(ya, yb + vs, vs)], hv)
                self.path([(x, yb) for x in range(xc_, bx + XC - st, -st)], G.HB)
                self.relay.setdefault((xc_, ya), {})["post"] = (G.HA, hv)
                self.relay.setdefault((xc_, yb), {})["pre"] = (hv, G.HB)
                self.cxq.setdefault((xa, ya), {})["out"] = G.HA
                self.cxq.setdefault((bx + XC, yb), {})["inn"] = G.HB
            else:
                assert ay == by, "a horizontal handoff joins units on the same rows"
                xb = bx + XC
                st = 1 if xb > xa else -1
                self.path([(x, ya) for x in range(xa, xb + st, st)], G.HX)
                self.cxq.setdefault((xa, ya), {})["out"] = G.HX
                self.cxq.setdefault((xb, ya), {})["inn"] = G.HX

    def ry(self, u, r):
        return self.origins[u][1] + self.wl(r, self.YC)

    def head_in(self, ul):
        X0 = self.origins[ul][0]
        xcx = X0 + self.XC
        for c, grp in enumerate(self.groups):
            root = grp[0]
            y = self.ry(ul, root)
            self.path([(x, y) for x in range(xcx, self.hx(c) + 1)], G.HX)
            self.cxq.setdefault((xcx, y), {})["out"] = G.HX
            self.hrole[(self.hx(c), y)] = 2
            self.mcast([(self.hx(c), yy) for yy in range(self.HR)], y, G.HM)
            for k, r in enumerate(grp):
                yk = self.ry(ul, r)
                ch = self.cxch.setdefault((xcx, yk), {})
                ch["nfx"] = sum(self.hc(rr) for rr in grp[k + 1:])
                if k > 0:   # hop r_k -> r_(k-1) on CH[k % 2]
                    self.path([(xcx, yk), (xcx, self.ry(ul, grp[k - 1]))], CH[k % 2])
                    ch["co"] = CH[k % 2]
                    ch["member"] = True
                if k + 1 < len(grp):
                    ch["ci"] = CH[(k + 1) % 2]

    def wloop(self):
        HR, YC = self.HR, self.YC
        o, h = self.line2([(self.HX0 + self.HXC, y) for y in range(HR)], HR // 2, G.WA, G.WB, G.WC2, G.WM, self.G2W)
        self.wrow.update(o)
        self.wroot = (self.HX0 + self.HXC, HR // 2, h)
        U0 = self.origins[0]
        assert U0[1] == self.origins[-1][1], "unit 0 shares the last unit's rows"
        xcx = U0[0] + self.XC
        for c, grp in enumerate(self.groups):
            x = self.hx(c)
            te = self.ry(0, grp[0])
            self.add(x, HR - 1, G.VA, RX["R"] | TX["N"])
            for y in range(1, HR - 1):
                self.add(x, y, G.VA, RX["S"] | TX["N"])
            self.add(x, 0, G.VA, RX["S"] | TX["R"])
            if te > 0:
                self.path([(x, y) for y in range(0, te + 1)], G.VS)
            self.path([(xx, te) for xx in range(x, xcx - 1, -1)], G.HE)
            self.cxq.setdefault((xcx, te), {})["emb"] = G.HE
            self.hturn[(x, 0)] = self.hturn.get((x, 0), 0) | 1
            self.hturn[(x, te)] = self.hturn.get((x, te), 0) | 2
            for k, r in enumerate(grp):
                yk = self.ry(0, r)
                ch = self.cxch.setdefault((xcx, yk), {})
                ch["nfe"] = sum(self.hc(rr) for rr in grp[k + 1:])
                if k > 0:   # hop r_(k-1) -> r_k on CH[k % 2]
                    self.path([(xcx, self.ry(0, grp[k - 1])), (xcx, yk)], CH[k % 2])
                    ch["ci"] = CH[k % 2]
                    ch["emember"] = True
                if k + 1 < len(grp):
                    ch["co"] = CH[(k + 1) % 2]

    def build_g12(self):
        self.build(True, False)

    # ------------------------------------------------------------ rcfg words
    WNR = 18   # weight PEs' route slots (weight_g12.csl NR)

    def words(self, x, y, nr):
        return super().words(x, y, self.WNR if nr == 16 else nr)

    def rcfg_weight_g(self, x, y, r, c, flush):
        w = self.rcfg_weight(x, y, r)
        n = 1 + 2 * self.WNR
        for o in (18, 21):
            if w[n + o]:
                w[n + o] = 2 * self.hc(r) + 1
        w[n + 11] = (1 if c == 0 else 0) | (4 if flush else 0)
        w[n + 12] = 0
        return w

    def rcfg_cx_g(self, x, y, first, flush):
        w = self.rcfg_cx(x, y, first)
        n = 1 + 2 * 12
        fl = w[n + 7]
        ch = self.cxch.get((x, y), {})
        if flush:
            fl |= 8 | 16
            if ch.get("member"):
                fl |= 64 | 2
        if first and ch.get("emember"):
            fl |= 4 | 128
        w[n + 7] = fl
        pk = self.parks(x, y, 4)
        ci = ch.get("ci", pk[3])
        co = ch.get("co", pk[3])
        # the spare colour of the queue binding: none of the six input queues' finals (iq7 the chain)
        w[n + 8] = self.tmpc(x, y, w[n:n + 4] + [w[n + 9], ci], (13, 14, 17, 18, 19, 20))
        return w + [ci, co, ch.get("nfx", 0) if flush else 0, ch.get("nfe", 0) if first else 0]

    def rcfg_cy_g(self, x, y, flush):
        w = self.rcfg_cy(x, y)
        return w + [4 if flush else 0, 0, 0, 0, 0]

    def rcfg_head(self, x, y):
        pk = self.parks(x, y, 7)
        w = self.words(x, y, 12)
        ro = self.hrow[(x, y)]
        xr = self.hrole.get((x, y), 0)
        q = [ro[2] if ro[2] is not None else pk[0], ro[3] if ro[3] is not None else pk[1], G.HX if xr == 2 else pk[2], pk[3],
             G.HM if xr == 0 else pk[4], ro[1], G.HM if xr == 2 else pk[0]]
        tr = self.hturn.get((x, y), 0) if self.WL else 0
        if tr & 1 and not tr & 2:
            q[3] = G.VA                     # the VA sink (iq5)
        q6 = (G.VA if y == 0 else G.VS) if tr & 2 else None
        if tr & 2:
            assert xr == 2; q[4] = q6       # te: a root's spare iq6
        q7 = G.HR3 if self.WL else pk[5]
        c = next(cc for cc in range(self.HC) if self.hx(cc) == x)
        fl = (1 if self.WL else 0) | (tr << 1)
        oq = [G.VA, G.VS, G.HE, G.HS] if self.WL else [pk[6]] * 4
        return (w + q + [1 if ro[0] == 3 else ro[0], xr, self.tmpc(x, y, q[:5] + [q7], (13, 14, 17, 18, 19, 20)), q7, fl, y] + oq
                + [self.nx(c)])

    def rcfg_hctl_g(self, x, y, vr):
        return self.rcfg_hctl(x, y, vr)


def groups_split(PY, YC, RA, ga, gw):
    """Head groups: the A rows in groups of ga, the W rows above the CY row and those below it in groups of gw (no
    group across the CY row or the A / W boundary)."""
    out = []
    for lo, hi, gs in ((0, min(RA, YC), ga), (min(RA, YC), YC, gw), (YC, PY, gw)):
        r = lo
        while r < hi:
            out.append(list(range(r, min(r + gs, hi))))
            r += gs
    return out


def tiny_geo(**kw):
    """The tiny simulation: units 8 x 8 (A rows 0-2 Hc 2, W rows 3-7 Hc 3; YC 4), SWA 4, 2 x 2 units (unit 0 at the
    origin, down column 0, up column 1: the last unit on unit 0's rows), head groups [0-2] [3] [4-5] [6-7]."""
    PX, PY, RA = 8, 8, 3
    origins = [(0, 0), (0, 9), (9, 9), (9, 0)]
    grp = groups_split(PY, PY // 2, RA, 3, 2)
    return GeoG12(PX, PY, len(grp), 18, 2, 3, RA, 4, origins, grp, K=3, **kw)


def g12_geo(**kw):
    """12B: units 344 x 283 (A rows 0-131 Hc 12, W rows 132-282 Hc 15), SWA 43, K 6; column 0 (x 0) units 0-3 top
    to bottom, column 1 (x 345) units 4-7 bottom to top (unit 7 on unit 0's rows); the head 53 + 1 columns at x 690
    (groups: the A rows in sixes, the W rows in fives)."""
    PX, PY, RA = 344, 283, 132
    UH = PY + 1
    origins = [(0, UH * u) for u in range(4)] + [(345, UH * (7 - u)) for u in range(4, 8)]
    grp = groups_split(PY, PY // 2, RA, 6, 5)
    return GeoG12(PX, PY, len(grp), 1153, 12, 15, RA, 43, origins, grp, K=6, **kw)


GEOS = dict(tiny=tiny_geo, g12=g12_geo)


def dump_rcfg(g, vr, path):
    """Every placed PE's rcfg words as "x y kind w..." lines (geo/g12.rs `rcfg_dump`'s format)."""
    last = len(g.origins) - 1
    with open(path, "w") as f:
        for x in range(g.W):
            for y in range(g.H):
                k = g.kind(x, y)
                if k in ("w", "cx", "cy"):
                    u = g.unit_of(x, y); X0, Y0 = g.origins[u]; lx, ly = x - X0, y - Y0
                    r, c = ly - (ly > g.YC), lx - (lx > g.XC)
                    w = g.rcfg_weight_g(x, y, r, c, u == last) if k == "w" else g.rcfg_cx_g(x, y, u == 0, u == last) if k == "cx" else g.rcfg_cy_g(x, y, u == last)
                elif k == "head":
                    w = g.rcfg_head(x, y)
                elif k == "hctl":
                    w = g.rcfg_hctl_g(x, y, vr)
                else:
                    continue
                f.write(f"{x} {y} {k} " + " ".join(str(int(v)) for v in w) + "\n")
