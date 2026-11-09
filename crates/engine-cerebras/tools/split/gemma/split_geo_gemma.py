"""Gemma 4 role-split geometry (src/spatial/split/README.md, "Gemma 4"): split_geo.Geo's units, chains, segments,
handoffs, head and token loop, plus
- explicit unit origins (a list of (x, y) top-left corners, any snake the handoffs allow),
- "ple" PEs (every PE outside the units and the head: PLE storage, or routes only),
- TB: the head control rows' token id, west along each storage row (storage PEs take a copy),
- PB: a PLE bus a unit column, down each weight column from the storage above the units (the owner switches its
  route to its ramp at run time); the CY PEs on it take a copy (an async receive),
- KVS: the shared-KV source unit's CY row sends north through the consumer units above it (consumer weight PEs
  take a copy; the source's weight PEs and the consumers' CY PEs pass it).
Handoff width 2 Hc + 1 (x | the pending term p | its row's |h|^2)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # split_geo.py (tools/split)
import split_geo as G
from split_geo import RX, TX

TB = 0
KVS = 10


class GeoG(G.Geo):
    def __init__(self, PX, PY, HC, HR, Hc, SWA, origins, cols, kvsrc=None, kvdst=(), W=None, K=5, **kw):
        """origins: unit u's (x, y). cols: unit columns, each dict(x=, units=[u top..bottom], srows=(y0, y1),
        pb=colour, nt=tokens a storage PE) (the storage rows are directly above the column's top unit)."""
        super().__init__(PX, PY, 1, 1, HC, HR, Hc, SWA, SWA, SWA, 1, 0, **kw)
        self.origins = list(origins)
        self.order = list(range(len(origins)))
        self.cols = cols
        self.kvsrc, self.kvdst = kvsrc, list(kvdst)
        self.HX0 = max(x for x, _ in origins) + self.UW
        self.HX0 = W if W is not None else self.HX0
        self.W = self.HX0 + HC + 1
        self.H = max(max(y for _, y in origins) + self.UH, HR)
        self.hand_n = 2 * Hc + 1
        self.K = K
        self.ple = {}        # (x, y) -> dict(t0=, nt=, lv=, top=, pb=)
        self.tbsend = set()  # hctl rows that send TB
        self.cyple = {}      # CY (x, y) -> (pb colour, its unit's index in the column, words on the bus)
        self.kvsend = set()  # CY PEs that send KVS
        self.kvrecv = set()  # weight PEs that take KVS

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
            return "ple"
        X0, Y0 = self.origins[u]
        lx, ly = x - X0, y - Y0
        if lx == self.XC and ly == self.YC:
            return "filler"
        if lx == self.XC:
            return "cx"
        if ly == self.YC:
            return "cy"
        return "w"

    def handoff(self, ua, ub):
        (ax, ay), (bx, by) = self.origins[ua], self.origins[ub]
        A0, B0 = (ax, ay), (bx, by)
        XC, YC, PY = self.XC, self.YC, self.PY
        G_ = G
        for r in range(PY):
            pr = self.wl(r, YC)
            xa, ya = A0[0] + XC, A0[1] + pr
            if ax == bx:
                hv = G_.HV[ua % 2]
                pc = self.wl(r, XC)
                xc_ = A0[0] + pc
                st = 1 if xc_ > xa else -1
                yb = B0[1] + pr
                vs = 1 if yb > ya else -1
                self.path([(x, ya) for x in range(xa, xc_ + st, st)], G_.HA)
                self.path([(xc_, y) for y in range(ya, yb + vs, vs)], hv)
                self.path([(x, yb) for x in range(xc_, B0[0] + XC - st, -st)], G_.HB)
                self.relay.setdefault((xc_, ya), {})["post"] = (G_.HA, hv)
                self.relay.setdefault((xc_, yb), {})["pre"] = (hv, G_.HB)
                self.cxq.setdefault((xa, ya), {})["out"] = G_.HA
                self.cxq.setdefault((B0[0] + XC, yb), {})["inn"] = G_.HB
            else:
                assert ay == by, "a horizontal handoff joins units on the same rows"
                xb = B0[0] + XC
                st = 1 if xb > xa else -1
                self.path([(x, ya) for x in range(xa, xb + st, st)], G_.HX)
                self.cxq.setdefault((xa, ya), {})["out"] = G_.HX
                self.cxq.setdefault((xb, ya), {})["inn"] = G_.HX

    def wloop(self):
        # split_geo's, without its unit-0-at-row-0 assertion (the paths' clash checks guard the layout)
        HR, YC = self.HR, self.YC
        o, h = self.line2([(self.HX0 + self.HXC, y) for y in range(HR)], HR // 2, G.WA, G.WB, G.WC2, G.WM, self.G2W)
        self.wrow.update(o)
        self.wroot = (self.HX0 + self.HXC, HR // 2, h)
        U0 = self.unit_origin(0)
        for c in range(self.HC):
            x = self.hx(c)
            te, to = U0[1] + self.wl(2 * c, YC), U0[1] + self.wl(2 * c + 1, YC)
            assert to == te + 1 and to < HR
            self.add(x, HR - 1, G.VA, RX["R"] | TX["N"])
            for y in range(1, HR - 1):
                self.add(x, y, G.VA, RX["S"] | TX["N"])
            self.add(x, 0, G.VA, RX["S"] | TX["R"])
            if te > 0:
                self.path([(x, y) for y in range(0, te + 1)], G.VS)
            self.path([(x, te), (x, to)], G.HS)
            for y in (te, to):
                self.path([(xx, y) for xx in range(x, U0[0] + self.XC - 1, -1)], G.HE)
                self.cxq.setdefault((U0[0] + self.XC, y), {})["emb"] = G.HE
            self.hturn[(x, 0)] = self.hturn.get((x, 0), 0) | 1
            self.hturn[(x, te)] = self.hturn.get((x, te), 0) | 2
            self.hturn[(x, to)] = self.hturn.get((x, to), 0) | 4

    def head_in(self, ul):
        A0 = self.unit_origin(ul)
        XC, YC, PY = self.XC, self.YC, self.PY
        assert self.HC * 2 == PY
        for r in range(PY):
            y = A0[1] + self.wl(r, YC)
            xa = A0[0] + XC
            c = r // 2
            self.path([(x, y) for x in range(xa, self.hx(c) + 1)], G.HX)
            self.cxq.setdefault((xa, y), {})["out"] = G.HX
            if r % 2 == 1:
                self.hrole[(self.hx(c), y)] = 1
                self.path([(self.hx(c), y), (self.hx(c), y - 1)], G.HN)
        for c in range(self.HC):
            src = A0[1] + self.wl(2 * c, YC)
            self.hrole[(self.hx(c), src)] = 2
            self.mcast([(self.hx(c), y) for y in range(self.HR)], src, G.HM)

    def build_gemma(self):
        self.build(True, False)
        XC, YC, PY, PX = self.XC, self.YC, self.PY, self.PX
        # ---- PLE storage, TB, PB
        # TB: one stream a storage row, west from the head control row to the westmost storage block on the row;
        # every storage PE (not a block's CX column) takes a copy
        trows = {}
        for col in self.cols:
            for y in range(*col["srows"]):
                trows.setdefault(y, []).append(col["x"])
        for y, xs0 in trows.items():
            xw = min(xs0)
            hc = self.HX0 + self.HXC
            pts = [(x, y) for x in range(hc, xw - 1, -1)]
            for i, p in enumerate(pts):
                rx = RX["R"] if i == 0 else RX["E"]
                tx = 0 if i == len(pts) - 1 else TX["W"]
                if p[0] < self.HX0 and any(X0 <= p[0] < X0 + self.UW and p[0] != X0 + XC for X0 in xs0):
                    tx |= TX["R"]
                self.add(*p, TB, rx | tx)
            self.tbsend.add(y)
        for col in self.cols:
            X0 = col["x"]
            y0, y1 = col["srows"]
            top = self.origins[col["units"][0]][1]
            assert y1 == top, "the storage sits directly above the column's top unit"
            nu = len(col["units"])
            lv = self.K * nu
            for c in range(PX):
                x = X0 + self.wl(c, XC)
                for y in range(y0, y1):
                    k = y - y0
                    self.add(x, y, col["pb"], (RX["R"] if y == y0 else RX["N"]) | TX["S"])
                    self.ple[(x, y)] = dict(t0=k * col["nt"], nt=col["nt"], lv=lv, top=y == y0, pb=col["pb"], col=c)
                prev_end = y1
                for ui, u in enumerate(col["units"]):
                    UX, UY = self.origins[u]
                    lastu = ui == nu - 1
                    for y in range(prev_end, UY):          # a gap between the column's units (route-only PEs)
                        self.add(x, y, col["pb"], RX["N"] | TX["S"])
                    prev_end = UY + self.UH
                    for ly in range(self.UH):
                        y = UY + ly
                        if ly == YC:
                            self.add(x, y, col["pb"], RX["N"] | TX["R"] | (0 if lastu else TX["S"]))
                            self.cyple[(x, y)] = (col["pb"], ui, lv)
                            if lastu:
                                break
                        else:
                            self.add(x, y, col["pb"], RX["N"] | TX["S"])
        # ---- KVS: the source unit's CY row north through the consumers (all in one unit column, above it)
        if self.kvsrc is not None:
            SX, SY = self.origins[self.kvsrc]
            ys = sorted(self.origins[u][1] for u in self.kvdst)
            assert all(self.origins[u][0] == SX for u in self.kvdst) and max(ys) < SY
            ytop = min(ys)
            for c in range(PX):
                x = SX + self.wl(c, XC)
                ysrc = SY + YC
                self.add(x, ysrc, KVS, RX["R"] | TX["N"])
                self.kvsend.add((x, ysrc))
                for y in range(ysrc - 1, ytop - 1, -1):
                    u = self.unit_of(x, y)
                    assert u is not None, ("KVS crosses a gap", x, y)
                    ly = y - self.origins[u][1]
                    take = u in self.kvdst and ly != YC
                    tx = (TX["R"] if take else 0) | (TX["N"] if y > ytop else 0)
                    self.add(x, y, KVS, RX["S"] | tx)
                    if take:
                        self.kvrecv.add((x, y))

    # ------------------------------------------------------------ rcfg words
    WNR = 18   # weight PEs' route slots (weight_gemma.csl NR)

    def words(self, x, y, nr):
        return super().words(x, y, self.WNR if nr == 16 else nr)

    def rcfg_weight_g(self, x, y, r, c, flush):
        w = self.rcfg_weight(x, y, r)
        # GDN slots repurposed: [RQ+11] flags (bit 0 column 0: its row sends the deferred scalar, bit 1 KVS consumer,
        # bit 2 the flush unit), [RQ+12] KVS colour
        n = 1 + 2 * self.WNR
        for o in (18, 21):
            if w[n + o]:
                w[n + o] = self.hand_n
        fl = (1 if c == 0 else 0) | (2 if (x, y) in self.kvrecv else 0) | (4 if flush else 0)
        w[n + 11] = fl
        w[n + 12] = KVS
        return w

    def rcfg_cx_g(self, x, y, first, flush):
        w = self.rcfg_cx(x, y, first)
        n = 1 + 2 * 12
        w[n + 7] |= (8 | 16) if flush else 0
        return w

    def rcfg_cy_g(self, x, y, flush):
        w = self.rcfg_cy(x, y)
        pb, ui, lv = self.cyple.get((x, y), (None, 0, 0))
        fl = (1 if pb is not None else 0) | (2 if (x, y) in self.kvsend else 0) | (4 if flush else 0)
        # [RQ+11] flags, [RQ+12] PB colour, [RQ+13] this unit's place in its column (PLE words 5 ui ..), [RQ+14] bus words
        return w + [fl, pb if pb is not None else 0, ui, lv, KVS]

    def rcfg_ple(self, x, y):
        w = self.words(x, y, 8)
        p = self.ple.get((x, y))
        r = self.routes.get((x, y), {})
        tb_in = TB in r and (r[TB] & TX["R"]) != 0
        pk = self.parks(x, y, 2)
        fl = (1 if p else 0) | (2 if tb_in else 0) | (4 if p and p["top"] else 0)
        return w + [fl, p["t0"] if p else 0, p["nt"] if p else 0, p["lv"] if p else 0, p["pb"] if p else pk[1],
                    TB if tb_in else pk[0], self.tmpc(x, y, [TB if tb_in else pk[0]], (13, 14, 17, 18, 19, 20))]

    def rcfg_hctl_g(self, x, y, vr):
        return self.rcfg_hctl(x, y, vr) + [1 if y in self.tbsend else 0, TB]


def ple_rects(g):
    """The "ple" PE rectangles (x, y, w, h, store): storage blocks and every other non-unit PE a route crosses;
    the remaining non-unit PEs west of the head are fillers."""
    out = []
    for col in g.cols:
        y0, y1 = col["srows"]
        out.append((col["x"], y0, g.UW, y1 - y0, True))
    return out


def pass_pes(g):
    """Non-unit, non-storage PEs west of the head that carry a route."""
    st = set()
    for (x0, y0, w, h, _) in ple_rects(g):
        for x in range(x0, x0 + w):
            for y in range(y0, y0 + h):
                st.add((x, y))
    return sorted(p for p in g.routes if p[0] < g.HX0 and g.kind(*p) == "ple" and p not in st)


def tiny_geo(**kw):
    """The tiny simulation, E2B's arrangement in small: PX 8 x PY 8 units; column A (x 0) units 4, 3, 2 (top to bottom,
    unit 2 the KVS source), storage rows 0-3; column B (x 9) units 0 and 1 level with units 4 and 2 (unit 0 shares the
    last unit's rows, as the head's turns need), storage rows 2-3; the head at x 18."""
    PX, PY = 8, 8
    origins = [(9, 4), (9, 22), (0, 22), (0, 13), (0, 4)]
    cols = [dict(x=0, units=[4, 3, 2], srows=(0, 4), pb=17, nt=128),
            dict(x=9, units=[0, 1], srows=(2, 4), pb=10, nt=256)]
    return GeoG(PX, PY, 4, 32, 3, 4, origins, cols, kvsrc=2, kvdst=(3, 4), **kw)


def e2b_geo(**kw):
    """E2B: column A (x 0) units 6..2 at rows 508.., storage rows 163-507; column B (x 257) units 0 (rows 508-636, level
    with unit 6) and 1 (rows 1024-1152, level with unit 2), storage rows 370-507; the head at x 514."""
    PX, PY = 256, 128
    UH = PY + 1
    oA = [(0, 1153 - UH * (5 - i)) for i in range(5)]            # top to bottom: units 6, 5, 4, 3, 2
    origins = [(257, 508), (257, 1024), oA[4], oA[3], oA[2], oA[1], oA[0]]
    cols = [dict(x=0, units=[6, 5, 4, 3, 2], srows=(163, 508), pb=17, nt=760),
            dict(x=257, units=[0, 1], srows=(370, 508), pb=10, nt=1900)]
    return GeoG(PX, PY, 64, 1130, 12, 128, origins, cols, kvsrc=2, kvdst=(3, 4, 5, 6), **kw)


def e4b_geo(**kw):
    """E4B: units 256 x 160 (Hc 16), K 6 (S x 5, G); column A (x 0) units 6, 5, 4, 3 (rows 509..; unit 3 the KVS source
    of layers 22 / 23), storage rows 177-508; column B (x 257) units 0, 1, 2 level with 6, 5, 3, storage rows 260-508;
    the head 80 + 1 columns at x 514."""
    PX, PY = 256, 160
    UH = PY + 1
    oA = [(0, 1153 - UH * (4 - i)) for i in range(4)]            # top to bottom: units 6, 5, 4, 3
    origins = [(257, oA[0][1]), (257, oA[1][1]), (257, oA[3][1]), oA[3], oA[2], oA[1], oA[0]]
    cols = [dict(x=0, units=[6, 5, 4, 3], srows=(oA[0][1] - 332, oA[0][1]), pb=17, nt=791),
            dict(x=257, units=[0, 1, 2], srows=(oA[0][1] - 249, oA[0][1]), pb=10, nt=1055)]
    return GeoG(PX, PY, 80, 1130, 16, 128, origins, cols, kvsrc=3, kvdst=(4, 5, 6), K=6, **kw)


GEOS = dict(tiny=tiny_geo, e2b=e2b_geo, e4b=e4b_geo)


def dump_rcfg(g, vr, path):
    """Every placed PE's rcfg words as "x y kind w..." lines (geo/gemma.rs `rcfg_dump`'s format)."""
    last = len(g.origins) - 1
    pp = set(pass_pes(g))
    rects = ple_rects(g)
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
                elif k == "ple" and ((x, y) in g.ple or (x, y) in pp or any(a <= x < a + ww and b <= y < b + hh for (a, b, ww, hh, _) in rects)):
                    w = g.rcfg_ple(x, y)
                else:
                    continue
                f.write(f"{x} {y} {k} " + " ".join(str(int(v)) for v in w) + "\n")
