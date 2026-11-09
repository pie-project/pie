"""Role-split wafer geometry: PE kinds, routes and queue colours (`rcfg`) of every PE.

Units of PX x PY weight PEs (+ the CX column at local x = PX / 2, the CY row at local y = PY / 2,
a filler corner), unit (ux, uy) at (ux (PX + 1), uy (PY + 1)), UNX x UNY of them visited as a
snake (down unit column 0, up column 1, ...). The head (HC weight columns + a control column in
the middle, HR rows) east of the units.
Collectives (single-level chains, first version): the rows' and columns' reductions into CX /
CY with their multicasts; on weight rows the scores' segments (SWA columns, a two-half root
in the middle, its multicast looping back) and GDN's (k, q) multicast (SWG columns, from the
column owning k index r, rows r < DKH); on the CY row the o-norm segments (SWV columns).
Handoffs: vertical between the units of a unit column (row r's chunk along row r to weight
column r, down / up it, along row r of the next unit; the turns are weight-PE relays),
horizontal at the snake's turns and into the head (along each row).
Every PE parks its unused input queues on colours nothing routes to its ramp."""

RX = dict(W=0x00, E=0x20, S=0x40, N=0x60, R=0x80)
TX = dict(W=0x01, E=0x02, S=0x04, N=0x08, R=0x10)
OPP = dict(W="E", E="W", N="S", S="N")
XA, XB, MX, YA, YB, MY, LB = 1, 2, 3, 4, 5, 6, 7
SA, SB, SM, SG = 8, 13, 9, 10                 # weight rows: scores segment chain / multicast, GDN (k, q)
CS, CS2, CM = 11, 17, 12                      # CY row: o-norm segment chain / multicast
HA, HB, HX = 0, 11, 12                        # handoffs (HB / HX never touch the CY row)
HV = (15, 14)                                 # vertical handoffs leaving even / odd units
X2 = (18, 19)                                 # two-level chains: the row's level-2 colours
Y2 = (20, 16)                                 # the column's
HN, HM, HR3 = 4, 6, 3                         # head: relay north, column multicast, row multicast (the wafer loop's row word)
# the wafer token loop (WL): the head control column's (value, id) max-reduce (two-level, root at
# the middle row) and its winner multicast; the winner row's head PEs send their embedding slices up
# their column (VA) to row 0, down again (VS) to the column's CX row pair of unit 0 (rows 2c, 2c + 1),
# the pair's halves split by HS (2c -> 2c + 1), then west along both rows (HE) into unit 0's CX
WA, WB, WC2, WM = 10, 11, (13, 14), 15
VA, VS, HS, HE = 5, 8, 9, 17
ALL = list(range(21))


class Geo:
    def __init__(self, PX, PY, UNX, UNY, HC, HR, Hc, SWA, SWG, SWV, DKG, DKH, units=None, G2=0, G2X=None, G2Y=None, G2H=None, WL=0, G2W=24):
        self.WL, self.G2W = WL, G2W
        self.G2 = G2
        self.G2X = G2 if G2X is None else G2X
        self.G2Y = G2 if G2Y is None else G2Y
        self.G2H = G2 if G2H is None else G2H
        self.PX, self.PY, self.UNX, self.UNY, self.HC, self.HR, self.Hc = PX, PY, UNX, UNY, HC, HR, Hc
        self.SWA, self.SWG, self.SWV, self.DKG, self.DKH = SWA, SWG, SWV, DKG, DKH
        self.XC, self.YC = PX // 2, PY // 2
        self.UW, self.UH = PX + 1, PY + 1
        self.HX0 = UNX * self.UW
        self.HXC = HC // 2
        self.W = self.HX0 + HC + 1
        self.H = max(UNY * self.UH, HR)
        order = []
        for ux in range(UNX):
            ys = range(UNY) if ux % 2 == 0 else range(UNY - 1, -1, -1)
            order += [(ux, uy) for uy in ys]
        self.order = order[:units] if units else order
        self.routes = {}
        self.relay = {}
        self.wrole, self.cxq, self.cyseg, self.wseg, self.gsrc, self.hrole, self.hrow = {}, {}, {}, {}, {}, {}, {}
        self.rootq = {}

    def wl(self, i, c):
        return i + (1 if i >= c else 0)

    def unit_origin(self, u):
        ux, uy = self.order[u]
        return ux * self.UW, uy * self.UH

    def unit_of(self, x, y):
        k = (x // self.UW, y // self.UH)
        return self.order.index(k) if k in self.order else None

    def kind(self, x, y):
        if x >= self.HX0:
            if x - self.HX0 > self.HC or y >= self.HR:
                return "filler"
            return "hctl" if x - self.HX0 == self.HXC else "head"
        if self.unit_of(x, y) is None:
            return "filler"
        lx, ly = x % self.UW, y % self.UH
        if lx == self.XC and ly == self.YC:
            return "filler"
        if lx == self.XC:
            return "cx"
        if ly == self.YC:
            return "cy"
        return "w"

    def add(self, x, y, c, b):
        d = self.routes.setdefault((x, y), {})
        if c in d and d[c] != b:
            raise AssertionError(f"route clash at {(x, y)} colour {c}: {d[c]:#x} vs {b:#x}")
        d[c] = b

    @staticmethod
    def dirn(a, b):
        if b[0] == a[0] + 1: return "E"
        if b[0] == a[0] - 1: return "W"
        if b[1] == a[1] + 1: return "S"
        return "N"

    def line(self, pts, root, A, B, M, loop=False, half_colours=None):
        """Single-level chains toward pts[root] from both sides on colours A / B (alternating
        by distance), the root's multicast on M (loop: also to its own ramp). Returns
        {point: (role, send colour, recv colour or None)} for the non-roots and the colours the
        root takes each half on (low, high)."""
        out = {}
        n = len(pts)
        for i, p in enumerate(pts):
            if i == root:
                continue
            low = i < root
            k = root - i if low else i - root
            kmax = root if low else n - 1 - root
            send = (A if k % 2 else B) if low else (B if k % 2 else A)
            recv = (A if (k + 1) % 2 else B) if low else (B if (k + 1) % 2 else A)
            tow = self.dirn(p, pts[i + 1] if low else pts[i - 1])
            awy = OPP[tow]
            self.add(*p, send, RX["R"] | TX[tow])
            if k < kmax:
                self.add(*p, recv, RX[awy] | TX["R"])
                self.add(*p, M, RX[tow] | TX["R"] | TX[awy])
            else:
                self.add(*p, M, RX[tow] | TX["R"])
            out[p] = (0 if k == kmax else 1, send, recv if k < kmax else None)
        r = pts[root]
        tx = TX["R"] if loop else 0
        lo = hi = None
        if root > 0:
            self.add(*r, A, RX[self.dirn(r, pts[root - 1])] | TX["R"]); lo = A
            tx |= TX[self.dirn(r, pts[root - 1])]
        if root < n - 1:
            self.add(*r, B, RX[self.dirn(r, pts[root + 1])] | TX["R"]); hi = B
            tx |= TX[self.dirn(r, pts[root + 1])]
        self.add(*r, M, RX["R"] | tx)
        return out, (lo, hi)

    def line2(self, pts, root, A, B, C2, M, G):
        """Two-level streamed halves toward pts[root]: each
        half in groups of G counted from the root (k = distance, group q = (k - 1) // G, its end at
        kk = 1); level-1 one-way chains inside a group into its end (A / B by k), the ends' level-2
        chain into the root (C2[(q + half) % 2]), passing the groups' other members. Returns
        {point: (role, send colour, level-1 recv or None, level-2 recv or None)}; roles 0 start,
        1 add-forward, 2 group end (both inputs), 3 the farthest end (level 1 only)."""
        if G <= 0:
            o, h = self.line(pts, root, A, B, M)
            return {p: (v[0], v[1], v[2], None) for p, v in o.items()}, h
        out = {}
        n = len(pts)
        for i, p in enumerate(pts):
            if i == root:
                continue
            low = i < root
            half = 0 if low else 1
            k = root - i if low else i - root
            kmax = root if low else n - 1 - root
            Q = (kmax + G - 1) // G
            q = (k - 1) // G
            kk = k - q * G
            gs = kmax - q * G if q == Q - 1 else G
            tow = self.dirn(p, pts[i + 1] if low else pts[i - 1])
            awy = OPP[tow]
            self.add(*p, M, RX[tow] | TX["R"] | (TX[awy] if k < kmax else 0))
            s2 = C2[(q + half) % 2]
            r2 = C2[(q + 1 + half) % 2]
            l1s = (A if k % 2 else B) if low else (B if k % 2 else A)
            l1r = (A if (k + 1) % 2 else B) if low else (B if (k + 1) % 2 else A)
            if kk > 1:
                self.add(*p, l1s, RX["R"] | TX[tow])
                if kk < gs:
                    self.add(*p, l1r, RX[awy] | TX["R"])
                if q + 1 < Q:
                    self.add(*p, r2, RX[awy] | TX[tow])        # the next end's level-2 passes
                out[p] = (0 if kk == gs else 1, l1s, l1r if kk < gs else None, None)
            else:
                self.add(*p, s2, RX["R"] | TX[tow])
                has1 = gs > 1
                if has1:
                    self.add(*p, l1r, RX[awy] | TX["R"])
                if q + 1 < Q:
                    self.add(*p, r2, RX[awy] | TX["R"])
                    out[p] = (2 if has1 else 1, s2, l1r if has1 else r2, r2 if has1 else None)
                else:
                    out[p] = (3 if has1 else 0, s2, l1r if has1 else None, None)
        r = pts[root]
        tx = 0
        lo = hi = None
        if root > 0:
            c = C2[0]; self.add(*r, c, RX[self.dirn(r, pts[root - 1])] | TX["R"]); lo = c
            tx |= TX[self.dirn(r, pts[root - 1])]
        if root < n - 1:
            c = C2[1]; self.add(*r, c, RX[self.dirn(r, pts[root + 1])] | TX["R"]); hi = c
            tx |= TX[self.dirn(r, pts[root + 1])]
        self.add(*r, M, RX["R"] | tx)
        return out, (lo, hi)

    def path(self, pts, c):
        for i, p in enumerate(pts):
            rx = RX["R"] if i == 0 else RX[self.dirn(p, pts[i - 1])]
            tx = TX["R"] if i == len(pts) - 1 else TX[self.dirn(p, pts[i + 1])]
            self.add(*p, c, rx | tx)

    def mcast(self, pts, src, c):
        """pts[src] multicasts on c to the other points (both directions)."""
        n = len(pts)
        for i, p in enumerate(pts):
            if i == src:
                tx = (TX[self.dirn(p, pts[i - 1])] if i > 0 else 0) | (TX[self.dirn(p, pts[i + 1])] if i < n - 1 else 0)
                self.add(*p, c, RX["R"] | tx)
                continue
            if i < src:
                prev, nxt = pts[i + 1], (pts[i - 1] if i > 0 else None)
            else:
                prev, nxt = pts[i - 1], (pts[i + 1] if i < n - 1 else None)
            self.add(*p, c, RX[self.dirn(p, prev)] | TX["R"] | (TX[self.dirn(p, nxt)] if nxt else 0))

    # ------------------------------------------------------------ build
    def build(self, has_att, has_gdn):
        PX, PY, XC, YC = self.PX, self.PY, self.XC, self.YC
        # a segment (scores, GDN's (k, q), the o-norm) is a line of adjacent PEs: it must not span the
        # control column (the gap at local x = XC)
        for n, sw in (("SWA", self.SWA), ("SWG", self.SWG), ("SWV", self.SWV)):
            assert sw > 0 and XC % sw == 0, f"{n} {sw} does not divide XC {XC} (a segment would span the control column)"
        for u in range(len(self.order)):
            X0, Y0 = self.unit_origin(u)
            for r in range(PY):
                y = Y0 + self.wl(r, YC)
                o, h = self.line2([(X0 + lx, y) for lx in range(PX + 1)], XC, XA, XB, X2, MX, self.G2X)
                self.rootq[(X0 + XC, y)] = h
                for p, v in o.items():
                    self.wrole.setdefault(p, {})["x"] = v
                self.add(X0 + XC, y, LB, RX["R"] | TX["R"])
                wpts = [(X0 + self.wl(c, XC), y) for c in range(PX)]
                if has_att:
                    for s0 in range(0, PX, self.SWA):
                        seg = wpts[s0:s0 + self.SWA]
                        root = len(seg) // 2
                        o, (lo, hi) = self.line(seg, root, SA, SB, SM)
                        # kernel (qmode 2): iq7 <- [0], iq6 <- [1] (a two-half root) or [3], oq4 <- [2]
                        for p, (role, snd, rcv) in o.items():
                            self.wseg[p] = (role, [rcv, None, snd, SM])
                        two = lo is not None and hi is not None
                        self.wseg[seg[root]] = (4 if two else 6, [lo or hi, hi if two else None, SM, None])
                if has_gdn and r < self.DKH:
                    for s0 in range(0, PX, self.SWG):
                        seg = wpts[s0:s0 + self.SWG]
                        if len(seg) == 1:
                            self.gsrc[seg[0]] = (0xC000 | (r % self.DKG), None)
                            continue
                        own = r // self.DKG
                        self.mcast(seg, own, SG)
                        for i, p in enumerate(seg):
                            self.gsrc[p] = ((0x8000 | (r % self.DKG)) if i == own else 0, SG)
                elif has_gdn:
                    for p in wpts:
                        self.gsrc[p] = (0x4000, None)       # no state row: k_r = q_r = 0
            for c in range(PX):
                x = X0 + self.wl(c, XC)
                o, h = self.line2([(x, Y0 + ly) for ly in range(PY + 1)], YC, YA, YB, Y2, MY, self.G2Y)
                self.rootq[(x, Y0 + YC)] = h
                for p, v in o.items():
                    self.wrole.setdefault(p, {})["y"] = v
            y = Y0 + YC
            for x in [X0 + self.wl(c, XC) for c in range(PX)]:
                self.add(x, y, LB, RX["R"] | TX["R"])
            if has_gdn:
                cpts = [(X0 + self.wl(c, XC), y) for c in range(PX)]
                for s0 in range(0, PX, self.SWV):
                    seg = cpts[s0:s0 + self.SWV]
                    if len(seg) == 1:
                        self.cyseg[seg[0]] = (5, None, None, None, None)
                        continue
                    root = len(seg) // 2
                    o, (lo, hi) = self.line(seg, root, CS, CS2, CM)
                    for p, (role, snd, rcv) in o.items():
                        self.cyseg[p] = (role, rcv, snd, CM, None)
                    two = lo is not None and hi is not None
                    self.cyseg[seg[root]] = (4 if two else 6, lo or hi, CM, None, hi if two else None)
        for u in range(len(self.order) - 1):
            self.handoff(u, u + 1)
        self.head_in(len(self.order) - 1)
        for y in range(self.HR):
            o, h = self.line2([(self.HX0 + i, y) for i in range(self.HC + 1)], self.HXC, XA, XB, X2, HR3, self.G2H)
            self.hrow.update(o)
            self.rootq[(self.HX0 + self.HXC, y)] = h
            self.add(self.HX0 + self.HXC, y, LB, RX["R"] | TX["R"])
        self.wrow = {}
        self.hturn = {}
        if self.WL:
            self.wloop()

    def wloop(self):
        HR, YC = self.HR, self.YC
        o, h = self.line2([(self.HX0 + self.HXC, y) for y in range(HR)], HR // 2, WA, WB, WC2, WM, self.G2W)
        self.wrow.update(o)
        self.wroot = (self.HX0 + self.HXC, HR // 2, h)
        U0 = self.unit_origin(0)
        assert U0[1] == 0 and self.unit_origin(len(self.order) - 1)[1] == 0
        for c in range(self.HC):
            x = self.hx(c)
            te, to = self.wl(2 * c, YC), self.wl(2 * c + 1, YC)
            assert to == te + 1 and to < HR
            # VA: every row north to row 0 (the winner's own route is rewritten to its ramp at run time)
            self.add(x, HR - 1, VA, RX["R"] | TX["N"])
            for y in range(1, HR - 1):
                self.add(x, y, VA, RX["S"] | TX["N"])
            self.add(x, 0, VA, RX["S"] | TX["R"])
            # VS: row 0 down to te (column 0: te is row 0 itself, VA ends there)
            if te > 0:
                self.path([(x, y) for y in range(0, te + 1)], VS)
            self.path([(x, te), (x, to)], HS)
            for y in (te, to):
                self.path([(xx, y) for xx in range(x, U0[0] + self.XC - 1, -1)], HE)
                self.cxq.setdefault((U0[0] + self.XC, y), {})["emb"] = HE
            self.hturn[(x, 0)] = self.hturn.get((x, 0), 0) | 1          # the VA sink (row 0)
            self.hturn[(x, te)] = self.hturn.get((x, te), 0) | 2        # takes VS (or VA at row 0), splits
            self.hturn[(x, to)] = self.hturn.get((x, to), 0) | 4        # takes HS

    def handoff(self, ua, ub):
        (ax, ay), (bx, by) = self.order[ua], self.order[ub]
        A0, B0 = self.unit_origin(ua), self.unit_origin(ub)
        XC, YC, PY = self.XC, self.YC, self.PY
        for r in range(PY):
            pr = self.wl(r, YC)
            xa, ya = A0[0] + XC, A0[1] + pr
            if ax == bx:
                hv = HV[ua % 2]
                pc = self.wl(r, XC)
                xc_ = A0[0] + pc
                st = 1 if xc_ > xa else -1
                yb = B0[1] + pr
                vs = 1 if yb > ya else -1
                self.path([(x, ya) for x in range(xa, xc_ + st, st)], HA)
                self.path([(xc_, y) for y in range(ya, yb + vs, vs)], hv)
                self.path([(x, yb) for x in range(xc_, B0[0] + XC - st, -st)], HB)
                self.relay.setdefault((xc_, ya), {})["post"] = (HA, hv)
                self.relay.setdefault((xc_, yb), {})["pre"] = (hv, HB)
                self.cxq.setdefault((xa, ya), {})["out"] = HA
                self.cxq.setdefault((B0[0] + XC, yb), {})["inn"] = HB
            else:
                xb = B0[0] + XC
                st = 1 if xb > xa else -1
                self.path([(x, ya) for x in range(xa, xb + st, st)], HX)
                self.cxq.setdefault((xa, ya), {})["out"] = HX
                self.cxq.setdefault((xb, ya), {})["inn"] = HX

    def hx(self, c):
        return self.HX0 + c + (1 if c >= self.HXC else 0)

    def head_in(self, ul):
        A0 = self.unit_origin(ul)
        XC, YC, PY = self.XC, self.YC, self.PY
        assert self.HC * 2 == PY, "head columns = CX rows / 2"
        for r in range(PY):
            y = A0[1] + self.wl(r, YC)
            xa = A0[0] + XC
            c = r // 2
            self.path([(x, y) for x in range(xa, self.hx(c) + 1)], HX)
            self.cxq.setdefault((xa, y), {})["out"] = HX
            if r % 2 == 1:
                self.hrole[(self.hx(c), y)] = 1
                self.path([(self.hx(c), y), (self.hx(c), y - 1)], HN)
        for c in range(self.HC):
            src = A0[1] + self.wl(2 * c, YC)
            self.hrole[(self.hx(c), src)] = 2
            self.mcast([(self.hx(c), y) for y in range(self.HR)], src, HM)

    # ------------------------------------------------------------ rcfg words
    def words(self, x, y, nr):
        r = self.routes.get((x, y), {})
        if len(r) > nr:
            raise AssertionError(f"{(x, y)} has {len(r)} routes > {nr}: {r}")
        w = [len(r)]
        for c, b in sorted(r.items()):
            w += [c, b]
        return w + [0] * (1 + 2 * nr - len(w))

    def parks(self, x, y, n):
        """n colours nothing routes to this PE's ramp."""
        r = self.routes.get((x, y), {})
        # colours with no route at this PE first (never this PE's own send colours), then
        # colours routed only through it
        free = [c for c in ALL if c not in r] + [c for c in ALL if c in r and not (r[c] & TX["R"]) and (r[c] & 0xe0) != RX["R"]]
        assert len(free) >= n, ((x, y), r)
        return free[:n]

    def tmpc(self, x, y, finals, comptime):
        """A spare colour for the kernel's queue binding: no ramp route here, no final, no comptime park."""
        r = self.routes.get((x, y), {})
        for c in ALL:
            if c in finals or c in comptime:
                continue
            if c not in r or not (r[c] & TX["R"]):
                return c
        for c in ALL:   # a comptime park colour that is free here (it only holds a queue until bound)
            if c not in finals and (c not in r or not (r[c] & TX["R"])):
                return c
        raise AssertionError((x, y, finals))

    def rcfg_weight(self, x, y, r):
        ro = self.wrole[(x, y)]
        rx, ry = ro["x"], ro["y"]
        need = sum(1 for v in (rx[2], rx[3], ry[2], ry[3]) if v is None)
        pk = self.parks(x, y, need)
        it = iter(pk)
        w = self.words(x, y, 16)
        q = [rx[2] if rx[2] is not None else next(it), rx[3] if rx[3] is not None else next(it), ry[2] if ry[2] is not None else next(it),
             ry[3] if ry[3] is not None else next(it), MX, MY, rx[1], ry[1]]
        role_s, sq = self.wseg.get((x, y), (5, [None, None, None, None]))
        # an unused segment slot keeps its queue's own colour (no new park: busy PEs have few spare colours)
        sq = [v if v is not None else (q[5], q[4], q[6], q[4])[i] for i, v in enumerate(sq)]
        gs, gc = self.gsrc.get((x, y), (0x4000, None))
        if gc is None:
            gc = q[4]
        rel = self.relay.get((x, y), {})
        pre = (self.Hc,) + rel["pre"] if "pre" in rel else (0, 0, 0)
        post = (self.Hc,) + rel["post"] if "post" in rel else (0, 0, 0)
        w += q + [rx[0], ry[0], role_s, gs, gc] + sq + [r] + list(pre) + list(post)
        return w + [self.tmpc(x, y, q[:6], (13, 14, 17, 18, 19, 20))]

    def rcfg_cx(self, x, y, first):
        pk = self.parks(x, y, 2)
        w = self.words(x, y, 12)
        h = self.cxq.get((x, y), {})
        lo, hi = self.rootq[(x, y)]
        q = [lo, hi, LB, h.get("inn", pk[0]), LB, MX, h.get("out", pk[1])]
        fl = (1 if first or "inn" not in h else 0) | (2 if "out" not in h else 0)
        em = h.get("emb")
        q6 = em if em is not None else self.parks(x, y, 3)[2]
        fl |= 4 if (self.WL and em is not None) else 0
        return w + q + [fl, self.tmpc(x, y, q[:4] + [q6], (13, 14, 17, 18, 19)), q6]

    def rcfg_cy(self, x, y):
        pk = self.parks(x, y, 4)
        w = self.words(x, y, 12)
        role, si, so, sm, s2 = self.cyseg.get((x, y), (5, None, None, None, None))
        lo, hi = self.rootq[(x, y)]
        q = [lo, hi, LB, si if si is not None else pk[0], LB, MY, so if so is not None else pk[3]]
        tail = [role, sm if sm is not None else pk[1], s2 if s2 is not None else pk[2]]
        return w + q + tail + [self.tmpc(x, y, q[:4] + tail[1:], (13, 14, 17, 18, 19, 20))]

    def rcfg_head(self, x, y):
        pk = self.parks(x, y, 7)
        w = self.words(x, y, 12)
        ro = self.hrow[(x, y)]
        xr = self.hrole.get((x, y), 0)
        q = [ro[2] if ro[2] is not None else pk[0], ro[3] if ro[3] is not None else pk[1], HX if xr in (1, 2) else pk[2], HN if xr == 2 else pk[3],
             HM if xr in (0, 1) else pk[4], ro[1], HN if xr == 1 else (HM if xr == 2 else pk[0])]
        tr = self.hturn.get((x, y), 0) if self.WL else 0
        if tr & 1 and not tr & 2:
            q[3] = VA                       # the VA sink (iq5; row 0 has no relay A)
        if tr & 4:
            assert xr == 1; q[3] = HS        # relay A's spare iq5
        q6 = (VA if y == 0 else VS) if tr & 2 else None
        if tr & 2:
            assert xr == 2; q[4] = q6        # relay B's spare iq6
        q7 = HR3 if self.WL else pk[5]
        c = None
        for cc in range(self.HC):
            if self.hx(cc) == x: c = cc
        fl = (1 if self.WL else 0) | (tr << 1)
        oq = [VA, VS, HE, HS] if self.WL else [pk[6]] * 4
        return w + q + [1 if ro[0] == 3 else ro[0], xr, self.tmpc(x, y, q[:5] + [q7], (13, 14, 17, 18, 19, 20)), q7, fl, y] + oq

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
        return w + [lo, hi, LB, LB, y * vr, self.tmpc(x, y, fin, (13, 14, 17, 18, 19, 20)), q5, q6, q7, oq3, oq4, role, 1 if self.WL else 0]
