"""host.py: what the bare runners (run_<family>.py, cs_python on the simulator or a CS-3) share: the runtime on a linked
program, symbol copies by rectangle, the prep's files, the placement wafer-compile's bindings state, init.

  h = host.Host(args)       args: a parse of host.parser()'s options (the runner adds its own)
  h.put / h.get             a symbol's words over a rectangle (put: 16-bit elements one a word with b16)
  h.place(skip=())          every binding in order (`skip`: symbols left out); h.init(): init and a barrier
  h.meta, h.npy(name), h.g  the prep's meta.json (--override applied), its arrays, the compiled program"""
import argparse, json, os, time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime  # pylint: disable=no-name-in-module
import compiler

HERE = os.path.dirname(os.path.abspath(__file__))


def parser(doc):
    """The options every runner takes."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name"); ap.add_argument("--cmaddr"); ap.add_argument("--report", default="g.json")
    ap.add_argument("--prep"); ap.add_argument("--prefix", default=""); ap.add_argument("--ntok", type=int, default=0)
    ap.add_argument("--greedy", type=int, default=0); ap.add_argument("--force", type=int, default=2)
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--override", default="")
    ap.add_argument("--xdbg", type=int, default=0, help="read every CX's x after each layer after each launch (XKEEP 1 builds)")
    ap.add_argument("--tdbg", type=int, default=0)
    ap.add_argument("--golden", default="golden.json", help="the golden in the prep dir (after the prefix)")
    ap.add_argument("--compiled", default=os.path.join(HERE, "compiled"), help="wafer-compile's output for this prep (build.py writes it)")
    return ap


K32 = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
K16 = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)


class Host:
    def __init__(self, a):
        t0 = time.time()
        self.a = a
        self.meta = json.load(open(self.file("meta.json")))
        for kv in filter(None, a.override.split(",")):
            k, v = kv.split("="); self.meta[k] = int(v)
        self.g = compiler.load(a.compiled)
        self.runner = SdkRuntime(a.name, cmaddr=a.cmaddr)
        self.runner.load(); self.runner.run()
        self.t_load = time.time() - t0
        self.t_place = 0.0
        self.ids = {}

    def file(self, name):
        return os.path.join(self.a.prep, self.a.prefix + name)

    def npy(self, name):
        return np.load(self.file(name + ".npy"), mmap_mode="r")

    def golden(self):
        return json.load(open(self.file(self.a.golden)))

    def sym(self, s):
        if s not in self.ids:
            self.ids[s] = self.runner.get_id(s)
        return self.ids[s]

    def put(self, s, x0, y0, w, h, arr, b16=False):
        if w * h == 0: return
        arr = np.array(arr, copy=True)
        if b16: arr = arr.view(np.uint16).astype(np.uint32)
        n = arr.size // (w * h)
        self.runner.memcpy_h2d(self.sym(s), arr.reshape(-1), x0, y0, w, h, n, **(K16 if b16 else K32))

    def get(self, s, x0, y0, w, h, n, dt=np.float32):
        b = np.zeros(w * h * n, dt)
        self.runner.memcpy_d2h(b, self.sym(s), x0, y0, w, h, n, **K32)
        return b.reshape(h, w, n)

    def barrier(self):
        """A read from every PE: each one has run what was launched before."""
        b = np.zeros(self.g.W * self.g.H, np.uint32)
        self.runner.memcpy_d2h(b, self.sym("rcfg"), 0, 0, self.g.W, self.g.H, 1, **K32)

    def place(self, skip=()):
        """Every copy as wafer-compile's bindings state them (the Rust family's placement, in its order), but `skip`'s
        symbols."""
        t0 = time.time()
        put = lambda s, *r: None if s in skip else self.put(s, *r)
        compiler.place(put, self.npy, self.g, lambda n: os.path.exists(self.file(n + ".npy")))
        self.t_place = time.time() - t0

    def init(self):
        self.runner.launch("init", nonblock=False)
        self.barrier()
        print(json.dumps({"load_s": round(self.t_load, 1), "place_s": round(self.t_place, 1)}), flush=True)

    def launch(self, steps, pos):
        self.runner.launch("compute", np.uint32(steps), np.uint32(pos), nonblock=False)

    def stop(self):
        self.runner.stop()
