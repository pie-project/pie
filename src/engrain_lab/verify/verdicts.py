"""Does the precomputed refusal ever disagree with an actual replay?

`precompute_verdicts` settles at compile time every replay whose answer does
not depend on the stack - 92.5% of them - and the kernels trust it. So it is
the one table whose correctness no runtime check would catch: a wrong
`refused` narrows a mask silently and every device verification agrees with
it, because they read the same table. This re-derives every refusal from the
ACTION table independently.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import engrain
import engrain.internals


def main() -> None:
    vocabulary = [bytes([b]) for b in range(256)]
    compiler = engrain.internals.Compiler(vocabulary)
    schema = json.dumps({
        "type": "object",
        "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["name", "id"],
    })
    grammar = compiler.compile_json_schema(schema)
    a = grammar.device_arrays()

    def u32(key):
        return np.frombuffer(a[key], dtype=np.uint32)

    def i32(key):
        return np.frombuffer(a[key], dtype=np.int32)

    go, ro, ri = u32('group_offsets'), u32('reading_offsets'), u32('reading_index')
    to, tm = u32('reading_term_offsets'), u32('reading_terminals')
    ao, at_, av = u32('action_offsets'), u32('action_terminals'), i32('action_values')
    vo, vs, vv = u32('verdict_offsets'), u32('verdict_stride'), u32('verdicts')
    L, P = len(go) - 1, len(ao) - 1

    def action(st, t):
        lo, hi = int(ao[st]), int(ao[st + 1])
        row = at_[lo:hi]
        k = int(np.searchsorted(row, t))
        if k >= hi - lo or int(row[k]) != t:
            return None
        return int(av[lo + k])

    bad = 0
    for lex in range(L):
        stride, base = int(vs[lex]), int(vo[lex])
        if stride == 0:
            continue
        for top in range(P):
            for slot, grp in enumerate(range(int(go[lex]), int(go[lex + 1]))):
                word = int(vv[base + top * stride + slot // 16])
                verdict = (word >> (2 * (slot % 16))) & 3
                if verdict != 1:
                    continue
                # It claims refused. Re-derive that: no reading of this group
                # may survive by shifting alone, and none may reach a reduce -
                # a reduce depends on the stack, so it cannot be settled here.
                # `reading_offsets[group]` names a length-prefixed block, not
                # the start of a CSR run: a block is shared with every group in
                # the pool that wants the same reading list, so it cannot say
                # how long it is by where the next one starts.
                at = int(ro[grp])
                for use in range(at + 1, at + 1 + int(ri[at])):
                    reading = int(ri[use])
                    state, alive, reduced = top, True, False
                    for k in range(int(to[reading]), int(to[reading + 1])):
                        value = action(state, int(tm[k]))
                        if value is None:
                            alive = False
                            break
                        if value > 0:
                            state = value - 1
                        else:
                            reduced = True
                            break
                    if reduced or alive:
                        bad += 1
                        if bad <= 3:
                            print(f"DISAGREE lex={lex} top={top} slot={slot} "
                                  f"group={grp} reading={reading} "
                                  f"reduced={reduced} alive={alive}")
                        break
    print("disagreements:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
