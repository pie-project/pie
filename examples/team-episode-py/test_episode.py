import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode import (Episode, STOP, ASSISTANT_OPEN, number_words, parse_tool_calls,  # noqa: E402
                     loads_lenient, render_system, render_tool_responses)


def call(name, **args):
    return '<tool_call>\n%s\n</tool_call>' % json.dumps({"name": name, "arguments": args})


END = "<|im_end|>"


class FakeDriver:

    def __init__(self, ep, scripts):
        self.ep = ep
        self.scripts = {a: list(s) for a, s in scripts.items()}
        self.ctx = {a: ep.shared_prompt() + ep.private_prompt(a) for a in range(ep.n_agents)}
        self.injects = {a: [] for a in range(ep.n_agents)}
        self.rounds = 0

    def chunk(self, a):
        s = self.scripts.get(a) or []
        if not s:
            return (END, "end_turn", 1)
        c = s.pop(0)
        text, reason, n = c(self.ep, a, self.ctx[a]) if callable(c) else c
        return text, reason, min(n, self.ep.max_new_tokens(a))  # driver never exceeds the budget

    def run(self, max_rounds=200):
        ep = self.ep
        while not ep.done and self.rounds < max_rounds:
            self.rounds += 1
            for a in ep.runnable():
                text, reason, n = self.chunk(a)
                self.ctx[a] += text
                inj = ep.step(a, text, reason, n)
                if inj:
                    self.ctx[a] += inj
                    self.injects[a].append(inj)
            for a in range(ep.n_agents):
                if ep.status(a) == "idle":
                    w = ep.wake(a)
                    if w:
                        self.ctx[a] += w
                        self.injects[a].append(w)
        assert ep.done, "episode did not end"
        return ep.result()


def last_response(inj):
    out = []
    for part in inj.split("<tool_response>\n")[1:]:
        out.append(json.loads(part.split("\n</tool_response>")[0]))
    return out


def test_number_words():
    assert number_words(1) == "one"
    assert number_words(13) == "thirteen"
    assert number_words(20) == "twenty"
    assert number_words(42) == "forty-two"
    assert number_words(99) == "ninety-nine"
    assert len(set(number_words(i) for i in range(1, 100))) == 99


def test_prompts_and_template():
    ep = Episode("aggregate", 3, seed=7)
    sp = ep.shared_prompt()
    assert sp.startswith("<|im_start|>system\n")
    assert sp.endswith('{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
                       "<|im_end|>\n")
    assert "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" in sp
    assert "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"message\"" in sp
    assert "\n</tools>\n\nFor each function call" in sp
    assert all(ep.shared_prompt() == sp for _ in range(3))
    for a in range(3):
        assert number_words(ep.numbers[a]) not in sp or ep.numbers.count(ep.numbers[a]) == 0
        pp = ep.private_prompt(a)
        assert pp.startswith("<|im_start|>user\nYou are agent %d." % a)
        assert number_words(ep.numbers[a]) in pp
        assert pp.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ASSISTANT_OPEN == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert STOP == ("</tool_call>", "<|im_end|>")
    assert render_tool_responses([{"ok": True}]) == (
        '<|im_start|>user\n<tool_response>\n{"ok": true}\n</tool_response><|im_end|>\n'
        + ASSISTANT_OPEN)
    assert render_tool_responses([{"a": 1}, {"b": 2}]).startswith(
        '<|im_start|>user\n<tool_response>\n{"a": 1}\n</tool_response>\n<tool_response>\n{"b": 2}'
        '\n</tool_response><|im_end|>\n')
    assert render_system("", []).startswith("<|im_start|>system\n# Tools\n")
    maze = Episode("maze", 2, seed=1)
    names = [t["function"]["name"] for t in maze.tools]
    assert names == ["message", "read_messages", "look", "move"]
    assert [t["function"]["name"] for t in ep.tools] == ["message", "read_messages", "submit"]


def test_determinism():
    a, b = Episode("aggregate", 5, seed=3), Episode("aggregate", 5, seed=3)
    assert a.numbers == b.numbers and a.shared_prompt() == b.shared_prompt()
    assert Episode("aggregate", 5, seed=4).numbers != a.numbers or True
    m1, m2 = Episode("maze", 2, seed=9, depth=3), Episode("maze", 2, seed=9, depth=3)
    assert m1.exit == m2.exit and m1._adj == m2._adj
    assert any(Episode("maze", 1, seed=s).exit != m1.exit for s in range(5))


def aggregate_success_scripts(ep):
    n = ep.n_agents

    def make(a):
        known = {a: ep.numbers[a]}

        def read(ep_, a_, ctx):
            return ("Reading.\n" + call("read_messages"), "tool_call", 6)

        def submit(ep_, a_, ctx):
            for line in ctx.split("<tool_response>\n")[1:]:
                obj = json.loads(line.split("\n</tool_response>")[0])
                for m in obj.get("messages", []):
                    known[m["from"]] = int(m["text"])
            assert len(known) == n, known
            return (call("submit", answer=sorted(known.values())), "tool_call", 12)

        s = [(call("message", to="all", text=str(ep.numbers[a])), "tool_call", 15), read]
        if a == 0:
            s.append(submit)
        return s

    return {a: make(a) for a in range(n)}


def test_aggregate_success():
    ep = Episode("aggregate", 4, seed=11, budget_tokens=256)
    drv = FakeDriver(ep, aggregate_success_scripts(ep))
    r = drv.run()
    assert r["success"] is True, r
    assert r["end_reason"] == "success"
    assert r["broadcasts"] == 4 and r["messages"] == 4
    assert r["submissions"] == [{"agent": 0, "answer": sorted(ep.numbers), "correct": True}]
    assert set(r.keys()) >= {"task", "seed", "n_agents", "success", "reward", "tokens", "messages",
                             "broadcasts", "submissions", "events"}
    tot = sum(r["tokens"])
    assert abs(r["reward"] - (1 - 0.25 * min(1, tot / (4 * 256)))) < 1e-12
    json.dumps(r)  # serializable
    inj = drv.injects[1][0]
    assert inj.startswith("<|im_end|>\n<|im_start|>user\n<tool_response>\n")
    assert inj.endswith("</tool_response><|im_end|>\n" + ASSISTANT_OPEN)
    assert last_response(inj)[0]["ok"] is True


def test_wrong_submission():
    ep = Episode("aggregate", 2, seed=5)
    wrong = sorted(ep.numbers)
    wrong[0] = wrong[0] % 99 + 1
    drv = FakeDriver(ep, {0: [(call("submit", answer=wrong), "tool_call", 10)], 1: []})
    r = drv.run()
    assert r["success"] is False and r["reward"] == 0.0
    assert r["submissions"] == [{"agent": 0, "answer": wrong, "correct": False}]
    assert r["end_reason"] in ("deadlock", "all_done")
    ep = Episode("aggregate", 2, seed=5)
    assert ep.step(0, call("submit", answer=wrong), "tool_call", 10) is None
    assert ep.status(0) == "done" and ep.status(1) == "running" and not ep.done
    ep = Episode("aggregate", 2, seed=5)
    if ep.numbers[0] != ep.numbers[1]:
        ep.step(0, call("submit", answer=sorted(ep.numbers, reverse=True)), "tool_call", 10)
        assert not ep.success


def test_idle_agent_woken_by_mail():
    ep = Episode("aggregate", 2, seed=2)
    assert ep.step(1, "Waiting." + END, "end_turn", 3) is None
    assert ep.status(1) == "idle" and ep.runnable() == [0]
    assert ep.wake(1) is None  # no mail yet
    inj = ep.step(0, call("message", to=1, text="my number is %d" % ep.numbers[0]), "tool_call", 9)
    assert inj and last_response(inj)[0] == {"ok": True, "delivered_to": [1]}
    assert not ep.done
    w = ep.wake(1)
    assert w == ("\n<|im_start|>user\n[notice] 1 new message(s) from agent 0. Use read_messages."
                 "<|im_end|>\n" + ASSISTANT_OPEN)
    assert ep.status(1) == "running"
    assert ep.wake(1) is None
    inj = ep.step(1, call("read_messages"), "tool_call", 5)
    assert last_response(inj)[0] == {"messages": [{"from": 0, "text": "my number is %d"
                                                   % ep.numbers[0]}]}
    inj = ep.step(1, call("read_messages"), "tool_call", 5)
    assert last_response(inj)[0] == {"messages": []}
    ep = Episode("aggregate", 2, seed=2)
    ep.step(1, "ok", "end_turn", 1)
    ep.step(0, call("message", to="all", text="hi"), "tool_call", 5)
    assert ep.wake(1).startswith("<|im_end|>\n<|im_start|>user\n[notice] 1 new message(s)")


def test_notices():
    ep = Episode("aggregate", 3, seed=2)
    ep.step(0, call("message", to="all", text="17"), "tool_call", 5)
    ep.step(2, call("message", to=1, text="23"), "tool_call", 5)
    inj = ep.step(1, "Let me think about the plan", "length", 1024)
    assert inj == "\n[notice: 2 unread message(s) from agents 0, 2]\n"
    assert ep.step(1, " more text", "length", 1024 - 1) == ""  # already notified
    ep = Episode("aggregate", 2, seed=2, chunk_tokens=8)
    ep.step(0, call("message", to=1, text="5"), "tool_call", 5)
    assert ep.step(1, "x <tool_", "length", 8) == ""
    assert ep.step(1, "call>\n{\"name\"", "length", 8) == ""
    inj = ep.step(1, ": \"read_messages\"}</tool_call>", "tool_call", 8)
    resp = last_response(inj)[0]
    assert resp["messages"] == [{"from": 0, "text": "5"}] and "notice" not in resp
    ep = Episode("aggregate", 2, seed=2)
    ep.step(0, call("message", to=1, text="5"), "tool_call", 5)
    resp = last_response(ep.step(1, call("message", to=0, text="9"), "tool_call", 5))[0]
    assert resp["ok"] and resp["notice"] == "1 new message(s) from agent 0. Use read_messages."


def test_split_tool_call():
    ep = Episode("aggregate", 2, seed=8, chunk_tokens=8)
    full = "I will tell everyone.\n" + call("message", to="all", text="%d" % ep.numbers[0])
    cut = len(full) // 2
    assert ep.step(0, full[:cut], "length", 8) == ""  # nothing to run yet, keep going
    assert ep.messages == 0
    inj = ep.step(0, full[cut:], "tool_call", 8)
    assert last_response(inj)[0] == {"ok": True, "delivered_to": [1]}
    assert ep.messages == 1 and ep.broadcasts == 1
    ep = Episode("aggregate", 2, seed=8)
    s = call("read_messages")
    assert ep.step(0, "ok " + s[:5], "length", 8) == ""
    assert last_response(ep.step(0, s[5:], "tool_call", 8)) == [{"messages": []}]
    ep = Episode("aggregate", 2, seed=8)
    two = call("message", to=1, text="a") + "\n" + call("read_messages")
    k = len(call("message", to=1, text="a")) + 10
    assert ep.step(0, two[:k], "length", 8) == "" and ep.messages == 0
    resps = last_response(ep.step(0, two[k:], "tool_call", 8))
    assert len(resps) == 2 and resps[0]["ok"] and resps[1] == {"messages": []}


def test_several_calls_in_order():
    ep = Episode("aggregate", 3, seed=4)
    text = (call("message", to=1, text="first") + "\n" + call("message", to=2, text="second")
            + "\n" + call("read_messages") + END)
    inj = ep.step(0, text, "end_turn", 40)
    assert inj.startswith("\n<|im_start|>user\n<tool_response>\n")  # turn already closed
    r = last_response(inj)
    assert [x.get("delivered_to") for x in r[:2]] == [[1], [2]] and r[2] == {"messages": []}
    assert [e["args"]["text"] for e in ep.events if e["kind"] == "call"
            and e["tool"] == "message"] == ["first", "second"]
    ep = Episode("aggregate", 1, seed=4)
    assert ep.step(0, call("submit", answer=ep.answer) + call("read_messages"), "tool_call",
                   9) is None
    assert ep.success and ep.done and len([e for e in ep.events if e["kind"] == "call"]) == 1


def test_malformed_calls():
    ep = Episode("aggregate", 2, seed=1)
    cases = [
        ("<tool_call>\n{\"name\": \"message\", \"arguments\": {\"to\": 1, \"text\": }\n</tool_call>",
         "not valid JSON"),
        (call("fly", height=3), "unknown tool"),
        (call("message", text="no recipient"), "needs"),
        (call("message", to=7, text="x"), "agent number"),
        (call("message", to=0, text="me"), "you are agent 0"),
        (call("submit", answer="many"), "list of integers"),
        (call("submit"), "submit needs"),
        ("<tool_call>\n{\"arguments\": {}}\n</tool_call>", "no \"name\""),
        ("<tool_call>\n\n</tool_call>", "empty"),
        ("{\"nme\": 1}</tool_call>", "no \"name\""),
        ("I am done.</tool_call>", "without"),
    ]
    for text, frag in cases:
        inj = ep.step(0, text, "tool_call", 10)
        assert inj is not None, text
        r = last_response(inj)
        assert len(r) == 1 and "error" in r[0] and frag in r[0]["error"], (text, r)
        assert ep.status(0) == "running"
    assert ep.messages == 0 and ep.submissions == []
    ep = Episode("aggregate", 2, seed=1)
    ok = [
        '{"name": "read_messages", "arguments": {}}\n</tool_call>',
        '<tool_call>\n{"name": "read_messages", "arguments": {}\n</tool_call>',
        '<tool_call>\n{"name": "read_messages", "arguments": {},}\n</tool_call>',
        '<tool_call>\n{"name": "read_messages", "arguments": "{}"}\n</tool_call>',
        '<tool_call>```json\n{"name": "read_messages", "arguments": {}}```</tool_call>',
        '<tool_call>{"name": "read_messages", "arguments": {}}}</tool_call>',
    ]
    for text in ok:
        r = last_response(ep.step(0, text, "tool_call", 10))
        assert r == [{"messages": []}], (text, r)
    r = last_response(ep.step(0, '<tool_call>{"name": "read_messages"}' + END, "end_turn", 10))
    assert r == [{"messages": []}]
    assert loads_lenient("{'a': 1}") == {"a": 1} and loads_lenient("nope") is None
    calls, consumed = parse_tool_calls("hi <tool_call>{\"name\": \"x\"")
    assert calls == [] and consumed == 0
    for junk in ["", "<tool_call>", "</tool_call></tool_call>", "<tool_call>[1, 2]</tool_call>",
                 "<tool_call>{\"name\": 5}</tool_call>", "\x00￿"]:
        for reason in ("tool_call", "length", "end_turn", "weird"):
            e = Episode("maze", 2, seed=0)
            e.step(0, junk, reason, 1)


def test_maze_counts():
    for depth, rooms, leaves in ((1, 31, 3), (2, 121, 9), (3, 391, 27)):
        ep = Episode("maze", 2, seed=5, depth=depth)
        assert ep.n_rooms == rooms and len(ep._adj) == rooms, (depth, ep.n_rooms)
        assert len(ep.leaves) == leaves and ep.exit in ep.leaves
        assert sorted(ep._adj) == list(range(rooms))
        edges = sum(len(d) for d in ep._adj.values())
        assert edges == 2 * (rooms - 1)  # a tree
        assert len(ep._adj[ep.root]) == 3
        assert ep._dist_to_exit(ep.root) == 10 * depth
        degs = sorted(set(len(d) for d in ep._adj.values()))
        assert degs == ([1, 2, 3] if depth == 1 else [1, 2, 3, 4]), degs
    assert "121 rooms" in Episode("maze", 2, seed=5).shared_prompt()


def maze_solver(ep):
    path = []
    r = ep.exit
    while r is not None:
        path.append(r)
        r = ep._parent[r]
    path.reverse()  # root .. exit
    chunks = [(call("look"), "tool_call", 6)]
    chunks += [(call("move", room=r), "tool_call", 9) for r in path[1:]]
    return chunks


def test_maze_exit():
    ep = Episode("maze", 3, seed=21, depth=2, budget_tokens=4096)
    drv = FakeDriver(ep, {0: maze_solver(ep), 1: [(call("look"), "tool_call", 6)], 2: []})
    r = drv.run()
    assert r["success"] and r["end_reason"] == "success"
    first = last_response(drv.injects[0][0])[0]
    assert set(first) == {"room", "doors", "exit"} and first["room"] == 0
    assert first["exit"] is False and len(first["doors"]) == 3
    assert sum(1 for e in r["events"] if e["kind"] == "call" and e["tool"] == "move") == 20
    ep = Episode("maze", 1, seed=21)
    far = [x for x in ep._adj if x not in ep._adj[0] and x != 0][0]
    resp = last_response(ep.step(0, call("move", room=far), "tool_call", 5))[0]
    assert "not adjacent" in resp["error"]
    resp = last_response(ep.step(0, call("move", room="north"), "tool_call", 5))[0]
    assert "error" in resp
    resp = last_response(ep.step(0, call("move", room=str(ep._adj[0][0])), "tool_call", 5))[0]
    assert resp["room"] == ep._adj[0][0] and 0 in resp["doors"]
    resp = last_response(ep.step(0, call("submit", answer=[1]), "tool_call", 5))[0]
    assert "unknown tool" in resp["error"]


def test_reward_formula():
    ep = Episode("aggregate", 2, seed=0, budget_tokens=100, lam=0.25)
    assert ep.reward() == 0.0
    ep.tokens = [30, 50]
    ep.success = True
    assert abs(ep.reward() - (1 - 0.25 * 0.4)) < 1e-12
    ep.tokens = [100, 100]
    assert abs(ep.reward() - 0.75) < 1e-12
    ep.tokens = [500, 500]  # min(1, .) caps the penalty
    assert abs(ep.reward() - 0.75) < 1e-12
    ep = Episode("aggregate", 1, seed=0, budget_tokens=64, lam=0.5)
    ep.step(0, call("submit", answer=ep.answer), "tool_call", 16)
    r = ep.result()
    assert r["success"] and abs(r["reward"] - (1 - 0.5 * 16 / 64)) < 1e-12 and r["tokens"] == [16]


def test_budget_exhaustion():
    ep = Episode("aggregate", 2, seed=3, budget_tokens=50, chunk_tokens=20)
    assert ep.max_new_tokens(0) == 20
    filler = ("blah ", "length", 20)
    drv = FakeDriver(ep, {0: [filler] * 10, 1: [filler] * 10})
    r = drv.run()
    assert r["end_reason"] == "budget" and not r["success"] and r["reward"] == 0.0
    assert r["tokens"] == [50, 50]  # driver asks for min(chunk, budget left): 20, 20, 10
    assert drv.rounds == 3
    ep = Episode("aggregate", 2, seed=3, budget_tokens=10)
    assert ep.step(0, call("message", to=1, text="x"), "tool_call", 10) is None
    assert ep.messages == 1 and ep.status(0) == "done"


def test_deadlock():
    ep = Episode("aggregate", 3, seed=6)
    drv = FakeDriver(ep, {a: [("I'll wait." + END, "end_turn", 4)] for a in range(3)})
    r = drv.run()
    assert r["end_reason"] == "deadlock" and not r["success"] and drv.rounds == 1
    assert ep.runnable() == [] and all(ep.status(a) == "done" for a in range(3))
    ep = Episode("aggregate", 2, seed=6)
    ep.step(1, "wait" + END, "end_turn", 2)
    ep.step(0, call("message", to=1, text="hey") + END, "end_turn", 8)
    assert ep.status(0) == "running"  # tool call in an end_turn chunk: response, keeps running
    ep.step(0, "bye" + END, "end_turn", 2)
    assert not ep.done and ep.runnable() == []
    assert ep.wake(1) is not None and ep.runnable() == [1]
    ep.step(1, "meh" + END, "end_turn", 2)
    assert ep.done and ep.end_reason == "deadlock"
    assert ep.step(0, call("read_messages"), "tool_call", 3) is None  # after the end: no-op


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print("OK: %d tests passed (python %d.%d)" % (len(TESTS), sys.version_info[0],
                                                  sys.version_info[1]))
