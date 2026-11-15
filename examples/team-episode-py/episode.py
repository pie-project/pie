import json
import re
import random
import math

STOP = ("</tool_call>", "<|im_end|>")  # drivers stop generation on either (include it in the text)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"  # thinking off (Qwen3 enable_thinking=False)
TASKS = ("aggregate", "maze")

MAZE_BRANCHING = 3
MAZE_CORRIDOR = 10
MAX_MESSAGE_CHARS = 1000


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n):
    n = int(n)
    if n < 0 or n > 99:
        raise ValueError("number_words supports 0-99")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _tojson(x):
    return json.dumps(x, ensure_ascii=False)


def render_system(content, tools):
    out = IM_START + "system\n"
    if content:
        out += content + "\n\n"
    out += ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
    for tool in tools:
        out += "\n" + _tojson(tool)
    out += ("\n</tools>\n\nFor each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            + IM_END + "\n")
    return out


def render_user(content):
    return IM_START + "user\n" + content + IM_END + "\n" + ASSISTANT_OPEN


def render_tool_responses(payloads):
    out = IM_START + "user"
    for p in payloads:
        out += "\n<tool_response>\n" + _tojson(p) + "\n</tool_response>"
    return out + IM_END + "\n" + ASSISTANT_OPEN


def _tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": list(required or [])}}}


TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _raw_decode(s):
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[min(starts):])
        return obj
    except ValueError:
        return None


def loads_lenient(s):
    s = _FENCE_RE.sub("", s.strip())
    variants = [s, _TRAILING_COMMA_RE.sub(r"\1", s)]
    if "'" in s and '"' not in s:
        variants.append(s.replace("'", '"'))
    for v in list(variants):
        missing = v.count("{") - v.count("}")
        if 0 < missing <= 4:
            variants.append(v.rstrip() + "}" * missing)
    for v in variants:
        obj = _raw_decode(v)
        if obj is not None:
            return obj
    return None


def _parse_call_body(body):
    raw = body.strip()
    if raw.endswith(IM_END):
        raw = raw[: -len(IM_END)].rstrip()
    if not raw:
        return [{"error": "empty tool call"}]
    obj = loads_lenient(raw)
    if obj is None:
        return [{"error": "tool call is not valid JSON; use {\"name\": <tool>, \"arguments\": {...}}",
                 "raw": raw[:200]}]
    items = obj if isinstance(obj, list) else [obj]
    out = []
    for it in items:
        if not isinstance(it, dict):
            out.append({"error": "tool call must be a JSON object", "raw": raw[:200]})
            continue
        fn = it.get("function") if isinstance(it.get("function"), dict) else it
        name = fn.get("name")
        args = None
        for key in ("arguments", "parameters", "args"):
            if key in fn:
                args = fn[key]
                break
        if isinstance(args, str):
            args = {} if not args.strip() else loads_lenient(args)
            if args is None:
                out.append({"error": "arguments are not valid JSON", "name": name})
                continue
        if args is None:
            args = {}
        if not isinstance(name, str) or not name.strip():
            out.append({"error": "tool call has no \"name\"", "raw": raw[:200]})
        elif not isinstance(args, dict):
            out.append({"error": "\"arguments\" must be a JSON object", "name": name})
        else:
            out.append({"name": name.strip(), "arguments": args})
    return out


def parse_tool_calls(text, final=False):
    calls = []
    consumed = 0
    for m in _CALL_RE.finditer(text):
        calls.extend(_parse_call_body(m.group(1)))
        consumed = m.end()
    rest = text[consumed:]
    i = rest.find(TOOL_OPEN)
    if i >= 0:
        if final:
            calls.extend(_parse_call_body(rest[i + len(TOOL_OPEN):]))
            consumed = len(text)
    elif TOOL_CLOSE in rest:
        j = rest.find(TOOL_CLOSE)
        body = rest[:j]
        k = body.find("{")
        if k >= 0 and loads_lenient(body[k:]) is not None:
            calls.extend(_parse_call_body(body[k:]))
        else:
            calls.append({"error": "found </tool_call> without a <tool_call>{...} before it"})
        consumed += j + len(TOOL_CLOSE)
    return calls, consumed


def _open_call_pending(text):
    i = text.rfind(TOOL_OPEN)
    if i >= 0 and text.find(TOOL_CLOSE, i) < 0:
        return True
    for k in range(1, len(TOOL_OPEN)):
        if text.endswith(TOOL_OPEN[:k]):
            return True
    return False


def _as_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str) and re.match(r"^\s*-?\d+\s*$", v):
        return int(v)
    return None


def _senders(msgs):
    ids = sorted(set(m["from"] for m in msgs))
    if len(ids) == 1:
        return "agent %d" % ids[0]
    return "agents " + ", ".join(str(i) for i in ids)


class Episode:
    def __init__(self, task, n_agents, seed, *, budget_tokens=2048, chunk_tokens=1024, depth=2,
                 lam=0.25):
        if task not in TASKS:
            raise ValueError("task must be one of %r" % (TASKS,))
        if int(n_agents) < 1:
            raise ValueError("n_agents must be >= 1")
        self.task = task
        self.n_agents = int(n_agents)
        self.seed = seed
        self.budget_tokens = int(budget_tokens)
        self.chunk_tokens = int(chunk_tokens)
        self.depth = int(depth)
        self.lam = float(lam)
        self._rng = random.Random(seed)
        n = self.n_agents
        self.tokens = [0] * n
        self._status = ["running"] * n
        self._done_reason = [None] * n      # "submitted" | "budget"
        self._buf = [""] * n                # text of the current assistant turn not yet executed
        self._turn_closed = [False] * n     # last text ended with <|im_end|>
        self._inbox = [[] for _ in range(n)]
        self._read = [0] * n                # inbox index read via read_messages
        self._seen = [0] * n                # inbox index the agent has been notified about
        self.success = False
        self.end_reason = None
        self.messages = 0
        self.broadcasts = 0
        self.submissions = []
        self.events = []
        self._t = 0
        if task == "aggregate":
            self.numbers = [self._rng.randint(1, 99) for _ in range(n)]
            self.answer = sorted(self.numbers)
            self._event(None, "start", numbers=list(self.numbers), answer=list(self.answer))
        else:
            if self.depth < 1:
                raise ValueError("depth must be >= 1")
            self._build_maze()
            self._pos = [self.root] * n
            self._event(None, "start", rooms=self.n_rooms, root=self.root, exit=self.exit,
                        exit_distance=self._dist_to_exit(self.root))
        self._tools = self._make_tools()
        self._tool_names = [t["function"]["name"] for t in self._tools]
        self._shared = render_system(self._system_text(), self._tools)

    def _build_maze(self):
        rng = self._rng
        n_rooms = 1 + MAZE_CORRIDOR * sum(MAZE_BRANCHING ** k for k in range(1, self.depth + 1))
        ids = list(range(1, n_rooms))
        rng.shuffle(ids)
        ids.reverse()  # pop() from the end yields the shuffled order
        self.root = 0
        adj = {0: []}
        parent = {0: None}
        leaves = []
        stack = [(0, 1)]  # (junction room, corridor level)
        while stack:
            junction, level = stack.pop(0)
            for _ in range(MAZE_BRANCHING):
                prev = junction
                for _ in range(MAZE_CORRIDOR):
                    r = ids.pop()
                    adj[r] = [prev]
                    adj[prev].append(r)
                    parent[r] = prev
                    prev = r
                if level < self.depth:
                    stack.append((prev, level + 1))
                else:
                    leaves.append(prev)
        assert not ids and len(adj) == n_rooms
        self.exit = rng.choice(leaves)
        for r in range(n_rooms):
            rng.shuffle(adj[r])  # door order fixed per seed, shared by every agent
        self.n_rooms = n_rooms
        self.leaves = leaves
        self._adj = adj
        self._parent = parent

    def _dist_to_exit(self, room):
        path = []
        r = self.exit
        while r is not None:
            path.append(r)
            r = self._parent[r]
        anc = {x: i for i, x in enumerate(path)}
        d = 0
        r = room
        while r not in anc:
            r = self._parent[r]
            d += 1
        return d + anc[r]

    def _look(self, a):
        r = self._pos[a]
        return {"room": r, "doors": list(self._adj[r]), "exit": r == self.exit}

    def _make_tools(self):
        n = self.n_agents
        tools = [
            _tool("message",
                  "Send a text message to one teammate (to = their agent number) or to every "
                  "teammate (to = \"all\"): plans, questions, findings, dead ends, checked "
                  "results. This is the only way to reach a teammate. Delivery is immediate; "
                  "sending does not read your own messages or wait for replies.",
                  {"to": {"type": ["integer", "string"],
                          "description": "agent number 0-%d, or \"all\"" % (n - 1)},
                   "text": {"type": "string", "description": "the message"}},
                  ["to", "text"]),
            _tool("read_messages",
                  "Read the messages teammates have sent you since you last read them. Use it "
                  "when you are notified of new messages."),
        ]
        if self.task == "aggregate":
            tools.append(_tool(
                "submit",
                "Submit the team's answer: all %d integers as digits, sorted ascending, duplicates "
                "kept. A submission is final and ends your work." % n,
                {"answer": {"type": "array", "items": {"type": "integer"},
                            "description": "the sorted list of all %d integers" % n}},
                ["answer"]))
        else:
            tools.append(_tool(
                "look",
                "Describe your current room without moving: its number, its doors (the numbers "
                "of the adjacent rooms, including the way back) and whether it is the exit."))
            tools.append(_tool(
                "move",
                "Walk through a door of your current room into the adjacent room with that "
                "number. Returns what look returns there. Your position is private: moving never "
                "moves a teammate.",
                {"room": {"type": "integer", "description": "an adjacent room number from doors"}},
                ["room"]))
        return tools

    def _system_text(self):
        n = self.n_agents
        team = (
            "You are one of %d identical agents (agent 0 to agent %d) working at the same time as "
            "a team. There is no leader. The whole team shares one reward, so act for the team. "
            "You cannot see your teammates' work; the only way to reach a teammate is the message "
            "tool, and messages sent to you are read with read_messages. Be brief: every token "
            "you write counts against a budget of %d tokens per agent, and the fewer tokens the "
            "team uses, the larger the reward."
            % (n, n - 1, self.budget_tokens))
        if self.task == "aggregate":
            task = (
                "Task: each agent privately holds one integer from 1 to 99, written in English "
                "words. The team must find all %d integers. Any agent may call submit with the "
                "full list of the %d integers as digits, sorted ascending, duplicates kept (a team "
                "of four might submit [3, 15, 15, 88]). If that list is correct, the "
                "whole team succeeds. A submission is final and ends your work, so collect every "
                "teammate's number first. Tip: send your number to everyone with "
                "message(to=\"all\"), then read_messages until you know all %d numbers."
                % (n, n, n))
        else:
            task = (
                "Task: the team explores a maze of %d rooms shaped like a tree, and everyone "
                "starts in room %d. From the start, %d corridors lead away. Every corridor is %d "
                "rooms long and ends either at a junction where %d new corridors branch off, or at "
                "a dead end. One dead end is the exit. Call look to see your room, its doors and "
                "whether it is the exit; call move with an adjacent room number to walk there. As "
                "soon as any agent reaches the exit, the whole team succeeds. Split up: tell your "
                "teammates which corridors you take and which ones are dead ends, so nobody "
                "searches the same place twice."
                % (self.n_rooms, self.root, MAZE_BRANCHING, MAZE_CORRIDOR, MAZE_BRANCHING))
        return team + "\n\n" + task

    def shared_prompt(self):
        return self._shared

    def private_prompt(self, agent):
        self._check_agent(agent)
        if self.task == "aggregate":
            body = "You are agent %d. Your private number is %s." % (
                agent, number_words(self.numbers[agent]))
        else:
            body = "You are agent %d. You are in room %d, the start." % (agent, self.root)
        return render_user(body)

    def prompt(self, agent):
        return self.shared_prompt() + self.private_prompt(agent)

    @property
    def tools(self):
        return [dict(t) for t in self._tools]

    def _check_agent(self, agent):
        if not isinstance(agent, int) or isinstance(agent, bool) or not 0 <= agent < self.n_agents:
            raise IndexError("agent must be an int in [0, %d)" % self.n_agents)

    def _event(self, agent, kind, **kw):
        ev = {"t": self._t, "agent": agent, "kind": kind}
        ev.update(kw)
        self.events.append(ev)

    def _new_mail(self, agent):
        return self._inbox[agent][self._seen[agent]:]

    def _end(self, reason):
        if self.end_reason is None:
            self.end_reason = reason
            self._event(None, "end", reason=reason, success=self.success)

    def _check_end(self):
        if self.end_reason is not None:
            return True
        if self.success:
            self._end("success")
        elif all(s == "done" for s in self._status):
            if all(r == "budget" for r in self._done_reason):
                self._end("budget")
            else:
                self._end("all_done")
        elif ("running" not in self._status
              and not any(self._status[a] == "idle" and self._new_mail(a)
                          for a in range(self.n_agents))):
            self._end("deadlock")
        return self.end_reason is not None

    @property
    def done(self):
        return self._check_end()

    def status(self, agent):
        self._check_agent(agent)
        if self.done:
            return "done"
        return self._status[agent]

    def runnable(self):
        if self.done:
            return []
        return [a for a in range(self.n_agents) if self._status[a] == "running"]

    def budget_left(self, agent):
        self._check_agent(agent)
        return max(0, self.budget_tokens - self.tokens[agent])

    def max_new_tokens(self, agent):
        return min(self.chunk_tokens, self.budget_left(agent))

    def _finish_agent(self, agent, reason):
        if self._status[agent] != "done":
            self._status[agent] = "done"
            self._done_reason[agent] = reason
            self._event(agent, "done", reason=reason)

    def step(self, agent, text, reason, n_tokens):
        self._check_agent(agent)
        self._t += 1
        text = text or ""
        self.tokens[agent] += int(n_tokens or 0)
        if self.done or self._status[agent] != "running":
            return None
        if reason not in ("tool_call", "end_turn", "length"):
            reason = "end_turn" if text.endswith(IM_END) else (
                "tool_call" if text.endswith(TOOL_CLOSE) else "length")
        buf = self._buf[agent] + text
        final = reason == "end_turn"
        calls, consumed = parse_tool_calls(buf, final=final)
        out_of_budget = self.tokens[agent] >= self.budget_tokens
        if calls and reason == "length" and not out_of_budget and _open_call_pending(buf[consumed:]):
            calls = []  # another call is still being written: run them all once it closes

        if calls:
            closed = buf.rstrip().endswith(IM_END)
            self._buf[agent] = ""
            responses = []
            for call in calls:
                if self._status[agent] != "running" or self.done:
                    break
                responses.append(self._execute(agent, call))
            if self.done or self._status[agent] != "running":
                return None
            if out_of_budget:
                self._finish_agent(agent, "budget")
                self._check_end()
                return None
            mail = self._new_mail(agent)
            if mail and responses:
                self._seen[agent] = len(self._inbox[agent])
                responses[-1] = dict(responses[-1])
                responses[-1]["notice"] = "%d new message(s) from %s. Use read_messages." % (
                    len(mail), _senders(mail))
            return ("\n" if closed else IM_END + "\n") + render_tool_responses(responses)

        if out_of_budget:
            self._buf[agent] = ""
            self._finish_agent(agent, "budget")
            self._check_end()
            return None
        if reason == "end_turn":
            self._buf[agent] = ""
            self._turn_closed[agent] = buf.rstrip().endswith(IM_END)
            self._status[agent] = "idle"
            self._event(agent, "idle")
            self._check_end()
            return None
        self._buf[agent] = buf
        mail = self._new_mail(agent)
        if mail and not _open_call_pending(buf):
            self._seen[agent] = len(self._inbox[agent])
            self._event(agent, "notice", inline=True, count=len(mail))
            return "\n[notice: %d unread message(s) from %s]\n" % (len(mail), _senders(mail))
        return ""

    def wake(self, agent):
        self._check_agent(agent)
        if self.done or self._status[agent] != "idle":
            return None
        mail = self._new_mail(agent)
        if not mail:
            return None
        self._t += 1
        self._seen[agent] = len(self._inbox[agent])
        self._status[agent] = "running"
        self._event(agent, "wake", count=len(mail))
        prefix = "\n" if self._turn_closed[agent] else IM_END + "\n"
        return prefix + render_user("[notice] %d new message(s) from %s. Use read_messages." % (
            len(mail), _senders(mail)))

    def _execute(self, agent, call):
        if "error" in call:
            resp = {"error": call["error"]}
            self._event(agent, "call", tool=call.get("name"), args=None, error=call["error"])
            return resp
        name, args = call["name"], call["arguments"]
        if name not in self._tool_names:
            resp = {"error": "unknown tool %r; available tools: %s"
                             % (name, ", ".join(self._tool_names))}
        else:
            try:
                resp = getattr(self, "_tool_" + name)(agent, args)
            except Exception as e:  # never raise into the driver
                resp = {"error": "%s failed: %s" % (name, e)}
        ev = {"tool": name, "args": args}
        if "error" in resp:
            ev["error"] = resp["error"]
        self._event(agent, "call", **ev)
        return resp

    def _tool_message(self, agent, args):
        if "to" not in args or "text" not in args:
            return {"error": "message needs \"to\" (agent number or \"all\") and \"text\""}
        to, text = args["to"], args["text"]
        if not isinstance(text, str):
            text = _tojson(text)
        if not text.strip():
            return {"error": "\"text\" is empty"}
        truncated = len(text) > MAX_MESSAGE_CHARS
        text = text[:MAX_MESSAGE_CHARS]
        if isinstance(to, str) and to.strip().lower() in ("all", "everyone", "everybody", "*"):
            targets = [b for b in range(self.n_agents) if b != agent]
            to_field = "all"
        else:
            b = _as_int(to)
            if b is None and isinstance(to, str):
                m = re.match(r"^\s*agent\s*(\d+)\s*$", to, re.IGNORECASE)
                b = int(m.group(1)) if m else None
            if b is None or not 0 <= b < self.n_agents:
                return {"error": "\"to\" must be an agent number 0-%d or \"all\", got %s"
                                 % (self.n_agents - 1, _tojson(to))}
            if b == agent:
                return {"error": "you are agent %d; message a teammate or \"all\"" % agent}
            targets = [b]
            to_field = b
        for b in targets:
            self._inbox[b].append({"from": agent, "to": to_field, "text": text})
        self.messages += 1
        if to_field == "all":
            self.broadcasts += 1
        resp = {"ok": True, "delivered_to": targets}
        if truncated:
            resp["note"] = "text truncated to %d characters" % MAX_MESSAGE_CHARS
        return resp

    def _tool_read_messages(self, agent, args):
        box = self._inbox[agent]
        new = box[self._read[agent]:]
        self._read[agent] = len(box)
        self._seen[agent] = max(self._seen[agent], len(box))
        return {"messages": [{"from": m["from"], "text": m["text"]} if m["to"] != "all"
                             else {"from": m["from"], "to": "all", "text": m["text"]}
                             for m in new]}

    def _tool_submit(self, agent, args):
        if "answer" not in args:
            return {"error": "submit needs \"answer\": the sorted list of all %d integers"
                             % self.n_agents}
        ans = args["answer"]
        if isinstance(ans, str):
            parsed = loads_lenient(ans)
            ans = parsed if isinstance(parsed, list) else [p for p in re.split(r"[\s,]+", ans) if p]
        if isinstance(ans, (int, float)) and not isinstance(ans, bool):
            ans = [ans]
        if not isinstance(ans, list):
            return {"error": "\"answer\" must be a list of integers"}
        vals = [_as_int(v) for v in ans]
        if any(v is None for v in vals):
            return {"error": "\"answer\" must be a list of integers (digits), got %s"
                             % _tojson(ans)[:200]}
        correct = vals == self.answer
        self.submissions.append({"agent": agent, "answer": vals, "correct": correct})
        self._finish_agent(agent, "submitted")
        if correct:
            self.success = True
        self._check_end()
        return {"ok": True, "submitted": vals}

    def _tool_look(self, agent, args):
        return self._look(agent)

    def _tool_move(self, agent, args):
        if "room" not in args:
            return {"error": "move needs \"room\": an adjacent room number"}
        r = _as_int(args["room"])
        here = self._pos[agent]
        if r is None or r not in self._adj[here]:
            return {"error": "room %s is not adjacent to room %d; doors: %s"
                             % (_tojson(args["room"]), here, _tojson(self._adj[here]))}
        self._pos[agent] = r
        if r == self.exit:
            self.success = True
            self._check_end()
        return self._look(agent)

    def reward(self):
        s = 1.0 if self.success else 0.0
        frac = sum(self.tokens) / float(self.n_agents * self.budget_tokens)
        return s * (1.0 - self.lam * min(1.0, frac))

    def result(self):
        self._check_end()
        return {
            "task": self.task,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "success": bool(self.success),
            "reward": self.reward(),
            "tokens": list(self.tokens),
            "messages": self.messages,
            "broadcasts": self.broadcasts,
            "submissions": [dict(s) for s in self.submissions],
            "events": [dict(e) for e in self.events],
            "end_reason": self.end_reason,
        }
