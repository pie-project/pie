"""Concurrency test — run two generate() tasks concurrently with asyncio.gather.

If async concurrency works, we should see interleaved output from both
contexts. If not, one context will finish completely before the other starts.
"""
import asyncio

from inferlet import Model, Context, Sampler, Event, runtime, session, set_return

log: list[str] = []


async def generate(ctx: Context, label: str) -> None:
    msg = f"[{label}] START"
    log.append(msg)
    session.send(msg)

    step_count = 0
    async for event in await ctx.generate(
        Sampler.top_p(0.6, 0.95),
        max_tokens=20,
        decode=True,
        reasoning=True,
    ):
        step_count += 1
        match event:
            case Event.Thinking(text=t) | Event.Text(text=t):
                msg = f"[{label}] step={step_count} {t}"
                log.append(msg)
                session.send(msg)
            case Event.Done():
                break

    msg = f"[{label}] END"
    log.append(msg)
    session.send(msg)


async def main():
    model = Model.load(runtime.models()[0])

    # Two separate contexts with different prompts
    ctx1 = Context(model)
    ctx1.system("You are helpful.")
    ctx1.user("Count from 1 to 5.")

    ctx2 = Context(model)
    ctx2.system("You are helpful.")
    ctx2.user("Name 3 colors.")

    session.send("[test] starting asyncio.gather")
    await asyncio.gather(
        generate(ctx1, "CTX1"),
        generate(ctx2, "CTX2"),
    )
    session.send("[test] asyncio.gather complete")

    # Analyze results: check if interleaving happened
    labels = ["1" if l.startswith("[CTX1]") else "2" if l.startswith("[CTX2]") else "_" for l in log]
    session.send(f"[test] order: {''.join(labels)}")

    switches = 0
    last = ""
    for l in labels:
        if l != "_" and l != last:
            switches += 1
            last = l
    session.send(f"[test] context switches: {switches}")
    session.send(f"[test] verdict: {'CONCURRENT' if switches > 2 else 'SEQUENTIAL'}")

    set_return("done")
