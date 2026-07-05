"""Concurrency test — run two generate() tasks concurrently with asyncio.gather.

If async concurrency works, we should see interleaved output from both
contexts. If not, one context will finish completely before the other starts.
"""
import asyncio

from inferlet import (
    Context,
    Sampler,
    chat,
    reasoning,
    session,
    set_return,
)

log: list[str] = []


async def generate(ctx: Context, label: str) -> None:
    msg = f"[{label}] START"
    log.append(msg)
    session.send(msg)

    chat_dec = chat.Decoder()
    think = reasoning.Decoder()
    step_count = 0

    g = ctx.generate(Sampler.top_p(0.6, 0.95), max_tokens=20)
    async for step in g:
        step_count += 1
        out = await step.execute()

        match think.feed(out.tokens):
            case reasoning.Event.Delta(text=t):
                m = f"[{label}] step={step_count} {t}"
                log.append(m)
                session.send(m)
            case _:
                pass

        match chat_dec.feed(out.tokens):
            case chat.Event.Delta(text=t):
                m = f"[{label}] step={step_count} {t}"
                log.append(m)
                session.send(m)
            case chat.Event.Done(_):
                break
            case _:
                pass

    msg = f"[{label}] END"
    log.append(msg)
    session.send(msg)


async def main(input: dict) -> None:
    ctx1 = Context()
    ctx1.system("You are helpful.").user("Count from 1 to 5.")

    ctx2 = Context()
    ctx2.system("You are helpful.").user("Name 3 colors.")

    session.send("[test] starting asyncio.gather")
    await asyncio.gather(
        generate(ctx1, "CTX1"),
        generate(ctx2, "CTX2"),
    )
    session.send("[test] asyncio.gather complete")

    labels = [
        "1" if l.startswith("[CTX1]") else "2" if l.startswith("[CTX2]") else "_"
        for l in log
    ]
    session.send(f"[test] order: {''.join(labels)}")

    switches = 0
    last = ""
    for l in labels:
        if l != "_" and l != last:
            switches += 1
            last = l
    session.send(f"[test] context switches: {switches}")
    session.send(
        f"[test] verdict: {'CONCURRENT' if switches > 2 else 'SEQUENTIAL'}"
    )

    set_return("done")
