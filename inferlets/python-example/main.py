"""Simple text completion — Python inferlet example.

Demonstrates:
- Loading a model
- Building chat context with chat fillers
- Streaming generation with independent chat + reasoning decoders
- Match-case dispatch on decoder events
"""

from inferlet import Context, Sampler, chat, reasoning, session


async def main(input: dict) -> str:
    ctx = Context()
    ctx.system("You are a helpful assistant.")
    ctx.user("What is the capital of France? Tell me a joke.")

    chat_dec = chat.Decoder()
    think = reasoning.Decoder()
    output = ""

    g = ctx.generate(Sampler.top_p(0.6, 0.95), max_tokens=256)
    async for step in g:
        out = await step.execute()

        match think.feed(out.tokens):
            case reasoning.Event.Delta(text=t):
                session.send(t)
            case _:
                pass

        match chat_dec.feed(out.tokens):
            case chat.Event.Delta(text=t):
                session.send(t)
                output += t
            case chat.Event.Done(text=full):
                output = full
                break
            case _:
                pass

    session.send("\n[done]")
    return output
