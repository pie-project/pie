"""Simple text completion — Python inferlet example.

Demonstrates:
- Loading a model
- Using Context for chat-style prompt building
- Streaming generation with EventStream
"""

from inferlet import Model, Context, Sampler, Event, runtime, session


async def main(args: list[str]) -> str:
    # Load model
    model = Model.load(runtime.models()[0])

    # Build context
    ctx = Context(model)
    ctx.system("You are a helpful assistant.")
    ctx.user("Hello, world! Tell me a joke.")

    # Stream the response
    output = ""
    async for event in await ctx.generate(
        Sampler.top_p(0.6, 0.95),
        max_tokens=256,
        decode=True,
    ):
        match event:
            case Event.Thinking(text=t):
                session.send(t)
            case Event.Text(text=t):
                session.send(t)
                output += t
            case Event.Done():
                break

    session.send("\n[done]")
    return output
