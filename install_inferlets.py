import asyncio
from pie_client import PieClient

async def main():
    async with PieClient("ws://127.0.0.1:8080") as client:
        await client.authenticate("local-dev")
        await client.install_program(
            "/home/dhruv/pie/build-out/self_consistency.wasm",
            "/home/dhruv/pie/inferlets/self-consistency/Pie.toml",
            force_overwrite=True,
        )
        print("self-consistency installed")
        await client.install_program(
            "/home/dhruv/pie/build-out/graph_of_thought.wasm",
            "/home/dhruv/pie/inferlets/graph-of-thought/Pie.toml",
            force_overwrite=True,
        )
        print("graph-of-thought installed")

asyncio.run(main())
