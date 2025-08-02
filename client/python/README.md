# PIE Python Client

This package provides the Python client for interacting with the PIE. It is required for connecting to and communicating with the PIE engine from any Python script, allowing you to launch and manage inferlets programmatically.

## Installation

To install the client, 

```bash
pip install -e .
````

## Quick Start

Here is a basic example of how to use the client to connect to a running PIE instance, upload a compiled inferlet, launch it, and receive its output.

```python
import asyncio
from pathlib import Path
from pie import PieClient

async def main():
    # Connect to the PIE engine (assumes it's running on localhost)
    async with PieClient("ws://localhost:8080") as client:
        print("Successfully connected to the PIE engine! ✅")

        # 1. Load your compiled inferlet (WASM file)
        # Make sure you have compiled the examples first.
        program_path = Path("../example-apps/target/wasm32-wasip2/release/text_completion.wasm")
        with open(program_path, "rb") as f:
            program_bytes = f.read()
        program_hash = blake3(program_bytes).hexdigest()

        if not await client.program_exists(program_hash):
            print("Program not found on server, uploading...")
            await client.upload_program(program_bytes)
            print("Upload complete.")

        # 3. Launch an instance of the inferlet with arguments
        print("Launching inferlet instance...")
        instance_args = ["--prompt", "What is the capital of France?", "--max-tokens", "16"]
        instance = await client.launch_instance(program_hash, arguments=instance_args)

        # 4. Receive output from the instance
        while True:
            event, message = await instance.recv()
            if event == "terminated":
                print(f"\n\nInstance finished. Reason: {message}")
                break
            else:
                print(message, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```