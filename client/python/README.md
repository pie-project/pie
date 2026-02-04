# Pie Python Client

A Python client for interacting with the Pie server.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import asyncio
from pie import PieClient, ParsedPrivateKey

async def main():
    async with PieClient("ws://127.0.0.1:8080") as client:
        # Authentication is always required.
        # If server auth is enabled, provide a valid private key:
        key = ParsedPrivateKey.from_file("~/.ssh/id_ed25519")
        await client.authenticate("username", key)
        
        # If server auth is disabled, any username works without a key:
        # await client.authenticate("any_username")
        
        # Install and launch a program
        with open("my_program.wasm", "rb") as f:
            await client.install_program(f.read())
        
        program_hash = "..."  # blake3 hash of the wasm binary
        instance = await client.launch_instance(program_hash)
        
        # Interact with the instance
        await instance.send("hello")
        event, message = await instance.recv()
        print(f"Received: {event.name} - {message}")

asyncio.run(main())
```

## API Reference

### PieClient

| Method | Description |
|--------|-------------|
| `authenticate(username, private_key)` | Public key authentication (challenge-response) |
| `internal_authenticate(token)` | Token-based internal authentication |
| `install_program(wasm_path, manifest_path)` | Install a WASM program from file paths |
| `program_exists(inferlet, wasm_path, manifest_path)` | Check if program is uploaded with optional file-based hash verification |
| `launch_instance(hash, args, detached)` | Launch a program instance |
| `attach_instance(instance_id)` | Attach to a detached instance |
| `list_instances()` | List running instances |
| `terminate_instance(instance_id)` | Terminate an instance |
| `ping()` | Check server connectivity |
| `query(subject, record)` | Generic server query |

### Instance

| Method | Description |
|--------|-------------|
| `send(message)` | Send a string to the instance |
| `upload_blob(bytes)` | Upload binary data |
| `recv()` | Receive next event (returns `(Event, data)`) |
| `terminate()` | Request termination |

### Event Types

| Event | Description |
|-------|-------------|
| `Message` | Text message from instance |
| `Completed` | Instance finished successfully |
| `Aborted` | Instance was aborted |
| `Exception` | Instance raised an exception |
| `ServerError` | Server-side error |
| `OutOfResources` | Resource limit reached |
| `Blob` | Binary data received |
| `Stdout` | Streaming stdout output |
| `Stderr` | Streaming stderr output |

### ParsedPrivateKey

Supports RSA (≥2048 bits), ED25519, and ECDSA (P-256, P-384) keys.

```python
# From file
key = ParsedPrivateKey.from_file("~/.ssh/id_ed25519")

# From string
key = ParsedPrivateKey.parse(key_content)
```

## Example

See [main.py](./main.py) for a complete usage example.