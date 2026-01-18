# HTTP Server Inferlet Example

This example demonstrates how to create an **HTTP server inferlet** using the `wasi:http/incoming-handler` interface. Unlike regular inferlets that use the `inferlet:core/run` interface and run to completion, server inferlets handle incoming HTTP requests.

## Key Differences from Regular Inferlets

| Aspect | Regular Inferlet | Server Inferlet |
|--------|------------------|-----------------|
| Interface | `inferlet:core/run` | `wasi:http/incoming-handler` |
| Entry Point | `#[inferlet::main]` | `#[wstd::http_server]` |
| Lifecycle | Runs once to completion | Handles each HTTP request |
| Instance | Single instance per launch | Fresh instance per request |

## Endpoints

This example provides the following endpoints:

- **`/`** - Home page with a greeting and list of available endpoints
- **`/wait`** - Demonstrates async sleep (sleeps for 1 second and reports elapsed time)
- **`/echo`** - Echoes back the request body (useful for testing POST requests)
- **`/echo-headers`** - Echoes back request headers as response headers
- **`/info`** - Returns server and request information as JSON

## Building

```bash
cd sdk/examples
cargo build -p http-server --target wasm32-wasip2 --release
```

## Running

Launch the server inferlet on a specific port:

```bash
pie http --path ./target/wasm32-wasip2/release/http_server.wasm --port 8080
```

## Testing

Once the server is running, you can test the endpoints:

```bash
# Home page
curl http://localhost:8080/

# Async sleep demo
curl http://localhost:8080/wait

# Echo back POST body
curl -X POST -d "Hello, World!" http://localhost:8080/echo

# Echo headers
curl -v http://localhost:8080/echo-headers

# Get request info as JSON
curl http://localhost:8080/info?foo=bar
```

## Code Structure

The example uses the `wstd` crate's `#[wstd::http_server]` macro, which:

1. Sets up the `wasi:http/incoming-handler` export
2. Converts the raw WASI HTTP types to ergonomic `Request` and `Responder` types
3. Handles the async execution of your handler function

### Key Components

- **`Request<IncomingBody>`** - The incoming HTTP request with its body
- **`Responder`** - Used to send the HTTP response
- **`Finished`** - Return type indicating the response has been completed
- **`Response`** - Builder for constructing HTTP responses
- **`BodyForthcoming`** - Used for streaming response bodies

## Dependencies

This example only requires:

```toml
[dependencies]
wstd = "0.5.6"
```
