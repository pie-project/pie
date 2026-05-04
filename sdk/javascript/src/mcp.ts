// MCP client wrapper — wraps pie:mcp/client WIT interface.
//
// All response payloads are returned as raw JSON strings — the WIT contract
// stays stable as MCP evolves; parse the JSON in your inferlet using
// whatever shape you need.

import * as _mcp from 'pie:mcp/client';
import type { Session as _Session } from 'pie:mcp/client';

/** Discover available MCP servers. */
export function availableServers(): string[] {
    return _mcp.availableServers();
}

/**
 * Open a session to a registered MCP server.
 * The MCP `initialize` handshake is performed by the host at registration
 * time; this is just a typed-handle constructor.
 */
export function connect(serverName: string): McpSession {
    return new McpSession(_mcp.connect(serverName));
}

/**
 * An active connection to an MCP server.
 *
 * All methods return the raw JSON-RPC `result` field as a string. Parse
 * with `JSON.parse(...)` — particularly to inspect `isError` / `content` /
 * `structuredContent` on a `callTool` response.
 *
 * @example
 * ```ts
 * const session = mcp.connect("my-mcp-server");
 * const tools = JSON.parse(session.listTools()).tools;
 * const result = JSON.parse(session.callTool("search", '{"query": "hi"}'));
 * if (result.isError) { ... }
 * ```
 */
export class McpSession {
    /** @internal */
    readonly _handle: _Session;

    /** @internal */
    constructor(handle: _Session) {
        this._handle = handle;
    }

    /** Raw `tools/list` JSON-RPC result. */
    listTools(): string {
        return this._handle.listTools();
    }

    /** Raw `tools/call` JSON-RPC result. Includes `isError` / `content`. */
    callTool(name: string, args: string): string {
        return this._handle.callTool(name, args);
    }

    /** Raw `resources/list` JSON-RPC result. */
    listResources(): string {
        return this._handle.listResources();
    }

    /** Raw `resources/read` JSON-RPC result. */
    readResource(uri: string): string {
        return this._handle.readResource(uri);
    }

    /** Raw `prompts/list` JSON-RPC result. */
    listPrompts(): string {
        return this._handle.listPrompts();
    }

    /** Raw `prompts/get` JSON-RPC result. */
    getPrompt(name: string, args: string): string {
        return this._handle.getPrompt(name, args);
    }
}
