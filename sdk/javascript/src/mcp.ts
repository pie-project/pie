// MCP client wrapper — wraps pie:mcp/client WIT interface.

import * as _mcp from 'pie:mcp/client';
import type { Session as _Session } from 'pie:mcp/client';
import type { Content } from 'pie:mcp/types';

/** Discover available MCP servers. */
export function availableServers(): string[] {
    return _mcp.availableServers();
}

/** Connect to a named MCP server, performing the MCP handshake. */
export function connect(serverName: string): McpSession {
    return new McpSession(_mcp.connect(serverName));
}

/**
 * An active connection to an MCP server.
 *
 * Wraps the `pie:mcp/client.Session` WIT resource.
 */
export class McpSession {
    /** @internal */
    readonly _handle: _Session;

    /** @internal */
    constructor(handle: _Session) {
        this._handle = handle;
    }

    /** List tools exposed by this server (returns JSON). */
    listTools(): string {
        return this._handle.listTools();
    }

    /** Call a tool by name with JSON arguments. */
    callTool(name: string, args: string): Content[] {
        return this._handle.callTool(name, args);
    }

    /** List resources exposed by this server (returns JSON). */
    listResources(): string {
        return this._handle.listResources();
    }

    /** Read a resource by URI. */
    readResource(uri: string): Content[] {
        return this._handle.readResource(uri);
    }

    /** List prompts exposed by this server (returns JSON). */
    listPrompts(): string {
        return this._handle.listPrompts();
    }

    /** Render a prompt template with the given arguments (returns JSON). */
    getPrompt(name: string, args: string): string {
        return this._handle.getPrompt(name, args);
    }
}

/** Re-export the Content type for external use. */
export type { Content };
