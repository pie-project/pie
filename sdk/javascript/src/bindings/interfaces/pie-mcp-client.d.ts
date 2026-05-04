/** @module Interface pie:mcp/client **/
/**
 * Names of MCP servers the host has registered for this session.
 */
export function availableServers(): Array<string>;
/**
 * Open a session to a registered MCP server.
 * 
 * The MCP `initialize` handshake is performed by the host when it
 * registers the server, not here. This call validates that
 * `server-name` is registered and returns a typed handle.
 */
export function connect(serverName: string): Session;
export type Error = import('./pie-mcp-types.js').Error;
export type Json = import('./pie-mcp-types.js').Json;

export class Session {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * List tools exposed by this server.
  * Returns the raw `tools/list` JSON-RPC `result` field.
  */
  listTools(): Json;
  /**
  * Call a tool by name with JSON-encoded arguments.
  * Returns the raw `tools/call` JSON-RPC `result` field on
  * transport success, including any `isError` / `content` /
  * `structuredContent` fields the server returned. Tool-level
  * failures should be detected by inspecting `isError` in the
  * returned JSON.
  */
  callTool(name: string, args: Json): Json;
  /**
  * List resources exposed by this server.
  */
  listResources(): Json;
  /**
  * Read a resource by URI.
  * Returns the raw `resources/read` JSON-RPC `result` field.
  */
  readResource(uri: string): Json;
  /**
  * List prompts exposed by this server.
  */
  listPrompts(): Json;
  /**
  * Render a prompt template with the given arguments.
  */
  getPrompt(name: string, args: Json): Json;
}
