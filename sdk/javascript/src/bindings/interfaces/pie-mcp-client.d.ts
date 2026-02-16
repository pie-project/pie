/** @module Interface pie:mcp/client **/
/**
 * Discover available MCP servers.
 */
export function availableServers(): Array<string>;
/**
 * Connect to a named MCP server, performing the MCP handshake.
 */
export function connect(serverName: string): Session;
export type Error = import('./pie-mcp-types.js').Error;
export type Json = import('./pie-mcp-types.js').Json;
export type Content = import('./pie-mcp-types.js').Content;

export class Session {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * List tools exposed by this server.
  */
  listTools(): Json;
  /**
  * Call a tool by name with JSON arguments.
  */
  callTool(name: string, args: Json): Array<Content>;
  /**
  * List resources exposed by this server.
  */
  listResources(): Json;
  /**
  * Read a resource by URI.
  */
  readResource(uri: string): Array<Content>;
  /**
  * List prompts exposed by this server.
  */
  listPrompts(): Json;
  /**
  * Render a prompt template with the given arguments.
  */
  getPrompt(name: string, args: Json): Json;
}
