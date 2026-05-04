// JavaScript mirror of inferlets/mcp-example.
//
// Connects to an MCP server registered on the client session, lists its
// tools, calls one, and demonstrates the canonical pattern for handling
// the raw JSON responses returned by `pie:mcp/client`.

import { mcp } from 'inferlet';

interface Input {
    server?: string;
    tool?: string;
    text?: string;
}

interface Output {
    servers: string[];
    tools_json: string;
    result: string;
}

export async function main(input: Input): Promise<Output> {
    const serverName = input.server ?? 'demo';
    const tool = input.tool ?? 'echo';
    const text = input.text ?? 'hello';

    const servers = mcp.availableServers();
    console.log(`available servers: ${JSON.stringify(servers)}`);

    if (!servers.includes(serverName)) {
        throw new Error(
            `MCP server '${serverName}' is not registered (have: ${JSON.stringify(servers)})`,
        );
    }

    const session = mcp.connect(serverName);

    // tools/list response is opaque JSON: `{"tools": [{"name": ..., ...}, ...]}`.
    const toolsJson = session.listTools();
    console.log(`tools: ${toolsJson}`);

    // tools/call: arguments must be JSON. The response carries
    // `content[]` and `isError`; we have to inspect both.
    const argsJson = JSON.stringify({ text });
    const raw = session.callTool(tool, argsJson);

    const result = extractToolResult(tool, raw);
    console.log(`result: ${result}`);

    return { servers, tools_json: toolsJson, result };
}

/**
 * Pull the user-visible string out of a `tools/call` response.
 * Honors `isError: true` by throwing. For success, returns the text of
 * the first text-typed content item.
 */
function extractToolResult(tool: string, raw: string): string {
    let v: any;
    try {
        v = JSON.parse(raw);
    } catch (e: any) {
        throw new Error(`call_tool('${tool}') returned non-JSON: ${e.message}`);
    }

    const firstText = (): string => {
        const items = Array.isArray(v?.content) ? v.content : [];
        for (const item of items) {
            if (item?.type === 'text') return String(item.text ?? '');
        }
        return '<no text content>';
    };

    if (v?.isError === true) {
        throw new Error(`tool '${tool}' reported failure: ${firstText()}`);
    }
    return firstText();
}
