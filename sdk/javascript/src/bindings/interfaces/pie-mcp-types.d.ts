/** @module Interface pie:mcp/types **/
/**
 * Opaque JSON carried as a string.
 */
export type Json = string;
/**
 * MCP error, mirrors JSON-RPC error shape.
 */
export interface Error {
  code: number,
  message: string,
  data?: Json,
}
export interface ImageContent {
  mimeType: string,
  data: Uint8Array,
}
export interface ResourceContent {
  uri: string,
  mimeType?: string,
  text?: string,
  blob?: Uint8Array,
}
/**
 * The fundamental unit of content in MCP responses.
 */
export type Content = ContentText | ContentImage | ContentEmbeddedResource;
export interface ContentText {
  tag: 'text',
  val: string,
}
export interface ContentImage {
  tag: 'image',
  val: ImageContent,
}
export interface ContentEmbeddedResource {
  tag: 'embedded-resource',
  val: ResourceContent,
}
