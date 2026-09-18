// The JavaScript client SDK (`sdk/client/javascript`) speaks MessagePack over
// a WebSocket. In the tab there is no socket: this shim is a WebSocket-shaped
// object over the runtime's frame API (`pie_send_frame`/`pie_recv_frames`),
// so `new PieClient("pie://local")` works unchanged. Frames are `Uint8Array`s
// and cross to the runtime's thread by transfer, not by copy (see `moved`).
// Install it once with `installInPageTransport(backend)`; any other URI still
// gets a real WebSocket.

let backend = null;

/**
 * A frame's bytes and what to transfer with them: the buffer itself when the
 * view owns all of it, otherwise a copy (the SDK's encoder hands out views
 * into a scratch buffer it may still be using).
 */
function moved(data) {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength && !bytes.buffer.detached) {
    return { bytes, transfer: [bytes.buffer] };
  }
  const copy = bytes.slice();
  return { bytes: copy, transfer: [copy.buffer] };
}

class PieSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url) {
    super();
    this.url = url;
    this.binaryType = "blob";
    this.readyState = PieSocket.CONNECTING;
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    this.session = null;
    backend.call("openSession").then(
      (id) => {
        if (this.readyState !== PieSocket.CONNECTING) {
          backend.call("closeSession", [id]);
          return;
        }
        this.session = id;
        this.readyState = PieSocket.OPEN;
        this.onopen?.(new Event("open"));
        this.pump();
      },
      (e) => {
        this.readyState = PieSocket.CLOSED;
        this.onerror?.(e);
      },
    );
  }

  send(data) {
    if (this.readyState !== PieSocket.OPEN) throw new Error("socket is not open");
    const { bytes, transfer } = moved(data);
    // A refused frame (malformed, session gone) surfaces as an error event;
    // there is no synchronous path across threads.
    backend.call("sendFrame", [this.session, bytes], { transfer }).catch((e) => this.onerror?.(e));
  }

  async pump() {
    while (this.readyState === PieSocket.OPEN) {
      let frames;
      try {
        frames = await backend.call("recvFrames", [this.session, 1000, 64]);
      } catch (e) {
        if (this.readyState === PieSocket.OPEN) this.onerror?.(e);
        break;
      }
      for (const frame of frames) {
        const data = this.binaryType === "blob" ? new Blob([frame]) : frame.buffer;
        this.onmessage?.({ data });
      }
    }
  }

  close() {
    if (this.readyState === PieSocket.CLOSED) return;
    this.readyState = PieSocket.CLOSED;
    if (this.session !== null) backend.call("closeSession", [this.session]);
    this.onclose?.(new Event("close"));
  }
}

/**
 * Route `pie://…` URIs to the runtime behind `runtimeBackend` (an object with
 * `call(op, args, { transfer })`, as pie.mjs makes); everything else is untouched.
 */
export function installInPageTransport(runtimeBackend) {
  backend = runtimeBackend;
  if (globalThis.WebSocket?.__pie) return;
  const Real = globalThis.WebSocket;
  function Routed(url, protocols) {
    if (String(url).startsWith("pie://")) return new PieSocket(url);
    return new Real(url, protocols);
  }
  Routed.__pie = true;
  Routed.CONNECTING = 0;
  Routed.OPEN = 1;
  Routed.CLOSING = 2;
  Routed.CLOSED = 3;
  Routed.prototype = Real?.prototype ?? {};
  globalThis.WebSocket = Routed;
}
