function moved(data) {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength && !bytes.buffer.detached) {
    return { bytes, transfer: [bytes.buffer] };
  }
  const copy = bytes.slice();
  return { bytes: copy, transfer: [copy.buffer] };
}

export function socketClass(backend) {
  return class PieSocket extends EventTarget {
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
  };
}
