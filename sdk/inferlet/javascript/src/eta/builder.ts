// The DSL channel and the neutral trace builder — ports of
// `eta-dsl/src/channel.rs` and `eta-dsl/src/builder.rs` (+ `lint.rs`).

import {
  ChannelDecl,
  Dtype,
  HostRole,
  Port,
  PortBinding,
  SinkScope,
  Stage,
  STAGE_NAMES,
  StageProgram,
  TraceContainer,
  containerHash,
  encode,
  numel,
  portConsumes,
  shapeOf,
  tags,
} from './ir.js';
import {
  ChannelState,
  SinkCall,
  StageResult,
  TraceError,
  channelStateByGid,
  internChannel,
  isTracing,
  nextGid,
  recordChannelPut,
  recordChannelRead,
  registerChannelState,
  registeredChannelCount,
  releaseChannelState,
  traceStage,
  withConstants,
  withSession,
} from './trace.js';
import { ConstData, Tensor, materialize, reshapeIdTo } from './value.js';

/** A handle to a channel's trace state. */
export class DslChannel {
  constructor(readonly state: ChannelState) {}

  private static build(shape: readonly number[], dtype: Dtype, capacity: number, seed: ConstData | null, seeded: boolean): DslChannel {
    const gid = nextGid();
    const state: ChannelState = {
      gid,
      name: `ch${gid}`,
      shape: shapeOf(shape),
      dtype,
      capacity,
      seeded: seeded || seed !== null,
      hasSeed: seed !== null,
      progPuts: [],
      progTakes: [],
      progReads: [],
      hostPuts: 0,
      hostTakes: 0,
      hostReads: 0,
      descTakes: 0,
      descReads: 0,
    };
    registerChannelState(state);
    return new DslChannel(state);
  }

  static new(shape: readonly number[], dtype: Dtype): DslChannel {
    return DslChannel.build(shape, dtype, 1, null, false);
  }
  static fromConst(data: ConstData): DslChannel {
    return DslChannel.build(data.shape, data.dtype, 1, data, true);
  }
  static seeded(shape: readonly number[], dtype: Dtype): DslChannel {
    return DslChannel.build(shape, dtype, 1, null, true);
  }
  static byGid(gid: number): DslChannel | undefined {
    const st = channelStateByGid(gid);
    return st ? new DslChannel(st) : undefined;
  }
  static release(gid: number): boolean {
    return releaseChannelState(gid);
  }
  static registeredCount(): number {
    return registeredChannelCount();
  }

  capacity(n: number): this {
    this.state.capacity = n;
    return this;
  }
  named(name: string): this {
    this.state.name = name;
    return this;
  }
  get gid(): number {
    return this.state.gid;
  }
  get name(): string {
    return this.state.name;
  }
  get shape() {
    return this.state.shape;
  }
  get dtype(): Dtype {
    return this.state.dtype;
  }
  get isSeeded(): boolean {
    return this.state.seeded;
  }

  noteHostPut(): void {
    this.state.hostPuts++;
  }
  noteHostTake(): void {
    this.state.hostTakes++;
  }
  noteHostRead(): void {
    this.state.hostReads++;
  }
  noteDescClaim(consumes: boolean): void {
    if (consumes) this.state.descTakes++;
    else this.state.descReads++;
  }

  take(): Tensor {
    if (!isTracing()) throw new TraceError(`channel ${this.name}: take() outside a stage body is a host operation; use ch.takeHost()`);
    const [vid, ty] = recordChannelRead(this.state, true);
    return Tensor.node(vid, ty);
  }
  read(): Tensor {
    if (!isTracing()) throw new TraceError(`channel ${this.name}: read() outside a stage body is a host operation; use ch.readHost()`);
    const [vid, ty] = recordChannelRead(this.state, false);
    return Tensor.node(vid, ty);
  }
  putTensor(t: Tensor): void {
    if (!isTracing()) throw new TraceError(`channel ${this.name}: put(Tensor) outside a traced stage`);
    const [vid, ty] = materialize(t);
    const fitted = reshapeIdTo(vid, ty, this.state.shape);
    recordChannelPut(this.state, fitted);
  }
}

/** A traced, linted forward pass: the canonical container plus the
 * dense-order channel identities (gids) and names. */
export class Traced {
  constructor(
    readonly container: TraceContainer,
    readonly channelOrder: number[],
    readonly channelNames: string[],
  ) {}

  encode(): Uint8Array {
    return encode(this.container);
  }
  identityHash(): bigint {
    return containerHash(this.encode());
  }
}

export class Builder {
  private ports: [Port, DslChannel][] = [];
  private stages: [Stage, () => void][] = [];

  constructor(
    readonly vocab: number,
    readonly pageSize: number,
  ) {}

  bindPort(port: Port, source: DslChannel): void {
    source.noteDescClaim(portConsumes(port));
    this.ports.push([port, source]);
  }

  bindPortRecorded(port: Port, source: DslChannel): void {
    this.ports.push([port, source]);
  }

  stage(stage: Stage, body: () => void): void {
    const i = this.stages.findIndex(([s]) => s === stage);
    if (i >= 0) this.stages[i] = [stage, body];
    else this.stages.push([stage, body]);
  }

  private channelPort(port: Port): DslChannel | undefined {
    return this.ports.find(([p]) => p === port)?.[1];
  }

  private rows(): number {
    const ro = this.channelPort(Port.READOUT);
    if (ro) return Math.max(Math.min(numel(ro.shape), 0xffff_ffff), 1);
    const ei = this.channelPort(Port.EMBED_INDPTR);
    if (ei) return Math.max(Math.min(numel(ei.shape), 0xffff_ffff) - 1, 1);
    return 1;
  }

  private record(rows: number): { results: StageResult[]; ports: [Port, number][] } {
    const ports: [Port, number][] = this.ports.map(([port, source]) => [port, internChannel(source.state)]);
    const results: StageResult[] = [];
    for (const stage of [Stage.PROLOGUE, Stage.ON_ATTN_PROJ, Stage.ON_ATTN, Stage.EPILOGUE]) {
      let body: (() => void) | undefined;
      for (const [s, b] of this.stages) if (s === stage) body = b;
      if (!body) continue;
      results.push(traceStage(stage, rows, body));
    }
    return { results, ports };
  }

  build(): Traced {
    const rows = this.rows();
    const {
      result: { results: stageResults, ports: rawPorts },
      channels: rawChannels,
      names: rawNames,
    } = withConstants(this.vocab, this.pageSize, () => withSession(() => this.record(rows)));

    // Re-key the container to gid (declaration) order.
    const order = rawChannels.map((_, i) => i).sort((a, b) => rawChannels[a].gid - rawChannels[b].gid);
    const remap = new Array<number>(rawChannels.length);
    order.forEach((oldIdx, newIdx) => (remap[oldIdx] = newIdx));
    const channels = order.map((i) => rawChannels[i]);
    for (const r of stageResults) for (const op of r.ops) if (op.chan >= 0) op.chan = remap[op.chan];
    const ports: [Port, number][] = rawPorts.map(([p, ci]) => [p, remap[ci]]);

    // Name table: strictly sorted and unique.
    const nameOrder = rawNames.map((_, i) => i).sort((a, b) => (rawNames[a] < rawNames[b] ? -1 : rawNames[a] > rawNames[b] ? 1 : 0));
    const nameRemap = new Array<number>(rawNames.length);
    nameOrder.forEach((oldIdx, newIdx) => (nameRemap[oldIdx] = newIdx));
    const names = nameOrder.map((i) => rawNames[i]);
    for (const r of stageResults) for (const op of r.ops) if (op.tag === tags.KERNEL_CALL || op.tag === tags.SINK_CALL) op.nameIdx = nameRemap[op.nameIdx];

    const sinks: [Stage, SinkCall][] = [];
    for (const r of stageResults) for (const s of r.sinks) sinks.push([r.stage, s]);

    const decls: ChannelDecl[] = channels.map((st) => {
      const hasProgPut = st.progPuts.length > 0;
      const hasProgConsume = st.progTakes.length > 0 || st.progReads.length > 0;
      const hasDescUse = st.descTakes > 0 || st.descReads > 0;
      const hasHostPut = st.hostPuts > 0;
      const hostConsumes = st.hostTakes > 0 || st.hostReads > 0;
      const isTerminalOutput = hasProgPut && !hasProgConsume && !hasDescUse && !hasHostPut && !st.seeded && !st.hasSeed;
      const seededDescriptorWriter = st.seeded && hasDescUse && !hasProgPut;
      let hostRole: HostRole;
      if ((hasHostPut || seededDescriptorWriter) && !hasProgPut) hostRole = HostRole.WRITER;
      else if (hostConsumes && (st.progTakes.length > 0 || hasProgPut)) hostRole = HostRole.READER;
      else if (isTerminalOutput) hostRole = HostRole.READER;
      else hostRole = HostRole.NONE;
      const seeded = st.seeded || (hasHostPut && hasProgPut);
      return { shape: st.shape, dtype: st.dtype, capacity: st.capacity, hostRole, seeded };
    });

    const stages: StageProgram[] = stageResults.map((r) => ({ stage: r.stage, ops: r.ops }));
    const portBindings: PortBinding[] = ports.sort((a, b) => a[0] - b[0]).map(([port, channel]) => ({ port, channel }));
    const container: TraceContainer = { names, channels: decls, ports: portBindings, stages, externs: [] };

    lint(channels, sinks);

    return new Traced(
      container,
      channels.map((st) => st.gid),
      channels.map((st) => st.name),
    );
  }
}

function lint(channels: ChannelState[], sinks: [Stage, SinkCall][]): void {
  const errs: string[] = [];
  for (const st of channels) {
    const hostWrites = st.hostPuts > 0;
    const hostConsumes = st.hostTakes > 0 || st.hostReads > 0;
    const stagePuts = st.progPuts.length > 0;
    const stageConsumes = st.progTakes.length > 0 || st.descTakes > 0;
    if (hostWrites && hostConsumes) errs.push(`channel \`${st.name}\` has two host endpoints (host writes and host consumes); SPSC needs one pass endpoint`);
    const produced = stagePuts || hostWrites || st.seeded || st.hasSeed;
    const consumed = stageConsumes || st.progReads.length > 0 || st.descReads > 0 || hostConsumes;
    if (consumed && !produced) errs.push(`channel \`${st.name}\` is consumed but never produced or seeded`);
  }
  for (const [stage, s] of sinks) {
    const ok = s.scope === SinkScope.PASS_WIDE ? stage === Stage.PROLOGUE : stage === Stage.PROLOGUE || stage === Stage.ON_ATTN_PROJ;
    if (!ok) errs.push(`sink \`${s.name}\` is misplaced in stage \`${STAGE_NAMES[stage]}\``);
  }
  if (errs.length) throw new TraceError(errs.join('; '));
}
