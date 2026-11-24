import type { RadarConfig, SimSettings } from '../config/radar';
import type { Target } from '../sim/targets';
import type { FrameMeta, OutBuffers } from './engine';

export type ToWorker =
  | { type: 'setup'; cfg: RadarConfig; settings: SimSettings; targets: Target[]; selectedId: number; keepTime?: boolean }
  | { type: 'targets'; targets: Target[]; resetInitial: boolean }
  | { type: 'select'; id: number }
  | { type: 'cut'; dBin: number }
  | { type: 'reset' }
  | { type: 'frame'; advance: boolean; recycle?: OutBuffers };

export type FromWorker = { type: 'frame'; meta: FrameMeta; bufs: OutBuffers; computeMs: number; targets: Target[] } | { type: 'error'; message: string };

export function transferList(b: OutBuffers): ArrayBuffer[] {
  return [b.rdDb, b.rangeDb, b.raDb, b.cutDb, b.cutThrDb, b.ifRe, b.ifIm, b.specDb].map((a) => a.buffer as ArrayBuffer);
}
