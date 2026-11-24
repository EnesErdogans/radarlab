/// <reference lib="webworker" />
// Module Web Worker running the radar engine. Output buffers are transferred to
// the main thread and handed back with the next request (ping-pong pool), so the
// steady state allocates no large arrays.

import { Engine, allocOut, type OutBuffers } from './engine';
import { transferList, type FromWorker, type ToWorker } from './protocol';

let engine: Engine | null = null;
const pool: OutBuffers[] = [];

function fits(b: OutBuffers, e: Engine): boolean {
  return b.rdDb.length === e.d.nRangeBins * e.cfg.nChirps && b.raDb.length === e.d.nRangeBins * e.A && b.ifRe.length === e.cfg.nSamples && b.specDb.length === e.spec.cols * e.spec.nfft;
}

function post(msg: FromWorker, transfer: Transferable[] = []): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg, transfer);
}

self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  try {
    switch (m.type) {
      case 'setup': {
        const prev = engine;
        engine = new Engine(m.cfg, m.settings, m.targets);
        engine.selectedId = m.selectedId;
        if (m.keepTime && prev) {
          engine.time = prev.time;
          engine.frame = prev.frame;
          engine.initial = prev.initial.map((t) => ({ ...t }));
        }
        pool.length = 0;
        break;
      }
      case 'targets':
        engine?.setTargets(m.targets, m.resetInitial);
        break;
      case 'select':
        if (engine) {
          if (engine.selectedId !== m.id) engine.spec.reset();
          engine.selectedId = m.id;
          engine.cutOverride = -1;
        }
        break;
      case 'cut':
        if (engine) engine.cutOverride = m.dBin;
        break;
      case 'reset':
        if (engine) {
          engine.setTargets(engine.initial, false);
          engine.tracker.reset();
          engine.spec.reset();
          engine.time = 0;
          engine.frame = 0;
        }
        break;
      case 'frame': {
        if (!engine) return;
        if (m.recycle && fits(m.recycle, engine)) pool.push(m.recycle);
        const out = pool.pop() ?? allocOut(engine.cfg, engine.d, engine.A, engine.spec.cols, engine.spec.nfft);
        const t0 = performance.now();
        const meta = engine.runFrame(out, m.advance, m.advance);
        const computeMs = performance.now() - t0;
        post({ type: 'frame', meta, bufs: out, computeMs, targets: engine.targets.map((t) => ({ ...t })) }, transferList(out));
        break;
      }
    }
  } catch (e) {
    post({ type: 'error', message: String((e as Error)?.stack ?? e) });
  }
};
