import { describe, it, expect } from 'vitest';
import { PRESETS, defaultSettings, derive, type PresetId } from '../src/config/radar';
import { Engine, allocOut, STAGES } from '../src/engine/engine';
import { defaultScene } from '../src/sim/scenes';

// Not a strict perf gate (CI machines vary) — logs per-stage timings and checks the chain runs end-to-end.
describe('engine benchmark', () => {
  for (const id of ['automotive', 'drone', 'micro'] as PresetId[]) {
    it(`runs ${id} preset`, () => {
      const p = PRESETS[id];
      const s = { ...defaultSettings(), ...p.settings };
      const e = new Engine(p.cfg, s, defaultScene(id));
      const out = allocOut(p.cfg, derive(p.cfg), s.angleFft, e.spec.cols, e.spec.nfft);
      for (let i = 0; i < 3; i++) e.runFrame(out); // warm-up
      const n = 8;
      const acc: Record<string, number> = {};
      let tot = 0;
      let meta;
      for (let i = 0; i < n; i++) {
        const t0 = performance.now();
        meta = e.runFrame(out);
        tot += performance.now() - t0;
        for (const k of STAGES) acc[k] = (acc[k] ?? 0) + meta.timings[k] / n;
      }
      const ms = tot / n;
      console.log(`${id}: ${ms.toFixed(1)} ms/frame (${(1000 / ms).toFixed(1)} fps) dets=${meta!.detections.length} tracks=${meta!.tracks.length}`,
        Object.entries(acc).map(([k, v]) => `${k}=${v.toFixed(1)}`).join(' '));
      expect(meta!.detections.length).toBeGreaterThan(0);
    });
  }
});
