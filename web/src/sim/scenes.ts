import type { PresetId } from '../config/radar';
import { defaultTarget, type Target } from './targets';

export function defaultScene(p: PresetId): Target[] {
  switch (p) {
    case 'automotive':
      return [
        { ...defaultTarget('car', 1), x: -1.8, y: 30, vx: 0, vy: -8, rcs: 10 },
        { ...defaultTarget('car', 2), x: 2.2, y: 14, vx: 0, vy: 4, rcs: 8 },
        { ...defaultTarget('pedestrian', 3), x: -6, y: 9, vx: 1.3, vy: 0, rcs: 0.6 },
        { ...defaultTarget('point', 4), x: 5, y: 22, vx: 0, vy: 0, rcs: 3 },
      ];
    case 'drone':
      return [
        { ...defaultTarget('drone', 1), x: -20, y: 180, z: 30, vx: 2, vy: -6, rcs: 0.05 },
        { ...defaultTarget('bird', 2), x: 40, y: 120, z: 20, vx: -5, vy: 3, rcs: 0.03 },
        { ...defaultTarget('car', 3), x: 60, y: 300, z: 0.5, vx: 0, vy: -12, rcs: 10 },
      ];
    case 'micro':
      return [
        { ...defaultTarget('drone', 1), x: -2, y: 25, z: 3, vx: 0, vy: -1.5, rcs: 0.1, rpm: 5400, bladeLength: 0.12 },
        { ...defaultTarget('bird', 2), x: 6, y: 40, z: 4, vx: -3, vy: 2, rcs: 0.04, flapHz: 7 },
        { ...defaultTarget('pedestrian', 3), x: 3, y: 15, vx: 0, vy: 1.4, rcs: 0.6 },
      ];
  }
}
