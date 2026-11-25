import './style.css';
import { PRESETS, checkWarnings, cloneConfig, defaultSettings, derive, ulaLayout, type CfarType, type PresetId, type RadarConfig, type SimSettings, type WindowType } from './config/radar';
import { TARGET_TYPES, defaultTarget, maxMicroRadialSpeed, targetKinematics, type Target, type TargetType } from './sim/targets';
import { defaultScene } from './sim/scenes';
import { transferList, type FromWorker, type ToWorker } from './engine/protocol';
import type { OutBuffers } from './engine/engine';
import { LEARN, applyI18n, getLang, setLang, t, type Lang } from './ui/i18n';
import type { CmapName } from './ui/colormap';
import { BevPanel, BudgetPanel, ChirpPanel, CutPanel, MicroPanel, RAPanel, RDPanel, RangePanel, type AppApi, type Frame, type Panel } from './ui/panels';

const GITHUB_URL = 'https://github.com/EnesErdogans/radarlab';

// ------------------------------------------------------------------ state
interface State {
  preset: PresetId;
  cfg: RadarConfig;
  s: SimSettings;
  targets: Target[];
  selectedId: number;
  cmap: CmapName;
  playing: boolean;
  speed: number; // 0 = as fast as possible
  learnStep: number; // -1 = off
}

const url = new URLSearchParams(location.search);
const startPreset = (url.get('preset') as PresetId) in PRESETS ? (url.get('preset') as PresetId) : 'automotive';

function presetState(id: PresetId, keep?: Partial<State>): State {
  const p = PRESETS[id];
  const s = { ...defaultSettings(), ...p.settings, seed: keep?.s?.seed ?? 1 };
  s.cfar = { ...defaultSettings().cfar, ...(p.settings.cfar ?? {}) };
  if (keep?.s?.imp) s.imp = { ...keep.s.imp, pn: { ...keep.s.imp.pn } };
  const targets = defaultScene(id);
  return {
    preset: id,
    cfg: cloneConfig(p.cfg),
    s,
    targets,
    selectedId: targets[0]?.id ?? -1,
    cmap: keep?.cmap ?? 'viridis',
    playing: keep?.playing ?? url.get('paused') === null,
    speed: keep?.speed ?? 1,
    learnStep: keep?.learnStep ?? -1,
  };
}

let st = presetState(startPreset);
if (url.get('lang') === 'tr') setLang('tr');

// ------------------------------------------------------------------ worker
const worker = new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' });
let current: Frame | null = null;
let previous: OutBuffers | null = null;
let inFlight = false;
let rerenderWanted = false;
let lastRequest = 0;
let computeMs = 0;
let dirty = false;
const uiTimes: number[] = [];
const simTimes: number[] = [];

function send(m: ToWorker, transfer: Transferable[] = []): void {
  worker.postMessage(m, transfer);
}

function setup(keepTime = true): void {
  send({ type: 'setup', cfg: st.cfg, settings: st.s, targets: st.targets, selectedId: st.selectedId, keepTime });
  previous = null;
  requestFrame(false);
}

function requestFrame(advance: boolean): void {
  if (inFlight) {
    if (!advance) rerenderWanted = true;
    return;
  }
  inFlight = true;
  lastRequest = performance.now();
  const rec = previous;
  previous = null;
  if (rec) send({ type: 'frame', advance, recycle: rec }, transferList(rec));
  else send({ type: 'frame', advance });
}

worker.onmessage = (ev: MessageEvent<FromWorker>) => {
  const m = ev.data;
  if (m.type === 'error') {
    console.error(m.message);
    inFlight = false;
    return;
  }
  inFlight = false;
  if (current) previous = current.bufs;
  current = { meta: m.meta, bufs: m.bufs, cfg: st.cfg, d: derive(st.cfg) };
  computeMs = m.computeMs;
  const now = performance.now();
  simTimes.push(now);
  while (simTimes.length && now - simTimes[0] > 1000) simTimes.shift();
  // positions advanced by the worker
  if (!dragging) {
    for (const wt of m.targets) {
      const lt = st.targets.find((q) => q.id === wt.id);
      if (lt) {
        lt.x = wt.x;
        lt.y = wt.y;
        lt.z = wt.z;
      }
    }
  }
  dirty = true;
  if (rerenderWanted) {
    rerenderWanted = false;
    requestFrame(false);
  }
};

// ------------------------------------------------------------------ app api for panels
let dragging = false;
let lastDragSend = 0;
const api: AppApi = {
  frame: () => current,
  cmap: () => st.cmap,
  selectedId: () => st.selectedId,
  select(id) {
    if (st.selectedId === id) return;
    st.selectedId = id;
    send({ type: 'select', id });
    renderTargetList();
    if (!st.playing) requestFrame(false);
    dirty = true;
  },
  setCut(dBin) {
    send({ type: 'cut', dBin });
    if (!st.playing) requestFrame(false);
  },
  dragTarget(id, x, y, done) {
    const tg = st.targets.find((q) => q.id === id);
    if (!tg) return;
    dragging = !done;
    tg.x = +x.toFixed(2);
    tg.y = +y.toFixed(2);
    const now = performance.now();
    if (done || now - lastDragSend > 40) {
      lastDragSend = now;
      send({ type: 'targets', targets: st.targets, resetInitial: true });
      if (!st.playing) requestFrame(false);
      updateTargetInputs();
    }
  },
  computeMs: () => computeMs,
  uiFps: () => uiTimes.length,
  simFps: () => simTimes.length,
  warnings: () => {
    const d = derive(st.cfg);
    return checkWarnings(
      st.cfg,
      d,
      st.targets.map((tg) => {
        const k = targetKinematics(tg);
        return { name: `${t(tg.type)} #${tg.id}`, range: k.range, vr: k.vr, azDeg: k.azDeg, maxMicroSpeed: maxMicroRadialSpeed(tg) };
      }),
    );
  },
};

// ------------------------------------------------------------------ DOM
const app = document.getElementById('app')!;
app.innerHTML = `
<header class="top">
  <div class="brand"><b>Radar<span>Lab</span></b><small data-i18n="tagline"></small></div>
  <div class="tb"><label data-i18n="preset"></label><select id="preset"></select></div>
  <div class="tb">
    <button class="btn primary" id="play"></button>
    <button class="btn" id="step" data-i18n="step"></button>
    <button class="btn" id="reset" data-i18n="reset"></button>
    <select id="speed" title="Speed"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="0">max</option></select>
  </div>
  <div class="tb seedbox"><label data-i18n="seed"></label><input type="number" id="seed" min="0" step="1"></div>
  <button class="btn" id="learn" data-i18n="learn"></button>
  <div class="spacer"></div>
  <span class="stat" id="stat"></span>
  <div class="seg" id="lang"><button data-l="en">EN</button><button data-l="tr">TR</button></div>
  <a class="gh" id="gh" target="_blank" rel="noopener" data-i18n="github"></a>
</header>
<aside class="side">
  <div class="tabs">
    <button data-tab="targets" class="on" data-i18n="tabTargets"></button>
    <button data-tab="radar" data-i18n="tabRadar"></button>
    <button data-tab="proc" data-i18n="tabProc"></button>
    <button data-tab="imp" data-i18n="tabImp"></button>
  </div>
  <div class="tabpane on" id="tab-targets">
    <p class="hint" data-i18n="dragHint"></p>
    <div class="addrow"><select id="addType"></select><button class="btn" id="addBtn" data-i18n="addTarget"></button></div>
    <div id="tlist"></div>
  </div>
  <div class="tabpane" id="tab-radar"></div>
  <div class="tabpane" id="tab-proc"></div>
  <div class="tabpane" id="tab-imp"></div>
</aside>
<main class="grid" id="grid">
  <div class="learn-card" id="learnCard"></div>
</main>`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const grid = $('grid');
const panels: Panel[] = [new BevPanel(api), new RDPanel(api), new RAPanel(api), new CutPanel(api), new MicroPanel(api), new ChirpPanel(api), new RangePanel(api), new BudgetPanel(api)];
for (const p of panels) grid.insertBefore(p.el, $('learnCard'));
const cutPanel = panels.find((p) => p.id === 'cut') as CutPanel;

($('gh') as HTMLAnchorElement).href = GITHUB_URL;

// header
const presetSel = $<HTMLSelectElement>('preset');
function fillPresets(): void {
  presetSel.innerHTML = (Object.keys(PRESETS) as PresetId[]).map((id) => `<option value="${id}">${PRESETS[id].name[getLang()]}</option>`).join('');
  presetSel.value = st.preset;
}
presetSel.onchange = () => {
  st = presetState(presetSel.value as PresetId, st);
  current = null;
  setup(false);
  buildEditors();
};
const playBtn = $<HTMLButtonElement>('play');
function syncPlay(): void {
  playBtn.textContent = st.playing ? '❚❚ ' + t('pause') : '▶ ' + t('play');
}
playBtn.onclick = () => {
  st.playing = !st.playing;
  syncPlay();
};
$('step').onclick = () => {
  st.playing = false;
  syncPlay();
  requestFrame(true);
};
$('reset').onclick = () => {
  send({ type: 'reset' });
  requestFrame(false);
};
const speedSel = $<HTMLSelectElement>('speed');
speedSel.value = String(st.speed);
speedSel.onchange = () => (st.speed = +speedSel.value);
const seedIn = $<HTMLInputElement>('seed');
seedIn.value = String(st.s.seed);
seedIn.onchange = () => {
  st.s.seed = Math.max(0, Math.floor(+seedIn.value || 0));
  setup();
};
$('lang')
  .querySelectorAll<HTMLButtonElement>('button')
  .forEach((b) => (b.onclick = () => switchLang(b.dataset.l as Lang)));
function switchLang(l: Lang): void {
  setLang(l);
  applyI18n();
  fillPresets();
  buildEditors();
  syncPlay();
  $('lang')
    .querySelectorAll<HTMLButtonElement>('button')
    .forEach((b) => b.classList.toggle('on', b.dataset.l === l));
  renderLearn();
  dirty = true;
}

// tabs
document.querySelectorAll<HTMLButtonElement>('.tabs button').forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
    document.querySelectorAll('.tabpane').forEach((x) => x.classList.toggle('on', x.id === 'tab-' + b.dataset.tab));
  };
});

// ------------------------------------------------------------------ target editor
const addType = $<HTMLSelectElement>('addType');
$('addBtn').onclick = () => {
  const id = Math.max(0, ...st.targets.map((q) => q.id)) + 1;
  const d = derive(st.cfg);
  const tg = defaultTarget(addType.value as TargetType, id, 0, Math.min(d.rMax * 0.5, 30));
  st.targets.push(tg);
  st.selectedId = id;
  send({ type: 'targets', targets: st.targets, resetInitial: true });
  send({ type: 'select', id });
  renderTargetList();
  if (!st.playing) requestFrame(false);
};

const num = (v: number, dp = 2) => String(+v.toFixed(dp));
function targetFields(tg: Target): [string, string, number, number][] {
  // key, label, value, step
  const f: [string, string, number, number][] = [
    ['x', 'x [m]', tg.x, 0.5],
    ['y', 'y [m]', tg.y, 0.5],
    ['z', 'z [m]', tg.z, 0.5],
    ['vx', 'vx [m/s]', tg.vx, 0.5],
    ['vy', 'vy [m/s]', tg.vy, 0.5],
    ['rcs', t('rcs'), 10 * Math.log10(tg.rcs), 1],
  ];
  if (tg.type === 'drone') {
    f.push(['rpm', t('rpm'), tg.rpm ?? 6000, 100], ['bladeLength', t('blade'), tg.bladeLength ?? 0.12, 0.01]);
  }
  if (tg.type === 'bird') f.push(['flapHz', t('flap'), tg.flapHz ?? 6, 0.5]);
  return f;
}

function renderTargetList(): void {
  const list = $('tlist');
  list.innerHTML = '';
  for (const tg of st.targets) {
    const card = document.createElement('div');
    card.className = 'tcard' + (tg.id === st.selectedId ? ' sel' : '');
    card.dataset.id = String(tg.id);
    card.innerHTML = `<div class="th"><b>#${tg.id}</b><select>${TARGET_TYPES.map((ty) => `<option value="${ty}" ${ty === tg.type ? 'selected' : ''}>${t(ty)}</option>`).join('')}</select><button class="x" title="${t('remove')}">×</button></div>
      <div class="tgrid">${targetFields(tg)
        .map(([k, lab, v, stp]) => `<label>${lab}<input type="number" data-k="${k}" step="${stp}" value="${num(v)}"></label>`)
        .join('')}</div>`;
    card.onclick = (e) => {
      if ((e.target as HTMLElement).closest('input,select,button')) return;
      api.select(tg.id);
    };
    card.querySelector('select')!.onchange = (e) => {
      const ty = (e.target as HTMLSelectElement).value as TargetType;
      const nt = { ...defaultTarget(ty, tg.id), x: tg.x, y: tg.y };
      Object.assign(tg, nt);
      commitTargets();
      renderTargetList();
    };
    (card.querySelector('.x') as HTMLButtonElement).onclick = () => {
      st.targets = st.targets.filter((q) => q.id !== tg.id);
      if (st.selectedId === tg.id) {
        st.selectedId = st.targets[0]?.id ?? -1;
        send({ type: 'select', id: st.selectedId });
      }
      commitTargets();
      renderTargetList();
    };
    card.querySelectorAll<HTMLInputElement>('input').forEach((inp) => {
      inp.onchange = () => {
        const k = inp.dataset.k!;
        const v = +inp.value;
        if (!isFinite(v)) return;
        if (k === 'rcs') tg.rcs = Math.pow(10, v / 10);
        else (tg as unknown as Record<string, number>)[k] = v;
        commitTargets();
      };
    });
    list.appendChild(card);
  }
}
function commitTargets(): void {
  send({ type: 'targets', targets: st.targets, resetInitial: true });
  if (!st.playing) requestFrame(false);
  dirty = true;
}
let lastInputSync = 0;
function updateTargetInputs(): void {
  const focused = document.activeElement;
  for (const card of Array.from($('tlist').children) as HTMLElement[]) {
    const tg = st.targets.find((q) => q.id === +card.dataset.id!);
    if (!tg) continue;
    card.querySelectorAll<HTMLInputElement>('input').forEach((inp) => {
      if (inp === focused) return;
      const k = inp.dataset.k!;
      if (k === 'x' || k === 'y' || k === 'z') inp.value = num((tg as unknown as Record<string, number>)[k], 1);
    });
  }
}

// ------------------------------------------------------------------ radar / processing editors
type FieldDef =
  | { kind: 'num'; label: string; get: () => number; set: (v: number) => void; step?: number; min?: number; max?: number }
  | { kind: 'sel'; label: string; opts: [string, string][]; get: () => string; set: (v: string) => void }
  | { kind: 'chk'; label: string; get: () => boolean; set: (v: boolean) => void }
  | { kind: 'pair'; label: string; get: () => [number, number]; set: (a: number, b: number) => void }
  | { kind: 'h'; label: string }
  | { kind: 'note'; html: string };

function buildForm(el: HTMLElement, defs: FieldDef[], onChange: () => void): void {
  el.innerHTML = '';
  for (const f of defs) {
    if (f.kind === 'h') {
      const h = document.createElement('h4');
      h.textContent = f.label;
      el.appendChild(h);
      continue;
    }
    if (f.kind === 'note') {
      const p = document.createElement('p');
      p.className = 'note';
      p.innerHTML = f.html;
      el.appendChild(p);
      continue;
    }
    const row = document.createElement('label');
    row.className = 'field' + (f.kind === 'pair' ? ' pair' : '');
    const sp = document.createElement('span');
    sp.textContent = f.label;
    row.appendChild(sp);
    if (f.kind === 'num') {
      const i = document.createElement('input');
      i.type = 'number';
      i.step = String(f.step ?? 'any');
      if (f.min !== undefined) i.min = String(f.min);
      if (f.max !== undefined) i.max = String(f.max);
      i.value = num(f.get(), 4);
      i.onchange = () => {
        let v = +i.value;
        if (!isFinite(v)) return;
        if (f.min !== undefined) v = Math.max(f.min, v);
        if (f.max !== undefined) v = Math.min(f.max, v);
        f.set(v);
        i.value = num(f.get(), 4);
        onChange();
      };
      row.appendChild(i);
    } else if (f.kind === 'sel') {
      const s = document.createElement('select');
      s.innerHTML = f.opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
      s.value = f.get();
      s.onchange = () => {
        f.set(s.value);
        onChange();
      };
      row.appendChild(s);
    } else if (f.kind === 'chk') {
      const c = document.createElement('input');
      c.type = 'checkbox';
      c.checked = f.get();
      c.onchange = () => {
        f.set(c.checked);
        onChange();
      };
      row.appendChild(c);
    } else {
      const [a, b] = f.get();
      const ia = document.createElement('input'),
        ib = document.createElement('input');
      for (const [inp, v] of [
        [ia, a],
        [ib, b],
      ] as const) {
        inp.type = 'number';
        inp.min = '0';
        inp.max = '16';
        inp.step = '1';
        inp.value = String(v);
        inp.onchange = () => {
          f.set(Math.max(0, Math.round(+ia.value)), Math.max(0, Math.round(+ib.value)));
          onChange();
        };
        row.appendChild(inp);
      }
    }
    el.appendChild(row);
  }
}

const pow2 = (lo: number, hi: number): [string, string][] => {
  const o: [string, string][] = [];
  for (let v = lo; v <= hi; v *= 2) o.push([String(v), String(v)]);
  return o;
};

function buildEditors(): void {
  addType.innerHTML = TARGET_TYPES.map((ty) => `<option value="${ty}">${t(ty)}</option>`).join('');
  renderTargetList();
  const c = st.cfg;
  const bEff = () => (c.slope * c.nSamples) / c.fs;
  const setB = (bHz: number) => (c.slope = (bHz * c.fs) / c.nSamples);
  const radar: FieldDef[] = [
    { kind: 'h', label: 'Waveform' },
    { kind: 'num', label: t('fcL'), get: () => c.fc / 1e9, set: (v) => (c.fc = v * 1e9), step: 0.5, min: 1, max: 300 },
    { kind: 'num', label: t('bandwidth'), get: () => bEff() / 1e6, set: (v) => setB(v * 1e6), step: 10, min: 1, max: 5000 },
    {
      kind: 'num',
      label: t('fsL'),
      get: () => c.fs / 1e6,
      set: (v) => {
        const b = bEff();
        c.fs = v * 1e6;
        setB(b);
      },
      step: 0.5,
      min: 0.1,
      max: 100,
    },
    {
      kind: 'sel',
      label: t('nL'),
      opts: pow2(32, 1024),
      get: () => String(c.nSamples),
      set: (v) => {
        const b = bEff();
        c.nSamples = +v;
        setB(b);
      },
    },
    { kind: 'num', label: t('tcL'), get: () => c.chirpPeriod * 1e6, set: (v) => (c.chirpPeriod = v * 1e-6), step: 1, min: 1, max: 2000 },
    { kind: 'num', label: t('rampL'), get: () => (c.rampTime ?? c.chirpPeriod) * 1e6, set: (v) => (c.rampTime = v * 1e-6), step: 1, min: 1, max: 2000 },
    { kind: 'sel', label: t('mL'), opts: pow2(16, 512), get: () => String(c.nChirps), set: (v) => (c.nChirps = +v) },
    { kind: 'num', label: t('framePL'), get: () => c.framePeriod * 1e3, set: (v) => (c.framePeriod = v * 1e-3), step: 5, min: 1, max: 1000 },
    { kind: 'h', label: 'MIMO' },
    {
      kind: 'sel',
      label: t('ntxL'),
      opts: [1, 2, 3, 4].map((v) => [String(v), String(v)]),
      get: () => String(c.txPos.length),
      set: (v) => Object.assign(c, ulaLayout(+v, c.rxPos.length), { txPosZ: undefined, rxPosZ: undefined }),
    },
    {
      kind: 'sel',
      label: t('nrxL'),
      opts: [1, 2, 3, 4, 6, 8].map((v) => [String(v), String(v)]),
      get: () => String(c.rxPos.length),
      set: (v) => Object.assign(c, ulaLayout(c.txPos.length, +v), { txPosZ: undefined, rxPosZ: undefined }),
    },
  ];
  buildForm($('tab-radar'), radar, () => {
    setup();
    dirty = true;
  });
  const s = st.s;
  const wins: [string, string][] = ['hann', 'hamming', 'blackman', 'rect'].map((w) => [w, w]);
  const proc: FieldDef[] = [
    { kind: 'h', label: 'Signal' },
    { kind: 'num', label: t('snrL'), get: () => s.snrRefDb, set: (v) => (s.snrRefDb = v), step: 1, min: -40, max: 60 },
    { kind: 'num', label: t('refRL'), get: () => s.refRange, set: (v) => (s.refRange = v), step: 1, min: 0.1 },
    { kind: 'chk', label: t('noiseL'), get: () => s.noise, set: (v) => (s.noise = v) },
    { kind: 'chk', label: t('clutterL'), get: () => s.clutter, set: (v) => (s.clutter = v) },
    { kind: 'chk', label: t('mtiL'), get: () => s.mti, set: (v) => (s.mti = v) },
    { kind: 'sel', label: t('rwinL'), opts: wins, get: () => s.rangeWindow, set: (v) => (s.rangeWindow = v as WindowType) },
    { kind: 'sel', label: t('dwinL'), opts: wins, get: () => s.dopplerWindow, set: (v) => (s.dopplerWindow = v as WindowType) },
    { kind: 'sel', label: t('awinL'), opts: wins, get: () => s.raWindow, set: (v) => (s.raWindow = v as WindowType) },
    { kind: 'h', label: 'CFAR' },
    {
      kind: 'sel',
      label: t('cfarL'),
      opts: ['CA', 'GO', 'SO', 'OS'].map((v) => [v, v]),
      get: () => s.cfar.type,
      set: (v) => (s.cfar.type = v as CfarType),
    },
    {
      kind: 'sel',
      label: t('pfaL'),
      opts: [3, 4, 5, 6, 7].map((e) => [`1e-${e}`, `10⁻${'⁰¹²³⁴⁵⁶⁷⁸⁹'[e]}`]),
      get: () => `1e-${Math.round(-Math.log10(s.cfar.pfa))}`,
      set: (v) => (s.cfar.pfa = +v),
    },
    { kind: 'pair', label: t('guardL'), get: () => [s.cfar.guardR, s.cfar.guardD], set: (a, b) => Object.assign(s.cfar, { guardR: a, guardD: b }) },
    { kind: 'pair', label: t('trainL'), get: () => [s.cfar.trainR, s.cfar.trainD], set: (a, b) => Object.assign(s.cfar, { trainR: Math.max(1, a), trainD: b }) },
    { kind: 'num', label: t('osRankL'), get: () => s.cfar.osRank, set: (v) => (s.cfar.osRank = v), step: 0.05, min: 0.05, max: 1 },
    { kind: 'h', label: 'Display' },
    {
      kind: 'sel',
      label: t('cmapL'),
      opts: [
        ['viridis', 'viridis'],
        ['inferno', 'inferno'],
      ],
      get: () => st.cmap,
      set: (v) => (st.cmap = v as CmapName),
    },
  ];
  buildForm($('tab-proc'), proc, () => {
    cutPanel.activeType = s.cfar.type;
    setup();
    dirty = true;
  });
  cutPanel.activeType = s.cfar.type;

  // impairments (roadmap A4; same models as the Python reference radarlab.impairments)
  const im = s.imp;
  const impDefs: FieldDef[] = [
    { kind: 'note', html: t('impNote') },
    { kind: 'h', label: t('impPn') },
    { kind: 'chk', label: t('impOn'), get: () => im.phaseNoise, set: (v) => (im.phaseNoise = v) },
    { kind: 'num', label: t('pnL0'), get: () => im.pn.l0DbcHz, set: (v) => (im.pn.l0DbcHz = v), step: 1, min: -140, max: -40 },
    { kind: 'num', label: t('pnBw'), get: () => im.pn.loopBwHz / 1e3, set: (v) => (im.pn.loopBwHz = v * 1e3), step: 50, min: 1, max: 10000 },
    { kind: 'num', label: t('pnFloor'), get: () => im.pn.floorDbcHz, set: (v) => (im.pn.floorDbcHz = v), step: 5, min: -400, max: -60 },
    { kind: 'h', label: t('impLeak') },
    { kind: 'chk', label: t('impOn'), get: () => im.leakage, set: (v) => (im.leakage = v) },
    { kind: 'num', label: t('leakD'), get: () => im.leakDistance, set: (v) => (im.leakDistance = v), step: 0.05, min: 0, max: 10 },
    { kind: 'num', label: t('leakP'), get: () => im.leakPowerDb, set: (v) => (im.leakPowerDb = v), step: 5, min: -20, max: 90 },
    { kind: 'h', label: t('impSw') },
    {
      kind: 'sel',
      label: t('swModel'),
      opts: [
        ['0', t('sw0')],
        ['1', t('sw1')],
        ['3', t('sw3')],
      ],
      get: () => String(im.swerling),
      set: (v) => (im.swerling = +v as 0 | 1 | 3),
    },
    { kind: 'h', label: t('impIq') },
    { kind: 'chk', label: t('impOn'), get: () => im.iq, set: (v) => (im.iq = v) },
    { kind: 'num', label: t('iqG'), get: () => im.iqGainDb, set: (v) => (im.iqGainDb = v), step: 0.1, min: -6, max: 6 },
    { kind: 'num', label: t('iqP'), get: () => im.iqPhaseDeg, set: (v) => (im.iqPhaseDeg = v), step: 0.5, min: -45, max: 45 },
    { kind: 'h', label: t('impIf') },
    { kind: 'chk', label: t('impOn'), get: () => im.ifFilter, set: (v) => (im.ifFilter = v) },
    { kind: 'num', label: t('hpfL'), get: () => im.hpfHz / 1e3, set: (v) => (im.hpfHz = v * 1e3), step: 50, min: 0, max: 50000 },
    { kind: 'num', label: t('lpfL'), get: () => im.lpfHz / 1e6, set: (v) => (im.lpfHz = v * 1e6), step: 0.5, min: 0, max: 100 },
    { kind: 'sel', label: t('fOrd'), opts: ['1', '2', '3', '4'].map((v) => [v, v]), get: () => String(im.filterOrder), set: (v) => (im.filterOrder = +v) },
    { kind: 'h', label: t('impAdc') },
    { kind: 'chk', label: t('impOn'), get: () => im.adc, set: (v) => (im.adc = v) },
    { kind: 'num', label: t('adcBits'), get: () => im.adcBits, set: (v) => (im.adcBits = Math.round(v)), step: 1, min: 2, max: 16 },
    { kind: 'num', label: t('adcFs'), get: () => im.adcFullScaleDb, set: (v) => (im.adcFullScaleDb = v), step: 1, min: -20, max: 100 },
  ];
  buildForm($('tab-imp'), impDefs, () => {
    setup();
    dirty = true;
  });
}

// ------------------------------------------------------------------ learn mode
$('learn').onclick = () => {
  st.learnStep = st.learnStep >= 0 ? -1 : 0;
  renderLearn();
};
function renderLearn(): void {
  const on = st.learnStep >= 0;
  document.body.classList.toggle('learn', on);
  $('learn').classList.toggle('on', on);
  const card = $('learnCard');
  for (const p of panels) p.el.classList.toggle('focus', on && LEARN[st.learnStep]?.panel === p.id);
  if (!on) {
    card.innerHTML = '';
    return;
  }
  const L = LEARN[st.learnStep];
  const lang = getLang();
  card.innerHTML = `<div class="stepn">${t('learnStep', { i: st.learnStep + 1, n: LEARN.length })}</div><h3>${L.title[lang]}</h3><div class="lbody">${L.body[lang]}</div>
    <div class="dots">${LEARN.map((_, i) => `<i class="${i === st.learnStep ? 'on' : ''}"></i>`).join('')}</div>
    <div class="nav"><button class="btn" id="lPrev" ${st.learnStep === 0 ? 'disabled' : ''}>← ${t('prev')}</button><button class="btn primary" id="lNext" ${st.learnStep === LEARN.length - 1 ? 'disabled' : ''}>${t('next')} →</button><span class="spacer"></span><button class="btn" id="lClose">${t('close')}</button></div>`;
  $('lPrev').onclick = () => {
    st.learnStep = Math.max(0, st.learnStep - 1);
    renderLearn();
  };
  $('lNext').onclick = () => {
    st.learnStep = Math.min(LEARN.length - 1, st.learnStep + 1);
    renderLearn();
  };
  $('lClose').onclick = () => {
    st.learnStep = -1;
    renderLearn();
  };
  dirty = true;
}
document.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).closest('input,select,textarea')) return;
  if (e.key === ' ') {
    e.preventDefault();
    playBtn.click();
  } else if (st.learnStep >= 0 && e.key === 'ArrowRight') $('lNext').click();
  else if (st.learnStep >= 0 && e.key === 'ArrowLeft') $('lPrev').click();
  else if (e.key === 'Escape' && st.learnStep >= 0) $('lClose').click();
});

// ------------------------------------------------------------------ main loop
const statEl = $('stat');
let lastStat = 0;
function loop(now: number): void {
  if (st.playing && !inFlight) {
    const period = st.speed > 0 ? (st.cfg.framePeriod * 1000) / st.speed : 0;
    if (now - lastRequest >= period - 4) requestFrame(true);
  }
  if (dirty && current) {
    dirty = false;
    for (const p of panels) if (p.el.offsetParent !== null) p.render();
    uiTimes.push(now);
  }
  while (uiTimes.length && now - uiTimes[0] > 1000) uiTimes.shift();
  if (now - lastStat > 250 && current) {
    lastStat = now;
    statEl.textContent = `t=${current.meta.time.toFixed(2)} s · ${computeMs.toFixed(1)} ms/frame`;
    if (now - lastInputSync > 300) {
      lastInputSync = now;
      updateTargetInputs();
    }
  }
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ boot
if (url.has('learn')) st.learnStep = Math.max(0, Math.min(LEARN.length - 1, +(url.get('learn') || 0)));
fillPresets();
buildEditors();
switchLang(getLang());
setup(false);
requestAnimationFrame(loop);
(window as unknown as { radarlab: unknown }).radarlab = { state: () => st, frame: () => current };
