export type Lang = 'en' | 'tr';

type Dict = Record<string, string>;

const en: Dict = {
  tagline: 'FMCW MIMO radar simulator',
  preset: 'Preset',
  play: 'Play',
  pause: 'Pause',
  step: 'Step',
  reset: 'Reset',
  seed: 'Seed',
  learn: 'Learn',
  github: 'GitHub',
  tabTargets: 'Targets',
  tabRadar: 'Radar',
  tabProc: 'Processing',
  pChirp: 'Chirp & IF signal',
  pRange: 'Range profile',
  pRD: 'Range–Doppler map + CFAR',
  pCut: 'CFAR cut',
  pRA: 'Range–angle map',
  pBev: "Bird's-eye view",
  pMicro: 'Micro-Doppler spectrogram',
  pBudget: 'Parameter budget & timings',
  range: 'Range',
  velocity: 'Radial velocity',
  angle: 'Azimuth',
  power: 'Power',
  time: 'Time',
  freq: 'Frequency',
  amplitude: 'Amplitude',
  sampleTime: 'Fast time',
  frames: 'frames',
  xAxis: 'x (cross-range)',
  yAxis: 'y (down-range)',
  tx: 'TX',
  rx: 'RX',
  adcWin: 'ADC',
  delayExag: 'τ ×{k}',
  beat: 'beat f_b',
  truth: 'Ground truth',
  dets: 'Detections',
  tracks: 'Confirmed tracks',
  tentative: 'tentative',
  fov: 'FOV',
  selected: 'Selected',
  noSel: 'Select a target (click in the bird’s-eye view or the target list).',
  cutAt: 'Doppler bin {d} (v = {v} m/s) — click the RD map to choose',
  active: 'active',
  cutGamma: 'all α exact for Gamma(K = {k}) noise',
  rangeRes: 'Range resolution ΔR',
  rMax: 'Max range R_max',
  vRes: 'Velocity resolution Δv',
  vMax: 'Max velocity ±v_max',
  angRes: 'Angle resolution',
  fovL: 'Unambiguous FOV',
  frameTime: 'Frame active / period',
  dataRate: 'Raw data rate',
  bEff: 'Effective bandwidth',
  lambda: 'Wavelength λ',
  virt: 'Virtual channels K',
  cube: 'Data cube K×M×N',
  timings: 'Worker timings / frame',
  total: 'total',
  fpsSim: 'sim fps',
  fpsUi: 'ui fps',
  warnings: 'Warnings',
  noWarnings: 'No warnings — configuration is consistent.',
  warnAdcRamp: 'ADC window ends at {adc} µs but the ramp ends at {ramp} µs: samples taken off the ramp.',
  warnRampTc: 'Ramp time is longer than the chirp period Tc.',
  warnFrame: 'Frame active time {act} ms exceeds the frame period.',
  warnPow2: 'N and M must be powers of two.',
  warnRmax: '{name}: R = {r} m is beyond R_max — it will fold / be cut off.',
  warnVel: '{name}: v_r = {v} m/s exceeds ±v_max — Doppler aliasing.',
  warnMicro: '{name}: micro-Doppler parts reach {v} m/s > v_max — spectrogram aliasing.',
  warnFov: '{name}: outside the unambiguous angular FOV.',
  addTarget: 'Add target',
  remove: 'Remove',
  type: 'Type',
  point: 'Point',
  car: 'Car',
  pedestrian: 'Pedestrian',
  drone: 'Drone',
  bird: 'Bird',
  rcs: 'RCS [dBsm]',
  rpm: 'Rotor rpm',
  blade: 'Blade [m]',
  flap: 'Flap [Hz]',
  bandwidth: 'Bandwidth B_eff [MHz]',
  fsL: 'Sample rate f_s [MHz]',
  nL: 'Samples / chirp N',
  tcL: 'Chirp period T_c [µs]',
  rampL: 'Ramp time [µs]',
  mL: 'Chirps / TX M',
  ntxL: 'TX antennas',
  nrxL: 'RX antennas',
  framePL: 'Frame period [ms]',
  fcL: 'Carrier f_c [GHz]',
  snrL: 'SNR @ ref. range [dB]',
  refRL: 'Reference range [m]',
  noiseL: 'Noise',
  mtiL: 'MTI (static clutter removal)',
  clutterL: 'Static clutter',
  rwinL: 'Range window',
  dwinL: 'Doppler window',
  awinL: 'Range–angle map window',
  cfarL: 'CFAR type',
  pfaL: 'P_fa',
  guardL: 'Guard (range, Doppler)',
  trainL: 'Training (range, Doppler)',
  osRankL: 'OS rank',
  cmapL: 'Colormap',
  dragHint: 'Drag targets in the bird’s-eye view. Click a target to select it for the IF and micro-Doppler panels.',
  learnStep: 'Step {i} of {n}',
  prev: 'Previous',
  next: 'Next',
  close: 'Close',
  detsN: '{n} detections',
  microParts: 'max |v_r| of parts ≈ {v} m/s',
};

const tr: Dict = {
  tagline: 'FMCW MIMO radar simülatörü',
  preset: 'Hazır ayar',
  play: 'Oynat',
  pause: 'Duraklat',
  step: 'Adım',
  reset: 'Sıfırla',
  seed: 'Tohum',
  learn: 'Öğren',
  github: 'GitHub',
  tabTargets: 'Hedefler',
  tabRadar: 'Radar',
  tabProc: 'İşleme',
  pChirp: 'Chirp ve IF sinyali',
  pRange: 'Menzil profili',
  pRD: 'Menzil–Doppler haritası + CFAR',
  pCut: 'CFAR kesiti',
  pRA: 'Menzil–açı haritası',
  pBev: 'Kuşbakışı görünüm',
  pMicro: 'Mikro-Doppler spektrogramı',
  pBudget: 'Parametre bütçesi ve süreler',
  range: 'Menzil',
  velocity: 'Radyal hız',
  angle: 'Azimut',
  power: 'Güç',
  time: 'Zaman',
  freq: 'Frekans',
  amplitude: 'Genlik',
  sampleTime: 'Hızlı zaman',
  frames: 'çerçeve',
  xAxis: 'x (çapraz menzil)',
  yAxis: 'y (menzil yönü)',
  tx: 'TX',
  rx: 'RX',
  adcWin: 'ADC',
  delayExag: 'τ ×{k}',
  beat: 'vuru f_b',
  truth: 'Gerçek konum',
  dets: 'Tespitler',
  tracks: 'Onaylı izler',
  tentative: 'aday',
  fov: 'Görüş alanı',
  selected: 'Seçili',
  noSel: 'Bir hedef seçin (kuşbakışı görünümde veya hedef listesinde tıklayın).',
  cutAt: 'Doppler hücresi {d} (v = {v} m/s) — seçmek için RD haritasına tıklayın',
  active: 'etkin',
  cutGamma: 'tüm α değerleri Gamma(K = {k}) gürültü için tam',
  rangeRes: 'Menzil çözünürlüğü ΔR',
  rMax: 'Azami menzil R_max',
  vRes: 'Hız çözünürlüğü Δv',
  vMax: 'Azami hız ±v_max',
  angRes: 'Açı çözünürlüğü',
  fovL: 'Belirsizliksiz görüş alanı',
  frameTime: 'Çerçeve aktif / periyot',
  dataRate: 'Ham veri hızı',
  bEff: 'Etkin bant genişliği',
  lambda: 'Dalga boyu λ',
  virt: 'Sanal kanal K',
  cube: 'Veri küpü K×M×N',
  timings: 'İşçi süreleri / çerçeve',
  total: 'toplam',
  fpsSim: 'sim fps',
  fpsUi: 'arayüz fps',
  warnings: 'Uyarılar',
  noWarnings: 'Uyarı yok — yapılandırma tutarlı.',
  warnAdcRamp: 'ADC penceresi {adc} µs’de bitiyor ama rampa {ramp} µs’de bitiyor: örnekler rampa dışında alınıyor.',
  warnRampTc: 'Rampa süresi chirp periyodu Tc’den uzun.',
  warnFrame: 'Çerçeve aktif süresi {act} ms, çerçeve periyodunu aşıyor.',
  warnPow2: 'N ve M ikinin kuvveti olmalı.',
  warnRmax: '{name}: R = {r} m, R_max’ın ötesinde — katlanır / kesilir.',
  warnVel: '{name}: v_r = {v} m/s, ±v_max’ı aşıyor — Doppler örtüşmesi.',
  warnMicro: '{name}: mikro-Doppler parçaları {v} m/s’ye ulaşıyor > v_max — spektrogram örtüşmesi.',
  warnFov: '{name}: belirsizliksiz açı görüş alanının dışında.',
  addTarget: 'Hedef ekle',
  remove: 'Kaldır',
  type: 'Tür',
  point: 'Nokta',
  car: 'Araba',
  pedestrian: 'Yaya',
  drone: 'Drone',
  bird: 'Kuş',
  rcs: 'RKA [dBsm]',
  rpm: 'Rotor dev/dk',
  blade: 'Pervane [m]',
  flap: 'Kanat çırpma [Hz]',
  bandwidth: 'Bant genişliği B_eff [MHz]',
  fsL: 'Örnekleme hızı f_s [MHz]',
  nL: 'Chirp başına örnek N',
  tcL: 'Chirp periyodu T_c [µs]',
  rampL: 'Rampa süresi [µs]',
  mL: 'TX başına chirp M',
  ntxL: 'TX anten',
  nrxL: 'RX anten',
  framePL: 'Çerçeve periyodu [ms]',
  fcL: 'Taşıyıcı f_c [GHz]',
  snrL: 'Ref. menzilde SNR [dB]',
  refRL: 'Referans menzil [m]',
  noiseL: 'Gürültü',
  mtiL: 'MTI (durağan kargaşa giderme)',
  clutterL: 'Durağan kargaşa',
  rwinL: 'Menzil penceresi',
  dwinL: 'Doppler penceresi',
  awinL: 'Menzil–açı haritası penceresi',
  cfarL: 'CFAR türü',
  pfaL: 'P_fa',
  guardL: 'Koruma (menzil, Doppler)',
  trainL: 'Eğitim (menzil, Doppler)',
  osRankL: 'OS sırası',
  cmapL: 'Renk haritası',
  dragHint: 'Hedefleri kuşbakışı görünümde sürükleyin. IF ve mikro-Doppler panelleri için bir hedefe tıklayarak seçin.',
  learnStep: 'Adım {i} / {n}',
  prev: 'Geri',
  next: 'İleri',
  close: 'Kapat',
  detsN: '{n} tespit',
  microParts: 'parçaların azami |v_r| ≈ {v} m/s',
};

const dicts: Record<Lang, Dict> = { en, tr };
let lang: Lang = 'en';

export function setLang(l: Lang): void {
  lang = l;
  document.documentElement.lang = l;
}
export function getLang(): Lang {
  return lang;
}
export function t(key: string, args?: Record<string, string | number>): string {
  let s = dicts[lang][key] ?? dicts.en[key] ?? key;
  if (args) for (const [k, v] of Object.entries(args)) s = s.split(`{${k}}`).join(String(v));
  return s;
}
export function applyI18n(root: ParentNode = document): void {
  root.querySelectorAll<HTMLElement>('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n!);
  });
  root.querySelectorAll<HTMLElement>('[data-i18n-title]').forEach((el) => {
    el.title = t(el.dataset.i18nTitle!);
  });
}

// ---------------------------------------------------------------- Learn mode
export interface LearnStep {
  panel: string; // panel id
  title: Record<Lang, string>;
  body: Record<Lang, string>; // HTML
}

export const LEARN: LearnStep[] = [
  {
    panel: 'chirp',
    title: { en: '1 · Chirp and beat (IF) signal', tr: '1 · Chirp ve vuru (IF) sinyali' },
    body: {
      en: `<p>The transmitter sweeps a linear frequency ramp (a <b>chirp</b>) of slope <i>S</i>. The echo from a target at range <i>R</i> returns after the delay</p>
<div class="eq">τ = 2R / c</div>
<p>Mixing TX with RX yields the <b>beat</b> or intermediate-frequency signal, a tone at</p>
<div class="eq">f<sub>b</sub> = S·τ = 2SR / c</div>
<p>RadarLab synthesises the complex IF sample of every virtual channel <i>k</i>, chirp <i>m</i> and sample <i>n</i> exactly:</p>
<div class="eq">x[k,m,n] = A·exp( j2π(f<sub>c</sub>τ + Sτt<sub>f</sub> − ½Sτ²) + j2π·p<sub>k</sub> sinθ )</div>
<p>with intra-chirp motion τ(t), the residual-video-phase term −½Sτ² and TDM timing t<sub>m,t</sub> = m·T<sub>e</sub> + t·T<sub>c</sub>. Only the part of the ramp inside the <b>ADC window</b> (N/f<sub>s</sub>) is sampled; its swept bandwidth B<sub>eff</sub> = S·N/f<sub>s</sub> sets the range resolution.</p>`,
      tr: `<p>Verici, eğimi <i>S</i> olan doğrusal bir frekans rampası (<b>chirp</b>) yayar. <i>R</i> menzilindeki hedeften yankı şu gecikmeyle döner:</p>
<div class="eq">τ = 2R / c</div>
<p>TX ile RX karıştırılınca <b>vuru</b> (ara frekans, IF) sinyali elde edilir; frekansı</p>
<div class="eq">f<sub>b</sub> = S·τ = 2SR / c</div>
<p>RadarLab her sanal kanal <i>k</i>, chirp <i>m</i> ve örnek <i>n</i> için karmaşık IF örneğini tam olarak üretir:</p>
<div class="eq">x[k,m,n] = A·exp( j2π(f<sub>c</sub>τ + Sτt<sub>f</sub> − ½Sτ²) + j2π·p<sub>k</sub> sinθ )</div>
<p>Burada chirp içi hareket τ(t), artık video fazı terimi −½Sτ² ve TDM zamanlaması t<sub>m,t</sub> = m·T<sub>e</sub> + t·T<sub>c</sub> hesaba katılır. Rampanın yalnızca <b>ADC penceresi</b> (N/f<sub>s</sub>) içindeki kısmı örneklenir; taranan B<sub>eff</sub> = S·N/f<sub>s</sub> bant genişliği menzil çözünürlüğünü belirler.</p>`,
    },
  },
  {
    panel: 'range',
    title: { en: '2 · Range FFT', tr: '2 · Menzil FFT’si' },
    body: {
      en: `<p>An FFT over the N fast-time samples of each chirp turns each beat tone into a peak. Bin <i>r</i> corresponds to</p>
<div class="eq">R = r·ΔR,&nbsp;&nbsp; ΔR = c / (2B<sub>eff</sub>),&nbsp;&nbsp; R<sub>max</sub> = f<sub>s</sub>c / (2S)</div>
<p>A periodic <b>Hann window</b> (normalised by Σw) lowers the sidelobes from −13 dB to −31 dB at the cost of a wider main lobe. Dashed markers show the true ranges; with MTI off, static clutter appears here too.</p>`,
      tr: `<p>Her chirp’in N hızlı-zaman örneği üzerindeki FFT, her vuru tonunu bir tepeye dönüştürür. <i>r</i> hücresi şu menzile karşılık gelir:</p>
<div class="eq">R = r·ΔR,&nbsp;&nbsp; ΔR = c / (2B<sub>eff</sub>),&nbsp;&nbsp; R<sub>max</sub> = f<sub>s</sub>c / (2S)</div>
<p>Periyodik <b>Hann penceresi</b> (Σw ile normalize) yan lobları −13 dB’den −31 dB’ye indirir; bedeli daha geniş ana lobdur. Kesikli işaretler gerçek menzilleri gösterir; MTI kapalıyken durağan kargaşa da burada görünür.</p>`,
    },
  },
  {
    panel: 'rd',
    title: { en: '3 · Doppler FFT and the range–Doppler map', tr: '3 · Doppler FFT’si ve menzil–Doppler haritası' },
    body: {
      en: `<p>A moving target changes its phase from chirp to chirp by 4π·v·T<sub>e</sub>/λ. A second FFT across the M chirps (slow time) measures this rate:</p>
<div class="eq">v = (d − M/2)·Δv,&nbsp;&nbsp; Δv = λ / (2M·T<sub>e</sub>),&nbsp;&nbsp; v<sub>max</sub> = λ / (4T<sub>e</sub>)</div>
<p>T<sub>e</sub> = n<sub>TX</sub>·T<sub>c</sub> is the repetition time of one transmitter. Positive velocity = receding. Faster targets <b>alias</b> (wrap around). MTI subtracts the slow-time mean, removing zero-Doppler clutter. The map shows P[r,d] = Σ<sub>k</sub>|X<sub>k</sub>[r,d]|²; circles are CFAR detections, + are ground truth.</p>`,
      tr: `<p>Hareketli bir hedefin fazı chirp’ten chirp’e 4π·v·T<sub>e</sub>/λ kadar değişir. M chirp boyunca (yavaş zaman) ikinci bir FFT bu hızı ölçer:</p>
<div class="eq">v = (d − M/2)·Δv,&nbsp;&nbsp; Δv = λ / (2M·T<sub>e</sub>),&nbsp;&nbsp; v<sub>max</sub> = λ / (4T<sub>e</sub>)</div>
<p>T<sub>e</sub> = n<sub>TX</sub>·T<sub>c</sub> bir vericinin tekrar süresidir. Pozitif hız = uzaklaşan. Daha hızlı hedefler <b>örtüşür</b> (sarar). MTI yavaş-zaman ortalamasını çıkararak sıfır-Doppler kargaşasını siler. Harita P[r,d] = Σ<sub>k</sub>|X<sub>k</sub>[r,d]|² gösterir; daireler CFAR tespitleri, + işaretleri gerçek konumlardır.</p>`,
    },
  },
  {
    panel: 'cut',
    title: { en: '4 · CFAR detection', tr: '4 · CFAR tespiti' },
    body: {
      en: `<p>A <b>constant false-alarm rate</b> detector estimates the local noise from N training cells around each cell under test (guard cells excluded) and declares a detection if P &gt; α·noise:</p>
<div class="eq">CA: α = N·(P<sub>fa</sub><sup>−1/N</sup> − 1)</div>
<div class="eq">OS: P<sub>fa</sub> = Π<sub>i=0..k−1</sub> (N−i)/(N−i+α),&nbsp; k = round(rank·N)</div>
<p>GO/SO take the greater/smaller mean of the leading and lagging halves. This cut shows power along range at one Doppler bin with all four thresholds: CA is masked by a strong neighbour, OS is robust to it. Summing K channels makes the noise Gamma(K), so the formulas above (valid for K = 1) are replaced by the exact Gamma(K) values: CA in closed form (Beta distribution), OS/GO/SO by numerical integration. All four thresholds therefore hold the same design P<sub>fa</sub>, and their differences in this cut come only from how each one estimates the noise. The detection SNR is P divided by the noise estimate of the active type (threshold/α).</p>`,
      tr: `<p><b>Sabit yanlış alarm oranlı</b> (CFAR) dedektör, test edilen her hücrenin çevresindeki N eğitim hücresinden (koruma hücreleri hariç) yerel gürültüyü kestirir ve P &gt; α·gürültü ise tespit bildirir:</p>
<div class="eq">CA: α = N·(P<sub>fa</sub><sup>−1/N</sup> − 1)</div>
<div class="eq">OS: P<sub>fa</sub> = Π<sub>i=0..k−1</sub> (N−i)/(N−i+α),&nbsp; k = round(rank·N)</div>
<p>GO/SO, öncü ve artçı yarıların büyük/küçük ortalamasını alır. Bu kesit bir Doppler hücresinde menzil boyunca gücü ve dört eşiği gösterir: CA güçlü bir komşu tarafından maskelenir, OS buna dayanıklıdır. K kanal toplanınca gürültü Gamma(K) olur; bu yüzden yukarıdaki formüller (K = 1 için geçerli) yerine tam Gamma(K) değerleri kullanılır: CA kapalı biçimde (Beta dağılımı), OS/GO/SO sayısal integralle. Böylece dört eşiğin hepsi aynı tasarım P<sub>fa</sub> değerini tutar; bu kesitteki farklar yalnızca her birinin gürültüyü nasıl kestirdiğinden gelir. Tespit SNR’si, P’nin etkin türün gürültü kestirimine (eşik/α) bölümüdür.</p>`,
    },
  },
  {
    panel: 'ra',
    title: { en: '5 · MIMO angle estimation', tr: '5 · MIMO açı kestirimi' },
    body: {
      en: `<p>n<sub>TX</sub> transmitters (time-multiplexed) and n<sub>RX</sub> receivers form a <b>virtual array</b> of K = n<sub>TX</sub>·n<sub>RX</sub> elements at p<sub>k</sub> = p<sub>t</sub> + p<sub>r</sub>. A target at azimuth θ produces a linear phase 2π·p<sub>k</sub>·sinθ across it. A zero-padded 64-point FFT gives</p>
<div class="eq">sinθ = (a − 32) / (64·d)</div>
<p>Because TX chirps are sent at different times, a moving target adds a phase 2π(2v/λ)·t·T<sub>c</sub> per TX; it is removed (<b>TDM compensation</b>) using the estimated velocity before the angle FFT. Resolution ≈ 2/K rad for d = λ/2.</p>`,
      tr: `<p>Zaman paylaşımlı n<sub>TX</sub> verici ve n<sub>RX</sub> alıcı, p<sub>k</sub> = p<sub>t</sub> + p<sub>r</sub> konumlarında K = n<sub>TX</sub>·n<sub>RX</sub> elemanlı bir <b>sanal dizi</b> oluşturur. θ azimutundaki bir hedef dizide 2π·p<sub>k</sub>·sinθ doğrusal fazı üretir. Sıfır dolgulu 64 noktalı FFT:</p>
<div class="eq">sinθ = (a − 32) / (64·d)</div>
<p>TX chirp’leri farklı zamanlarda gönderildiğinden hareketli hedef her TX için 2π(2v/λ)·t·T<sub>c</sub> ek faz üretir; açı FFT’sinden önce kestirilen hızla giderilir (<b>TDM telafisi</b>). d = λ/2 için çözünürlük ≈ 2/K rad.</p>`,
    },
  },
  {
    panel: 'bev',
    title: { en: '6 · Point cloud and tracking', tr: '6 · Nokta bulutu ve izleme' },
    body: {
      en: `<p>Each detection (R, θ, v<sub>r</sub>) becomes a point x = R·sinθ, y = R·cosθ. Points of one object are clustered, then an <b>extended Kalman filter</b> with a constant-velocity model tracks state [x, y, v<sub>x</sub>, v<sub>y</sub>] from polar measurements:</p>
<div class="eq">h(s) = [ √(x²+y²),&nbsp; atan2(x, y),&nbsp; (x·v<sub>x</sub> + y·v<sub>y</sub>)/R ]</div>
<p>Global-nearest-neighbour assignment (Hungarian algorithm) uses a χ² gate of 9.21 (2 DOF, 99 %). Tracks are confirmed after 3 hits and deleted after consecutive misses. Drag the targets to see the tracker follow.</p>`,
      tr: `<p>Her tespit (R, θ, v<sub>r</sub>) x = R·sinθ, y = R·cosθ noktasına dönüşür. Bir nesnenin noktaları kümelenir, ardından sabit hız modelli <b>genişletilmiş Kalman filtresi</b> kutupsal ölçümlerden [x, y, v<sub>x</sub>, v<sub>y</sub>] durumunu izler:</p>
<div class="eq">h(s) = [ √(x²+y²),&nbsp; atan2(x, y),&nbsp; (x·v<sub>x</sub> + y·v<sub>y</sub>)/R ]</div>
<p>Küresel en yakın komşu ataması (Macar algoritması) 9.21’lik χ² kapısı (2 SD, %99) kullanır. İzler 3 isabetten sonra onaylanır, art arda kaçırmalardan sonra silinir. İzleyicinin takibini görmek için hedefleri sürükleyin.</p>`,
    },
  },
  {
    panel: 'micro',
    title: { en: '7 · Micro-Doppler', tr: '7 · Mikro-Doppler' },
    body: {
      en: `<p>Rotating or vibrating parts — drone rotor blades, bird wings, swinging limbs, car wheels — add time-varying Doppler on top of the bulk motion. A short-time Fourier transform of the slow-time signal at the target’s range bin reveals them:</p>
<div class="eq">v<sub>tip</sub> = 2π·(rpm/60)·L</div>
<p>A drone at 6000 rpm with 12 cm blades has tips at ±75 m/s — far beyond the automotive v<sub>max</sub>, which is why counter-UAS radars use fast chirps (small T<sub>e</sub>). Compare the drone and bird signatures in the micro-Doppler preset.</p>`,
      tr: `<p>Dönen veya titreşen parçalar — drone pervaneleri, kuş kanatları, sallanan uzuvlar, araba tekerlekleri — gövde hareketinin üstüne zamanla değişen Doppler ekler. Hedefin menzil hücresindeki yavaş-zaman sinyalinin kısa zamanlı Fourier dönüşümü bunları gösterir:</p>
<div class="eq">v<sub>uç</sub> = 2π·(dev/dk / 60)·L</div>
<p>6000 dev/dk ve 12 cm pervaneli bir drone’un uçları ±75 m/s hızdadır — otomotiv v<sub>max</sub>’ının çok ötesinde; bu yüzden İHA karşıtı radarlar hızlı chirp (küçük T<sub>e</sub>) kullanır. Mikro-Doppler hazır ayarında drone ve kuş imzalarını karşılaştırın.</p>`,
    },
  },
  {
    panel: 'budget',
    title: { en: '8 · Parameter budget', tr: '8 · Parametre bütçesi' },
    body: {
      en: `<p>Every waveform choice is a trade-off: more bandwidth → finer ΔR but smaller R<sub>max</sub> at fixed f<sub>s</sub>; more TX antennas → better angle resolution but longer T<sub>e</sub> → smaller v<sub>max</sub>; more chirps → finer Δv but a longer frame. Edit the radar parameters in the side panel and watch the budget and warnings update live. The timing bars show the cost of each processing stage in the Web Worker.</p>`,
      tr: `<p>Her dalga biçimi seçimi bir ödünleşimdir: daha fazla bant genişliği → daha ince ΔR ama sabit f<sub>s</sub>’de daha küçük R<sub>max</sub>; daha fazla TX anteni → daha iyi açı çözünürlüğü ama daha uzun T<sub>e</sub> → daha küçük v<sub>max</sub>; daha çok chirp → daha ince Δv ama daha uzun çerçeve. Yan paneldeki radar parametrelerini değiştirin; bütçe ve uyarılar anında güncellenir. Süre çubukları Web Worker’daki her işleme aşamasının maliyetini gösterir.</p>`,
    },
  },
];
