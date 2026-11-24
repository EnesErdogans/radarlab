# RadarLab — Paper Positioning, Candidate Papers and Venues

*Prepared 2026-11-24. Venue facts come from web pages fetched on that date (URLs inline). "Not found" means no authoritative page was found; estimates are labelled as such.*

---

## 1. Gap analysis: what is genuinely new?

**What is *not* new (do not claim it):**
- FMCW/TDM-MIMO raw-signal simulation. RadarSimPy, MATLAB `radarTransceiver`, mmSim, the Blender-based simulator and SHENRON all do it.
- Phase noise and interference simulation (RadarSimPy lists both).
- FFT/CFAR/MUSIC processing (OpenRadar).
- Micro-Doppler blade models (Chen 2006; Martin & Mulgrew).
- Ray-traced scene realism (Schüßler 2021, RadarSimPy). RadarLab deliberately does not compete here.

**What appears new as a combination, and is defensible:**
1. **Theory-anchored verification as a deliverable.** No open radar simulator found publishes Monte-Carlo agreement with detection theory (Pfa/Pd incl. CFAR variants) and estimation bounds (CRLB for R, v, θ). Validation in the literature is either one-scene measurement comparisons (Schüßler 2021, Tschapek 2022), downstream-task equivalence (SHENRON), or nothing (mmSim README; Blender sim). The Magosi et al. 2022 survey notes that validation metrics are not standardised.
2. **Correct statistics for MIMO non-coherent detection.** The Gamma(K)-exact CA-CFAR threshold and the *correlated-across-channels* Swerling-1 Pd are textbook-derivable but are usually ignored in tools and student code. TI-style detection sums |X_k|² over virtual channels and then applies exponential-noise CFAR formulas. A first calculation in this review shows the cost: for K = 4–8, N = 16–32 and a design Pfa of 10⁻⁴, the single-channel α is 4.1–6.1 dB too high (Gamma-exact α, MC-verified). Quantifying this, together with the Hann-window training-cell correlation (−2/3 between adjacent bins), is a small but real and citable result for an SIU-length paper. (Note: to my knowledge, TI's SDK demo CFAR works on log-magnitude data. This was not verified in this review. The comparison should be stated for linear-domain CA-CFAR, with the log-domain variant as a follow-up.)
3. **A zero-install browser twin verified by cross-language golden vectors.** Interactive radar tools exist only as desktop GUIs (sar-sim, IET Radar 2022; MATLAB apps). Cross-implementation equivalence testing (Python ↔ TypeScript) is a reproducibility contribution in its own right, and an educational one (step-by-step chain for students).
4. **A closed real-data loop with TI hardware formats.** The simulator emits DCA1000-format int16 `.bin` files from a TI `.cfg`, and the *same* reader + DSP chain processes simulated files, public raw datasets (RaDICaL, ColoRadar, UW AWR1843 set) and the owner's own captures. Exact round-trip tests plus sim-vs-real comparisons (noise floor, phase-noise skirt, corner reflector, fan/drone micro-Doppler) can be reproduced by anyone who has a $300 EVM.
5. **One framework for automotive (77 GHz) and counter-UAS (24 GHz) scenarios.** This includes aliasing-aware micro-Doppler that reproduces both the aliased K-band spectra and the flash-resolved fast-chirp spectra reported by Rahman & Robertson (2018).

**Honest risks:**
- Reviewers may see items 1–3 as "engineering", so the paper must lead with quantitative validation results, not with features.
- Without Phase 3 real-data results, international radar venues (IET Radar, EuRAD) will likely judge it incremental. Target SIU first.
- Heavy AI assistance in development must be disclosed: JOSS requires an AI-usage disclosure, and IEEE's generative-AI policy applies to SIU (see §4).

---

## 2. Candidate papers

### Paper A — SIU 2027 (4 pages, **in Turkish**, double-blind)
**TR title:** *RadarLab: FMCW MIMO Radar için Teorik Sınırlarla Doğrulanmış Açık Kaynak Simülatör ve Tarayıcı Tabanlı İkizi*
**EN title (for the IEEE Xplore English abstract):** *RadarLab: An Open-Source FMCW MIMO Radar Simulator Verified Against Detection and Estimation Theory, with a Zero-Install Browser Twin*

**Abstract (EN draft, ~150 words):** Radar simulators are widely used to develop and teach FMCW MIMO signal processing, but their statistical fidelity is rarely verified. We present RadarLab, an open-source simulator with a NumPy reference implementation and an interactive TypeScript twin running in the browser, kept numerically equivalent by cross-language golden test vectors. The simulator implements the exact beat-signal model with residual video phase, TDM-MIMO timing with motion-induced phase compensation, CA/OS/GO/SO-CFAR and FFT/MVDR/MUSIC direction finding. We verify it against closed-form theory: false-alarm and detection probabilities for non-coherent integration over K virtual channels (Gamma-exact CFAR thresholds and channel-correlated Swerling-1 targets), and estimator RMSE against Cramér–Rao bounds for range, velocity and angle. We show that applying the single-channel (exponential-noise) CA-CFAR scale factor after non-coherently summing K channels sets the threshold 4–6 dB too high for K = 4–8 (design Pfa = 10⁻⁴, N = 16–32). The realised Pfa drops below 10⁻¹², which costs an equivalent amount of detection SNR. We also quantify the effect of window-induced training-cell correlation. Code and browser demo are publicly available.

**Figures (≤ 4 pages → 3–4 figures):**
1. Signal-chain block diagram + browser screenshot.
2. Pfa vs threshold factor: simulated vs analytic for CA/OS/GO/SO, K = 1 vs K = 8. Show the "naive exponential formula" error and the Hann-correlation effect.
3. Pd vs SNR: SW0/SW1-correlated/SW3 sim vs theory (K = 8).
4. RMSE/CRLB for R, v, θ vs SNR (threshold effect visible).

Table: golden-vector Py↔TS max error; runtime (Python vs browser worker) per frame.

### Paper B — IET Radar 2027 or EuRAD 2027 (5–6 pages, English)
**Title:** *From TI mmWave Configuration to Raw ADC and Back: Hardware-Anchored Validation of an Open FMCW MIMO Simulator for Automotive and Counter-UAS Scenarios*

**Abstract (draft):** We close the loop between an open-source FMCW MIMO simulator and commercial single-chip radar hardware. Starting from a TI mmWave CLI configuration, RadarLab synthesises raw ADC data in the DCA1000 binary format, including LO phase noise with range correlation, IF high-pass/anti-alias filtering, 12-bit quantisation, IQ imbalance, Tx–Rx leakage, two-ray ground multipath and FMCW mutual interference. The same reader and processing chain handle simulated captures, public raw datasets and our AWR-series measurements. We compare simulation and measurement on noise-floor statistics, phase-noise skirts versus range, corner-reflector range/angle accuracy, multipath fading versus distance and the micro-Doppler of rotating blades. We also reproduce the aliasing-dependent drone and bird micro-Doppler signatures reported at K-band. All models are parameterised from datasheet values and validated individually against closed-form predictions.

**Experiments:**
- (E1) DCA1000 round-trip.
- (E2) Noise-floor KS test sim vs real (terminated/empty-room capture).
- (E3) Corner reflector at 1–10 m: range/angle error, peak SNR vs R⁻⁴.
- (E4) Phase-noise skirt vs range: sim vs real vs L·4 sin²(πfτ).
- (E5) Two-ray fading (corner reflector on a tripod, radar at height h, sweep d) vs closed form.
- (E6) Desk fan / small drone micro-Doppler: blade-flash rate and max Doppler vs measured RPM.
- (E7) Simulated drone vs bird spectrograms + a simple classifier trained on sim, tested on public data (KTH 77 GHz Zenodo set, or the 60 GHz DataPort set).

### Paper C — JOSS (750–1,750 words), later SoftwareX / IEEE Access
**Title:** *RadarLab: reproducible FMCW MIMO radar simulation and processing in Python and the browser*
JOSS sections required: Summary, Statement of need, State of the field (with a build-vs-contribute justification vs RadarSimPy/OpenRadar), Software design, Research impact statement, AI usage disclosure (https://joss.readthedocs.io/en/latest/paper.html).

---

## 3. Proposed experiment/figure inventory (all reproducible from `python/examples/`)

| ID | Figure | Needs phase | Used in |
|---|---|---|---|
| F1 | Block diagram + browser UI | 1 | A, B, C |
| F2 | Pfa curves, 4 CFAR types × K ∈ {1, 8} × window ∈ {rect, Hann} | 1 | A |
| F3 | Pd vs SNR, SW0/1/3 correlated, theory overlay | 1–2 | A |
| F4 | RMSE/√CRLB for R, v, θ | 1 | A, B |
| F5 | TDM angle bias vs v/v_max, with/without Bechter compensation | 1 | A or B |
| F6 | Phase-noise residual PSD vs range (theory + sim [+ real]) | 2 (+3) | B |
| F7 | Two-ray fading vs distance (theory + sim [+ real]) | 2 (+3) | B |
| F8 | Interference: burst in fast time + noise-floor rise; parallel-ramp ghost | 2 | B |
| F9 | Drone spectrograms: CRF 4.25 kHz (aliased) vs 50 kHz (flashes); bird wingbeat | 1–2 | B |
| F10 | Sim-vs-real corner reflector + noise floor (KS) | 3 | B |
| F11 | Runtime: Python vs browser (frames/s) vs cube size | 1 | A, C |
| F12 | Golden-vector error histogram Py vs TS | 1 | A, C |

---

## 4. Venue table (facts with sources; dates as published on the pages consulted)

| Venue | Type / indexing | Length / language | Key dates found | Fees | Fit | Source |
|---|---|---|---|---|---|---|
| **IEEE SIU 2027** (35th Sinyal İşleme ve İletişim Uygulamaları Kurultayı) | National conference, IEEE Xplore | 4 pages max, double-blind | **SIU 2027 announcement: not found** (no site or CFP as of 2026-11-24). Pattern: SIU 2026 at Piri Reis Univ., Tuzla/Istanbul, **7–10 Jul 2026**; CFP deadline **13 Mar 2026**, the submission page later shows **29 Mar 2026** (likely an extension); notification 18 May; camera-ready 1 Jun. SIU 2025 at Işık Univ., Şile, 25–28 Jun 2025. **Estimate for SIU 2027: deadline mid/late Mar 2027, conference Jun–Jul 2027** | Registration (not checked) | **High**: radar SP track and a MIMO radar special session existed in 2026 | https://siu2026.pirireis.edu.tr/en/wp-content/uploads/2026/03/SIU_2026_CFP_EN_v1.4-1.pdf ; https://siu2026.pirireis.edu.tr/en/?p=108 ; https://siu2025.isikun.edu.tr/cfp_en.php |
| SIU language rule | — | **"Papers should be written in Turkish. English papers are also reviewed if at least one author's mother tongue is not Turkish or the first author's affiliation is outside Turkey."** Template: A4 two-column IEEE, 10 pt, no math in title/abstract | (SIU 2025 guidelines; assume the same for 2027 and check when published) | — | Enes must write Paper A in Turkish | https://siu2025.isikun.edu.tr/guidelines_en.php |
| **JOSS** | Journal (software paper), DOI via Crossref | 750–1,750 words, Markdown `paper.md` in the repo | Rolling. **Eligibility: repo public > 6 months with sustained development**; evidence of research impact (not aspirational); tests, CI, docs, CONTRIBUTING; **AI usage disclosure required** | **No fees** | High, after 6 months public and some external use | https://joss.readthedocs.io/en/latest/submitting.html ; https://joss.readthedocs.io/en/latest/paper.html |
| **SoftwareX** (Elsevier) | Journal | Short descriptive paper, **3,000-word limit**, plus a code release | Rolling; IF 2.4 (5-yr 3.1) | APC: not found on the page consulted | Medium-high (after JOSS or instead of it) | https://shop.elsevier.com/journals/softwarex/2352-7110 |
| **IEEE Access** | OA journal | No page limit (< 20 pages recommended) | Rolling | **APC USD 2,160**. IEEE member −5 %, member + society −20 %; member discounts do not apply to student members | Medium: needs Phase 3–4 results (sim vs real + validation) | https://ieeeaccess.ieee.org/about/article-processing-charges/ |
| **IEEE RadarConf 2027** | AESS flagship, IEEE Xplore | 2026 edition: 4–6 pages full paper | Aggregator only: **Bangalore, India, ~1 May 2027; submission deadline 31 Oct 2026 (already passed)**; notification 15 Jan 2027. **Official RadarConf'27 CFP not found**; the AESS page found is only a 2024 call for hosting proposals ("April/May timeframe") | — | Missed for 2027; target RadarConf 2028 | https://www.myhuiban.com/conference/2080.md ; https://ieee-aess.org/node/3039 ; https://ieee-aess.org/files/ieeeaess/2025-08/radar26_CFP_475142dbf6.pdf |
| **IET Radar 2027** (Int. Conf. on Radar Systems) | IET DL + **IEEE Xplore**, Scopus/Inspec | Full paper, 5–6 pages recommended | **Glasgow, 25–28 Oct 2027. Full paper deadline 22 Feb 2027**; acceptance 28 May 2027; final paper 1 Sep 2027. Best student paper prize €500 | Registration (not checked) | **High for Paper B** if Phase 3 results exist by Feb 2027 | https://radar.theiet.org/ ; https://radar.theiet.org/call-for-submissions |
| **EuRAD 2027** (within EuMW 2027) | EuMA/IEEE | (4 pages typical; not verified for 2027) | **EuMW 2027: Milan, Italy, 3–8 Oct 2027** (aggregator). **2027 paper deadline: not found**. For reference, the EuMW 2026 (London, 4–9 Oct 2026) paper deadline was **27 Mar 2026** | — | Medium-high for Paper B | https://www.showsbee.com/fairs/95043-European-Microwave-Week-2027.html ; https://www.eumw.eu/ |
| **URSI-Türkiye** Bilimsel Kongresi | National (Turkish) | Not stated | XIII congress held **7–9 Sep 2026, Kastamonu Univ.** (deadline shown as 30 Jun / 20 Jul 2026). Past editions 2010, 2016, 2018, 2021, 2023, 2026 → **next edition not found** (possibly 2028) | — | Low-medium (radar/satellite systems topic) | https://ursitr26.kastamonu.edu.tr/index.php/en/ |
| **ELECO** (EMO, Bursa) | National/International alternating | Not checked | 13th National ELECO 28–30 Nov 2024; 14th International ELECO 27–29 Nov 2025 (both at Bursa). **2026/2027 calls not found** | — | Medium (Turkish EE community) | https://www.emo.org.tr/ekler/0c8bd920cf407ae_ek.pdf |
| "Elektrik-Elektronik-Bilgisayar Sempozyumu" (EEB) | — | — | **Not found** in searches | — | — | — |

---

## 5. Recommendation and timeline

**Primary venue: IEEE SIU 2027, Paper A (Turkish, 4 pages).** The deadline is not yet announced; based on the 2025/2026 pattern, **expect mid/late March 2027 (plan for 1 March 2027 internally)**. Reasons:
- Phase-1 + Phase-4 theory validation alone is enough content.
- The national venue suits a first paper by an undergraduate.
- IEEE Xplore indexing.
- A MIMO-radar special session existed in 2026.
- The required Turkish language is not an obstacle for the owner.

**Parallel actions:**
1. **Make the GitHub repo public now** (with releases, CI, CONTRIBUTING). This starts JOSS's ">6 months public history" clock, so a JOSS submission becomes possible around **June 2027**, once some external use can be shown (course use at AGÜ, DENEYAP/student users, citations from Paper A).
2. **IET Radar 2027 (deadline 22 Feb 2027)** for Paper B, *only if* Phase-3 real-data results (TI EVM + DCA1000) exist by mid-February. Otherwise target EuRAD 2027 (expected deadline ~Feb–Mar 2027, unconfirmed) or RadarConf 2028.
3. **Dual-submission rule:** SIU 2026 forbade submitting a closely resembling paper to another conference during evaluation. Keep Papers A and B clearly distinct: A = theory validation + browser twin; B = hardware loop + impairments + C-UAS.
4. Journal (IEEE Access or SoftwareX) after Phase 3–4, merging A + B with full sim-vs-real results.
5. Write an AI-usage statement now (tools, versions, where used, how humans verified) and reuse it for SIU (IEEE policy), JOSS (mandatory section) and journals.
