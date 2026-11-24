# RadarLab — Literature Review

*Scope:* the background a research-grade FMCW TDM-MIMO radar simulator must rest on, and the sources a paper about it must cite. Each section ends with **Implications for RadarLab**. BibTeX keys in `[brackets]` refer to `references.bib`. Inline URLs point to the page actually consulted (Nov 2026). Where a detail could not be checked against a primary source, the text says so and the bib entry carries `note = {unverified ...}`.

*Status of verification:* all closed-form detection formulas quoted in §4 and in `fidelity_requirements.md` were checked by Monte-Carlo simulation (scripts: Swerling 0/1/2/3 with K channels, Gamma-exact CA-CFAR, GO/SO/OS-CFAR, phase-noise synthesis normalisation). Agreement was within MC error (≤ 0.5 % relative at 4·10⁵–2·10⁶ trials).

---

## 1. FMCW / MIMO radar signal model and processing fundamentals

### 1.1 Tutorials and textbooks
- **Patole, Torlak, Wang & Ali (2017)** [patole2017automotive], *IEEE Signal Processing Magazine* 34(2):22–35, DOI 10.1109/MSP.2016.2628914 (metadata: https://searchworks.jioinstitute.edu.in/articles/edsair__edsair.doi...........1f5727b55939701858c1047a529442db). The standard citation for the automotive FMCW chain: chirp-sequence waveform, 2-D FFT range–Doppler processing, CFAR, DoA, and the waveform trade-offs (ΔR = c/2B, v_max, R_max).
- **Engels et al. (2017)** [engels2017advances], *IEEE SPM* 34(2):36–46 (https://researchr.org/publication/EngelsHZJW17/bibtex). The same special issue; covers high-resolution frequency estimation in automotive radar.
- **Sun, Petropulu & Poor (2020)** [sun2020mimo], *IEEE SPM* 37(4):98–117, DOI 10.1109/MSP.2020.2978507 (https://researchr.org/publication/SunPP20/bibtex). A MIMO automotive tutorial covering virtual arrays, TDM/DDM/BPM multiplexing and sparse arrays.
- **Engels et al. (2021)** [engels2021automotive], *IEEE JSTSP* 15(4) (https://signalprocessingsociety.org/publications-resources/ieee-journal-selected-topics-signal-processing/2021/06/automotive-radar-signal-processing-research-directions-and-practical-challenges). Covers practical challenges, including interference and the limits of the idealised models.
- **Winkler (2007)** [winkler2007range], "Range Doppler detection for automotive FMCW radars", EuRAD 2007. This is the classic chirp-sequence (fast-chirp) range–Doppler paper. The bibliographic record was located (scilit), but page and DOI details could not be fetched, so they are marked unverified.
- **Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill 2014** [richards2014fundamentals] (ISBN 978-0-07-179832-7, https://booksrun.com/9780071798327-fundamentals-of-radar-signal-processing-second-edition-mcgraw-hill-professional-engineering). Reference for the detection theory (Swerling models, non-coherent integration, Albersheim/Shnidman) and CFAR analysis used in §4.
- **TI app notes.** "The fundamentals of millimeter wave sensors" (Iovescu & Rao, SPYY005, https://www.tij.co.jp/jp/lit/SPYY005) [ti_spyy005]. "MIMO Radar", SWRA554A, S. Rao, May 2017 rev. Jul 2018 (https://ti.com/lit/swra554) [ti_swra554]. SWRA554A states that in TDM-MIMO "a Doppler correction step must be performed in order to correct for any velocity induced phase change" before the angle FFT. It does **not** give a closed-form correction; that comes from Bechter et al. (§5.3).

### 1.2 Signal model adopted in RadarLab (cross-check against literature)
Transmit phase φ_T(t) = 2π(f_c t + ½ S t²). Mixing Tx with the conjugate of the delayed echo gives the beat phase 2π(f_c τ + S τ t − ½ S τ²). This is exactly the model in `PROJECT_BRIEF.md`, including the residual video phase (RVP) term −½Sτ². It is the model used in the TI white paper and in Patole et al. With a linear R(t) inside the chirp, the f_c·τ(t) term produces the Doppler shift (2v/λ, a phase rotation from chirp to chirp). The S·τ(t)·t term produces (i) the intra-chirp range–Doppler coupling, a beat-frequency offset of 2v/λ, i.e. a range bias ΔR_bias = c·(2v/λ)/(2S) = v·f_c/S (≈ 0.05 m at 77 GHz, 30 MHz/µs, 20 m/s); and (ii) range migration across chirps of v·M·T_e. Both come out automatically from the exact model and must *not* be removed in golden-vector expectations.

**Spatial-phase sign (flag for engineers).** In the far field, element k at position x_k (in λ) sees a path that is shorter by x_k sinθ cosφ. This gives τ_k = τ − (x_t + x_r) λ sinθ cosφ / c, so the physically consistent spatial term is **−2π·pos_k·sinθ cosφ** under the brief's own mixing convention (the convention that makes receding targets have positive Doppler). The brief writes **+**. That is a valid *display convention*: it makes θ>0 map to a positive angle bin under a forward FFT. But it is not what an exact per-channel-delay (near-field) implementation would produce. Pick one of two fixes: (a) keep + and document it as "equivalent to an inverse spatial FFT"; or (b) use − and flip the angle axis. Either way, real TI data must be calibrated with a corner reflector at a known azimuth (§10), because TI board antenna ordering and the IQ-swap setting also flip signs.

### 1.3 Key derived quantities (match the brief)
ΔR = c/(2B_eff); R_max = f_s c/(2S) (complex IF); Δv = λ/(2 M T_e); v_max = λ/(4 T_e) with T_e = n_Tx T_c. The v_max reduction by n_Tx in TDM is confirmed by Roos et al. 2018 (https://mwt-www.e-technik.uni-ulm.de/world/data/downloads/papers/2018/2018_ICMIM_Roos_Enhancement_Doppler_Unambiguioty_Chirp_Sequence_Modulated_TDM_MIMO_Radars.pdf): "f_D,max = ±1/(2·M·T_r)" for M transmitters [roos2018enhancement].

**Implications for RadarLab.** The brief's beat model is the literature-standard model. Fix or document the spatial-phase sign. Cite Patole 2017 and Sun 2020 as the tutorial anchors, and Richards for detection theory.

---

## 2. Radar simulators — state of the art

### 2.1 Taxonomy
Holder et al. (2018) [holder2018measurements] (https://tuprints.ulb.tu-darmstadt.de/id/eprint/9277) identify three hard problems for radar sensor models, all shown with measurements: multipath, target separability, and aspect-dependent RCS. The survey by Magosi et al. (2022) [magosi2022survey], *Sensors* 22(15):5693, DOI 10.3390/s22155693 (https://www.mdpi.com/1424-8220/22/15/5693), uses Holder's **ideal / phenomenological / physical** model classes and the detection-level vs raw-signal-level output distinction. It reports that quantitative validation metrics are rarely standardised. Ngo et al. compare clustering outputs on real vs synthetic data. RadarLab is a **physical, raw-signal-level** model with a **point/extended-scatterer scene** (no ray tracer). Its novelty is therefore not scene realism but *signal-chain fidelity + verifiability* (see `paper_positioning.md`).

### 2.2 Tools

| Tool | Licence / access | Level | Waveforms / MIMO | Impairments | Scene model | Validation published | Interactive/browser |
|---|---|---|---|---|---|---|---|
| **RadarSimPy** [radarsimpy] (https://github.com/rookiepeng/radarsimpy; Zenodo DOI 10.5281/zenodo.6792269) | Repo states GPL-3.0; the runtime module is distributed as **pre-built binaries** downloaded from radarsimx.com. The relation of the binaries to the repo source is not stated on the page | Raw baseband | Arbitrary (CW, FMCW, PMCW, pulse); TDM/CDM/FDM/DDM phase/amplitude modulation; TDM-MIMO FMCW example | Tx phase noise, mutual interference | Point targets + 3-D mesh ray tracing (ray density, fidelity level) | No validation paper found; no citation info on GitHub | No (Python) |
| **MATLAB Radar Toolbox** [mathworks_radar] (https://uk.mathworks.com/help/radar/automotive-radar.html) | Proprietary | Both: `radarDataGenerator` (statistical detections/tracks incl. multipath), `radarTransceiver` (I/Q waveform level) | FMCW, MFSK, MIMO virtual arrays, 77 GHz patch arrays | Antenna patterns, receiver noise; multipath ghost examples | Point targets, plus examples of pedestrian/bicyclist micro-Doppler | Vendor examples only | MATLAB apps (Radar Designer); not web/zero-install |
| **OpenRadar** [gupta2019openradar] (arXiv:1912.12395, https://ar5iv.labs.arxiv.org/html/1912.12395) | Apache-2.0 | **Processing only** (no simulator) | Range/Doppler FFT, Capon/Bartlett/MUSIC, CFAR | – | – | – | No |
| **mmSim** [cui2023mmsim] (https://github.com/yizzfz/mmSim) | Open (Python) | IF signal | TI array layouts, configurable chirps | Not stated | Points + 3-D models | "No validation against hardware" in README | No |
| **Blender-based FMCW channel sim** [liu2023blender] (arXiv:2307.09226) | Research code | Raw TDM-MIMO beat signal | TDM-MIMO | – | Blender ray tracing | Simulation-only "validation" per abstract | No |
| **Schüßler et al. 2021** SBR ray tracer [schussler2021realistic], *IEEE J. Microwaves* 1(4):962–974 (https://doaj.org/article/6cd5ab720a4e41caa1dc0932f1ebe12e) | Not public | Raw, large MIMO | Arbitrary Tx/Rx | – | SBR, urban scenes | **Yes: vs high-res automotive MIMO measurement** | No |
| **SHENRON** [bansal2023shenron] (https://wcsng.ucsd.edu/shenron/, RA-L 2023) / **C-Shenron** [srivastava2025cshenron] (VTC 2025, https://wcsng.ucsd.edu/c-shenron/) | Open source | MIMO radar data from lidar+camera; C-Shenron outputs range–AoA maps in CARLA | MIMO | – | Lidar point clouds | SHENRON claims sim evaluates algorithms "as effectively as real data"; C-Shenron evaluated in-sim only | No |
| **CARLA built-in radar** [dosovitskiy2017carla] (https://arxiv.org/pdf/1711.03938) | MIT | Ideal/phenomenological point cloud (raycast) | – | – | Game-engine geometry | – | No |
| **sar-sim** [fahnemann2022sarsim] (https://github.com/ims-as-luh/sar-sim; IET Radar 2022, DOI 10.1049/ICP.2023.1281) | Open (Python GUI) | FMCW SAR raw data + backprojection | SAR (not MIMO) | – | Algorithmic scene | – | Desktop GUI (closest "interactive" precedent) |
| **RadarSim (neural fields)** [chen2026radarsim] (arXiv:2605.26328, May 2026) | Research | Learned Doppler-range images | Single-chip | Learned | Neural field from RGB | vs own rig | No |

Observations from the table:
1. No open tool publishes a **systematic statistical validation against closed-form theory** (Pfa/Pd vs Marcum/Gamma, RMSE vs CRLB). Validation in the literature is either against measurements on one scene (Schüßler) or none.
2. No tool offers a **zero-install, in-browser twin** of a raw-signal simulator whose numerical equivalence to the reference is enforced by tests. The closest precedents are desktop GUIs (sar-sim; MATLAB apps).
3. Several tools target learning pipelines (SHENRON, RadarSim). None targets **TI cfg → simulated cube → same DSP chain as real DCA1000 data**, although OpenRadar shows community demand for the processing half.
4. RadarSimPy is the closest feature competitor (phase noise, interference, ray tracing). Its licence/binary model and the absence of a validation paper are the differentiators to state politely and factually.

**Implications for RadarLab.** Do not claim "first FMCW simulator", or "first with impairments" (RadarSimPy has phase noise and interference). Claim instead: (i) a fully-open reference implementation (MIT, pure numpy); (ii) theory-anchored verification as a first-class deliverable; (iii) a browser twin verified by cross-language golden vectors; (iv) a real-data path through the TI cfg/DCA1000 format.

---

## 3. FMCW impairments and how to model them

### 3.1 Phase noise and range correlation
- **Budge & Burt (1993)** [budge1993range], "Range correlation effects in radars", IEEE National Radar Conf., pp. 212–216, DOI 10.1109/NRC.1993.270463 (https://manuscript.isc.ac/Inventory/49/1609423.htm). Origin of the range-correlation argument: the IF phase noise is φ(t) − φ(t−τ).
- **Siddiq, Hobden, Pennock & Watson (2019)** [siddiq2019phase], "Phase noise in FMCW radar systems", *IEEE TAES* 55(1):70–81, DOI 10.1109/TAES.2018.2847999 (https://core.ac.uk/download/161919001.pdf). The IF phase-noise PSD is L_IF(f) = L_Tx(f)·4 sin²(πfτ). There is cancellation for fτ ≤ 1/6, and up to +6 dB at fτ = 1/2. The model is valid for τ below the coherence time. The paper uses a modified-Lorentzian pedestal S(f) = 2L_p/(1+(|f|/(0.5W_p))^k), k = 2–4, and gives a stability bound L_p < 2/(πW_p).
- **Tschapek, Körner, Carlowitz & Vossiek (2022)** [tschapek2022detailed], "Detailed analysis and modeling of phase noise and systematic phase distortions in FMCW radar systems", *IEEE J. Microwaves* 2(4):648–659 (https://doaj.org/article/50e653986ce84482b5570e88f0ea9fa3). Argues that white-Gaussian phase-noise models are too simple, and that systematic (deterministic) phase errors must be separated from random ones. Validated at 24 GHz with fiber-optic delay targets from 38 to 880 m.
- **Melzer, Onic, Starzer & Huemer (2015)** [melzer2015short], *IEEE JSTSP* 9(8):1650–1660, DOI 10.1109/JSTSP.2015.2465298 (https://research.jku.at/en/publications/short-range-leakage-cancelation-in-fmcw-radar-transceivers-using-/). Short-range leakage (bumper/radome reflection) combined with decorrelated phase noise raises the noise floor. They report ~6 dB of sensitivity loss, recovered by their canceller. This is the main reason to model **leakage and phase noise together**.
- **Synthesis method:** frequency-domain shaping of white noise (verified normalisation in `fidelity_requirements.md` §1). Kasdin (1995) [kasdin1995discrete], *Proc. IEEE* 83(5):802–827, DOI 10.1109/5.381848 (https://collaborate.princeton.edu/en/publications/discrete-simulation-of-colored-noise-and-stochastic-processes-and/) is the reference for power-law (1/f^α) components.
- **Typical values:** TI AWR1843 datasheet (SWRS222D, https://www.ti.com/document-viewer/AWR1843/datasheet/GUID-E54FF503-F97F-4440-A9AF-E01B8AD9B96A) gives **−95 dBc/Hz @ 1 MHz (76–77 GHz), −93 dBc/Hz (77–81 GHz)** [ti_awr1843_ds].

### 3.2 ADC quantisation, clipping, IF filtering
The AWR1843 datasheet [ti_awr1843_ds] specifies a **12-bit ADC**, up to 12.5 Msps complex-1x / 25 Msps real/complex-2x, **IF bandwidth up to 10 MHz**, **HPF1 corners 175/235/350/700 kHz and HPF2 350/700/1400/2800 kHz**, IF gain 24 dB range in 2 dB steps (max 48 dB), noise figure 14/15 dB, and **image rejection ratio 30 dB**. The mmWave SDK CLI exposes the same HPF corners via `profileCfg` (https://software-dl.ti.com/ra-processors/esd/MMWAVE-SDK-AOP/03_02_00_06_AOP/exports/mmwave_sdk_user_guide.pdf) [ti_mmwave_sdk_ug]. DCA1000 samples are 16-bit two's complement [ti_swra581]. Standard uniform-quantiser theory (SQNR ≈ 6.02b + 1.76 dB for a full-scale sinusoid) applies; no FMCW-specific modelling paper beyond this is needed.

### 3.3 Tx–Rx leakage
See Melzer 2015 above. Model it as a strong scatterer at a very short range (on-chip coupling: τ ≈ 0; bumper: R ≈ 0.05–0.3 m) with its own decorrelated phase noise φ(t) − φ(t−τ_leak). The HPF then attenuates its low beat frequency.

### 3.4 IQ imbalance
Feger, Wagner & Stelzer (2011) [feger2011wband], *Frequenz* 65(7–8):183–191, DOI 10.1515/freq.2011.026 (https://www.degruyterbrill.com/document/doi/10.1515/freq.2011.026/html), derive the IQ-imbalance signal model for an FMCW radar and suppress the image by >45 dB with predistortion. On a TI chip, IMRR = 30 dB, i.e. the mirror (ghost) target sits at about −30 dBc at the negated beat frequency.

### 3.5 Chirp nonlinearity
Wang, Millar, Parsons, Ma & Orlik (2019) [wang2019range], ICMIM 2019, MERL TR2019-018 (https://www.merl.com/publications/docs/TR2019-018.pdf). Adds a phase term φ(t) to the chirp, so the beat gains φ(t) − φ(t−τ) and is "no longer a single-tone signal". Nonlinearity is modelled as random (phase noise) plus periodic (PLL transient / switching) deviations, often parametrised as polynomial or sinusoidal. The effect grows with range. Their 79 GHz/4 GHz example shows effective-frequency deviation of +0.15 % at the start of the sweep and −2.5 % at the end.

### 3.6 Multipath (two-ray ground reflection)
Kohnert, Vogt & Stolle [kohnert2023height] (arXiv:2312.11148, https://arxiv.org/pdf/2312.11148) give the four-path FMCW transfer function F_p(f) = 1 + 2Γe^{−j2πfΔτ} + Γ²e^{−j2πf·2Δτ}, with Δτ ≈ 2h_S h_T/(c d) and Γ → −1 for grazing incidence on a smooth road. This gives |F_p| ≈ 4 sin²(2π h_S h_T/(λ d)), i.e. deep range-dependent fades. Holder et al. 2018 [holder2018measurements] show measured multipath ghosts.

### 3.7 Mutual interference
- Goppelt, Blöcher & Menzel (2010) [goppelt2010automotive], *Adv. Radio Sci.* 8:55–60, DOI 10.5194/ars-8-55-2010 (https://ars.copernicus.org/articles/8/55/2010/ars-8-55-2010.pdf). Two cases. With **identical (parallel) ramps**, there is a constant IF, which creates a ghost "second line in the range profile", broadened by uncorrelated phase noise. With **crossing ramps**, there is a short V-shaped IF chirp burst that gives a "uniform increase of the noise floor over the whole IF-band". The paper gives no closed-form burst duration.
- Toth et al. [toth2018analytical] (https://www.spsc.tugraz.at/posts/2019/06/01/analytical-investigation-of-non-coherent-mutual-FMCW-radar-interference.html). Model the non-coherent interference as a "chirp burst" with a linear-phase spectrum whose magnitude fluctuates with the burst phase parameters, i.e. it is not white. The IEEE Xplore record is document 8546588; the venue could not be confirmed (marked unverified).
- Burst duration: the interferer's beat frequency sweeps at rate (S_v − S_i) and passes the receiver's IF band [−f_IF, f_IF], so T_burst ≈ 2 f_IF / |S_v − S_i|. *This relation is derived here from the geometry above.* It is consistent with Goppelt's qualitative statement that the burst is short when the IF bandwidth ≪ the sweep bandwidth, but it was not found as an equation in the sources fetched.

**Implications for RadarLab.** Every Phase-2 impairment has a citable, closed-form model with typical TI parameter values (see `fidelity_requirements.md`). The two most consequential choices are: (1) synthesise **one LO phase-noise process per frame shared by all Tx/Rx channels and all scatterers**, then apply φ(t) − φ(t−τ_i) per scatterer, so range correlation and leakage-induced noise come out automatically; (2) use the Toth/Goppelt chirp-burst model for interference, generated in the time domain and band-limited by the same IF LPF.

---

## 4. Detection: CFAR and analytic Pfa/Pd

- **Rohling (1983)** [rohling1983radar], "Radar CFAR thresholding in clutter and multiple target situations", *IEEE TAES* 19:608– (https://ui.adsabs.harvard.edu/abs/1983ITAES..19..608R/abstract). Introduces **OS-CFAR**, "a single value drawn from the ordered statistic", which is robust to multiple targets in the reference window and to clutter edges.
- **GO/SO-CFAR** analytic Pfa for exponential noise: Gandhi & Kassam (1988) [gandhi1988analysis] (*IEEE TAES*; details not fetched → unverified). The closed forms used in `fidelity_requirements.md` §8 were **verified numerically** (n = 8 cells/side: GO 0.01663 vs MC 0.01670; SO 0.06141 vs 0.06168).
- **2-D CFAR for range–Doppler maps:** Kronauge & Rohling (2013) [kronauge2013fast], *IEEE TAES* 49(3):1817–1823, DOI 10.1109/TAES.2013.6558022 (https://manuscript.isc.ac/Inventory/49/35137.htm).
- **Swerling models:** Swerling (1960) [swerling1960probability], *IRE Trans. IT* 6(2):269–308, DOI 10.1109/TIT.1960.1057561. Required-SNR approximation: Shnidman (2002) [shnidman2002determination], *IEEE TAES* 38(3):1059–1064, DOI 10.1109/TAES.2002.1039422 (https://manuscript.isc.ac/Inventory/49/835279.htm). MATLAB's `shnidman` documents its inputs (Swerling 0–4, N pulses, square-law detector; https://www.mathworks.com/help/phased/ref/shnidman.html) and refers to Richards 2005 for the equations.
- **Non-coherent integration over K virtual channels.** The brief's detection statistic P = Σ_k|X_k|² is K-fold non-coherent integration. With unit-variance complex noise it is Gamma(K,1) distributed under H0. Therefore (i) the fixed-threshold Pfa = Q(K, T) (regularised upper incomplete gamma); (ii) CA-CFAR's α must use the **Beta-distribution-exact** formula (derived and MC-verified, `fidelity_requirements.md` §8.2), not N(Pfa^{−1/N} − 1), which is exact only for K = 1. For a steady target, Pd = Q_K(√(2Kχ), √(2T)) (generalised Marcum Q). **Important modelling point:** all K virtual channels see the *same* target fluctuation within a frame, so the right model for a fluctuating target is **Swerling 1 (fully correlated across the K channels)**, not Swerling 2. The closed form for that case (DiFranco & Rubin / Richards style) was MC-verified (e.g. K = 4, χ = 6 dB, Pfa = 10⁻³: 0.5548 theory vs 0.5555 MC).

**Implications for RadarLab.** An ROC figure "simulated vs analytic" for CA/OS/GO/SO, K = 1 and K = 8, SW0/SW1, is cheap to produce and is something no competitor publishes. It should be the centrepiece of the SIU paper.

---

## 5. Direction of arrival and TDM-MIMO motion compensation

### 5.1 Estimators
- **FFT/Bartlett** (Patole 2017; TI SWRA554).
- **MVDR/Capon** [capon1969high], *Proc. IEEE* 57(8):1408–1418, DOI 10.1109/PROC.1969.7278.
- **MUSIC** [schmidt1986multiple], *IEEE TAP* 34(3):276–280, DOI 10.1109/TAP.1986.1143830. For single-snapshot automotive data, spatial smoothing or chirp-snapshots are needed. OpenRadar implements Capon/Bartlett/MUSIC [gupta2019openradar].
- **CRB for DoA:** Stoica & Nehorai (1989) [stoica1989music], *IEEE TASSP* 37(5):720–741, DOI 10.1109/29.17564.

### 5.2 Snapshots
With one frame, the usual snapshots are the M chirps (after range FFT at the target's range bin), or the range–Doppler cell across K channels (one snapshot, which requires smoothing). Document the choice: the CRLB comparison depends on it.

### 5.3 TDM motion-induced phase
Bechter, Roos & Waldschmidt (2017) [bechter2017compensation], *IEEE MWCL* 27(12):1164–1166 (citation verified via Roos et al. 2018 reference list). Each Tx's chirps are taken at their true times; the Doppler DFT is S_Txm(f_D) = Σ_k s_Txm[k]·exp(−j2πf_D(Mk + m)T_r). This is equivalent to the brief's per-Tx phase correction exp(−j2π(2v/λ)·t·T_c) using the estimated v. When |v| > v_max, the residual Doppler ambiguity appears as a Tx-dependent phase ramp e^{j2πξm/M}, which the method of Roos et al. (2018) [roos2018enhancement] resolves.

**Implications for RadarLab.** Add a test in which a target at v near ±v_max with n_Tx = 2–3 shows the angle bias without compensation and its removal with it. Optionally add the Roos ambiguity hypothesis test as a "beyond v_max" feature.

---

## 6. Target tracking for automotive radar

- EKF with measurement vector z = [r, θ, v_r], where v_r = (x v_x + y v_y)/r for a static radar. Bühren & Yang (2007) [buhren2007global], ICASSP 2007, pp. II-313–316 (https://www.iss.uni-stuttgart.de/forschung/publikationen/buehren_icassp2007.pdf), use distance, angle and relative speed (range rate) in an EKF. They show that the common linear range-rate model fails when headings differ, and use a nonlinear range-rate model with ego/target speeds and headings.
- Textbook reference for EKF/IMM, gating and NIS/NEES consistency tests: Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley 2001 [barshalom2001estimation] (details from memory, unverified).
- TI's group tracker documentation (https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/1023/Tracking-radar-targets-with-multiple-reflection-points.pdf) shows the industrial practice for extended targets: many reflection points per target, so tracking happens after clustering.

**Implications for RadarLab.** A constant-velocity EKF in Cartesian coordinates with the polar measurement [r, θ, v_r], plus NEES/NIS consistency plots, is a sufficient and defensible Phase-1 tracker.

---

## 7. Micro-Doppler: drones, birds, pedestrians

### 7.1 Foundations
- **Chen, Li, Ho & Wechsler (2006)** [chen2006micro], "Micro-Doppler effect in radar: phenomenon, model, and simulation study", *IEEE TAES* 42(1):2–21, DOI 10.1109/TAES.2006.1603402 (https://www.scilit.net/publications/f1f0bca0bedeb4d1dc31af169912d5ef). Rotating-blade, vibration and human models.
- **Chen, *The Micro-Doppler Effect in Radar*, 2nd ed., Artech House 2019**, ISBN 978-1-63081-546-2 [chen2019book] (https://www.microwavejournal.com/articles/31892-updated-and-expanded-edition-of-artech-house-classic-now-available).

### 7.2 Rotor-blade model
- **Martin & Mulgrew.** The blade-return model with a sinc envelope (a thin rigid blade of length L rotating at Ω) is attributed to Martin & Mulgrew, *Analysis of the theoretical radar return signal from aircraft propeller blades*, IEEE Int. Radar Conf. 1990 [martin1990analysis]. The 1990 record could not be fetched, so it is marked unverified. The verified companion paper is Martin & Mulgrew, "Analysis of the effects of blade pitch on the radar return signal from rotating aircraft blades", Radar 92, IET, pp. 446–449 (https://manuscript.isc.ac/Inventory/49/279503.htm) [martin1992analysis].
- Modern variants: Costa et al. [costa2025modeling] (arXiv:2504.05168) give the monostatic thin-wire multi-blade model h(t) = Σ_i γ_i e^{j(ω0/c)(−2R_O + L_B cos(Ωt+φ_i)cosψ)}·L_B·sinc[(ω0 L_B/c) cos(Ωt+φ_i) cosψ], with φ_i = φ0 + 2π(i−1)/N_B, extended to bistatic geometry and to multiple propellers with independent ω_p. Westerkam et al. [westerkam2025second] (arXiv:2506.00497) derive the PSD of the point-blade model: comb spacing N_bΩ̄ and bandwidth consistent with 2LΩ/λ. Their example uses L = 21 cm, Ω̄ = 523 s⁻¹, at X-band.

### 7.3 Drone vs bird measurements and datasets
- **Rahman & Robertson (2018)** [rahman2018radar], *Scientific Reports* 8, DOI 10.1038/s41598-018-35880-9 (https://www.nature.com/articles/s41598-018-35880-9). **K-band** FMCW: 24 GHz, 150 MHz, chirp 234.8 µs (CRF 4.25 kHz), +25 dBm, 24.5 dBi horns. **W-band**: 94 GHz, 150 MHz, chirp 80.5 µs (CRF 12.4 kHz); one capture at a 20 µs chirp. Targets: DJI Phantom 3 Standard, DJI Inspire 1, DJI S900; four raptors/owls. Key findings: blade flashes are visible only with fast Doppler sampling (the 20 µs chirp); at lower CRF the propeller return aliases into a spread around the bulk Doppler; HERM lines appear with longer STFT windows; **bird wingbeat 4–6 Hz, wing micro-Doppler 0–10 dB below the body, without the wide drone spread**. Public availability of the data is not stated.
- **Rahman & Robertson (2020)** [rahman2020classification], *IET RSN* 14(5):653–661, DOI 10.1049/iet-rsn.2019.0493. CNN on spectrograms: 94.4 % test accuracy over 4 classes and 98.3 % for drone vs non-drone (custom net); ~99 % with GoogLeNet.
- **Molchanov et al. (2014)** [molchanov2014classification], *Int. J. Microw. Wirel. Technol.* 6(3–4):435–444, DOI 10.1017/S1759078714000282. X-band CW, correlation-matrix eigen-features, ~92 % correct classification.
- **Harmanny, de Wit & Prémel Cabic (2014)** [harmanny2014radar], EuRAD 2014, pp. 165–168, DOI 10.1109/EuRAD.2014.6991233. Spectrogram and cepstrogram features, the latter for blade-rate estimation.
- **Open data.** Karlsson (KTH), *Radar measurements on drones, birds and humans with a 77 GHz FMCW sensor*, Zenodo 2021, DOI 10.5281/zenodo.5511912, CC-BY-4.0 (SAAB SIRS 1600, 160 MHz, PRF 17 kHz, scanning; six drone types, gulls, heron, pigeon, raven, humans; 1.5 GB .mat) [karlsson2021radar]. Kang, Forsten, Semkin & Rangan, *Millimeter Wave 60 GHz Radar Measurements: UAS and Birds*, IEEE DataPort 2024, DOI 10.21227/c8r5-6h79, CC-BY-4.0 (drones + bionic bird, 15 GB .pkl, amplitude and phase) [kang2024millimeter].
- **Drone RCS.** Ezuma, Funderburk & Guvenc [ezuma2019compact] (arXiv:1911.05926). At 25 GHz: DJI Phantom 4 Pro −12.4 dBsm, Inspire 1 Pro −11.1 dBsm, Trimble zx5 −9.6 dBsm. They cite about −13 dBsm for a Phantom 3 Pro at 24 GHz. The RCS fluctuation is best fitted by a GEV distribution (log-normal at 15 GHz for some drones), **not** a Swerling model, so Swerling-1 is only an approximation for drones.

### 7.4 Birds
Gong et al. [gong2023formation] (arXiv:2309.15415) use a wingtip-velocity model v(t) = ωL cos(ωt + φ0), giving f_mD(t) = −(2Lω/λ) sin(ωt + φ0) (here ω is the flapping angular rate). Example values: L = 0.6 m, wingbeat ~7 Hz (gulls, X-band). Combined with Rahman 2018's 4–6 Hz for raptors, a **2–10 Hz wingbeat range** is a reasonable default prior. Note that Rahman reports flapping-gliding alternation, which should be modelled as an on/off flapping envelope.

### 7.5 Pedestrians
- Boulic, Magnenat-Thalmann & Thalmann (1990) [boulic1990global], *The Visual Computer* 6(6):344–358, DOI 10.1007/BF01901021. Empirical "global human walking model" parametrised by relative velocity, used by Chen's book code.
- van Dorp & Groen (2003) [vandorp2003human], *IEE Proc. RSN* 150(5):356–365, DOI 10.1049/ip-rsn:20030568. Use it to estimate walking parameters from radar.
- Abdulatif et al. [abdulatif2017realtime] (arXiv:1711.09175) use Boulic with 16 ellipsoid body segments and the ellipsoid RCS formula σ = πa²b²c² / (a² sin²θ cos²φ + b² sin²θ sin²φ + c² cos²θ)², at 25 GHz with a 1.75 m subject at 1 m/s. They report feet reaching about 4×v_body (~4.5–6 m/s).

**Implications for RadarLab.** (i) At 24 GHz with L = 0.12 m and 100 rev/s, the tip speed is 75 m/s, so f_D,max = 2·75/0.0125 = 12 kHz. That needs a chirp rate ≥ 24 kHz (T_c ≤ ~40 µs) for an unaliased blade spread. The simulator must reproduce Rahman's *aliased* spectra at T_c = 234.8 µs as well as the clean flashes at 20 µs, and this pair makes a good validation figure. (ii) Use the sinc blade model, not point blades, otherwise flashes are absent. (iii) Default bird model: two-segment wing, sinusoidal flapping, 2–10 Hz, with an on/off glide envelope, and body RCS ≫ wing RCS (0–10 dB gap).

---

## 8. Cramér–Rao lower bounds (validation targets)

- **Single complex tone** in complex WGN (Rife & Boorstyn 1974 [rife1974single], *IEEE TIT* 20(5):591–598, DOI 10.1109/TIT.1974.1055282). With unknown amplitude, phase and frequency, N samples, and per-sample SNR ρ = |A|²/σ²: var(ω̂) ≥ 6/(ρ N(N²−1)) rad²/sample². The derivation for general sampling positions gives var ≥ σ²/(2|A|² Σ(n − n̄)²), which reduces to the above.
- Range: R = c f_b/(2S), f_b = ω f_s/(2π), so **var(R̂) ≥ (c f_s/(4πS))² · 6/(ρ N(N²−1))**.
- Velocity, using M chirps with the post-range-FFT SNR ρ_r = ρN (rectangular window): **var(v̂) ≥ (λ/(4π T_e))² · 6/(ρ_r M(M²−1))**.
- Angle, using K virtual elements at positions d_k (in λ) with the post-2D-FFT SNR ρ_rd = ρNM: **var(sin θ̂) ≥ 1/(2ρ_rd (2π)² Σ_k(d_k − d̄)²)**. For a ULA with d = ½ this is 6/(ρ_rd π² K(K²−1)). Then var(θ̂) = var(sin θ̂)/cos²θ. The multi-source DoA CRB is in Stoica & Nehorai 1989 [stoica1989music]; Kay Vol. I (1993) [kay1993fundamentals] is the textbook reference.
- Caveats for the validation study: (a) phase must be treated as unknown; if the golden φ0 = 0 is "known", carrier phase gives a much tighter bound. (b) Windowing costs SNR (Hann processing loss ≈ 1.76 dB), so compare unwindowed ML (zero-padded FFT peak + Newton/parabolic refinement) against the CRLB, and report the windowed estimator separately. (c) The threshold effect at low SNR is expected and should be shown, not hidden. (d) In TDM the slow-time sampling instants differ per Tx, which couples Doppler and angle slightly. Either use single-Tx data for the velocity CRLB or include the per-Tx time offsets in the Fisher information matrix.

---

## 9. Validation methodology used by simulator papers
- **Measurement comparison of one complex scene.** Schüßler et al. 2021 compare simulated and measured high-resolution MIMO radar images of a realistic scenario. Tschapek 2022 compares predicted and measured range correlation using a fiber delay line.
- **Phenomenon reproduction.** Holder et al. 2018 use measured multipath, separability and RCS-aspect effects as test cases.
- **Downstream-task equivalence.** SHENRON: algorithms evaluated on sim vs real data. C-Shenron: driving metrics only.
- **Statistical agreement on point clouds / clusters.** Ngo et al., as summarised by Magosi et al. 2022.
- **Gap.** None of the tools above report Monte-Carlo agreement with detection theory or estimation bounds. RadarLab can contribute a **three-tier validation protocol**: (T1) analytic/theory (Pfa, Pd, CRLB, phase-noise PSD, two-ray fading curve); (T2) cross-implementation (Python ↔ TS golden vectors, max abs error); (T3) sim vs real on the same TI configuration (range profile of a corner reflector, noise floor, phase-noise skirt, micro-Doppler of a fan or drone). Proposed metrics: Pfa ratio and Pd curve deviation; RMSE/CRLB efficiency; KS test on noise-floor distributions; peak-position/SNR error; spectrogram SSIM or envelope correlation for micro-Doppler.

---

## 10. Real-data datasets usable for sim-vs-real

| Dataset | Radar | Raw ADC? | Content | Access |
|---|---|---|---|---|
| **RADIal** [rebut2022raw] (https://github.com/valeoai/RADIal; CVPR 2022, arXiv:2112.10646) | HD radar, 12 Tx × 16 Rx = 192 virtual, 4 chips | **Yes** (per-chip files) | ~2 h, 91 sequences, 8,252 labelled frames (9,550 vehicles) + free space; camera + lidar | Public; licence not stated on README |
| **CARRADA** [ouaknine2021carrada] (arXiv:2005.01456; ICPR 2020, DOI 10.1109/ICPR48806.2021.9413181) | 77 GHz, 2Tx×4Rx, 4 GHz BW, 64 chirps × 256 samples, 0.20 m / 0.42 m/s | RA/RD/RAD tensors (raw ADC not clearly stated) | 30 sequences, 12,666 frames; pedestrian/car/cyclist | Public |
| **RaDICaL** [lim2021radical] (https://publish.illinois.edu/radicaldata/) | TI IWR1443BOOST + DCA1000, 3Tx/4Rx (mostly 2 Tx) | **Yes** | RGB-D + IMU, "hundreds of thousands of frames" | Illinois Data Bank |
| **UW raw ADC** [gao2022raw] (https://ieee-dataport.org/node/10295, DOI 10.21227/xm40-jx59) | TI AWR1843, 2Tx×4Rx | **Yes** (.mat) | ~19,800 frames, 6 road-user classes | IEEE DataPort subscription |
| **ColoRadar** [kramer2022coloradar] (arXiv:2103.04510) | TI AWR1843BOOST + DCA1000; MMWCAS cascade | **Yes** | Indoor/outdoor 3-D, lidar ground truth | Public |
| **KTH 77 GHz drones/birds/humans** [karlsson2021radar] | SAAB SIRS 1600 FMCW | Range-azimuth complex samples | Drones, gulls, humans | Zenodo CC-BY |
| **60 GHz UAS & birds** [kang2024millimeter] | 60 GHz | Amp/phase | Drones + bionic bird | DataPort CC-BY |

**Best first targets for T3 validation:** RaDICaL, ColoRadar or the UW set. They are DCA1000/AWR-family raw ADC, so the TI `.cfg` parser + `.bin` reader can be exercised on public data before the owner records his own captures.

---

## 11. TI DCA1000 raw format and mmWave `.cfg`

**DCA1000 binary (SWRA581B, Oct 2018, https://ti.com/lit/SWRA581) [ti_swra581]:**
- All samples are int16 two's complement.
- **xWR12xx/14xx (4 LVDS lanes, complex):** lane ↔ Rx. For each sample time the file holds I(Rx0..Rx3) then Q(Rx0..Rx3), i.e. 8 int16 words per complex sample-time. Then samples in a chirp, chirps in order, frames in order. Unused lanes are zero-filled.
- **xWR16xx / IWR6843 (2 LVDS lanes, complex):** non-interleaved per Rx inside each chirp (Rx0 block of N samples, then Rx1, …). Within a block, words come in groups of 4: I(n), I(n+1), Q(n), Q(n+1).
- **xWR18xx:** TI E2E staff state that the AWR1843 "ADC data format is same as AWR1642" (2 lanes) (https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/842747/dca1000evm-data-unpacking-format-in-complex-format).
- **IQ swap caveat:** mmWave Studio's IQ-swap option changes the order and SWRA581's description is ambiguous, so verify the order on a capture (corner reflector at known range: the correct order gives a single peak at the positive range bin; the wrong order mirrors it).
- With TDM, chirps from successive Tx are consecutive in the file (chirp index = loop·n_Tx + tx, following `frameCfg` chirp start/end).

**mmWave SDK CLI (v3.2 user guide) [ti_mmwave_sdk_ug]:**
- `profileCfg <id> <startFreq GHz> <idleTime µs> <adcStartTime µs> <rampEndTime µs> <txOutPower> <txPhaseShifter> <freqSlopeConst MHz/µs> <txStartTime µs> <numAdcSamples> <digOutSampleRate ksps> <hpfCornerFreq1 idx> <hpfCornerFreq2 idx> <rxGain dB>`. The guide lists 15 parameters including txCalibEnCfg, which the CLI does not expose; the CLI line has 14 values.
- `chirpCfg <startIdx> <endIdx> <profileId> <startFreqVar Hz> <slopeVar kHz/µs> <idleVar µs> <adcStartVar µs> <txEnableMask>`
- `frameCfg <chirpStart> <chirpEnd> <numLoops> <numFrames> <framePeriod ms> <trigger> <triggerDelay ms>`
- `channelCfg <rxMask> <txMask> <cascading>`; `adcCfg <numADCBits: 2=16-bit> <adcOutputFmt: 1=complex1x, 2=complex2x>`; `adcbufCfg <subFrameIdx> <fmt 0=complex> <sampleSwap> <chanInterleave 1=non-interleaved> <chirpThreshold>`.

**Mapping to RadarLab config:** the brief's fc is the frequency at chirp start t_{m,t}, and the brief's adcStartTime then shifts sampling. So set fc = startFreq·1e9 and adcStartTime = adcStartTime_µs·1e-6, which reproduces TI timing exactly. (Do *not* also move fc to the start of sampling; that would double-count. For λ in the Doppler/angle formulas, the centre of the sampled band, fc + slope·(adcStartTime + N/(2fs)), is the more accurate choice; document which is used.) slope = freqSlopeConst·1e12 Hz/s. fs = digOutSampleRate·1e3. N = numAdcSamples. Tc = idleTime + rampEndTime. M = numLoops. n_Tx = chirpEnd − chirpStart + 1 (if each chirp uses one Tx). framePeriod = ms·1e-3. iqMode = complex for adcOutputFmt 1/2. adcOutputFmt 0 (real) maps to iqMode = real. The precise behaviour of complex2x (image band) was not verified in a primary source; treat it as complex and check on real data.

---

## 12. Summary of what the literature says RadarLab must get right
1. The literature-standard beat model, including RVP and the shared LO phase noise (range correlation). The brief already has the beat model; decide the spatial sign.
2. Detection theory with **Gamma(K)** noise and **SW1-correlated** fluctuations across virtual channels.
3. TDM motion compensation (Bechter 2017) and the n_Tx-reduced v_max.
4. A sinc-envelope blade model with realistic K-band CRF aliasing (Rahman 2018).
5. Reproducible validation: theory → golden vectors → TI raw data. This combination is the publishable gap.
