# RadarLab — Fidelity Requirements (implementable model specifications)

Audience: the Python (reference) and TypeScript engineers. Conventions follow `docs/PROJECT_BRIEF.md` (SI units; noise variance σ² = 1 per complex sample; beat phase 2π(f_c τ + S τ t_f − ½ S τ²)). Sources are in `review.md` / `references.bib`.

Every equation tagged **[MC✓]** was checked by Monte Carlo or numerical integration in the literature-review sandbox, with agreement to MC error. Values tagged **(assumption)** are engineering defaults that have no primary source in this review. Expose them as parameters and do not present them as facts in the paper.

Contents: 0 conventions · 1 phase noise · 2 Tx–Rx leakage · 3 chirp nonlinearity · 4 IF filters · 5 ADC · 6 IQ imbalance / DC · 7 channel gain/phase errors & antenna patterns · 8 detection theory (Pd/Pfa, CFAR) · 9 Swerling fluctuation · 10 two-ray multipath · 11 interference · 12 micro-Doppler (rotor, bird, pedestrian) · 13 CRLB · 14 distributed clutter · 15 validation test matrix.

---

## 0. Cross-cutting conventions that the models below rely on

**0.1 Order of operations (per chirp, per Rx channel).**

> Σ_scatterers (ideal beat · antenna gains · fluctuation · phase-noise residual · nonlinearity residual) + leakage + interference + thermal noise → IQ imbalance + DC offset → IF HPF → IF LPF (anti-alias) → IF gain → ADC (clip + quantise) → (optional) int16 DCA1000 packing.

Thermal noise is added **before** the IF filters, so the filters shape it the way they do in hardware. In the `'snr'` amplitude mode with all impairments off, the chain must reduce exactly to the brief's ideal model. A regression test must enforce this.

**0.2 Two execution modes.**
- *Tone mode (fast, browser default).* Each scatterer's beat is nearly a tone, so the IF filter is applied as a complex gain H(f_b) evaluated at the scatterer's beat frequency. This is exact for linear filters in steady state.
- *Time-domain mode (reference).* The whole fast-time record is filtered with the actual IIR/FIR, including the HPF start-up transient and its interaction with adcStartTime. It is needed for interference, leakage and nonlinearity.

Golden tests must show the two modes agree to < 0.1 dB when there is no transient (adcStartTime ≥ 5 HPF time constants).

**0.3 FFT noise statistics with the brief's window normalisation.** The brief divides by Σw. After that normalisation, white complex noise of variance σ² has per-cell variance σ²·Σw²/(Σw)² in each FFT dimension:

| Window | Per-cell variance factor | Correlation of adjacent noise cells |
|---|---|---|
| rect | 1/N | 0 |
| periodic Hann | 1.5/N | **−2/3** [MC✓] |

Consequences:
- (a) The post-range-FFT SNR is ρN for rect and ρN/1.5 for Hann (Hann processing loss 1.76 dB per dimension, 3.52 dB for 2-D).
- (b) **With Hann windows, neighbouring training cells are strongly correlated.** CFAR Pfa therefore deviates from the i.i.d. formulas. Analytic CFAR tests must use either a **rect window** or training cells spaced ≥ 2 bins apart. Otherwise document the measured deviation as a known effect, which is itself a nice paper figure.

---

## 1. Phase noise (LO), with range correlation

**PSD model (single-sideband L(f), dBc/Hz), PLL-like with a floor:**

  L(f) = L_pll / (1 + (|f|/f_c)^2) + L_floor   (two-sided phase PSD S_φ(f) = L(f) for |f|>0)

| Parameter | 77 GHz default (AWR1843-like) | 24 GHz default | Source |
|---|---|---|---|
| L_pll | −87 dBc/Hz | −80 dBc/Hz (assumption) | fitted so L(1 MHz) ≈ −95 dBc/Hz, matching TI AWR1843 datasheet: −95 dBc/Hz @1 MHz (76–77 GHz), −93 (77–81 GHz) |
| f_c (loop BW) | 450 kHz | 300 kHz (assumption) | (assumption) |
| L_floor | −140 dBc/Hz | −140 dBc/Hz | (assumption) |
| Optional pedestal | modified Lorentzian 2L_p/(1+(|f|/(0.5W_p))^k), k = 2–4 | | Siddiq et al. 2019 |

**Synthesis (per frame; shared by all Tx, all Rx, all scatterers) [MC✓]:**
1. Choose a simulation rate f_pn ≥ max(4·f_s, 4·f_c) and a grid covering the frame [0, M·T_e + max τ]. N_pn = 2^⌈log2⌉.
2. Draw w[n] ~ N(0,1) real, compute W = FFT(w), and set Φ = W·sqrt(L(|f_k|)·f_pn) with Φ[0] = 0.
3. φ[n] = Re(IFFT(Φ)). Check: var(φ) = Σ_k L(f_k)·f_pn/N_pn = ∫L df. The sandbox check gave 2.79e-3 vs 2.81e-3 rad².
4. For each scatterer i and ADC instant t: φ_res,i(t) = φ(t) − φ(t − τ_i(t)), with linear interpolation of φ on its grid. Multiply the scatterer's beat by exp(j φ_res,i(t)).

**Analytic check (unit test):** var(φ_res) = ∫ L(f)·4 sin²(πfτ) df (Budge & Burt 1993; Siddiq et al. 2019). The sandbox gave 4.751e-3 MC vs 4.763e-3 theory at τ = 0.67 µs. The PSD of φ_res must follow L(f)·4 sin²(πfτ). Test at fτ = ½: +6 dB above L(f).

**Why shared synthesis:** it reproduces (i) range correlation, (ii) the identical phase-noise realisation in all virtual channels (so it does not average out across MIMO channels and appears as correlated "skirts" in angle), and (iii) the leakage-induced noise-floor rise (§2). Optional extension (Tschapek et al. 2022): a *deterministic* systematic phase error per chirp (e.g. a sinusoid at the PLL settling frequency), kept separate from the random part.

**Cost note (TS):** for a 1-frame, 128 × 256 configuration at f_pn = 50 MHz the grid is ~2^20 points per frame: OK in a worker. Provide a cheaper "per-scatterer direct synthesis" option: generate φ_res directly with PSD L(f)·4 sin²(πfτ_i) on the ADC grid. This is exact marginally, but it loses cross-scatterer correlation; flag it as approximate.

---

## 2. Tx→Rx leakage

Model leakage as an additional deterministic "scatterer" per (Tx, Rx) pair with:
- delay τ_L (on-chip coupling: 10–50 ps; bumper/radome reflection: 2R_L/c with R_L = 0.05–0.3 m) (assumption);
- complex amplitude A_L = 10^{(P_L,dBc)/20}·A_ref with a random fixed phase per pair;
- P_L **30–60 dB above the thermal noise per sample** (assumption; hardware dependent; expose as a parameter);
- **the same phase-noise residual** φ(t) − φ(t − τ_L) as §1.

Its beat f_L = Sτ_L lies near DC, so the HPF attenuates it (§4). With τ_L tiny the residual is small, but with radome reflections the noise skirt near DC rises. That is the effect reported by Melzer et al. 2015 (~6 dB sensitivity loss).

Validation: close-in noise floor vs leakage power and τ_L, compared with ∫L(f)·4 sin²(πfτ_L) df·|A_L|².

---

## 3. Chirp (VCO) nonlinearity

Instantaneous frequency f(t) = f_c + S t + δf(t), so the extra Tx phase is φ_nl(t) = 2π∫₀ᵗ δf(u) du. The beat gains φ_nl(t) − φ_nl(t − τ) ≈ 2π δf(t)·τ for small τ. Use the exact difference in the reference implementation.

Two parametric forms (Wang et al. 2019 describe both polynomial and periodic deviations):
- Polynomial: δf(t) = B·[a₂ (2t/T_r − 1)² + a₃ (2t/T_r − 1)³], with |a₂|, |a₃| ~ 1e-4 – 1e-3 (assumption).
- Periodic (PLL ringing / switching): δf(t) = Δf_p e^{−t/τ_s} sin(2π f_p t + ϑ). Defaults: Δf_p = 0.1 % of B, f_p = 1–5 MHz, τ_s = 2 µs (assumption).

Expected effect: range-dependent broadening and spurs; the defocusing grows ∝ τ. Unit test: a peak-to-sidelobe-ratio degradation that increases with range.

---

## 4. IF filtering

- **HPF.** Cascade of two first-order sections with corner indices that match TI `profileCfg` (HPF1 ∈ {175, 235, 350, 700} kHz, HPF2 ∈ {350, 700, 1400, 2800} kHz, AWR1843 datasheet). H_HP(f) = Π_i (j f/f_i)/(1 + j f/f_i). The exact TI filter order is not documented in the sources fetched, so first-order sections are an **(assumption)**. Complex IF: apply the same real-coefficient filter to I and Q.
- **LPF (anti-alias).** Butterworth, order 4 (assumption), corner f_LP = min(IF bandwidth, 0.45·f_s for complex sampling). AWR1843: IF bandwidth up to 10 MHz.
- **Gain.** IF gain in dB (TI: 24 dB range, 2 dB steps, max 48 dB). Gain matters only together with ADC full scale (§5).
- **Time-domain mode:** run the IIR (bilinear-transformed) starting at chirp start t_{m,t}, simulated at ≥ 4·f_s and then decimated. ADC sampling starts at adcStartTime, so the transient is partially captured, as in hardware.

Tests: |H(f)| at the corner indices; tone-mode vs time-domain agreement (§0.2); the near-range dip in the noise floor.

---

## 5. ADC: quantisation and clipping

Parameters: bits b (default 12 per AWR1843; DCA1000 stores int16) and fullScale. Define full scale relative to the thermal-noise rms at the ADC input: `noiseRmsLSB` (default 4 LSB, assumption), i.e. FS = 2^{b−1} LSB.

Per component (I and Q separately):
- y = clip(round(x / LSB), −2^{b−1}, 2^{b−1} − 1)
- LSB = σ_I / noiseRmsLSB, where σ_I = sqrt(σ²/2)·g_IF.

Checks:
- SQNR of a full-scale sinusoid ≈ 6.02 b + 1.76 dB.
- With noiseRmsLSB ≥ 1 the quantisation noise adds ~LSB²/12 per component.
- Clipping produces harmonics/intermod: a near strong target should create spurious peaks at multiples of its beat frequency.

Output option: int16 packed exactly as DCA1000 (§ review 11) so the **real-data reader is round-trip tested on simulated files**. This is a very valuable test.

---

## 6. IQ imbalance and DC offset

Received complex IF x → x_IQ = μ x + ν x* + d, with

  μ = (1 + g e^{−jφ_e})/2, ν = (1 − g e^{+jφ_e})/2, IMRR = |μ|²/|ν|².

Defaults: g = 1.05 (≈0.42 dB), φ_e = 3° → IMRR = 28.9 dB [computed], close to the AWR1843 datasheet IMRR of 30 dB. DC offset d = (d_I + j d_Q), a few LSB (assumption).

Effect: a mirror target at −f_b (negative range bin, i.e. top of the range axis after wrap) at −IMRR dBc. For real-IF mode (iqMode 'real') there is no IQ imbalance; the spectrum is Hermitian.

---

## 7. Antenna patterns and channel errors

- **Element pattern (power).** G(θ, φ) = G₀·cos^{q_az}(θ)·cos^{q_el}(φ) for |θ|, |φ| < 90°, otherwise 0. The two-way amplitude factor is sqrt(G_t G_r). Defaults (assumption): q_az = 1.2, q_el = 4 (patch array narrow in elevation). Alternative: a tabulated measured pattern (CSV) with linear interpolation.
- **Per-channel complex gain error** (unequal Rx/Tx paths; TI compensates with a calibration vector): c_k = (1 + ε_k) e^{jδ_k}, ε ~ N(0, 0.05²), δ ~ N(0, (5°)²) (assumption). Provide a "calibrate" step that estimates c_k from a boresight corner reflector, mirroring TI's procedure. The angle-sidelobe rise vs δ_rms is a clean figure.
- Optional mutual coupling: K×K coupling matrix C, x ← C x.

---

## 8. Detection theory (Pd/Pfa) — the analytic targets for Monte-Carlo validation

Notation: K = number of non-coherently summed channels; P = Σ_k |X_k|² normalised so that the noise per cell has unit variance; χ = per-channel post-integration SNR in the cell; T = threshold.

### 8.1 Fixed threshold
- Noise only: P ~ Gamma(K,1). **Pfa = Q(K, T)** (regularised upper incomplete gamma), so T = Q⁻¹(K, Pfa) (`scipy.special.gammainccinv`). [MC✓]
- **SW0 (steady).** Pd = Q_K(√(2Kχ), √(2T)) = `ncx2.sf(2T, 2K, 2Kχ)`. [MC✓: K = 4, χ = 0 dB: 0.0849 theory vs 0.0849 MC]
- **SW1 correlated across the K channels** (the physically correct MIMO case: one complex amplitude per frame, shared by all virtual channels) [MC✓]:
  - K = 1: Pd = exp(−T/(1+χ)).
  - K ≥ 2: **Pd = 1 − P(K−1, T) + (1 + 1/(Kχ))^{K−1}·P(K−1, T/(1 + 1/(Kχ)))·exp(−T/(1 + Kχ))**, with P = regularised *lower* incomplete gamma (`gammainc`). Checks: K = 4, χ = 6 dB → 0.5548 vs MC 0.5555; K = 8, χ = 12 dB → 0.9061 vs 0.9062. Limit χ→0 gives Pfa (verified analytically).
- **SW2 (independent per channel).** Pd = Q(K, T/(1+χ)) [MC✓]. Use this only if channels decorrelate, e.g. frequency-diverse or very wide arrays.
- **SW3 single channel.** Pd = (1 + 2Tχ/(2+χ)²)·exp(−2T/(2+χ)) [MC✓: χ = 12 dB: 0.7781 vs 0.7785].
- **SW3 correlated, K > 1.** Pd = ∫₀^∞ Q_K(√(2Ks), √(2T)) · Gamma(s; 2, χ/2) ds, by numerical quadrature [MC✓: K = 4, χ = 6 dB: 0.6508 vs 0.6500].
- **Shnidman (2002) approximation** for required SNR (sanity only, not as ground truth). Equations as reproduced in Richards (2005/2014); constants below quoted from memory → **unverified**, test against the exact forms above before use:
  - η = sqrt(−0.8 ln(4Pfa(1−Pfa))) + sgn(Pd−0.5)·sqrt(−0.8 ln(4Pd(1−Pd)))
  - X∞ = η(η + 2 sqrt(N/2 + α − 1/4)), with α = 0 (N < 40) or ¼
  - C₁ = (((17.7006 Pd − 18.4496) Pd + 14.5339) Pd − 3.525)/κ, with κ = ∞, 1, N, 2, 2N for SW0–4
  - C₂ = (1/κ)(e^{27.31 Pd − 25.14} + (Pd − 0.8)(0.7 ln(1e-5/Pfa) + (2N − 20)/80))
  - C_dB = C₁ (0.1 ≤ Pd ≤ 0.872) or C₁ + C₂ (0.872 < Pd ≤ 0.99)
  - SNR₁ = 10^{C_dB/10}·X∞/N

### 8.2 CFAR (cell under test X, N training cells; noise i.i.d.; rect window, see §0.3)
- **CA-CFAR, K = 1:** α = N(Pfa^{−1/N} − 1) (brief). SW1 detection: **Pd = (1 + α/(N(1+χ)))^{−N}** [MC✓ 0.4646 vs 0.4652].
- **CA-CFAR, K > 1 (Gamma-exact) [MC✓].** With threshold α·(mean of training cells), X/(X+Y) ~ Beta(K, NK) where Y = Σ training. Hence
  - **Pfa = 1 − I_u(K, NK), u = c/(1+c), c = α/N**
  - α = N·u*/(1−u*), where u* = I⁻¹(K, NK, 1−Pfa) (`betaincinv`). This reduces to the brief's formula at K = 1. Checks: K = 4, N = 16, Pfa = 1e-3 → α = 3.533 (MC Pfa 1.02e-3); K = 8, N = 24 → α = 2.535.
- **GO / SO-CFAR** (two halves of n cells each, T multiplies the half *sums*) [MC✓ n = 8]:
  - S = Σ_{k=0}^{n−1} C(n−1+k, k)·(2+T)^{−(n+k)}
  - **Pfa_GO = 2(1+T)^{−n} − 2S, Pfa_SO = 2S**
  - Solve T by bisection. With the brief's "mean" convention, T = α/n.
- **OS-CFAR** (k-th order statistic, threshold α·X_(k)): Pfa = Π_{i=0}^{k−1} (N−i)/(N−i+α) (brief) [MC✓: 1.88e-4 vs 2.0e-4 at 4·10⁵ trials]. With K > 1 there is no simple closed form: obtain α by numerical integration of the Gamma order statistic, or by a 10⁷-sample MC table. Document which.
- **Edge cells:** the brief clips the range axis. Recompute α for the actual N at edges; test that Pfa stays flat across range.
- **2-D CFAR:** cross-shaped or rectangular training windows (Kronauge & Rohling 2013). Same formulas with N = total training cells, provided the cells are independent.

### 8.3 Coherent gains to convert per-sample SNR ρ into χ
- χ = ρ·N·M·(window losses) per virtual channel, with K-channel non-coherent integration on top. Window loss as in §0.3.
- For an off-bin target add the scalloping loss: up to 3.92 dB (rect) or 1.42 dB (Hann) per dimension. Validation MC should put targets on bin centres, or average over uniformly random offsets and compare with the averaged theory.

---

## 9. Swerling fluctuation (amplitude per frame)

| Model | Implementation (per scatterer, per frame, shared by all K channels and M chirps) |
|---|---|
| SW0 | a = 1 |
| SW1 | a = CN(0,1), so |a|² ~ Exp(1) |
| SW3 | |a|² ~ Gamma(shape 2, scale ½), phase U(0,2π) |
| SW2/SW4 | Pulse-to-pulse. In chirp-sequence FMCW this destroys Doppler coherence and is rarely physical. Offer only as "per-frame-per-channel" decorrelation (SW2-like across channels) for theory tests. |
| Drone RCS option | GEV or log-normal per Ezuma et al. (best fits at 25 GHz/15 GHz). Mean RCS defaults: Phantom-class −12.4 dBsm, Inspire-class −11.1 dBsm (Ezuma et al., 25 GHz) |

Mean RCS scales as σ̄. The brief's A ∝ sqrt(σ) then multiplies by a.

---

## 10. Two-ray (ground) multipath

**Geometry.** Radar at height h_r above a flat ground plane z = −h_r (radar coordinates). For a target at p = (x, y, z), the image is p' = (x, y, −2h_r − z).
- Direct one-way length R₁ = |p|; ground-bounce one-way length R₂ = |p'|.
- Four round-trip paths per scatterer:

| Path | Delay | Amplitude factor |
|---|---|---|
| D–D | 2R₁/c | 1 |
| D–G | (R₁+R₂)/c | Γ |
| G–D | (R₁+R₂)/c | Γ |
| G–G | 2R₂/c | Γ² |

- Treat each as a pseudo-scatterer with its own Doppler: d/dt of the respective path length / 2.
- Spatial phase: departure direction on the Tx side and arrival direction on the Rx side. Azimuth is identical for both; elevation differs. This matters only for z-offset arrays.

**Reflection coefficient (Fresnel, grazing angle ψ, relative permittivity ε_r):**
- Γ_H = (sinψ − √(ε_r − cos²ψ))/(sinψ + √(ε_r − cos²ψ))
- Γ_V = (ε_r sinψ − √(ε_r − cos²ψ))/(ε_r sinψ + √(ε_r − cos²ψ))
- Optional roughness factor exp(−2(2π σ_h sinψ/λ)²).
- Defaults: ε_r = 4 − 0.1j for asphalt (assumption), σ_h = 0.5 mm (assumption). At grazing incidence Γ → −1, as in Kohnert et al.

**Validation (closed form).** For d ≫ h, the received-power factor |F_p|² with |F_p| ≈ 4 sin²(2π h_r h_t/(λ d)) (Kohnert, Vogt & Stolle). Plot the simulated target peak power vs d against this curve: nulls at d = 2h_r h_t/(nλ). This is a clean figure.

---

## 11. FMCW mutual interference

**Interferer parameters:** f_c,i, S_i, T_c,i, idle, start-time offset t₀,i, P_t,i, G_t,i. Direct one-way path, distance d_i (and direction for spatial phase).
- Received amplitude (one-way radar equation): P_r = P_t,i G_t,i G_r λ²/((4π)² d_i²). In 'physical' mode, A_i = sqrt(P_r/P_n). In 'snr' mode use an INR parameter.
- Victim IF (brief's convention Tx_v·conj(Rx)): x_int(t) = A_i·exp(j[φ_v(t) − φ_i(t − d_i/c) + θ_i]), where φ_v and φ_i are the full chirp-train phases (including each one's own phase noise, *independent* of the victim's).
- Instantaneous IF frequency: f_int(t) = f_v(t) − f_i(t − d_i/c).
- **Implementation:** find intervals where |f_int| < f_LP·(1 + margin). Synthesise only there at an oversampled rate (≥ 8·f_s), apply the IF LPF (§4), decimate to the ADC grid, and add. Outside these intervals the contribution is filtered out.

**Expected behaviours (tests):**
- Crossing ramps (S_v ≠ S_i): a short chirp burst of duration **T_burst ≈ 2 f_LP/|S_v − S_i|** (derived; consistent with Goppelt et al. 2010 and Toth et al.). This raises the noise floor across all range bins of the affected chirps, and the floor is not white (Toth et al.).
- Parallel ramps (S_v = S_i, same timing): a constant IF, i.e. a **ghost target** at range c·Δf/(2S), broadened by the uncorrelated phase noise (Goppelt et al. 2010).

Mitigation baselines to include (optional): zeroing/windowing of detected burst samples.

---

## 12. Micro-Doppler target models

All models produce a set of point scatterers (position, velocity, complex amplitude) **per chirp time t_{m,t}**. The brief's "linear R(t) within a chirp" model then applies. Intra-chirp motion of fast blade tips (75 m/s over 50 µs = 3.7 mm ≈ 0.3λ at 24 GHz) is captured by the per-chirp radial velocity.

### 12.1 Rotor blades (Martin & Mulgrew; Chen 2006/2019)
Rotor with hub at range R₀, N_b blades of length L, rotation rate Ω (rad/s), initial angle φ₀, and angle β between the rotor plane and the radar LOS (β = 0: LOS in the rotor plane). Integrating a uniform line blade from root to tip gives, with the brief's phase convention (+4π r/λ for a radial displacement r away from the radar):

  s_k(t) = A_b · exp(jΦ_k(t)) · sinc(Φ_k(t)),  Φ_k(t) = (2πL/λ)·cosβ·cos(Ωt + φ₀ + 2πk/N_b),  sinc(x) = sin(x)/x

**[numerically verified: the line integral matches the closed form to 1e-9.]** Total rotor return = exp(j4πR₀(t)/λ)·Σ_k s_k(t), plus the body return.
- **Blade flash** when Φ_k = 0, i.e. the blade is ⟂ to the LOS. Flash rate = N_b·Ω/(2π) for even N_b, 2N_b·Ω/(2π) for odd N_b.
- **Max micro-Doppler:** f_mD,max = 2LΩ cosβ/λ.
- Implementation options:
  - (a) *analytic* (above), evaluated at each chirp; range = hub range;
  - (b) *discretised reference*: N_s ≥ ⌈8L cosβ/λ⌉ point scatterers per blade with amplitude A_b/N_s, which also captures the blade's range extent.

  A test asserts (a) ≈ (b) when L ≪ ΔR.
- Multirotor: each rotor has its own Ω_p ~ N(Ω̄, (0.03Ω̄)²) and φ₀ (independent), consistent with the "irregular flash spacing" in Rahman & Robertson 2018. Westerkam et al. use Ω_p ~ N(Ω̄, σ²) likewise.

| Parameter | Default (Phantom-class quadcopter) | Note |
|---|---|---|
| N_rotors, N_b | 4, 2 | |
| L | 0.12 m | (assumption; ~9.4" props) |
| Ω̄ | 2π·90 rad/s (5400 rpm) | (assumption) |
| Body RCS | −12.4 dBsm mean | Ezuma et al. 2019 (25 GHz) |
| Blade RCS (peak, per blade) | 10–20 dB below body | (assumption) |

**Sampling consequence:** at 24 GHz, f_mD,max = 2·0.12·565/0.0125 ≈ 10.9 kHz, so a chirp rate ≥ 22 kHz is needed to avoid aliasing. Rahman & Robertson's K-band CRF of 4.25 kHz *aliases*; their 20 µs W-band chirp shows flashes. The sim must reproduce both regimes. Spectrogram defaults: STFT window 5–20 ms (Rahman: longer windows reveal HERM lines).

### 12.2 Bird (flapping wings)
Body: one scatterer on a trajectory (RCS σ_body). Each wing has 2 segments (inner length L₁, outer L₂) hinged at the shoulder at lateral offset ±w_s.
- Shoulder (flap) angle: ψ₁(t) = ψ_A cos(2π f_w t) + ψ_off
- Elbow/wrist angle relative to the inner segment: ψ₂(t) = ψ_B cos(2π f_w t − Δ)
- Scatterers at the inner-segment midpoint and the outer-segment midpoint/tip; positions in the body frame are rotated to the flight heading.
- Glide envelope: flapping on for T_flap, then a glide (ψ frozen at ψ_off) for T_glide, as in Rahman & Robertson 2018's observation of intermittent flapping.

| Parameter | Default | Source |
|---|---|---|
| f_w | 2–10 Hz (default 5 Hz) | Rahman & Robertson 2018: 4–6 Hz (raptors/owls); Gong et al.: ~7 Hz (gulls) |
| L₁ + L₂ | 0.3–0.8 m (default 0.5 m) | Gong et al. simulate L = 0.6 m |
| ψ_A, ψ_B, Δ | 40°, 20°, 30° | (assumption) |
| Wing RCS rel. body | −10 to 0 dB | Rahman & Robertson 2018 ("0–10 dB below the bulk") |
| Body RCS | −30 to −20 dBsm | (assumption) |

Reference for a wingtip-only check: f_mD(t) = −(2Lω/λ) sin(ωt + φ₀) (Gong et al.).

### 12.3 Pedestrian
Implement the Boulic–Magnenat-Thalmann–Thalmann (1990) global walking model as used in Chen's book code (12–16 body segments; ellipsoid RCS σ = πa²b²c²/(a² sin²θ cos²φ + b² sin²θ sin²φ + c² cos²θ)², per Abdulatif et al.). Key parameters:
- Height H (default 1.75 m), leg length H_t ≈ 0.53 H (assumption).
- Walking speed v (default 1.4 m/s); relative velocity v_rel = v/H_t.
- Gait cycle duration T_cyc = 1.346/√v_rel s (Boulic model as reproduced in Chen's code; **unverified against the 1990 paper — check**).

Fallback (if the Boulic tables are not yet coded): a sinusoidal kinematic model with gait frequency 1/T_cyc and
- hip flexion ±25°, knee 0–60° (phase-lagged), arm swing ±20° in anti-phase with the legs;
- torso vertical oscillation 2–3 cm at twice the gait frequency (assumption).

Validation targets:
- torso Doppler ≈ v;
- feet peak ≈ 3–4·v (Abdulatif et al.: "4×V ≈ 6 m/s"; toe 4.5 m/s at 1 m/s walking per their text);
- cycle ~1.1 s at 1.4 m/s.

---

## 13. CRLB formulas for estimator validation

Per-sample complex SNR ρ = |A|²/σ² (σ² = 1 in the brief). Phase unknown. Rect windows for CRLB comparisons.

| Parameter | CRLB (variance) |
|---|---|
| Beat freq. (rad/sample) | 6/(ρ N(N²−1)) (Rife & Boorstyn 1974) |
| Range R | (c f_s/(4πS))² · 6/(ρ N(N²−1)) |
| Radial velocity v (M chirps at spacing T_e, post-range-FFT SNR ρ_r = ρN) | (λ/(4πT_e))² · 6/(ρ_r M(M²−1)) |
| sin θ (K elements at d_k in λ; post-2-D-FFT SNR ρ_rd = ρNM) | 1/(2ρ_rd(2π)² Σ_k(d_k − d̄)²); ULA d = ½: 6/(ρ_rd π² K(K²−1)) |
| θ | var(sin θ)/cos²θ |
| General sampling positions t_n | σ²/(2|A|²(2π)² Σ(t_n − t̄)²) per unit of frequency (cycles/unit t) |

**Monte-Carlo protocol (Phase 4 figure):**
- Estimator: ML via zero-padded FFT (×16) + parabolic/Newton refinement, unwindowed.
- Sweep ρ from −20 to +20 dB, ≥ 2000 trials per point, random φ₀ and sub-bin offsets.
- Report RMSE/√CRLB. Expect efficiency → 1 above the threshold SNR; show the threshold region; also show the Hann-windowed estimator.
- TDM caveat: for velocity use a single Tx, or include the per-Tx time offsets t·T_c in Σ(t_n − t̄)².
- Multi-target / MUSIC: Stoica & Nehorai 1989.

---

## 14. Distributed clutter (Phase 2, minimal)
Ground patch scatterers on a polar grid (Δr = ΔR/2, Δθ = 1°) with σ_cell = σ⁰·A_cell. The amplitude is complex Gaussian (Rayleigh); optional K-distributed texture (shape ν = 0.5–5).
- σ⁰ for asphalt at 77 GHz and low grazing: −35 to −20 dB (assumption, unverified).
- Clutter is static, so it appears at zero Doppler; the MTI/zero-Doppler notch is the test.

---

## 15. Validation test matrix (minimum set for the paper)

| # | Model | Test | Pass criterion |
|---|---|---|---|
| V1 | Noise + CFAR | Fixed-threshold and CA/OS/GO/SO Pfa vs analytic, K = 1 and K = 8 (Gamma-exact) | |log10(Pfa_sim/Pfa_th)| < 0.1 at Pfa = 1e-4 (≥ 1e6 cells) |
| V2 | Detection | Pd vs SNR for SW0/SW1(correlated)/SW3, K = 1, 8 | within the 95 % binomial CI |
| V3 | Estimation | RMSE vs CRLB for R, v, θ | efficiency ≥ 0.9 above threshold |
| V4 | Phase noise | residual PSD = L·4 sin²(πfτ); var matches integral | ±0.5 dB / ±2 % |
| V5 | Multipath | peak power vs d follows 4 sin²(2πh_r h_t/(λd)) | null positions ±1 range-bin equivalent |
| V6 | IQ imbalance | image at −IMRR dBc | ±0.5 dB |
| V7 | Interference | burst length 2f_LP/|ΔS|; ghost for equal slopes | ±1 sample; ghost at c Δf/(2S) |
| V8 | Micro-Doppler | f_mD,max = 2LΩcosβ/λ; flash rate N_bΩ/2π (even), 2N_bΩ/2π (odd); bird wingbeat line | ±1 Doppler bin; ±1 % rate |
| V9 | TDM | angle bias without/with Bechter compensation at v = 0.8 v_max | bias removed to < 0.1 bin |
| V10 | Cross-language | Python vs TS golden cubes | max |Δ| < 1e-9 (float64) or 1e-5 (float32) relative to peak |
| V11 | Real data | DCA1000 round-trip (sim → int16 .bin → reader → identical cube); corner-reflector range/angle on public raw sets | exact round-trip; range error < ΔR/4 |
