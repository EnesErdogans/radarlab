# Audit round 2: specialist feature audit + dataset test report (planned for Mon 2026-11-30)

Trigger: the supervising professor found errors (e.g. in micro-Doppler) and says there are more of the same kind. Some errors come from the system being effectively 2-D. This round finds them **before** any new features are built. The fixes come in a separate round, after Enes reviews the findings.

## Inputs
- `docs/audit/professor_notes.md`: the professor's findings, written down by Enes. If this file is missing, the auditors start without it and the lead asks Enes for it.
- Repository state at the start of the round. Read `docs/PROJECT_BRIEF.md`, `ROADMAP.md`, `docs/review/*`, `docs/real_data/report.md`, and both `STATUS.md` files.

## Method (every specialist follows these steps)
1. **Specialise first.** Before reading any RadarLab code, study the feature from the literature: textbooks, the key papers, and the canonical test cases. Write `docs/audit/<feature>/expert_brief.md`. It must contain:
   - the governing equations in full 3-D form;
   - the known pitfalls;
   - what a correct implementation must reproduce (numbers or plots from the literature);
   - which assumptions break in 2-D or in a planar/azimuth-only geometry.
2. **Audit.** Compare the Python and TypeScript code against the brief, line by line. Run your own independent scripts from `/tmp/audit/<feature>/`, not from inside the repository. Check whether each test really tests what it claims, and whether the tolerances are honest.
3. **Report** in `docs/audit/<feature>/findings.md`. For each finding give:
   - an ID;
   - severity: critical, major or minor;
   - the evidence (code location plus a script or numbers);
   - the physical explanation;
   - a proposed fix, including a test that would have caught it;
   - whether it is a "2-D/geometry" issue.
4. Do not change code in `python/src` or `web/src`.

## Specialists (one agent each, running in parallel)
| ID | Feature | Scope |
|---|---|---|
| S1 | Geometry and signal model (3-D) | Radar height and mounting tilt; R, θ, φ and v_r with z; the beat model; TDM timing; RVP; real/complex IF; range from slant versus ground distance; target aspect in 3-D (yaw, pitch, roll) |
| S2 | Micro-Doppler | Drone (rotor plane orientation versus LOS, elevation aspect β, blade flash and sinc, HERM lines, aliasing); bird (flapping plane versus LOS); pedestrian (Boulic/Chen 3-D limb kinematics, gait versus speed); cyclist; STFT parameters and display; agreement with published spectrograms |
| S3 | Antennas, array and DoA | Element patterns in azimuth **and elevation**; the 2-D virtual array (AWR1843 elevated TX); the cone-angle ambiguity (sinθ·cosφ) for azimuth-only arrays; elevation estimation; MVDR/MUSIC; calibration |
| S4 | Detection and CFAR | Gamma(K) thresholds; window effects; Swerling statistics; edge handling; the CFAR-after-beamforming option; Pd/Pfa claims in the paper |
| S5 | Tracking and point cloud | 2-D versus 3-D state (climbing drone, z velocity); polar-to-Cartesian with elevation; gating and association; track initiation |
| S6 | Channel and impairments | Ground plane and two-ray multipath (needs height); clutter versus grazing angle; phase noise; leakage; IF filters; ADC; IQ imbalance; DC; interference |
| S7 | Web parity and visualisation | Whether the UI's 2-D bird's-eye view, the panels and the Learn texts mislead (for example, elevated targets drawn at the wrong x/y, or wrong axis units); Python↔TS behaviour beyond the golden vectors |

## Test agent (T1): independent of the specialists
- Builds a **benchmark scenario suite**: canonical scenes with known ground truth. Examples: a corner reflector at known R/θ/φ; a drone hovering at 30 m altitude seen from 1 m radar height; a drone climbing; a bird; a pedestrian walking towards the radar and across it; two close targets; an off-boresight elevated target.
- Runs both engines (Python and the browser) on these scenes. It also runs the public real data: the UW AWR1843 pedestrian, cyclist and car captures, plus any reachable public micro-Doppler dataset.
- Produces `docs/audit/test_report/report.md` with **screenshots**: headless-Chrome screenshots of the web app for every scenario (all panels and Learn mode), plus Python figures. Each scenario is compared with the ground truth or the literature, in a pass/fail table with numbers.

## Synthesis (lead)
- Merge all findings into `docs/audit/SUMMARY.md`: de-duplicated, ranked, with a "2-D limitations" section and a fix plan, and send Enes a Turkish summary.
- Decide the 3-D roadmap: 3-D scene and state, a ground plane, a radar mounting pose, elevation patterns, 3-D tracking, and a 3-D scene view in the UI.
- No code fixes in this round.
