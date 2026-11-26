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

## Follow-up approved by Enes: 2-D → 3-D migration, targets first (ROADMAP Phase G)
Enes has approved this direction. It starts right after the audit summary exists. The S1/S2/S3/S5/S6 expert briefs serve as its requirement documents.
1. **G0 Design doc** `docs/design/3d.md`. Contents:
   - frames: world (ENU, ground z=0), radar mount pose (height, tilt, yaw), target body frame (yaw/pitch/roll plus rates);
   - conventions: a brief Revision 2 with `conventionVersion` 3;
   - a migration plan for py/ts/golden;
   - the list of tests that must catch the audit findings.
2. **G1 3-D targets (priority):**
   - Every target has a full 6-DoF pose and trajectory: position/velocity/acceleration in 3-D, attitude and angular rates, and waypoints.
   - Scatterers are defined in the body frame and rotated to world coordinates.
   - Drone: rotor discs follow body pitch/roll (a multirotor tilts forward when it moves), real arm geometry, aspect-dependent blade flash (β), hover/climb/forward-flight presets.
   - Bird: flapping plane from the body attitude; glide/flap.
   - Pedestrian: full 3-D Boulic/Chen limb kinematics, swing amplitude depending on speed, walking at any heading.
   - Cyclist: proper wheel, pedal and leg model.
   - Car: 3-D box scatterers, aspect-dependent RCS.
   - Simple self-occlusion and visibility.
3. **G2 Radar pose and environment:** mounting height/tilt; ground plane with true two-ray multipath; clutter depending on grazing angle.
4. **G3 Antennas and array:** element patterns in azimuth and elevation; 2-D virtual arrays (AWR1843 elevated TX, IWR6843AOP); joint azimuth/elevation DoA; a 3-D point cloud without the cone-angle error.
5. **G4 Tracking:** 3-D constant-velocity/constant-acceleration EKF with (R, θ, φ, v_r).
6. **G5 UI:** a 3-D scene view (three.js) with orbit camera, a side view, target editor for z/attitude/trajectory, and an elevation display in the panels.
7. **G6 Regenerate golden vectors (conventionVersion 3), update the validation report, README and the paper.**

Every step needs tests for both py and ts, and every finding from the audit needs a test that would have caught it.
