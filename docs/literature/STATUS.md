# docs/literature — STATUS

**Done (2026-11-24)**
- `review.md`: structured literature review, 12 sections, inline URLs.
- `references.bib`: 70 entries. Fields not checked online carry `note = {unverified ...}`.
- `fidelity_requirements.md`: implementable model specs. The detection formulas (Swerling 0/1-correlated/2/3, Gamma-exact CA-CFAR, GO/SO/OS) and the phase-noise synthesis normalisation were Monte-Carlo verified.
- `paper_positioning.md`: gap analysis, 3 candidate papers, figure inventory, venue table with sources.

**Known issues / open items**
- SIU 2027 CFP not yet published. Dates in the venue table are extrapolated from SIU 2025/2026.
- IEEE RadarConf 2027 info comes only from an aggregator (myhuiban); the official CFP was not found.
- Some bibliographic details are unverified (Martin & Mulgrew 1990, Gandhi & Kassam 1988, Winkler 2007 pages/DOI, Toth 2018 venue). Check them on IEEE Xplore with university access before the paper is submitted.
- Defaults marked "(assumption)" in `fidelity_requirements.md` (HPF order, LPF order, leakage power, blade RCS, bird body RCS, clutter σ⁰) have no primary source. Expose them as parameters and do not quote them as facts.
- Brief inconsistency flagged: the sign of the spatial-phase term (+ in the brief vs − from far-field τ_k physics). See review §1.2.

**Next**
- Once Phase 3 hardware is available: the TI AWR/IWR EVM model and its antenna layout determine which public raw datasets to use for sim-vs-real.
