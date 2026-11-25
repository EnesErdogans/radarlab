# SIU 2027 paper draft (Turkish, 4 pages, double-blind)

Roadmap item C1. Draft of *"RadarLab: FMCW MIMO Radar için Algılama ve Kestirim Kuramıyla Doğrulanmış Açık Kaynaklı Simülatör ve Tarayıcı Tabanlı İkizi"*. The plan is in `docs/literature/paper_positioning.md` (Paper A).

## Files

| File | Content |
|---|---|
| `main.tex` | Paper (IEEEtran `conference`, A4, 10 pt). Turkish title, Özetçe and Anahtar Kelimeler, then English title, Abstract and Keywords |
| `refs.bib` | Subset of `docs/literature/references.bib` plus one entry from `docs/real_data/report.md`. Entries that are not fully verified carry a `% UNVERIFIED:` comment |
| `IEEEtran.cls`, `IEEEtran.bst` | IEEEtran V1.8b and bst 1.14 (LPPL). Included because this sandbox's TeX Live lacks them. Replace them with the official SIU 2027 template files once the template is published |
| `figures/*.pdf` (+ `.png` previews) | Vector figures, made by the scripts in `python/examples/paper/` |
| `figures/*.json` | Every number in the figures and the text that comes from these scripts |
| `build.sh` | pdflatex → bibtex → pdflatex ×2 |

## Build

```sh
# 1. figures (about 4 min on 2 cores; all seeded, so the output is deterministic)
python -I python/examples/paper/fig_cfar_threshold.py      # Fig. 2, ~15 s
python -I python/examples/paper/fig_beamforming_vs_sum.py  # Fig. 3, ~1.5 min
python -I python/examples/paper/fig_validation.py          # Fig. 4 (V2, V3 full mode), ~2 min
# 2. paper
docs/paper/siu2027/build.sh        # or: latexmk -pdf main.tex (Overleaf: upload the folder)
```

Fig. 1 (block diagram) is TikZ inside `main.tex`.

**Turkish hyphenation.** `main.tex` loads `babel` with `turkish` when `turkish.ldf` exists, as in a full TeX Live, MiKTeX or Overleaf. The sandbox used for this draft has neither `babel-turkish` nor the Turkish hyphenation patterns, so there `main.tex` falls back to babel's `turkish` ini locale with *English* hyphenation patterns. The draft PDF built here is therefore a preview: line breaks, and possibly the page fill, will change slightly in a full installation. Check the final page count on Overleaf.

**Build status (2026-11-26, this sandbox).** pdflatex succeeds: 4 pages, A4, no overfull boxes. About a third of the last column is still free.

## Where the numbers come from

| Text / figure | Source |
|---|---|
| Table I, V1–V11, V1c Hann ratios 1.61/1.57, V2 29/30, V3 0.956–1.041 | `python/examples/out/validation_report.md` (full mode) |
| Fig. 4 | `fig_validation.json` (calls `validation.v2_pd` / `v3_crlb` with their fixed seeds, so it is identical to the report) |
| Threshold error 4.07–6.10 dB, realised Pfa, SNR loss 3.9–7.4 dB (SW0) / 4.6–7.4 dB (SW1) | `fig_cfar_threshold.json` → `table_K4_K8` |
| Hann ratios at K = 8 (1.07 … 2.44), naive-α Monte-Carlo 1.92e-3 vs 1.94e-3 | `fig_cfar_threshold.json` → `mc` |
| Power sum vs beamforming (5.0 / 4.2 / 4.3 / 5.7 / 6.9 / 10.8 dB) | `fig_beamforming_vs_sum.json` → `required_snr_db` (SW0, Pd = 0.9) |
| Golden-vector tolerances, the review finding (×2.6–3.3), web frame time ≈ 46 ms | `README.md` §6.1, §7; `docs/review/review_round1.md` (M2/M3) |
| Real-data paragraph | `docs/real_data/report.md` §1, §6 |

**Correction to earlier docs.** `paper_positioning.md` and README §5.3 give the realised Pfa of the naive threshold as "≈ 5·10⁻¹³" for K = 4–8. The exact value is 4·10⁻¹³–6·10⁻¹³ for K = 4 but about 10⁻²³ for K = 8 (`fig_cfar_threshold.json`). The paper uses the exact values. README §5.3 should be updated too (not done here: README is outside this folder's scope).

## SIU requirements checklist

Based on SIU 2025/2026 rules (`paper_positioning.md` §4). Re-check when the SIU 2027 call for papers is out.

- [x] Written in Turkish, with a Turkish title, Özetçe and keywords.
- [x] English title, abstract and keywords (needed for IEEE Xplore).
- [x] IEEE two-column conference format, A4, 10 pt.
- [x] At most 4 pages including references (4 pages in this build).
- [x] Double-blind: no names, affiliation, repo URL or self-identifying acknowledgements. The camera-ready author block is commented out in `main.tex`.
- [x] No maths in the title. The abstracts contain only light inline maths ($K$, $N$); **TODO:** check whether SIU 2027 forbids maths in abstracts, and if so write "K" and "N" as plain text.
- [ ] Use the official SIU 2027 LaTeX template once it is published (it may differ from plain IEEEtran, e.g. in the Özetçe/Abstract layout or the copyright notice).
- [ ] Run IEEE PDF eXpress compatibility (fonts embedded; matplotlib PDFs use TrueType, type 42) when the conference requires it.
- [ ] AI-use disclosure (IEEE policy): add it to the acknowledgements in the camera-ready version (placeholder in `main.tex`). For review, check whether SIU wants it in the anonymous submission.

## Open TODOs for the author (Enes)

1. **Read and correct the Turkish.** Check the terminology against recent SIU radar papers: *cıvıltı* (chirp), *vuru işareti* (beat signal), *evreuyumsuz tümleme*, *kargaşa* (clutter), *hüzme*, *kutu* (bin), *başvuru gerçeklemesi* (reference implementation).
2. **Double-blind and the name "RadarLab".** The public repository carries the author's name. If the SIU rules are strict, rename the tool to e.g. "[Simülatör]" in the review version, or make sure that the repo name cannot be found from the paper.
3. **References.** Verify the entries flagged `UNVERIFIED` in `refs.bib`: Patole (first names), mmSim (arXiv id), Rohling (DOI), Gandhi–Kassam (exact title, DOI), Kay (edition details), Bechter (DOI), Gao 2019 (pages). Also read Shor & Levanon (1991) and Ritcey (1990) and confirm that they cover the K-pulse non-coherent CA/OS and GO cases as the paper says.
4. **TI SDK claim.** The paper does *not* say that TI's demo makes the single-channel mistake. As far as we know, TI's CFAR runs on log-magnitude data (not verified). Do not add such a claim without checking the SDK source.
5. **Hann remedy.** The draft suggests recalibrating α by Monte-Carlo or using wider guard bands. Neither is implemented or tested in the repo yet. Either add a short experiment or soften the sentence.
6. **Beamforming comparison scope.** Only orthogonal beams (K-point DFT, no zero-padding, rectangular angle window) and CA-CFAR were simulated. Zero-padded or windowed beams, and OS-CFAR, are discussed only qualitatively.
7. **Browser frame time (≈ 46 ms).** This comes from Node on a 2-core sandbox (README §7). Re-measure in a real browser on the author's laptop and state the device, or drop the sentence.
8. **Space.** About a third of a column is free. Candidates: a browser screenshot (positioning plan F1), a sentence on the 24 GHz micro-Doppler validation (V8), or a log-domain CFAR remark.
9. **Camera-ready.** Uncomment the author block, add an e-mail address, the repository URL, the acknowledgements and the AI disclosure.
