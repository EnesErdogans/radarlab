# Real FMCW raw-ADC data for sim-vs-real validation (T3): candidates

Checked on 2026-11-24. Shell network: only `raw.githubusercontent.com` and PyPI/npm work.
`github.com`, Google Drive, Zenodo, IEEE DataPort, ti.com, SharePoint and university hosts are not reachable from the shell.
"HEAD" below is `curl -sSIL --max-time 20 <url>`: HTTP status and `content-length` in bytes.
"LFS" means the first bytes were checked so that a Git-LFS pointer is not mistaken for data.

## Ranking

| # | Candidate | Use | Shell-reachable? |
|---|---|---|---|
| 1 | UW "Experiments with mmWave Automotive Radar Test-bed": template raw ADC (`pms/bms/cms1000_30fs.mat`) | (b) micro-Doppler (pedestrian, cyclist); (a) simple single-target scene | **Yes**, 16–17 MB each |
| 2 | UWCR / RAMP-CNN raw ADC dataset (same radar and config, 19,800 frames + labels) + one DCA1000 `.bin` | (a) and (b) at scale; `.bin` exercises `read_dca1000` | No (Google Drive / DataPort) |
| 3 | RaDICaL (IWR1443 + DCA1000, CC BY 4.0) | (a) indoor scenes with depth ground truth | Config yes; data no |
| 4 | ColoRadar (AWR1843 + DCA1000 and a cascade, Apache-2.0) | (a) static scenes with lidar/mocap truth | No (SharePoint, GB per sequence) |
| 5 | Glasgow "Radar signatures of human activities" (5.8 GHz Ancortek, CC BY 4.0) | (b) walking micro-Doppler, true raw beat samples with header | No (2 GB `.7z`) |
| 6 | OpenRadar teaching assets (AWR16xx/18xx frames, Apache-2.0) | Smoke test only: no scene truth, partial config | **Yes**, 0.26–2.1 MB |
| – | Drones: KTH 77 GHz Zenodo (1.5 GB, range-azimuth, not ADC); Kang 60 GHz DataPort (15 GB); Safa et al. IWR1443 on a drone (radar is the carrier, not a target); Sci. Data 2026 multi-radar drone set (page not readable) | (b) drone | No |

**No official corner-reflector capture with a full `.cfg` was found on any reachable host**, from TI or from a dataset author.
TI sample captures sit in mmWave Studio / Radar Toolbox, on ti.com and dev.ti.com, and are not reachable.

## Recommendation

* **(b) micro-Doppler: #1 `pms1000_30fs.mat` (pedestrian)**, plus `bms1000_30fs.mat` (cyclist). They can be downloaded now (HTTP 200, not LFS) and are verified below. This is the authors' own release, MIT licensed and cited by an Asilomar 2019 paper. Simulate a `Pedestrian` with the same config and compare spectrograms with envelope correlation or SSIM.
* **(a) simple scene: #1 as well, for now** (one pedestrian at about 7 m on campus): range profile, noise floor, sidelobes and Doppler spread. For a real static/corner-reflector test, the best options are, in order: record our own AWR/IWR capture of a trihedral reflector at a measured range; download the UW DCA1000 `.bin` (#2); or download one ColoRadar sequence (#4).
* **For the drone case**, no small official raw-ADC drone set exists. Use the Rahman & Robertson regimes (simulation only) until we record our own data.

## 1. UW template raw ADC (Gao, Xing, Roy, Liu): **reachable**

* Source/owner: Xiangyu Gao (University of Washington), repo `Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification`, folder `template data/`. License: MIT (repo `LICENSE`).
* Paper: X. Gao, G. Xing, S. Roy, H. Liu, "Experiments with mmWave Automotive Radar Test-bed," Asilomar 2019, DOI 10.1109/IEEECONF44664.2019.9048939, arXiv:1912.12566.
* Radar: TI AWR1843 + DCA1000, 2 TX (horizontal, TDM: odd chirp = first TX, even = second) × 4 RX.
* Config (repo `config/get_params_value.m` and the UWCR README `config.py`): startFreq 77 GHz, slope 21.0017 MHz/µs, fs 4 Msps complex, N 128, 255 loops, chirp period 60 µs per TX (120 µs per TX cycle), frame period 33.33 ms, B_eff ≈ 0.672 GHz, ΔR ≈ 0.223 m.
  **Not given:** adcStartTime, idle/ramp split, rxGain. Assume adcStartTime ≈ 6 µs and document it.
* Scenes: `pms1000` = pedestrian, `bms1000` = bicycle, `cms1000` = car, 30 frames each, outdoor campus. These are named after UWCR sequences (`2019_04_09_pms1000`…). Labels and camera images are in the full UWCR set (#2), not in the template.
* Format: MATLAB v5 `.mat`, variable `data_frames`, complex128, shape (4 RX, F·M·nTx·N) = (4, 1,958,400). This is already readDCA1000-decoded: columns run frame → loop → TX → sample.
* URLs and HEAD (all 200, real MAT header, no LFS; `.gitattributes` has no LFS rule):
  * https://raw.githubusercontent.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification/main/template%20data/pms1000_30fs.mat (200, 16,832,765 B, md5 7d89f3e9e4d78bbd37d35e9fdb1a42fa)
  * …/template%20data/bms1000_30fs.mat (200, 17,071,118 B)
  * …/template%20data/cms1000_30fs.mat (200, 16,926,643 B)
* Verified (2026-11-24, scratch script outside the repo):
  * reshaping gives (F, K, M, N) = (30, 8, 255, 128);
  * after static removal, the strongest moving return is at about 6.9–7.1 m (range bin 31–32), −0.1 to −0.2 m/s;
  * about 70 % of the power is in the positive-range half, so I/Q order is correct with no conjugation;
  * values are small (max |x| ≈ 166), so the dataset is scaled.
* Mapping to `radarlab.io`: this is not a `.bin`, so build the `RadarConfig` by hand. Use the values above, plus:
  * `tx_pos=[0, 2]` (AWR1843 TX0/TX2) and `rx_pos=[0, .5, 1, 1.5]`;
  * `chirp_period=60e-6`, `n_chirps=255`, `iq_mode='complex'`.
  * Cube: `d.reshape(4, F, 255, 2, 128).transpose(1, 3, 0, 2, 4).reshape(F, 8, 255, 128)`. This is the same (F, K, M, N) layout that `read_dca1000` returns.
  * Alternatively, `dca1000_encode` the cube to a `.bin` and round-trip it through `read_dca1000`.

## 2. UWCR / RAMP-CNN raw ADC dataset (same group): manual download

* Repo: https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection (MIT code). Data: IEEE DataPort DOI 10.21227/xm40-jx59 (Gao, Luo, Xing, Roy, Liu 2022).
* Paper: Gao et al., "RAMP-CNN," IEEE Sensors J. 21(4):5119–5132, 2021, DOI 10.1109/JSEN.2020.3036047.
* Content: about 19,800 frames; per-frame `.mat` `adcData` (128 samples, 255 chirps, 4 RX, 2 TX); camera `.jpg`; per-frame `.csv` labels. Six classes, moving and static.
* Download: Google Drive https://drive.google.com/file/d/1OfqXXgoFg6xRYZkRqPJye4cQ29Fomh3l/view (very large).
* Real DCA1000 `.bin`: `2019_04_30_pbms002_raw_900fs.zip` in https://drive.google.com/drive/folders/1CC3nluGDral__geL6zIzCK2t5Jrhfwex. It has 900 frames with the #1 config. Estimated ~0.9 GB raw (not checked: Drive is not reachable). It is the cleanest real test of `read_dca1000` (AWR18xx 2-lane layout).

## 3. RaDICaL (Lim et al., UIUC): config reachable, data manual

* Data: Illinois Data Bank DOI 10.13012/B2IDB-3289560_V1, CC BY 4.0, ROS bags of 15–332 GB. SDK: https://github.com/moodoki/radical_sdk.
* Paper: T. Y. Lim et al., "RaDICaL: A Synchronized FMCW Radar, Depth, IMU and RGB Camera Data Dataset with Low-Level FMCW Radar Signals," IEEE JSTSP 2021.
* Radar: IWR1443BOOST + DCA1000. Config: https://raw.githubusercontent.com/moodoki/radical_sdk/master/samples/indoor_human_rcs.cfg (HEAD 200, 1,101 B, plain text).
  * `profileCfg 0 77 58 7 40 0 0 100 1 304 9499 0 0 30`
  * `channelCfg 15 5 0` (TX0 + TX2, 4 RX), 32 loops, 33.333 ms
* 50-frame sample `.h5` (indoor): https://fireball.teckyianlim.me/file/flaming-cake/indoor_sample_50.h5 (md5 b195ff422cc4c979eeb81623899050cb, personal host, not reachable, size unknown). The radar array is (slow_time, antenna, fast_time).
* Mapping: `parse_ti_cfg` parses the `.cfg` once an `iwr1443` geometry entry exists. Currently only awr1843 / iwr1843 / iwr6843isk / awr1642 exist. Use TX0 = 0 and TX2 = 2λ, with no z offset.
  Raw xWR14xx `.bin` (4-lane, I/Q-interleaved) is **not** supported by `dca1000_decode`, but the `.h5` cube does not need it.

## 4. ColoRadar (Kramer et al., CU Boulder ARPG): manual

* https://arpg.colorado.edu/coloradar/ (Apache-2.0). Hosted on SharePoint, one zip per sequence. 52 sequences, more than 145 min.
* Paper: Kramer et al., "ColoRadar: The direct 3D millimeter wave radar dataset," IJRR 41(4), 2022, arXiv:2103.04510.
* Radar: AWR1843BOOST + DCA1000, and an MMWCAS cascade. Raw ADC plus heatmaps. `calib` holds the waveform and antenna config. There is also a decluttering calibration set. Lidar and mocap ground truth (the `aspen` mocap runs) suit static-scene comparison.
* Sizes: GB per sequence (not checked).

## 5. Glasgow radar signatures of human activities: manual

* https://researchdata.gla.ac.uk/848/, DOI 10.5525/gla.researchdata.848, CC BY 4.0. Fioranelli, Shah, Li, Shrestha, Yang, Le Kernec (2019).
* Radar: Ancortek, 5.8 GHz, 400 MHz bandwidth, 1 ms chirp, 128 samples, +18 dBm, Yagi antennas.
* `.dat` header: fc, Tsweep, N, B, followed by complex beat samples.
* Content: 1,754 recordings. Activity A01 = walking (file name `1PxxA01Rz`); A02–A06 are sitting, standing, picking up, drinking and falling.
* Download: https://researchdata.gla.ac.uk/848/20/Dataset_848.7z (2 GB). Readme: https://researchdata.gla.ac.uk/848/21/Readme_848.pdf.
* Mapping: SISO, `iq_mode='complex'`, `n_samples=128`, `chirp_period=1e-3`, `fc=5.8e9`, `slope=4e11`. A simple `.dat` reader is needed (future code).

## 6. OpenRadar teaching assets (PreSense): **reachable**, smoke test only

* Repo `PreSenseRadar/OpenRadar`, Apache-2.0. Also inside the PyPI sdist `openradar-1.0.1.tar.gz` (1.9 MB).
* Files (all HEAD 200, real `.npy`, no LFS):
  * `Presense%20Applied%20Radar/assets/simple_frame_1.npy` (262,272 B; (128 chirps, 128 samples))
  * `doppler_example.npy` (524,416 B; (128, 256))
  * `doppler_example_1.npy` (2,097,280 B; (128, 8 virtual, 128); 2 TX × 4 RX)
  * `chirp.txt` (6,912 B)
  * `frame.txt` (1,769,472 B)

  The prefix is https://raw.githubusercontent.com/PreSenseRadar/OpenRadar/master/. The values are uint16 and must be cast to int16.
* Config: the notebooks give only fs = 2500 ksps, slope 60 MHz/µs, N = 128. There is no scene description.
* mmWave Studio scripts:
  * `scripts/1843_mmwavestudio_config.lua` (200, 21,960 B): 77 GHz, idle 100 µs, adcStart 6 µs, rampEnd 62 µs, 60.012 MHz/µs, 128 samples, 2500 ksps, 3 TX in order TX0, TX2, TX1, 128 loops, 70 ms.
  * `scripts/1642_mmwavestudio_config.lua` (200, 5,736 B).
* The matching walking-person DCA1000 capture `1_person_walking_128loops.bin` is on Google Drive, file id `16rdlZSHpIzYXEBHTW40V75RPN-yzn0oZ`, from `demo/visualizer/download_data.py`. It is AWR1843 data, 300 frames, about 236 MB (computed), and not reachable. It is a good second micro-Doppler `.bin` that tests `read_dca1000` with 3 TX.

## Manual-download list (put outside git or in a git-ignored folder; do not commit data)

1. `2019_04_30_pbms002_raw_900fs.zip`: UW Drive folder above. Use it for the DCA1000 `.bin` reader and micro-Doppler.
2. `1_person_walking_128loops.bin`: https://drive.google.com/uc?export=download&id=16rdlZSHpIzYXEBHTW40V75RPN-yzn0oZ
3. `indoor_sample_50.h5`: RaDICaL sample URL above.
4. One ColoRadar `aspen` (mocap) sequence plus `calib`.
5. `Dataset_848.7z` (Glasgow), for walking only.
