# The PVQD study

English | [繁體中文](README.zh-TW.md)

These are the plan and the scripts behind the report in [`../../paper/`](../../paper/). On the 296 recordings of the Perceptual Voice Qualities Database (PVQD), the study compares recording conditions, MFCCs, clinical acoustic measures and three frozen speech representation networks (HuBERT, WavLM, Whisper) as predictors of the CAPE-V severity rating of the sustained /a/, in 14 pre-specified contrasts, and sets them against the agreement among the raters. The recordings, every feature computed from them, the model weights and all results stay out of the repository.

The analysis plan (`analysis_plan.md`) and the record of deviations (`experiments/deviations.md`) are in Traditional Chinese; `analysis_plan.yaml` is the machine-readable copy of the plan that the scripts read.

## Contents

| Step | Files |
| --- | --- |
| S0, data audit: identifiers, demographics, ratings, recordings, reliability, rater ceiling | `s0_audit/audit.py`; the six decisions taken on its findings are in `s0_audit/decisions.csv` |
| S1, segmentation into /a/, /i/, sentences and the whole file, without reading a rating | `s1_segments/segment.py` with `test_segment.py`, the review figures `figures.py`, and `analysis_set.py` for the analysis set |
| S2, analysis plan, frozen on 2026-11-02 | `analysis_plan.md`, `analysis_plan.yaml` |
| S3, evaluation code | the package itself: `compare`, `bootstrap.py`, `reliability.py` |
| S4, features: recording conditions (M0), MFCC-13 and MFCC-39 (A1, A2), clinical measures (A3), HuBERT, WavLM and Whisper (A4–A6) | `features/common.py`, `extract_mfcc_metadata.py`, `extract_clinical.py`, `extract_ssl.py`, `download_models.py`, `run_s4.sh`, `verify_s4.py`, `tests/test_features.py`; `models/manifest.json` holds the revision, licence and SHA-256 of every model file |
| S5, freeze | `experiments/freeze.py`, which wrote `experiments/freeze.json` |
| S5–S6, experiments E1–E7 and the pre-specified analysis | `experiments/harness.py`, `experiments/analyze.py` |
| Deviations from the plan | `experiments/deviations.md` |

The report uses E1, the primary comparison, and from the others only what its sections name. The out-of-fold predictions, the result tables and the post hoc analyses are not in the repository; ask me if you need them.

## The freeze

`freeze.py` wrote `freeze.json` on 2026-11-03, before the first experiment ran, and refuses to write it again. It holds the commit of the package and the git tree of its `src/`, the Python version and every installed package, the SHA-256 of 52 files (the two plan files, the feature and experiment scripts, the input tables, the indexes, the feature files and the two manifests), the hash of the job list, the number of jobs (59) and the master seed. `harness.py run` starts only when all of it still matches.

The plan was frozen on my machine and its hash recorded there; it was not deposited in a public registry, so when it was frozen cannot be checked from outside. Line 3 of the plan calls this 預先登記 (pre-registration); the report describes it, more exactly, as frozen and pre-specified.

Four of the frozen files here are byte for byte the files the freeze checked, and their SHA-256 match `freeze.json`: `analysis_plan.yaml`, `s0_audit/decisions.csv`, `data/pvqd_manifest.json` and `models/manifest.json`. The other frozen files here (`analysis_plan.md`, the four scripts in `features/` that it lists, `freeze.py`, `harness.py`, `analyze.py`) and `freeze.json` itself do not reproduce the recorded hashes, and the commit `freeze.json` names is not in this repository's history; section II-D of the report explains why. I keep the frozen files. The code in `src/` is the code used for the results, with support for class labels added afterwards; with that addition, a rerun of E1 gave the frozen result files byte for byte. `.gitattributes` keeps every frozen file with its original bytes and line endings.

## Rerunning

The scripts find the package as `../voice-severity-bench` and record its commit, and the commands below run from the study folder with its own `.venv`. Copy the folder next to a clone:

```bash
git clone https://github.com/recklight/voice-severity-bench
cp -r voice-severity-bench/studies/voice-severity voice-severity-study
cd voice-severity-study
```

**Data.** Download PVQD from Mendeley Data ([10.17632/9dz247gnyb.4](https://doi.org/10.17632/9dz247gnyb.4), CC BY 4.0) into `data/pvqd/`, so that the paths in `data/pvqd_manifest.json` resolve under it. The manifest gives the size and SHA-256 of every file; I downloaded version 1, whose recordings and rating sheets are identical to those of version 4.

**Environments.** S0 and S1 run on Python 3.9 or newer with numpy, pandas, scipy, soundfile, openpyxl, librosa and [kudio](https://github.com/recklight/Kudio) (I used Python 3.9.13, kudio 3.9.1, librosa 0.11.0). S4–S6 run in `.venv`, Python 3.10, with `pip install -e ../voice-severity-bench` plus praat-parselmouth, torch (CPU), transformers, safetensors and openpyxl; `freeze.json` lists the version of every package in my run. parselmouth is GPL-licensed, which is why the clinical measures are computed here and not in the MIT package.

```bash
python s0_audit/audit.py                                    # S0, tables in s0_audit/out/
python -m pytest s1_segments/test_segment.py                # synthetic checks of the segmenter
python s1_segments/segment.py                               # S1, analysis_intervals.csv and the other tables
python s1_segments/analysis_set.py                          # analysis_set.csv
.venv/Scripts/python.exe features/download_models.py        # the three models at the revisions in models/manifest.json
.venv/Scripts/python.exe features/extract_mfcc_metadata.py --materials a i sentences whole
bash features/run_s4.sh                                     # A3 and A4–A6 for the four materials
.venv/Scripts/python.exe features/verify_s4.py --materials a
.venv/Scripts/python.exe experiments/harness.py smoke --jobs 8
.venv/Scripts/python.exe experiments/freeze.py              # after moving freeze.json aside
.venv/Scripts/python.exe experiments/harness.py run all --jobs 8
.venv/Scripts/python.exe experiments/analyze.py             # tables, figures and summary.md in experiments/analysis/
```

These are the commands as I ran them on Windows; on Linux and macOS the interpreter is `.venv/bin/python`, also in the `PY=` line of `run_s4.sh`. `download_models.py` fetches about 1.7 GB, and `extract_ssl.py` runs on the CPU. `freeze.py` never overwrites an existing `freeze.json`, so a rerun writes its own freeze and is checked against that one. Repeating the numbers bit for bit needs one BLAS thread per worker; under that condition a rerun of E1 with 4 workers gave the frozen result files byte for byte (`experiments/deviations.md`, 2026-11-04).

## References

- P. R. Walden, "Perceptual Voice Qualities Database (PVQD): Database characteristics," *Journal of Voice*, vol. 36, no. 6, pp. 875.e15–875.e23, 2022, doi: 10.1016/j.jvoice.2020.10.001.
- P. Walden, "Perceptual Voice Qualities Database (PVQD)," Mendeley Data, version 4, 2022, doi: 10.17632/9dz247gnyb.4. CC BY 4.0.
