# PVQD 研究

[English](README.md) | 繁體中文

這裡是 [`../../paper/`](../../paper/) 那份報告背後的分析計畫和腳本。研究用 Perceptual Voice Qualities Database（PVQD）的 296 個錄音，比較錄音條件、MFCC、臨床聲學量測和三個凍結的語音表徵網路（HuBERT、WavLM、Whisper）預測持續 /a/ 的 CAPE-V 嚴重度，共 14 個事先定好的對比，並和評分者之間的一致性對照。錄音、從錄音算出來的特徵、模型權重和所有結果都不在 repo 裡。

分析計畫（`analysis_plan.md`）和偏離紀錄（`experiments/deviations.md`）是中文；`analysis_plan.yaml` 是計畫給腳本讀的版本。

## 內容

| 步驟 | 檔案 |
| --- | --- |
| S0，資料稽核：ID、人口學、評分、錄音、信度、評分者上限 | `s0_audit/audit.py`；依稽核結果做的六項決定在 `s0_audit/decisions.csv` |
| S1，切出 /a/、/i/、句子和整段錄音，全程不讀評分 | `s1_segments/segment.py` 和 `test_segment.py`、檢查用的圖 `figures.py`、產生分析集的 `analysis_set.py` |
| S2，分析計畫，2026-11-02 凍結 | `analysis_plan.md`、`analysis_plan.yaml` |
| S3，評估程式 | 就是套件本身：`compare`、`bootstrap.py`、`reliability.py` |
| S4，特徵：錄音條件（M0）、MFCC-13 和 MFCC-39（A1、A2）、臨床量測（A3）、HuBERT、WavLM、Whisper（A4–A6） | `features/common.py`、`extract_mfcc_metadata.py`、`extract_clinical.py`、`extract_ssl.py`、`download_models.py`、`run_s4.sh`、`verify_s4.py`、`tests/test_features.py`；`models/manifest.json` 記了每個模型檔的版本、授權和 SHA-256 |
| S5，凍結 | `experiments/freeze.py`，寫出 `experiments/freeze.json` |
| S5–S6，實驗 E1–E7 和事先定好的分析 | `experiments/harness.py`、`experiments/analyze.py` |
| 和計畫不同的地方 | `experiments/deviations.md` |

報告用的是 E1（主要比較），其他實驗只用到報告各節點名的部分。折外預測、結果表和事後分析都不在 repo 裡，需要的話可以向我索取。

## 凍結

`freeze.py` 在 2026-11-03 第一個實驗開跑前寫出 `freeze.json`，之後不會再寫。裡面有套件的 commit 和 `src/` 的 git tree、Python 版本和所有安裝的套件、52 個檔案的 SHA-256（兩個計畫檔、特徵和實驗腳本、輸入表、索引、特徵檔和兩個 manifest）、工作清單的雜湊、工作數（59）和主種子。`harness.py run` 要這些都還對得上才會開跑。

計畫是在我的電腦上凍結並記下雜湊，沒有送到公開的登錄處，所以凍結的時間外人無法查證。計畫第 3 行寫的「預先登記」指的就是這件事；報告用比較精確的說法：凍結、事先定好。

這裡有四個凍結檔和當初核對的檔案逐位元組相同，SHA-256 也和 `freeze.json` 一致：`analysis_plan.yaml`、`s0_audit/decisions.csv`、`data/pvqd_manifest.json`、`models/manifest.json`。其他凍結檔（`analysis_plan.md`、`features/` 裡它列的四個腳本、`freeze.py`、`harness.py`、`analyze.py`）和 `freeze.json` 本身都對不上記下的雜湊，`freeze.json` 記的 commit 也不在這個 repo 的歷史裡，原因寫在報告的 II-D 節。凍結檔我都留著。`src/` 的程式就是產生結果的程式，另外加了之後才寫的分類支援；加上之後重跑 E1，結果檔和凍結的逐位元組相同。`.gitattributes` 讓每個凍結檔保留原本的位元組和換行。

## 重跑

腳本會到 `../voice-severity-bench` 找套件並記下它的 commit，下面的指令也都在研究資料夾裡用它自己的 `.venv` 跑。所以先把資料夾複製到 clone 的旁邊：

```bash
git clone https://github.com/recklight/voice-severity-bench
cp -r voice-severity-bench/studies/voice-severity voice-severity-study
cd voice-severity-study
```

**資料。** 從 Mendeley Data 下載 PVQD（[10.17632/9dz247gnyb.4](https://doi.org/10.17632/9dz247gnyb.4)，CC BY 4.0）放到 `data/pvqd/`，讓 `data/pvqd_manifest.json` 裡的路徑都對得到。manifest 有每個檔案的大小和 SHA-256；我下載的是第 1 版，錄音和評分表都和第 4 版相同。

**環境。** S0、S1 用 Python 3.9 以上，需要 numpy、pandas、scipy、soundfile、openpyxl、librosa 和 [kudio](https://github.com/recklight/Kudio)（我用 Python 3.9.13、kudio 3.9.1、librosa 0.11.0）。S4–S6 用 `.venv`（Python 3.10），`pip install -e ../voice-severity-bench`，再加 praat-parselmouth、torch（CPU）、transformers、safetensors 和 openpyxl；我那次每個套件的版本都在 `freeze.json`。parselmouth 是 GPL 授權，所以臨床量測放在這裡算，不放進 MIT 授權的套件。

```bash
python s0_audit/audit.py                                    # S0，表格在 s0_audit/out/
python -m pytest s1_segments/test_segment.py                # 切段程式的合成資料檢查
python s1_segments/segment.py                               # S1，analysis_intervals.csv 等表格
python s1_segments/analysis_set.py                          # analysis_set.csv
.venv/Scripts/python.exe features/download_models.py        # 三個模型，版本照 models/manifest.json
.venv/Scripts/python.exe features/extract_mfcc_metadata.py --materials a i sentences whole
bash features/run_s4.sh                                     # 四種材料的 A3 和 A4–A6
.venv/Scripts/python.exe features/verify_s4.py --materials a
.venv/Scripts/python.exe experiments/harness.py smoke --jobs 8
.venv/Scripts/python.exe experiments/freeze.py              # 先把原本的 freeze.json 移開
.venv/Scripts/python.exe experiments/harness.py run all --jobs 8
.venv/Scripts/python.exe experiments/analyze.py             # 表、圖和 summary.md 寫到 experiments/analysis/
```

這是我在 Windows 上跑的指令；Linux 和 macOS 的直譯器是 `.venv/bin/python`，`run_s4.sh` 的 `PY=` 那行也要跟著改。`download_models.py` 要下載約 1.7 GB，`extract_ssl.py` 在 CPU 上跑。`freeze.py` 不會覆寫已經存在的 `freeze.json`，所以重跑會寫出自己的凍結檔，再拿它來核對。要逐位元重現數字，每個工作行程只能有 1 條 BLAS 執行緒；在這個條件下用 4 個行程重跑 E1，結果檔和凍結的結果逐位元組相同（`experiments/deviations.md`，2026-11-04）。

## References

- P. R. Walden, "Perceptual Voice Qualities Database (PVQD): Database characteristics," *Journal of Voice*, vol. 36, no. 6, pp. 875.e15–875.e23, 2022, doi: 10.1016/j.jvoice.2020.10.001.
- P. Walden, "Perceptual Voice Qualities Database (PVQD)," Mendeley Data, version 4, 2022, doi: 10.17632/9dz247gnyb.4. CC BY 4.0.
