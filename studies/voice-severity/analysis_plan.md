# 分析計畫

狀態：**凍結**（2026-11-02 起草，同日 RL 簽可）。這份檔案和 `analysis_plan.yaml` 的 SHA-256 記在 `PLAN.md`，S5 的 `freeze.json` 會再記一次，預先登記就以這兩處的雜湊為準。之後的任何改動都寫進 `experiments/deviations.md`，不回頭改這兩個檔。

機器讀的版本是 `analysis_plan.yaml`，兩者不一致時以這份文字為準，並記為偏離。

## 1. 要回答的問題

1. **確認性**：在 PVQD 的持續 /a/ 上，MFCC、臨床聲學量測和凍結的 SSL 表徵預測 CAPE-V 嚴重度，RMSE 差多少？各組是否勝過只用錄音條件的對照組？
2. **上限**：各組離評分者一致性的上限多遠？和單一評分者相比如何？
3. **混淆**：差距有多少來自錄音群組（錄音條件與族群）？

只有問題 1 的 14 個對比是確認性檢定（第 9 節），其餘都是探索性分析，論文裡照這個分法寫。

## 2. 凍結前已經看過的資料

- **S0**：看過評分的信度、評分者上限、錄音群組與嚴重度的關係、性別與年齡。沒有算過任何聲學特徵和評分的關係。
- **S1**：切段全程沒讀評分，有原始碼檢查（`s1_segments/test_segment.py`）。S1 的參數和核准計畫不同的地方記在 `s1_segments/s1_notes.md`。
- **凍結前不看**：特徵與評分的任何關聯，以及 9 個缺 /a/ 檔案的評分。

## 3. 樣本與分析單位

- **資料**：PVQD 296 檔，不排除任何檔案（D2）。
- **主要分析集**：287 檔，條件是 S1 判定 /a/ 區間有效（`s1_segments/analysis_set.csv` 的 `primary_set`）。

  | 錄音群組 | BL | LA | NYU | PT | SJ |
  |---|---|---|---|---|---|
  | 檔數 | 20 | 46 | 29 | 125 | 67 |

  女 190 檔、男 97 檔。
- **交叉驗證分組**：PT 群組人口學資料相同的 9 對檔案各合成一組（D4），其餘一檔一組。287 檔共 278 組；切分和 bootstrap 都以組為單位。

## 4. 目標值

- **主要目標**：CAPE-V Severity，取該檔全部評分的平均（D6，6 或 8 次評分），範圍 0–100。
- **次要目標**（探索性）：
  - CAPE-V Roughness、Breathiness、Strain、Pitch、Loudness；
  - GRBAS Grade、Roughness、Breathiness、Asthenia、Strain。

  GRBAS 是 0–3 分，RMSE 用原單位報，另外報 RMSE 除以目標標準差，方便跨向度比較。

## 5. 材料與共同前處理

- **主要材料**：/a/ 的分析區間（`s1_segments/analysis_intervals.csv`，material = a），長 0.6–3.0 s。
- **共同前處理**：所有特徵組都用同一套，由 `features/common.py` 統一呼叫。
  1. 雙聲道依 D3 處理：NYU1015 取右聲道，PT119 兩聲道平均。
  2. 在原取樣率下先裁出區間，再重取樣到 16 kHz。
  3. 區間 RMS 正規化到 −26 dBFS。

  臨床量測在原取樣率下另算一次，當敏感度分析（E5f）。

## 6. 特徵組

| 代號 | 內容 | 欄數 | 區塊 |
|---|---|---|---|
| B0 | 訓練集平均值（只當參考列，不進檢定） | 0 | — |
| M0 | 錄音條件：錄音群組 one-hot（5）、是否 25 kHz、是否雙聲道、整檔長度、整檔原始 RMS（dBFS）、底噪（dBFS）；取自 `s0_audit/out/recordings.csv` | 10 | 1 |
| A1 | MFCC-13（c0＋c1–c12），mean＋std | 26 | 1 |
| A2 | MFCC-39（再加一階、二階差分），mean＋std | 78 | 1 |
| A3 | 臨床量測（Praat，見下） | 18 | 1 |
| A4 | HuBERT base（`facebook/hubert-base-ls960`）13 層 | 每層 1536 | 14 |
| A5 | WavLM base+（`microsoft/wavlm-base-plus`）13 層 | 每層 1536 | 14 |
| A6 | Whisper-small 編碼器（`openai/whisper-small`）13 層 | 每層 1536 | 14 |

- **A1、A2**：用 voice-severity-bench 的 `prepare`，前端全用套件預設值。
  - 音框 30 ms、位移 15 ms、23 個 mel 濾波器（64 Hz 到 8 kHz）。
  - 預強調 0.97、對稱 Hamming 窗、lifter 22。
  - 差分視窗 7 和 5。
- **A3**：parselmouth 呼叫的 Praat。實際的 Praat 版本和每條指令的參數寫進特徵檔的 `extractor_json`。
  - **音高**：兩階段範圍（De Looze & Hirst, 2008）。第一次用 To Pitch (cc)，50–700 Hz；第二次的下限是第一次 F0 的 0.75 × Q1，上限 1.5 × Q3。
  - **F0 統計量**：平均（半音，相對 100 Hz）、標準差（半音）、第 5–95 百分位範圍（半音）、有聲音框比例、voice breaks 程度。共 5 欄。
  - **Jitter**：local、local absolute、rap、ppq5、ddp，共 5 欄。參數 0.0001、0.02、1.3。
  - **Shimmer**：local、local dB、apq3、apq5、apq11、dda，共 6 欄。參數 0.0001、0.02、1.3、1.6。
  - **HNR**：To Harmonicity (cc) 的平均，參數 0.01 s、第二次的音高下限、0.1、1.0。
  - **CPPS**：Maryn & Weenink（2015）的設定。PowerCepstrogram 參數 60 Hz、0.002 s、5000 Hz、預強調從 50 Hz；平滑視窗 0.02 s 和 0.0005 s；峰值搜尋 60–330 Hz；趨勢線用 robust 直線。
  - **未定義值**：Praat 回傳未定義時記為 NaN，原因寫進 log，不丟檔。
- **A4–A6**：模型版本固定在 S4 下載時的 commit，雜湊和授權記進 `freeze.json`。
  - **輸入**：前處理後的波形，各模型的 feature extractor 照模型自己的設定。
  - **層**：取 13 個 hidden state（第 0 層是第一個 Transformer 層的輸入），每層對時間取 mean＋std（768 × 2）。
  - **AVG 區塊**：先在音框層級平均 13 層，再取 mean＋std。
  - **Whisper**：輸入補到 30 s，輸出依實際長度截到 ⌈樣本數 / 320⌉ 個音框。
  - **長輸入**：超過 30 s 的輸入（只出現在句子和整段錄音）每 30 s 切一塊，音框接起來再統計。
  - **不微調**：只在 CPU 上抽特徵。

## 7. 模型與交叉驗證

- **管線**：在訓練折內依序學習以下四步：
  1. 區塊選擇
  2. 中位數補值，加缺值指標
  3. 標準化
  4. Ridge
- **內層**：在外層的訓練部分做 5 折分組交叉驗證，以 RMSE 選出「區塊 × alpha」，再用整個訓練部分重新擬合。
  - alpha 取 logspace(−3, 7, 21)。
  - 單一區塊的特徵組只選 alpha。
- **外層**：5 折 × 10 次重複，以組為單位。
  - 種子用套件的 `repeat_seeds(20261102, 10)`。
  - 同一個分析裡，每組特徵、每個目標都用同一批切分，切分只看分組和種子。
- **紀錄**：每一折選到的區塊和 alpha 都記下來；選到網格兩端的比例會報出來。

## 8. 指標

- **主要指標**：RMSE。另外報 MAE、R²、Pearson r、Spearman ρ 和 CCC（Lin, 1989）。
- **點估計**：每次重複把 287 檔的折外預測合起來算一次指標，再對 10 次重複取平均。
  - 不先平均各次重複的預測，避免把集成效果算進去。
  - R² = 1 − SSE／SST，SST 用全部 287 檔的平均。
- **折層級分數**：每折各算一次，共 50 個，給校正 t 檢定用。

## 9. 確認性檢定

**對比**：共 14 個，Δ = RMSE(前者) − RMSE(後者)，負值表示前者比較好。

- A1、A2、A3、A4、A5、A6 對 M0（6 個）
- A1、A2、A4、A5、A6 對 A3（5 個）
- A4、A5、A6 對 A2（3 個）

**檢定**：

- **校正 t 檢定**（Nadeau & Bengio, 2003）：用 50 個折層級差值，n_test／n_train 取實際平均（約 0.25），雙尾。14 個 p 值一起做 Holm 校正（Holm, 1979）。
- **配對 bootstrap**：
  - 以第 3 節的 278 組為單位，有放回抽 10,000 次，種子 20261103。
  - 每次重算第 8 節的點估計差值 Δ，取 percentile 95% CI。
  - 程式改寫自 SaDAE-py 的 `stats.paired_diff`。

**判定**：

| 結論 | 條件 |
|---|---|
| 前者優於後者（或反之） | Holm 校正後 p < .05，且 bootstrap 95% CI 整段在 0 的同一側，方向和 t 檢定的平均差一致 |
| 實質差距 | 上一列成立，且點估計 \|Δ\| ≥ 2 分 |
| 實質相當 | bootstrap 90% CI 整段落在 (−2, 2) 內 |
| 無法判定 | 都不符合 |

- **實質相當**：等同兩個單尾 5% 檢定，沒做多重比較校正，論文會註明。
- **最小有意義差距 2 分**：S0 的雜訊下限是 7.05，單一評分者（LORO）是 15.00，兩者差 7.95，2 分約是這段距離的四分之一，也約是目標標準差（25.14）的 8%。

## 10. 評分者上限（E6，預先規定，不進 Holm 族）

- **重算上限**：在 287 檔上重算雜訊下限、r_max = √ICC(1,k) 和 ICC(1,k)，並和 S0 在 296 檔上的數字並列。
- **人類基準（LORO）**：比法和 S0 相同。
  - 每對（檔案, 評分者）的參考值是其他評分者兩次平均的平均，人的分數是該評分者自己的兩次平均。
  - 以配對為單位等權平均：3 位評分者的檔有 3 對，4 位的有 4 對。
- **模型對人**：模型用同一批參考值算 RMSE，每次重複各算一次再取平均。
  - Δ = 模型 − 人類，用第 9 節的 bootstrap（以組抽樣，該檔的全部配對一起抽）附 95% CI。
  - 另外報模型 RMSE 減雜訊下限。

## 11. 探索性分析

- **E2 SSL 逐層曲線**：A4–A6 的 14 個區塊各自固定，只在內層選 alpha。11 個目標都跑，畫逐層 RMSE（附 bootstrap CI）和「層 × 向度」熱圖。
- **E3 次要目標**：7 組特徵 × 10 個次要目標，報第 8 節的全部指標和 bootstrap CI。第 9 節的 14 個對比在每個目標內各自做 Holm，一律標為探索性。
- **E4 其他材料**：

  | 材料 | 檔數 | 說明 |
  |---|---|---|
  | /i/ | 134 | 只涵蓋 BL、LA、SJ 和 1 個 NYU 檔，樣本和主要分析不同 |
  | 句子 | 296 | |
  | 整段錄音 | 296 | 切段的穩健性檢查（S1 驗證的 (c)），14 個對比全部重跑，報方向是否和主要分析一致 |

  句子和整段錄音上的 jitter、shimmer 照算，但在連續語音上本來就不穩，論文會註明。
- **E7 MFCC 設定組合**：用套件的 `run_ablation` 在 287 檔的 /a/ 區間上跑 0.1.0 的預設網格。
  - 網格：倒頻譜 12／6／18 × 差分 0／1／2 × ridge／svr，共 18 組。
  - 設定：`model.tune: true`，5 折 × 10 次。
  - 結果比對 0.1.0 README 裡合成資料的結論。

## 12. 敏感度分析（E5，全部探索性）

每項都報 RMSE 和 14 個對比的方向是否和主要分析一致，對比在該項內各自做 Holm。

| 代號 | 內容 |
|---|---|
| E5a | 只用 PT 群組（125 檔）。M0 剩是否 25 kHz、是否雙聲道、長度、原始音量、底噪 |
| E5b | 留一錄音群組：用其他 4 群訓練，在留下的那一群測試，報各群的 RMSE 和平均偏差（預測減觀測） |
| E5c | 性別：主要分析的折外預測依性別分開算 RMSE；另外在女（190）、男（97）內各自做交叉驗證 |
| E5d | 目標改用第一次評分的平均（`capev_severity__session1`） |
| E5e | 排除有旗標的檔案，分三組各跑一次：/a/ 區間旗標（glide 15、under_1s 10、low_confidence 1，剩 261 檔）；錄音旗標（削波 3、雙聲道 2）；評分者分歧（嚴重度評分 SD ≥ 20） |
| E5f | A3 改在原取樣率（25 或 44.1 kHz）下計算，和 A3 比，並重做 5 個以 A3 為基準的對比 |
| E5g | 各組加上 M0 後和 M0 比（A1⊕M0 … A6⊕M0 對 M0，6 個）；SSL 的區塊選擇固定包含 M0 |
| E5h | 殘差與錄音群組：各組的折外殘差（10 次重複平均）對錄音群組做 Kruskal–Wallis，並報群組解釋的殘差變異比例 |
| E5i | 分組改回一檔一組（不合併 D4 的 9 對） |
| E5j | 缺 /a/ 的 9 檔（LA9027、PT036、PT066、PT081、PT096、PT118、PT123、PT126、PT130）和 287 檔比嚴重度：中位數、Mann–Whitney U、錄音群組組成。用來判斷主要分析集是否偏輕 |

## 13. 缺值與失敗

- **A3 的 NaN**：在訓練折內補值並加缺值指標，報各群組、各欄的 NaN 數量。整列都是 NaN 的檔案照樣保留。
- **A1、A2、A4–A6 出現 NaN 或無限值**：停下來查原因，處理方式寫進 deviations。
- **擬合失敗**：留下紀錄，不靜默丟掉。

## 14. 不做的事

- 不試其他模型或其他 alpha 網格。
- 除了區塊選擇，不做其他特徵選擇。
- 不針對錄音群組調整目標值。
- 不微調 SSL。
- 不在測試折上做任何選擇。

如果之後要加，一律放進 `post_review/`，標明未預先登記。

## 15. 軟體與可重現性

- **環境**：
  - S4–S6 用 `paper-voice-severity/.venv`（Python 3.10），套件版本全部固定。
  - voice-severity-bench 固定在 S3 的 commit。
- **freeze.json**：記錄以下各項。
  - pip freeze
  - 模型 commit 和權重雜湊
  - 各實驗的工作清單雜湊
  - 資料 manifest 雜湊
  - 本計畫兩個檔的雜湊
  - 主種子 20261102
- **開跑前檢查**：`verify_freeze()` 對不上就不跑。

## 16. 偏離與事後分析

- **偏離**：任何偏離凍結計畫的地方，寫進 `experiments/deviations.md`，內容包括日期、內容、原因，以及是在看過結果之前還是之後。
- **事後追加**：審稿或看過結果後才追加的分析，放 `post_review/`。

## 參考文獻

- De Looze, C., & Hirst, D. J. (2008). Detecting changes in key and range for the automatic modelling and coding of intonation. *Speech Prosody 2008*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics*, 45(1), 255–268.
- Maryn, Y., & Weenink, D. (2015). Objective dysphonia measures in the program Praat: Smoothed cepstral peak prominence and acoustic voice quality index. *Journal of Voice*, 29(1), 35–43.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281.
