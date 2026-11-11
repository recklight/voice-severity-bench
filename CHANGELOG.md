# 變更紀錄

本檔案記錄 `voice-severity-bench` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### 新增

- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖也有英文版（`docs/workflow.html`、`docs/images/workflow.png`），中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。
- `dataset.csv` 可加 `start_s`／`end_s`，只分析檔案的一段；在原取樣率下先裁切再重新取樣。同一個檔案的不同區間可以各占一列。
- `audio.rms_dbfs`：把每段錄音調到固定的 RMS 位準。
- `feature_set_from_table`：把其他前端（臨床量測、語音表徵網路的各層）算好的特徵存成 `features.npz`，可帶欄名、`kind`、欄位區塊與來源資訊；`FeatureSet` 多了 `keys`、`take` 與 `select_blocks`。
- `model.impute: median`（訓練折中位數補值加缺值指標）、`model.block_selection`（在內層交叉驗證選區塊，`BlockSelector`）與 `model.ridge_alphas`（自訂 Ridge 網格）。
- `compare` 指令與 `compare_feature_sets`、`pairwise_contrasts`：多個特徵檔在同一組重複切分上比較，保留每次重複的折外預測；成對檢定（修正 t、Wilcoxon、Holm）加上以受試者為單位的 bootstrap 區間，可設等價邊界。新的設定鍵 `evaluation.n_repeats` 與 `analysis.n_resamples`。
- `bootstrap.py`（以組為單位的 bootstrap）與 `reliability.py`（ICC(1,1)／ICC(1,k)、評分雜訊下限、留一評分者）；`reference_data.shrout_fleiss_ratings` 是 Shrout & Fleiss（1979）的範例評分。
- `repeated_split_indices`：重複 k 折的共用切分，`ablate` 與 `compare` 都用它。
- Ridge 的內層調參每折只做一次 SVD 就得到所有懲罰的解，選擇結果和 GridSearchCV 相同。
- `examples/compare_feature_families.py` 與 `configs/compare_demo.yaml`。
- `compare` 與 `compare_feature_sets` 可比較分類目標：每次重複保留預測類別和各類的 `score_<class>`，summary 有 UAR、敏感度、特異度與 ROC-AUC；`strata` 讓切分依「標籤 × 分層」分層；`compare --metric` 選排名指標。
- `weighted_metric`、`bootstrap_metric` 與 `paired_bootstrap` 可算 UAR、準確率、敏感度、特異度與 ROC-AUC；`BootstrapInterval.n_undefined` 記錄指標算不出來的重抽次數。
- `permutation.py`：`permute_group_labels` 以受試者為單位交換標籤，可限定在分層內；`permutation_test` 每次排列都重跑含內層調參的交叉驗證，p = (1 + #null ≥ obs) / (1 + n)。
- `model.class_weight: balanced`（`logistic`、`svc`、`random_forest`）。
- `studies/voice-severity/`：PVQD 上的嚴重度研究，包含凍結的分析計畫、凍結紀錄 `freeze.json` 和每一步的腳本；`paper/` 是這個研究的四頁報告（PDF 與 LaTeX）。錄音和特徵不在 repo 裡。

### 修正

- 有 `group` 欄的分類切分：scikit-learn 1.7 的 `StratifiedGroupKFold(shuffle=True)` 洗的是每組的類別計數表，組別沒有跟著動，折間類別比例會跑掉（40 位受試者各 2 段、兩類各半、5 折，某一類在各折有 4 到 10 段，其實每折都可以剛好 8 段）。切分時先洗組別標籤，再用 `shuffle=False`。有 `group` 欄的分類結果因此和 0.1.0 不同，迴歸的切分沒有動。
- 有 `group` 欄時，每一類的數量算受試者而不是錄音，受試者少於折數就報錯。`configs/quick_classification.yaml` 只有 3 位受試者屬於較少的一類，所以用 3 折。

---

## [0.1.0] - 2026-10-09

### 新增

- 首次釋出。
- 可設定的倒頻譜前端：框長與位移依實際取樣率換算、對稱 Hamming 窗、64 Hz 起的 23 個梅爾濾波器、HTK 縮放的 DCT 與 `c0`、正弦倒頻譜提升、可選 `c0`／對數能量／不加能量項，以及 13／26／39 維的 Δ／ΔΔ 版面，並以 `feature_layout` 追溯每一欄。
- 語句層級彙整（平均、標準差、極值、百分位數、偏態、峰態），可先做 CMN 或 CMVN，並拒絕會讓統計量變成常數的組合。
- 特徵設定的 ablation 框架：倒頻譜數 × 濾波器數 × 動態階數 × 能量項 × 框長位移 × 提升 × 彙整統計量 × 降維 × 估計器，所有設定在相同的重複分組 k 折上評估，並以 Nadeau–Bengio 修正 t 檢定與 Wilcoxon 符號等級檢定比較基準設定，Holm 校正。
- 連續嚴重度迴歸（線性、Ridge、SVR、隨機森林、MLP）與二類偵測（Logistic、SVC、隨機森林、MLP），標準化與 PCA／LDA 只在訓練折擬合，可選巢狀交叉驗證調參；PyTorch 感知器放在 `dl` extra。
- 最小平方推論：係數標準誤、t、p、信賴區間、R²、調整後 R²、整體 F 檢定、AIC、BIC；殘差自由度為 0 時輸出 NaN 並標記為不可估計。
- 迴歸診斷：Shapiro–Wilk、Jarque–Bera、Breusch–Pagan、Durbin–Watson、leverage、Cook's distance、VIF 與條件數。
- 偏 F 檢定逐步選擇、正交多項式基底的階數掃描（樣本內指標與 LOOCV／k 折 RMSE 並列）、與閉式解比對的梯度下降。
- 特徵與嚴重度的相關分析（Pearson 的 Fisher z 信賴區間、Spearman、Holm／BH 校正），以及預測值與評分的一致性分析（Lin's CCC、Bland–Altman）。
- 批次擷取資料夾樹的音框層級特徵，輸出 npz、HTK（大端序、100 ns 單位、依實際版面計算 `parmKind`）或文字矩陣，每個檔案都有狀態紀錄。
- 可控嚴重度的合成母音（Rosenberg 聲門脈衝、jitter、shimmer、共振峰濾波與雜訊）與合成迴歸資料，供測試與範例使用。
- typer CLI：`synthesize`、`index`、`prepare`、`evaluate`、`train`、`predict`、`extract`、`ablate`、`correlate`、`regress`、`stepwise`、`order-sweep`；除了只寫出 `dataset.csv` 的 `index`，每個指令都在輸出資料夾寫出 `config.yaml` 與 `manifest.json`。
