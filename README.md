# voice-severity-bench

[![CI](https://github.com/recklight/voice-severity-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/voice-severity-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [繁體中文](README.zh-TW.md)

Represents voices by their mel-frequency cepstral coefficients (MFCCs), compares feature settings, and predicts continuous clinical severity ratings with regression models.

The input is a set of sustained-vowel recordings, each with a severity rating (for example CAPE-V's 0–100 visual analog scale). The cepstral front end computes frame-level features and pools them into one vector per recording, and regression or classification models are then evaluated with cross-validation grouped by speaker. The same pipeline can expand an ablation grid (number of cepstra × Δ order × energy term × pooling statistics × model) and test the differences for significance. For regression analysis it has least-squares inference, residual diagnostics, stepwise selection and polynomial order selection, and it also measures the correlation and agreement between features and ratings. Results are written as CSV, JSON and figures, and every output folder keeps the configuration and a manifest of the run, so it can be repeated later. The Python package is `voice_severity_bench`; the command-line tool is `voice-severity-bench`.

Author: RL

---

## Contents

- [Overview](#overview)
- [Method](#method)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [CLI reference](#cli-reference)
- [Project layout](#project-layout)
- [Preparing data](#preparing-data)
- [Configuration](#configuration)
- [Results and metrics](#results-and-metrics)
- [Known limitations](#known-limitations)
- [Design notes](#design-notes)
- [Development and testing](#development-and-testing)
- [References](#references)
- [License](#license)

---

## Overview

[![voice-severity-bench workflow](docs/images/workflow.png)](https://raw.githack.com/recklight/voice-severity-bench/master/docs/workflow.html?theme=dark)

The interactive diagram can be zoomed and searched, and it highlights the path between any two steps: [view it online](https://raw.githack.com/recklight/voice-severity-bench/master/docs/workflow.html?theme=dark), or open the local [`docs/workflow.html`](docs/workflow.html) in a browser.

Clinical voice assessment usually scores a voice on a perceptual scale: the G score of GRBAS is an ordinal grade from 0 to 3 (Hirano, 1981), and CAPE-V marks overall severity on a 0–100 scale (Kempster et al., 2009). Predicting such continuous scores from acoustic features is a common research question (Maryn et al., 2010, for example, combine several acoustic measures in one model), but MFCCs alone come with a list of details that change the result: frame length and hop, the number of filters and their lower frequency limit, the number of cepstral coefficients, `c0` or log energy, whether to add Δ and ΔΔ (that is, 13, 26 or 39 dimensions), and how the frames are pooled into one utterance vector. When RMSE moves by a point or two, a single split cannot tell whether the setting caused it or the cross-validation itself varies that much.

So every one of these choices is a key in the configuration file. All combinations are evaluated on the same speaker-grouped splits and compared with a corrected t-test that accounts for the overlap between training sets, with Holm's method for multiple comparisons. Regression analysis reports residual diagnostics and out-of-sample error alongside the in-sample $R^2$. The shared building blocks (pre-emphasis, framing, the mel filterbank, Δ regression, HTK file I/O, endpoint detection and UAR) come from [`speechdsp`](https://pypi.org/project/speechdsp/).

| Stage | What it does | Module |
| --- | --- | --- |
| 1. Load recordings | Mono float64, optional resampling, head and tail trimming in seconds, optional endpoint detection | `corpus` |
| 2. Cepstral front end | Pre-emphasis, symmetric Hamming window, mel filterbank, HTK-scaled DCT, liftering, energy term, Δ/ΔΔ | `cepstral_frontend` |
| 3. Utterance pooling | Mean, standard deviation, extremes, percentiles, skewness, kurtosis, with optional CMN/CMVN first | `pooling` |
| 4. Dataset and feature file | `dataset.csv` index, `features.npz`, building the index from a ratings table | `dataset`, `pipeline` |
| 5. Models | Regression and classification estimators, PCA/LDA, tuning by nested cross-validation, model files | `models`, `neural` |
| 6. Evaluation | Grouped or stratified k-fold, regression and classification metrics | `splitting`, `evaluation` |
| 7. Ablation | Expanding the feature-setting grid, shared splits, paired significance tests | `ablation` |
| 8. Regression analysis | Least-squares inference, residual diagnostics, stepwise selection, polynomial order, gradient descent | `ols`, `regression_diagnostics`, `stepwise`, `design_matrix`, `polynomial_order`, `gradient_descent` |
| 9. Correlation and agreement | Pearson (Fisher z interval), Spearman, Lin's CCC, Bland–Altman | `association` |
| 10. Output | Figures with a CSV of the same data, `config.yaml`, `manifest.json` | `figures`, `manifest` |
| 11. Synthetic data | Synthetic sustained vowels with controllable severity, and toy regression datasets, used by the tests, the examples and the quick start | `synthetic` |

### The PVQD study

[`studies/voice-severity/`](studies/voice-severity/) holds a study on real recordings, the one `compare`, `bootstrap` and `reliability` were written for. On the sustained /a/ of the Perceptual Voice Qualities Database (PVQD; Walden, 2022) it compares MFCCs, clinical acoustic measures and three frozen speech representation networks (HuBERT, WavLM, Whisper) as predictors of the CAPE-V severity rating, next to the agreement among the raters, under an analysis plan frozen before any feature was related to a rating. The four-page report is in [`paper/`](paper/). The recordings and every feature computed from them stay out of the repository; the study's README says where to get PVQD and how to rerun each step.

---

## Method

### 1. Cepstral front end (MFCC)

The whole signal is first pre-emphasized, $y[n] = x[n] - a\,x[n-1]$ (the first sample is kept as it is), then cut into frames by `frame_ms`/`hop_ms`. Milliseconds are converted to samples at **each file's actual sample rate**, rounding down, so 30/15 ms is 1323/661 samples at 44.1 kHz and 480/240 samples at 16 kHz. Each frame is multiplied by the analysis window and goes through an `n_fft`-point FFT, and the power spectrum $|X_k|^2$ is weighted by `n_mels` triangular filters, each with a peak of 1. The filter edges are equally spaced on the mel scale $m(f) = 1127 \ln(1 + f/700)$ (Stevens et al., 1937; O'Shaughnessy, 1987), from `fmin_hz` to `fmax_hz`. The filter energies get a floor, `log_floor`, before the natural log, so an all-zero frame does not produce $-\infty$. The cepstrum is then taken with the DCT-II in its HTK form (Davis & Mermelstein, 1980; Young et al., 2006):

$$c_m = \sqrt{\frac{2}{N}} \sum_{n=1}^{N} \log E_n \cos\left(\frac{\pi m (n - 0.5)}{N}\right), \qquad m = 0, \dots, M$$

For $m \ge 1$ this scaling matches SciPy's orthonormal DCT, while $c_0$ is larger by a factor of $\sqrt 2$ (a doctest checks the relation). $c_1 \dots c_M$ are then multiplied by the sinusoidal lifter $1 + \tfrac{L}{2}\sin(\pi m / L)$ (Juang et al., 1987). The energy term comes after the cepstra, in HTK's order: `c0`, the log energy of the raw frame `log_e` (computed before pre-emphasis and windowing, like logE in the ETSI front end and HTK's default raw energy), or nothing.

The defaults follow the filterbank of the distributed speech recognition front end (ETSI ES 201 108): 23 filters between 64 Hz and the Nyquist frequency, pre-emphasis 0.97, a symmetric Hamming window, and 12 cepstral coefficients plus `c0`. It is **not** a full reproduction of that standard: frames are 30/15 ms (the standard uses 25/10 ms), the filters act on the power spectrum rather than the magnitude spectrum, and the cepstrum uses HTK scaling and is liftered.

`mfcc()` in `speechdsp` is one fixed recipe (periodic Hamming window, filters starting at 0 Hz, orthonormal DCT, log energy instead of `c0` in dimension 0), and those choices are exactly what is being compared here. So the front end borrows only pre-emphasis, framing, the mel filterbank and Δ from `speechdsp`, and handles the window, power spectrum, DCT scaling and energy term itself. With the settings matched to `speechdsp.mfcc()`, the two agree value for value on $c_1 \dots c_{12}$ (checked in `tests/test_cepstral_frontend.py`).

| Key | Default | Effect |
| --- | --- | --- |
| `frontend.frame_ms` / `frontend.hop_ms` | 30.0 / 15.0 | Frame length and hop in milliseconds, converted at the actual sample rate |
| `frontend.n_fft` | `null` | FFT size; `null` takes the smallest power of 2 that holds one frame |
| `frontend.window` | `hamming_symmetric` | Symmetric Hamming, periodic Hamming, Hann or rectangular window |
| `frontend.preemphasis` | 0.97 | Pre-emphasis coefficient; 0 turns it off |
| `frontend.n_mels` | 23 | Number of triangular filters |
| `frontend.fmin_hz` / `frontend.fmax_hz` | 64.0 / `null` | Frequency range of the filterbank; `null` is the Nyquist frequency |
| `frontend.n_ceps` | 12 | Number of cepstral coefficients $c_1 \dots c_M$ (not counting $c_0$) |
| `frontend.energy_term` | `c0` | `c0`, `log_energy` or `none` |
| `frontend.lifter` | 22 | Sinusoidal lifter length $L$; 0 turns liftering off |
| `frontend.log_floor` | 1e-10 | Energy floor before the log |

### 2. Dynamic features and the 13/26/39-dimensional layouts

Δ is computed as a regression coefficient (Furui, 1986), with the first and last frames repeated at the edges:

$$d_t = \frac{\sum_{n=1}^{N} n\,(c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2}$$

Δ uses $N = 3$ (a 7-frame window), and ΔΔ applies the same formula to Δ with $N = 2$ (a 5-frame window). With `delta_order` set to 0, 1 or 2, each frame has 13, 26 or 39 dimensions, ordered `[static | Δ | ΔΔ]`. `feature_layout()` produces column names such as `c1 … c12, c0, d_c1 …, dd_c0`, so every result can be traced back to the feature column it came from.

| Key | Default | Effect |
| --- | --- | --- |
| `frontend.delta_order` | 2 | 0 for static features only, 1 adds Δ, 2 adds Δ and ΔΔ |
| `frontend.delta_widths` | `[7, 5]` | Regression window lengths for Δ and ΔΔ (odd) |

### 3. Utterance-level pooling

A regression model needs one vector per recording, so each column is summarized over all frames of the recording: mean, standard deviation, minimum, maximum, percentiles, skewness and excess kurtosis (the last two are biased moment ratios, defined as 0 for a constant column). The output is ordered statistic-major, with column names like `mean_c1`, `std_d_c0` and `p90_c12`. Before pooling, each recording can be given cepstral mean normalization (CMN) or cepstral mean and variance normalization (CMVN). Combinations that would make a statistic identical for every recording (the mean after CMN, the mean or standard deviation after CMVN) raise an error rather than produce a whole column of constants.

| Key | Default | Effect |
| --- | --- | --- |
| `pooling.statistics` | `[mean, std]` | Statistics, output in this order |
| `pooling.percentiles` | `[10.0, 50.0, 90.0]` | Percentiles used by `percentile` |
| `pooling.normalization` | `none` | `none`, `cmn` or `cmvn` |

### 4. Regression and classification models

Every model is a scikit-learn `Pipeline` (Pedregosa et al., 2011): `StandardScaler` → dimensionality reduction (none, PCA, or LDA for classification) → estimator, so scaling and reduction are learned on the training folds only. For regression there are least squares, Ridge (Hoerl & Kennard, 1970), support vector regression (Drucker et al., 1997), random forest (Breiman, 2001) and a multilayer perceptron; for classification, logistic regression, a support vector classifier, random forest and a multilayer perceptron. The activation of the perceptron's hidden layers is set by `model.activation` and defaults to logistic for both regression and classification. The scikit-learn perceptron sets its weights with the normalized initialization of Glorot & Bengio (2010) (range $\pm\sqrt{6/(n_{in}+n_{out})}$, with 2 in place of 6 for logistic activation). There is also a PyTorch perceptron in the `dl` extra (Adam optimizer, Kingma & Ba, 2015); its weights keep PyTorch's default uniform initialization in $\pm 1/\sqrt{n_{in}}$, the common heuristic from Glorot & Bengio's equation (1), and it trains on the CPU. SVR and the regression perceptron predict the standardized target, so `svr_epsilon` is in units of the rating's standard deviation. LDA has at most $C - 1$ discriminant directions (Fisher, 1936); asking for more raises an error. With `model.tune` on, each training fold runs an inner cross-validation, picks hyperparameters from a small grid and refits, so the outer folds evaluate the whole procedure, tuning included (Varma & Simon, 2006).

| Key | Default | Effect |
| --- | --- | --- |
| `data.task` | `regression` | `regression` or `classification` |
| `data.target` | `severity` | Target column in `dataset.csv` |
| `model.name` | `ridge` | Regression: `linear`, `ridge`, `svr`, `random_forest`, `mlp`, `torch_mlp`; classification: `logistic`, `svc`, `random_forest`, `mlp`, `torch_mlp` |
| `model.reducer` / `model.n_components` | `none` / `null` | Reduction method and dimension; `null` keeps 95% of the variance for PCA, or $C-1$ for LDA |
| `model.tune` | `false` | Tuning by nested cross-validation |
| `model.class_weight` | `null` | `balanced` weights the classes by their frequency in the training fold (`logistic`, `svc`, `random_forest` only) |
| `model.ridge_alpha`, `model.svr_c`, `model.svr_epsilon` | 1.0, 1.0, 0.1 | Regression hyperparameters |
| `model.hidden_units`, `model.activation`, `model.mlp_alpha` | `[100]`, `logistic`, 1e-4 | Perceptron architecture and L2 penalty |
| `model.early_stopping` / `model.patience` | `false` / 50 | scikit-learn perceptron only: whether to stop early on a validation set of 10% of the training fold; `patience` is the number of epochs allowed without improvement, judged on the training loss when early stopping is off |

The inner tuning grids are: Ridge `alpha` ∈ {0.01, 0.1, 1, 10, 100}; SVR `C` ∈ {0.1, 1, 10} × `epsilon` ∈ {0.05, 0.1, 0.5}; random forest `max_depth` ∈ {unlimited, 4, 8} × `min_samples_leaf` ∈ {1, 3}; perceptron `alpha` ∈ {1e-4, 1e-2, 1}; logistic `C` ∈ {0.01, 0.1, 1, 10}; SVC `C` ∈ {0.1, 1, 10}. Selection is by RMSE for regression and by UAR for classification.

### 5. Cross-validation and metrics

Both the outer evaluation and the inner tuning use k-fold cross-validation (Stone, 1974). Classification uses stratified k-fold, or stratified group k-fold when the dataset has a `group` column (the speaker). With speakers, a class is counted in speakers, not recordings: ten recordings of three speakers cannot fill five folds, and the split stops with an error saying so. I shuffle the group labels and call `StratifiedGroupKFold` with `shuffle=False`, because in scikit-learn 1.7 its `shuffle=True` moves the table of class counts per group without the groups. With 40 speakers of 2 recordings each, half of them in one class, 5 folds then hold 4 to 10 recordings of that class where 8 is possible. Regression uses shuffled k-fold; with groups it follows the `GroupKFold` rule, placing each speaker's recordings as a whole, largest group first, into the fold that currently holds the fewest recordings. Among groups of equal size the seed decides which goes first, rather than the sort algorithm: `argsort` orders ties differently from one NumPy version to the next, and leaving it to `argsort` would give the same seed different folds in different environments. All recordings of one speaker always fall in the same fold, which avoids the optimistic bias of a model that has learned to recognize the speaker; splitting by recording instead of by speaker overestimates performance on new speakers (Saeb et al., 2017).

The regression metrics are MAE, RMSE, $R^2$, Pearson $r$, Spearman $\rho$ and Lin's concordance correlation coefficient (CCC). The classification metrics are unweighted average recall (UAR, Schuller et al., 2009), accuracy and macro F1, plus sensitivity, specificity and ROC-AUC for two-class problems, where `evaluation.positive_class` names the positive class. Every metric is reported per fold, as the mean and sample standard deviation over the folds (`ddof=1`), and as a pooled value computed from all out-of-fold predictions together.

| Key | Default | Effect |
| --- | --- | --- |
| `evaluation.n_splits` | 5 | Number of outer folds |
| `evaluation.n_inner_splits` | 3 | Number of inner tuning folds |
| `evaluation.positive_class` | `dysphonic` | Positive class for two-class problems |
| `seed` | 0 | Master seed for the splits, model initialization and synthetic data |

### 6. Feature-setting ablation and significance tests

Each list in the `ablation` section is one axis of the grid: `n_ceps`, `n_mels`, `delta_order`, `energy_term`, `frame_hop_ms`, `lifter`, `statistics`, `reducer`, `models`. An empty list keeps the value from `frontend`/`pooling`/`model`. The grid expands into every combination, and **the first value of each list makes up the baseline**. Each combination is evaluated with `evaluation.n_splits` folds repeated `ablation.n_repeats` times. The splits depend only on the target, the groups and the seed, so every setting sees **exactly the same** folds and the scores can be compared in pairs. Static cepstra are computed only once per front-end setting (Δ settings aside), and Δ is added for each combination; `--cache-dir` stores them on disk for the next run. `ablate` checks that the grid fits the other sections (for example that `ablation.models` matches `data.task`, and that the largest number of cepstra is below the smallest number of filters) when it expands the grid, before any computation starts. A grid written for another task or front end therefore does not get in the way of `prepare`, `evaluate` and `train`.

For a setting $s$ and the baseline $b$, the per-fold differences $d_j = \text{metric}_s - \text{metric}_b$ over $J = k \times r$ folds are tested with the resampled t-test, using the Nadeau–Bengio correction (Nadeau & Bengio, 2003):

$$t = \frac{\bar d}{\sqrt{\left(\frac{1}{J} + \frac{n_{test}}{n_{train}}\right) s_d^2}}, \qquad \text{df} = J - 1$$

The term $n_{test}/n_{train}$ reflects that the training sets of the folds overlap, so the scores are not independent; an ordinary paired t-test that ignores it badly overstates significance. A Wilcoxon signed-rank test (Demšar, 2006) is included as a nonparametric reference, and each of the two sets of p-values is corrected with Holm's method (Holm, 1979) or with Benjamini–Hochberg (Benjamini & Hochberg, 1995).

| Key | Default | Effect |
| --- | --- | --- |
| `ablation.n_ceps` | `[12, 6, 18]` | Numbers of cepstra to compare (the first is the baseline) |
| `ablation.delta_order` | `[0, 1, 2]` | 13/26/39 dimensions |
| `ablation.models` | `[ridge, svr]` | Estimators to compare |
| `ablation.n_repeats` | 3 | Number of k-fold repeats |
| `ablation.metric` | `null` | Metric for ranking and testing; `null` means RMSE (regression) or UAR (classification) |
| `analysis.correction` | `holm` | `holm` or `fdr_bh` |

### 7. Least-squares inference and residual diagnostics

For a design matrix $X$ ($n \times p$, intercept included), $\hat\beta = \arg\min \lVert y - X\beta \rVert^2$ is solved by QR decomposition (Draper & Smith, 1998), and the report gives

$$\hat\sigma^2 = \frac{\text{RSS}}{n-p}, \quad \text{SE}(\hat\beta_j) = \hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}, \quad R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}, \quad \bar R^2 = 1 - (1-R^2)\frac{n-1}{n-p}, \quad F = \frac{\text{ESS}/(p-1)}{\text{RSS}/(n-p)}$$

together with $t$-based confidence intervals, the Gaussian log-likelihood, AIC (Akaike, 1974), BIC (Schwarz, 1978) and the condition number of the design matrix. When there are as many parameters as observations, the residual degrees of freedom are 0 and the model merely interpolates the data: standard errors, tests and $R^2$ are all output as NaN and marked as not estimable. More parameters than observations, or linearly dependent columns, raise an error.

The residual diagnostics are the Shapiro–Wilk (Shapiro & Wilk, 1965) and Jarque–Bera (Jarque & Bera, 1980) normality tests, the Breusch–Pagan test for heteroscedasticity computed as $nR^2$ (Breusch & Pagan, 1979; Koenker, 1981), the Durbin–Watson statistic (Durbin & Watson, 1950), leverage from the diagonal of the hat matrix, Cook's distance (Cook, 1977) $D_i = \frac{e_i^2}{p\hat\sigma^2}\frac{h_{ii}}{(1-h_{ii})^2}$, and the variance inflation factor $\text{VIF}_j = 1/(1 - R_j^2)$ (Marquardt, 1970). The second-order response surface $y = \beta_0 + \sum_i \beta_i x_i + \sum_i \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j$ (Box & Wilson, 1951) is built from centered variables, so that $x$ and $x^2$ are not nearly collinear.

| Key | Default | Effect |
| --- | --- | --- |
| `analysis.ci_level` | 0.95 | Level of the confidence intervals for coefficients and correlation coefficients |

### 8. Stepwise selection, polynomial order and gradient descent

Stepwise selection starts from the empty model (or the full one). At each step it first tries to add the variable with the smallest partial-F p-value below `p_enter`; only when nothing can be added does it try to remove the variable with the largest p-value above `p_remove`, and it stops when neither is possible (Efroymson, 1960). It requires `p_enter ≤ p_remove` and records the models it has visited, so it cannot cycle forever. The action, F, p, $R^2$ and adjusted $R^2$ of every step are written out. Check the VIF of the selected model and validate it out of sample, because p-values that the selection has already used cannot be read as inference.

The polynomial order sweep lists, for every order, the in-sample $R^2$, adjusted $R^2$, overall F-test, SSE, AIC and BIC, next to the out-of-sample leave-one-out RMSE (in closed form from the PRESS residuals $e_i/(1-h_{ii})$) and k-fold RMSE (with the polynomial basis rebuilt on each training fold). By default the basis is a set of discrete orthogonal polynomials built by the three-term recurrence

$$p_{k+1}(x) = (x - a_k)\,p_k(x) - b_k\,p_{k-1}(x)$$

whose columns are mutually orthogonal, with a condition number of 1, whereas an uncentered $x^6$ (with years, say, where $x \approx 2000$) makes the design matrix numerically singular.

Gradient descent minimizes $J = \frac{1}{2m}\lVert Z\theta - y\rVert^2$ with the vectorized simultaneous update $\theta \leftarrow \theta - \frac{\alpha}{m} Z^\top (Z\theta - y)$ (Cauchy, 1847). The variables are standardized first and the coefficients converted back to the original units afterwards. Convergence is judged by the relative decrease of the cost, with an iteration cap and a divergence check, and the result is compared with the closed-form least-squares solution.

| Key | Default | Effect |
| --- | --- | --- |
| `analysis.p_enter` / `analysis.p_remove` | 0.05 / 0.10 | Entry and removal thresholds for stepwise selection |
| `analysis.stepwise_start` | `empty` | `empty` (forward) or `full` (backward) |
| `analysis.max_degree` | 8 | Highest order in the sweep |
| `analysis.polynomial_basis` | `orthogonal` | `orthogonal`, `centered` or `raw` |
| `analysis.order_criterion` | `kfold_rmse` | Criterion for the recommended order: `kfold_rmse`, `loocv_rmse`, `bic`, `aic` |

### 9. Correlation and agreement

Whether a feature moves together with the rating, and whether the predictions **equal** the clinical rating, are two different questions. For the first, every feature column gets Pearson $r$ with the Fisher $z$ confidence interval $\tanh(\operatorname{atanh} r \pm z_{1-\alpha/2}/\sqrt{n-3})$ (Fisher, 1915) and the Spearman rank correlation $\rho$ (Spearman, 1904), and the whole set of p-values is corrected for multiple comparisons. The second uses Lin's concordance correlation coefficient (Lin, 1989)

$$\rho_c = \frac{2 s_{xy}}{s_x^2 + s_y^2 + (\bar x - \bar y)^2}$$

which penalizes scatter and systematic bias at once (if every prediction is 10 points too high, Pearson $r$ is still 1 but $\rho_c$ drops). The second question also gets the Bland–Altman mean difference with 95% limits of agreement $\bar d \pm 1.96\,s_d$ (Bland & Altman, 1986). `evaluate` writes both automatically for regression tasks.

### 10. Synthetic data

All tests and examples run on sustained vowels synthesized with a simple source–filter model. The source is the Rosenberg glottal pulse (Rosenberg, 1971), with the opening phase taking 40% of the period and the closing phase 16%; the length and amplitude of each period get Gaussian perturbations whose relative standard deviations are the jitter and the shimmer. The vocal tract is three second-order formant filters in cascade, with the formants of /a/, /i/ and /u/ taken from the male averages of Peterson & Barney (1952), followed by a first difference for lip radiation. Finally, white noise filtered by the same vocal tract is added at a level set by the harmonics-to-noise ratio (HNR).

Severity $s \in [0, 100]$ moves jitter, shimmer and HNR linearly between the two ends of `synthetic.*_range`. Each speaker has their own mean pitch and severity, and one speaker's recordings vary in severity, vowel, pitch and loudness, so a model has to find the severity cues among nuisance factors, while the `group` column lets cross-validation keep speakers apart. Recordings with a severity at or above `label_threshold` are labeled `dysphonic` and the rest `healthy`, for classification. There are also three toy datasets for the regression tools: a second-order response surface with known coefficients, a cubic polynomial, and four mixture proportions that add up to nearly 100% (multicollinear on purpose).

| Key | Default | Effect |
| --- | --- | --- |
| `synthetic.n_speakers` / `synthetic.n_recordings_per_speaker` | 20 / 3 | Number of speakers and recordings per speaker |
| `synthetic.duration_s` / `synthetic.sample_rate` | 1.0 / 16000 | Recording length and sample rate |
| `synthetic.jitter_range` | `[0.002, 0.03]` | Period perturbation at severity 0 and 100 |
| `synthetic.shimmer_range` | `[0.02, 0.3]` | Amplitude perturbation at severity 0 and 100 |
| `synthetic.hnr_range_db` | `[30.0, 6.0]` | Harmonics-to-noise ratio at severity 0 and 100 |
| `synthetic.label_threshold` | 35.0 | Threshold for the classification label |

### 11. Comparing feature families on shared folds

`ablate` varies the MFCC settings inside the package. To set MFCCs against features from elsewhere, such as clinical voice measures from Praat or the layers of a speech representation network, `compare` takes several feature files of the same recordings and evaluates the configured model on each with **the same** `evaluation.n_splits` × `evaluation.n_repeats` folds. Files from other front ends come in through `feature_set_from_table` (see Preparing data, section 4) and carry their own column names and, optionally, column blocks. Rows are matched by `path`, `start_s` and `end_s`, so the files may list the recordings in any order, but they must hold the same recordings, targets and groups.

Everything that is chosen is chosen inside the training folds: median imputation with a 0/1 indicator for every column that had gaps (`model.impute: median`, for measures such as jitter that are undefined when a voice has too few periods), standardization, the ridge penalty from `model.ridge_alphas`, and, with `model.block_selection`, the block. For a network with 13 hidden layers the inner loop picks the layer; picking the layer with the best test-fold score would make every layer-wise model look better than it is. For ridge regression the inner search factorizes each imputed and scaled inner training fold once, $X = USV^T$, and gets every penalty from the same factors, $\hat\beta_\alpha = V\,\mathrm{diag}(s/(s^2+\alpha))\,U^T y$ (Hastie et al., 2009, section 3.4.1). It picks what GridSearchCV would pick (`tests/test_models.py` compares the two), and with 14 blocks of 1536 columns and 21 penalties one training fold took 10.7 s instead of 90.8 s on my machine.

The point estimate of a file is the metric of all out-of-fold predictions of one repetition, averaged over the repetitions. Its interval comes from a speaker-level bootstrap: each resample draws whole groups with replacement (Davison & Hinkley, 1997, section 3.8), and the percentile interval is reported (Efron & Tibshirani, 1993). For a pair of files, `compare` reports the bootstrap interval of the difference, with both files scored on the same resamples, together with the corrected t-test and the Wilcoxon test on the paired fold scores (section 6), Holm-corrected over the contrasts tested. With `--margin m` a contrast is also marked `equivalent` when its 90% interval lies inside $(-m, m)$, the interval form of two one-sided tests at 5% (Schuirmann, 1987).

Class labels work the same way. With `data.task: classification` the files hold class names, and every repetition keeps the predicted class and a `score_<class>` per class. The summary gives UAR (the default metric; `--metric` picks another), accuracy, macro F1 and, for two classes, sensitivity, specificity and ROC-AUC with `evaluation.positive_class` as the positive class. `strata` in `compare_feature_sets` stratifies the folds on label × stratum, for example label × corpus. The bootstrap uses weighted recalls and the weighted Mann–Whitney form of the ROC-AUC (Hanley & McNeil, 1982). With few speakers a resample can lose a whole class, and its UAR, sensitivity, specificity or ROC-AUC is then undefined. Those resamples stay out of the interval and are counted in `<metric>_n_undefined` of the summary and `n_undefined` of the contrasts; a large count means the interval rests on the resamples that happened to keep every class.

`permutation_test` asks what score the whole procedure reaches when the labels carry no information (Ojala & Garriga, 2010). Speakers swap labels, all recordings of a speaker moving together, and with `strata` they swap only inside their stratum, so what the stratum explains stays in the null. Every permutation repeats the cross-validation with its inner tuning, because that selection is part of what is tested, so it costs `n_permutations` × folds × repetitions fits. The p-value is $(1 + \#\{\text{null} \ge \text{observed}\}) / (1 + n)$ (Phipson & Smyth, 2010). `permute_group_labels` is the swap on its own, for scoring stored predictions under permuted labels.

`--jobs` does not change which settings the folds choose, but it can change the last bits of the numbers, because joblib gives each worker `cpu_count // jobs` BLAS threads. A comparison of 8 feature files on 287 recordings, run with 16 workers (1 thread each on my 20 logical CPUs) and again with 4 workers (5 threads each), differed by at most 5.3e-13 in a prediction and chose the same settings in all 400 folds. With 1 thread per worker in both runs (`joblib.parallel_config(backend="loky", inner_max_num_threads=1)` around the call) the tables were identical byte for byte, so to repeat a run exactly, keep the threads per worker the same.

| Key | Default | Effect |
| --- | --- | --- |
| `model.impute` | `none` | `none`, or `median` with missing-value indicators |
| `model.block_selection` | `false` | Choose the block in the inner loop (needs `model.tune`) |
| `model.ridge_alphas` | `[]` | Ridge penalties searched by `model.tune`; `[]` keeps 0.01–100 |
| `evaluation.n_repeats` | 3 | Repetitions of the k-fold split in `compare` |
| `analysis.n_resamples` | 10000 | Bootstrap resamples |

### 12. Rater agreement and the ceiling it sets

A model is scored against the mean of a few clinicians' ratings, and that mean is noisy too. `icc_oneway` gives the one-way random-effects ICC(1,1) and ICC(1,k) (Shrout & Fleiss, 1979) with F-based intervals. The model is one-way because the rater columns of a clinical corpus are usually positions: the second rater of one recording need not be the second rater of the next. With unequal numbers of ratings, $k$ becomes $k_0$ (McGraw & Wong, 1996). Two numbers follow for a model:

- Noise floor: even a perfect predictor of the true score has an expected RMSE of $\sqrt{\overline{s_i^2/k_i}}$ against the mean of $k_i$ ratings with sample variance $s_i^2$ (`rating_noise_floor`). No model can be expected to go below it, and its correlation with the mean rating is bounded by $\sqrt{ICC(1,k)}$.
- One rater: `leave_one_rater_out` pairs every rating with the mean of the other ratings of the same recording. The RMSE of those pairs is how far one clinician is from the panel, and a model scored on the same references (the `reference` column) is compared with one clinician on equal terms.

`tests/test_reliability.py` reproduces ICC(1,1) = .17 and ICC(1,4) = .44 from the worked example of Shrout & Fleiss (1979).

---

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/recklight/voice-severity-bench.git
cd voice-severity-bench
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .                 # core (also installs speechdsp from PyPI)
pip install -e ".[dl]"           # PyTorch models (optional)
pip install -e ".[dev]"          # development tools: pytest, ruff
```

Without PyTorch, everything except `torch_mlp` works; if `torch_mlp` is configured but PyTorch is not installed, the error message says to run `pip install -e ".[dl]"` in the project folder. You can also skip installing and run `PYTHONPATH=src python -m voice_severity_bench --help` from inside the project folder.

---

## Quick start

No data is needed: the first command generates a synthetic dataset.

```bash
# 1. Generate a synthetic dataset
voice-severity-bench synthesize --config configs/quick_demo.yaml --out data/synthetic
# 2. Prepare the model inputs from the dataset index
voice-severity-bench prepare --config configs/quick_demo.yaml --out outputs/demo/prepare
# 3. Cross-validate
voice-severity-bench evaluate outputs/demo/prepare/features.npz --config configs/quick_demo.yaml --out outputs/demo/evaluate
# 4. Train on all the data
voice-severity-bench train outputs/demo/prepare/features.npz --config configs/quick_demo.yaml --out outputs/demo/train
# 5. Predict a new input file
voice-severity-bench predict data/synthetic/items/item_000.wav --model outputs/demo/train/model.joblib --out outputs/demo/predict
```

Or run the whole pipeline in one go: `python examples/end_to_end_synthetic.py`.

> The scores from these commands come from synthetic vowels and only confirm that the pipeline runs. The first point under [Known limitations](#known-limitations) explains why they are not representative of real recordings.

---

## Usage

### CLI

A typical study with your own recordings and ratings table:

```bash
# 1. Build the dataset index from the ratings table (ratings.csv columns: id, severity, group)
voice-severity-bench index data/recordings --targets data/ratings.csv
# 2. Prepare features: 25 ms / 10 ms, 26 filters, log energy (configs/compact_frontend.yaml)
voice-severity-bench prepare --config configs/compact_frontend.yaml --dataset data/recordings/dataset.csv --out outputs/study/prepare
# 3. Correlation of every feature column with severity, Holm-corrected
voice-severity-bench correlate outputs/study/prepare/features.npz --out outputs/study/correlate
# 4. Cross-validate SVR: --set switches the model and turns on nested tuning, --seed changes the splits
voice-severity-bench evaluate outputs/study/prepare/features.npz --config configs/compact_frontend.yaml --set model.name=svr --set model.tune=true --seed 7 --out outputs/study/evaluate_svr
# 5. Compare feature settings: number of cepstra × 13/26/39 dimensions × model, with the static cepstra cached
voice-severity-bench ablate --config configs/compact_frontend.yaml --dataset data/recordings/dataset.csv --out outputs/study/ablate --cache-dir outputs/study/cache
# 6. Least-squares inference and residual diagnostics for a few features
voice-severity-bench regress outputs/study/prepare/features.npz --predictors std_c8,mean_c10 --out outputs/study/regress
# 7. Train on all the data, then predict new recordings
voice-severity-bench train outputs/study/prepare/features.npz --config configs/compact_frontend.yaml --out outputs/study/train
voice-severity-bench predict data/new/visit_01.wav data/new/visit_02.wav --model outputs/study/train/model.joblib --out outputs/study/predict
```

`regress`, `stepwise` and `order-sweep` also accept an ordinary CSV table, for example `voice-severity-bench stepwise table.csv --target heat`, `voice-severity-bench order-sweep curve.csv --x year --target population` or `voice-severity-bench regress surface.csv --target y --design quadratic`.

### Python API

```python
import pandas as pd

from voice_severity_bench import (
    Config,
    FrontendConfig,
    cross_validate,
    feature_layout,
    frame_features,
    make_vowel,
    pool_frames,
    prepare_features,
    run_ablation,
    stepwise_select,
    write_synthetic_dataset,
)

# 1. Low level: a synthetic /a/ at severity 60, 39-dimensional frame features
signal = make_vowel(60.0, f0_hz=140.0, vowel="a", rng=0)
frontend = FrontendConfig()
frames = frame_features(signal, 16_000, frontend)
print(frames.shape)                                  # (65, 39)
names = feature_layout(frontend)
print(names[:2], names[12], names[13], names[-1])     # ('c1', 'c2') c0 d_c1 dd_c0

# 2. Pool all frames into one utterance vector (39 columns × mean and std)
print(pool_frames(frames, ["mean", "std"]).shape)    # (78,)

# 3. High level: synthetic dataset → features → speaker-grouped cross-validation
config = Config.from_yaml("configs/quick_demo.yaml")
index = write_synthetic_dataset("data/synthetic", config.synthetic, rng=config.seed)
features = prepare_features(index, config)
print(features.X.shape)                              # (24, 78)
result = cross_validate(features, config)
print(result.metrics["summary"]["ccc"])              # {'mean': 0.835..., 'std': 0.166...}

# 4. Feature-setting ablation: 13 vs. 39 dimensions on the same splits
ablation = run_ablation(index, config)
print(ablation.summary[["config_id", "n_features", "rmse_mean"]])
#        config_id  n_features  rmse_mean
# 0  delta_order=0          26  14.008524
# 1  delta_order=2          78  16.212205

# 5. Stepwise selection (p-values after selection are not inference; validate out of sample)
X = pd.DataFrame(features.X, columns=features.feature_names)
selection = stepwise_select(X, features.y, X.columns)
print(selection.selected)                            # ('mean_c10', 'std_c4', 'std_c10', 'std_d_c9')
```

### Example scripts

| Script | What it shows | Run time |
| --- | --- | --- |
| `examples/end_to_end_synthetic.py` | Synthetic data → features → grouped cross-validation → training → prediction | about 2 s |
| `examples/compare_feature_settings.py` | Ablation of number of cepstra × Δ order × model, with ranking and corrected t-tests | about 3 s |
| `examples/compare_feature_families.py` | MFCC-13, MFCC-39, periodicity measures with gaps and blocked band energies on the same folds, next to the rating noise floor and one simulated rater | about 12 s |
| `examples/compare_severity_regressors.py` | RMSE and CCC of five regressors, Bland–Altman, feature correlations | about 3 s |
| `examples/diagnose_regression_models.py` | Response-surface OLS, VIF and stepwise selection on the cement data, residual diagnostics | about 2 s |
| `examples/select_polynomial_order.py` | Orders 1–8: in-sample $R^2$ vs. LOOCV/k-fold error | about 2 s |
| `examples/compare_gradient_descent_with_ols.py` | How the learning rate and standardization affect convergence of gradient descent, checked against the closed form | about 2 s |

Every script takes `--out`, `--seed` and `--quick` (which reads `configs/quick_demo.yaml`, or for `compare_feature_families.py` shrinks `configs/compare_demo.yaml` to the same size; the gradient-descent and regression-diagnostics scripts use none of the settings in it that differ from the defaults, so their results are the same either way), and writes to `examples/output/<script name>/` by default. The run times were measured on my machine and include Python start-up.

---

## CLI reference

| Command | Purpose | Main input | Main output |
| --- | --- | --- | --- |
| `synthesize` | Generate a synthetic sustained-vowel dataset | Configuration file | `items/item_NNN.wav`, `dataset.csv` |
| `index` | Build the dataset index from a ratings table | Recording folder, `--targets` ratings table | `<folder>/dataset.csv` |
| `prepare` | One pooled cepstral vector per recording | `dataset.csv` | `features.npz`, `items.csv` |
| `evaluate` | Grouped or stratified k-fold cross-validation | `features.npz` | `metrics.json`, `folds.csv`, `predictions.csv`, figures and CSVs |
| `train` | Train on all the data | `features.npz` | `model.joblib` (`model.pt` for `torch_mlp`) |
| `predict` | Predict new recordings with the settings stored in the model file | WAV files, `--model` | `predictions.csv` |
| `extract` | Export frame-level features for every file in a folder tree | Recording folder | Mirrored `features/` folder, `extraction.csv` |
| `ablate` | Compare feature settings and models | `dataset.csv` | `ablation_folds.csv`, `ablation_summary.csv`, `ablation_comparison.csv`, `ablation.json`, figures |
| `compare` | Compare feature files from different front ends on the same folds | `NAME=features.npz` for each file | `comparison_folds.csv`, `comparison_items.csv`, `comparison_predictions.csv`, `comparison_summary.csv`, `comparison_contrasts.csv`, `comparison.json`, figures |
| `correlate` | Correlation of each feature column with the rating | `features.npz` or CSV | `correlations.csv`, `feature_correlations.*` |
| `regress` | Least-squares inference and residual diagnostics | `features.npz` or CSV | `coefficients.csv`, `regression_summary.json`, `vif.csv`, `regression_diagnostics.*` |
| `stepwise` | Stepwise selection by partial F-test | `features.npz` or CSV | `stepwise_steps.csv`, `coefficients.csv`, `regression_summary.json`, `vif.csv` |
| `order-sweep` | Polynomial order sweep | CSV (or `features.npz`) | `order_sweep.csv`, `order_sweep.*` |

| Option | Commands | Description |
| --- | --- | --- |
| `--log-level LEVEL` | Global (before the subcommand) | `DEBUG`, `INFO` (default), `WARNING` or `ERROR`; logs go to stderr |
| `--version` | Global | Print the version and exit |
| `--config/-c PATH` | All except `predict` | YAML configuration file; the defaults are used without it |
| `--set/-s KEY=VALUE` | Same as above | Override one setting, for example `--set model.name=svr`; can be repeated |
| `--seed INT` | `synthesize`, `evaluate`, `train`, `ablate`, `compare`, `order-sweep` | Same as `--set seed=INT` |
| `--dataset/-d PATH` | `prepare`, `ablate` | Same as `--set data.dataset=PATH` |
| `--out/-o PATH` | Commands that write files, except `index` | Output folder; without it, `<command>_<UTC time>` is created under `output.root` (`synthesize` defaults to `data/synthetic`). `index` always writes `dataset.csv` inside the recording folder |
| `--model/-m PATH` | `predict` | Model file written by `train` (required) |
| `--figures/--no-figures` | `evaluate`, `ablate`, `compare`, `correlate`, `regress`, `stepwise`, `order-sweep` | Whether to draw figures (each figure comes with a CSV of the same name) |
| `--target/-t COLUMN` | `correlate`, `regress`, `stepwise`, `order-sweep` | Response column of a CSV table (not needed for `.npz`) |
| `--predictors/-p A,B` | `regress`, `stepwise` | Comma-separated predictors; all numeric columns by default |
| `--design` | `regress` | `linear` or `quadratic` (second-order response surface) |
| `--x COLUMN` | `order-sweep` | The single predictor column |
| `--format` | `extract` | `npz`, `htk` or `text` |
| `--cache-dir PATH` | `ablate` | Disk cache for the static cepstra |
| `--contrast FIRST:SECOND` | `compare` | Pair of feature files to test, repeatable; every file against the first by default |
| `--margin M` | `compare` | Equivalence margin in units of the metric |
| `--metric NAME` | `compare` | Ranking metric; `rmse` for regression and `uar` for classification by default |
| `--jobs N` | `compare` | Folds fitted in parallel; the chosen settings do not depend on it (Method, section 11) |
| `--top N` | `correlate` | Number of features shown in the figure |
| `--targets PATH`, `--id-column` | `index` | Ratings table and its ID column |

Exit codes: `0` means success; `2` means a problem with the input or the configuration (only a single `error: ...` line is printed; add `--log-level DEBUG` to see the full traceback); `1` means an interruption or a bug in the program itself.

---

## Project layout

```text
voice-severity-bench/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml                    # bug report form
│   │   ├── config.yml                        # disables blank issues
│   │   └── feature_request.yml               # feature request form
│   └── workflows/
│       └── ci.yml                            # lint, 3.10 syntax gate, Linux/Windows × Python 3.10–3.13, oldest dependencies, PyTorch
├── configs/
│   ├── default.yaml                          # identical to Config(), with a comment on every key
│   ├── quick_demo.yaml                       # small synthetic set, runs in a few seconds
│   ├── compare_demo.yaml                     # settings of examples/compare_feature_families.py
│   ├── quick_classification.yaml             # the same, as binary detection
│   ├── classification.yaml                   # binary detection on the label column
│   └── compact_frontend.yaml                 # 25 ms / 10 ms, 26 filters, log energy
├── data/
│   └── .gitkeep                              # default place for data; the contents stay out of git
├── outputs/
│   └── .gitkeep                              # default root for run folders; the contents stay out of git
├── docs/
│   ├── images/workflow.png                   # workflow figure in README.md
│   ├── images/workflow.zh-TW.png             # workflow figure in README.zh-TW.md
│   ├── workflow.html                         # interactive workflow diagram (open in a browser)
│   └── workflow.zh-TW.html                   # the same diagram in Traditional Chinese
├── examples/
│   ├── end_to_end_synthetic.py               # the whole pipeline
│   ├── compare_feature_settings.py           # feature-setting ablation
│   ├── compare_feature_families.py           # feature files from four front ends on shared folds
│   ├── compare_severity_regressors.py        # regressor comparison, agreement and correlation
│   ├── diagnose_regression_models.py         # response surface, collinearity, stepwise selection, residuals
│   ├── select_polynomial_order.py            # polynomial order selection
│   └── compare_gradient_descent_with_ols.py  # gradient descent vs. the closed form
├── paper/                                    # the report on the PVQD study: PDF and LaTeX source
├── studies/
│   └── voice-severity/                       # the PVQD study: frozen plan, scripts, freeze record
├── src/
│   └── voice_severity_bench/
│       ├── __init__.py                       # public API
│       ├── __main__.py                       # python -m voice_severity_bench
│       ├── py.typed                          # marker for type information
│       ├── cli.py                            # typer command line
│       ├── config.py                         # configuration dataclasses and YAML I/O
│       ├── errors.py                         # exception classes
│       ├── logging_utils.py                  # logging setup
│       ├── manifest.py                       # run folders and manifest.json
│       ├── dataset.py                        # dataset.csv, features.npz, tables for analysis
│       ├── synthetic.py                      # synthetic vowels and synthetic regression data
│       ├── corpus.py                         # loading recordings, batch extraction, HTK/text output
│       ├── cepstral_frontend.py              # MFCC front end and Δ/ΔΔ
│       ├── pooling.py                        # utterance-level pooling
│       ├── pipeline.py                       # prepare/train/predict high-level API
│       ├── models.py                         # estimator registry, reduction, tuning, model files
│       ├── neural.py                         # PyTorch perceptron (dl extra)
│       ├── splitting.py                      # grouped and stratified cross-validation splits
│       ├── evaluation.py                     # cross-validation and metrics
│       ├── ablation.py                       # feature-setting grid and paired tests
│       ├── comparison.py                     # feature files compared on shared folds
│       ├── bootstrap.py                      # speaker-level bootstrap intervals
│       ├── permutation.py                    # speaker-level permutation test of a classifier
│       ├── reliability.py                    # ICC, rating noise floor, leave one rater out
│       ├── association.py                    # correlation, CCC, Bland–Altman, p-value correction
│       ├── ols.py                            # least-squares inference
│       ├── regression_diagnostics.py         # residual, influence and collinearity diagnostics
│       ├── design_matrix.py                  # orthogonal polynomials and second-order response surface
│       ├── polynomial_order.py               # polynomial order sweep
│       ├── stepwise.py                       # stepwise selection
│       ├── gradient_descent.py               # gradient descent
│       ├── reference_data.py                 # cement heat-of-hardening data, Shrout & Fleiss ratings
│       └── figures.py                        # figures and tables with the same data
├── tests/                                    # a test_<module>.py for most modules, plus examples, 3.10 compatibility and documentation checks
├── .gitattributes
├── .gitignore
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── README.zh-TW.md
└── pyproject.toml
```

---

## Preparing data

**No data comes with this project.** Prepare your own recordings and ratings, put them in any folder (the default `data/` is kept out of git), and describe them with a dataset index.

### 1. The dataset index, dataset.csv

| Column | Required | Description |
| --- | --- | --- |
| `path` | Yes | Path to the recording, relative to the folder that holds `dataset.csv`, with `/` as the separator |
| `severity` (or the column named by `data.target`) | Yes | Numeric rating for regression, for example CAPE-V 0–100 or the GRBAS G score 0–3 |
| `label` | Only for classification | Class name (a string), used with `data.task: classification` and `data.target: label` |
| `group` | No | Speaker or recording session; recordings of one group never appear in a training fold and a test fold at the same time |
| `start_s`, `end_s` | No | Interval of the recording to analyse, in seconds (section 4) |

```csv
path,severity,label,group
visit1/p001_a.wav,12.5,healthy,p001
visit1/p001_i.wav,15.0,healthy,p001
visit1/p002_a.wav,68.0,dysphonic,p002
```

Labels and groups are read only from columns; file names are never parsed. If the ratings are in a separate table (columns `id, severity, group`), `voice-severity-bench index <recording folder> --targets <ratings table>` matches them to the files by file stem or relative path and writes `<recording folder>/dataset.csv`. A missing file, a duplicate ID, an ID that matches several files, or two IDs that match the same file (for example `p001` by stem and `sub/p001` by relative path, both reaching the same recording) is an error. So is a file listed twice in `dataset.csv`, because that recording could then land in a training fold and a test fold at the same time.

### 2. File formats

- Recordings: WAV (8/16/32-bit PCM or floating point). Multichannel audio is averaged to mono and converted to float64 in $[-1, 1]$.
- Sample rate: any. Frame length, hop and mel filters are all computed at each file's actual sample rate. A dataset that mixes sample rates logs a warning; I recommend resampling everything to one rate with `audio.sample_rate` (polyphase filtering).
- Length and content: at least one analysis frame. For files that are shorter than one frame, cannot be read, or contain NaN/infinite samples (which can happen in floating-point WAV), `prepare` reports the number of problem files and the first five names, then stops; `extract` marks them `too_short` or `error` in `extraction.csv` and goes on with the other files.
- Trimming: `audio.trim_s` cuts a fixed number of seconds from each end, and `audio.trim_silence` removes leading and trailing silence by endpoint detection. Both are off by default.

### 3. Public reference data

`reference_data.py` includes the 13 observations of the Portland cement heat-of-hardening data (Woods et al., 1932; Hald, 1952): the percentages of four clinker components and the heat given off while the cement hardens. It is the classic example for stepwise regression and multicollinearity, used in `examples/diagnose_regression_models.py`, in doctests and in `tests/test_reference_data.py` (which checks the regression results given in textbooks). The 6 × 4 rating matrix of Shrout & Fleiss (1979) is there too, for the ICC tests. All the other tests use synthetic data.

### 4. Intervals and feature files from other front ends

`dataset.csv` can carry `start_s` and `end_s` in seconds; an empty cell leaves that end open. Only that part of the file is analysed, cut at the file's own sample rate before any resampling, so the resampling filter sees no edge inside the recording. One recording can then appear in several rows, for example its sustained vowel and its sentences; give those rows the same `group` so they never end up on both sides of a fold. The same file with the same interval twice is still an error. `audio.rms_dbfs` scales every interval to a fixed RMS level, so differences in recording level cannot reach `c0`.

```csv
path,severity,group,start_s,end_s
p001.wav,12.5,p001,1.20,4.05
p001.wav,12.5,p001,6.10,
p002.wav,68.0,p002,0.85,2.40
```

Features computed outside the package become a `features.npz` through `feature_set_from_table` and `save_features`. The table holds `path`, the target, optionally `group`, `start_s` and `end_s`, and the feature columns; `blocks` names the columns of each block, for example one block of 1536 columns per network layer. NaN is allowed and handled by `model.impute`. The file also stores its `kind` and an `extractor` dictionary for provenance (versions, settings, checksums). `evaluate`, `train` and `compare` read these files like the ones from `prepare`; only `compare` uses the blocks, and only MFCC files have their front-end settings checked against the configuration.

```python
import pandas as pd
from voice_severity_bench import feature_set_from_table, save_features

table = pd.read_csv("clinical_measures.csv")   # path, group, severity, cpps_db, hnr_db, ...
clinical = feature_set_from_table(
    table, target="severity", kind="clinical", extractor={"script": "measure_voice.py"}
)
save_features("features/clinical.npz", clinical)
```

```bash
voice-severity-bench compare mfcc=features/mfcc.npz clinical=features/clinical.npz \
    --set model.tune=true --set model.impute=median --margin 2 --out outputs/compare
```

---

## Configuration

`configs/default.yaml` is identical to `Config()` and has a comment on every key, so it can serve as a reference when you write your own configuration. `configs/quick_demo.yaml` lists only the keys that differ from the defaults and shrinks the synthetic data to 8 speakers and 24 recordings of 0.5 s; the quick start, the examples' `--quick` and CI use it. `configs/quick_classification.yaml` is binary detection at the same size, `configs/classification.yaml` switches any configuration to binary detection on the `label` column, and `configs/compact_frontend.yaml` shows another common set of MFCC parameters, and `configs/compare_demo.yaml` holds the settings of `examples/compare_feature_families.py`. Without `--config`, the CLI uses `Config()` and does not read any file on its own.

| Section | Main keys | Description |
| --- | --- | --- |
| (top level) | `seed` | Master seed |
| `data` | `dataset`, `task`, `target` | Dataset index, task and target column |
| `synthetic` | `n_speakers`, `jitter_range`, `hnr_range_db` | Size and generation parameters of the synthetic data |
| `audio` | `sample_rate`, `trim_s`, `trim_silence`, `rms_dbfs` | Resampling, trimming and level when recordings are loaded |
| `frontend` | `frame_ms`, `n_mels`, `n_ceps`, `energy_term`, `delta_order` | Cepstral front end |
| `pooling` | `statistics`, `normalization` | Utterance-level pooling |
| `analysis` | `ci_level`, `correction`, `p_enter`, `max_degree`, `n_resamples` | Statistical inference, stepwise selection, the order sweep and the bootstrap |
| `ablation` | `n_ceps`, `delta_order`, `models`, `n_repeats` | Ablation grid |
| `model` | `name`, `reducer`, `tune`, `impute`, `class_weight`, `block_selection` | Estimator, hyperparameters, missing values, class weights and block choice |
| `evaluation` | `n_splits`, `n_repeats`, `positive_class` | Cross-validation |
| `output` | `root`, `figure_format`, `dpi` | Output location and figure format |

The `audio`, `frontend` and `pooling` sections, together with `data.task` and `data.target`, determine what goes into `features.npz`. If `evaluate` or `train` finds that these settings differ from the ones `prepare` used, it stops with an error. `predict` uses only the settings stored in the model file, which is why it has no `--config`.

An unknown key is an error, and the message lists the valid keys, for example:

```text
$ voice-severity-bench prepare --set frontend.n_cepz=3
error: unknown configuration key 'frontend.n_cepz'; valid keys here are ['delta_order', 'delta_widths', 'energy_term', 'fmax_hz', 'fmin_hz', 'frame_ms', 'hop_ms', 'lifter', 'log_floor', 'n_ceps', 'n_fft', 'n_mels', 'preemphasis', 'window']
```

---

## Results and metrics

The run folder from a single `evaluate` run on a regression task:

```text
outputs/demo/evaluate/
├── config.yaml          # the full configuration actually used
├── manifest.json        # command, time, versions, git commit, seed, SHA-256 of the input files
├── metrics.json         # per-fold, mean ± std and pooled metrics
├── folds.csv            # one row per fold
├── predictions.csv      # one out-of-fold prediction per recording
├── residuals.png        # predicted against actual, residuals against predicted
├── residuals.csv
├── bland_altman.png     # agreement analysis
└── bland_altman.csv
```

| File | Contents |
| --- | --- |
| `metrics.json` | `task`, `n_items`, `n_splits`, `seed`, `folds`, `summary` (`mean` and `std` of each metric), `pooled` (with `bland_altman` added for regression; `classes`, `confusion_matrix` and `per_class` for classification) |
| `folds.csv` | `fold`, `n_train`, `n_test` and every metric |
| `predictions.csv` | `path`, `group`, `fold`, the target column, `predicted`, plus a `score_<class>` per class for classification |
| `residuals.*`, `bland_altman.*` | Figures for regression tasks, with tables of the same data |
| `confusion_matrix.*`, `roc_curve.*` | Figures for classification tasks, with tables of the same data (ROC only for two classes) |

| Metric | Definition | Why look at it |
| --- | --- | --- |
| MAE | $\frac1n\sum \lvert \hat y_i - y_i\rvert$ | Average error in rating units, easy to explain to clinicians |
| RMSE | $\sqrt{\frac1n\sum(\hat y_i - y_i)^2}$ | More sensitive to large errors; the ablation ranks by it by default |
| $R^2$ | $1 - \text{SSE}/\text{SST}$ (each fold against that fold's mean) | How much of the variance is explained; swings widely on small folds |
| Pearson $r$/Spearman $\rho$ | Linear and rank correlation | Whether predictions and ratings move in the same direction |
| CCC | $2s_{xy}/(s_x^2 + s_y^2 + (\bar x - \bar y)^2)$ | Whether the predictions **equal** the ratings, penalizing bias and scatter together |
| Bland–Altman | Mean difference and $\pm 1.96\,s_d$ limits of agreement | Systematic bias and the range of individual errors |
| UAR | Mean of the per-class recalls (computed with `speechdsp.uar`, averaged only over the classes that actually occur in the truth; the same as balanced accuracy) | Not inflated by the majority class when the classes are unbalanced |
| Sensitivity/specificity | Recall of the positive class and of the negative class | In screening, the two kinds of error have different costs |
| ROC-AUC | How well the positive scores are ranked | Does not depend on a single decision threshold |

### Regression: the quick-start run

```bash
voice-severity-bench synthesize --config configs/quick_demo.yaml --out data/synthetic
voice-severity-bench prepare --config configs/quick_demo.yaml --out outputs/demo/prepare
voice-severity-bench evaluate outputs/demo/prepare/features.npz --config configs/quick_demo.yaml --out outputs/demo/evaluate
```

24 recordings, 8 speakers, 39-dimensional frame features × mean and standard deviation (78 columns), Ridge, 4-fold grouped cross-validation:

| Metric | Per-fold mean ± std | Pooled |
| --- | --- | --- |
| MAE | 9.563 ± 1.769 | 9.563 |
| RMSE | 13.027 ± 3.141 | 13.308 |
| $R^2$ | 0.501 ± 0.674 | 0.838 |
| Pearson $r$ | 0.886 ± 0.132 | 0.917 |
| Spearman $\rho$ | 0.829 ± 0.168 | 0.907 |
| CCC | 0.835 ± 0.166 | 0.909 |

The pooled Bland–Altman mean difference is 1.547, with 95% limits of agreement from −24.915 to 28.010. The per-fold $R^2$ has a large standard deviation because each fold holds only 6 recordings and the spread of the ratings differs from fold to fold, so read it together with the pooled metrics and CCC.

### Classification: binary detection

```bash
voice-severity-bench prepare --config configs/quick_classification.yaml --out outputs/demo/prepare_cls
voice-severity-bench evaluate outputs/demo/prepare_cls/features.npz --config configs/quick_classification.yaml --out outputs/demo/evaluate_cls
```

The same synthetic recordings, with logistic regression on the `label` column (`dysphonic` at severity ≥ 35):

| Metric | Per-fold mean ± std | Pooled |
| --- | --- | --- |
| UAR | 0.889 ± 0.096 | 0.889 |
| Accuracy | 0.907 ± 0.085 | 0.917 |
| macro F1 | 0.897 ± 0.091 | 0.906 |
| Sensitivity (`dysphonic`) | 1.000 ± 0.000 | 1.000 |
| Specificity | 0.778 ± 0.192 | 0.778 |
| ROC-AUC | 1.000 ± 0.000 | 0.978 |

The 3 folds hold 9, 6 and 9 recordings, each with one of the 3 `healthy` speakers. The speakers are also why this configuration has 3 folds: the smaller class has 3 of the 8 speakers, and every fold needs one. The pooled confusion matrix (rows are true classes, columns predicted classes):

| True \ predicted | `dysphonic` | `healthy` |
| --- | --- | --- |
| `dysphonic` | 15 | 0 |
| `healthy` | 2 | 7 |

### Feature-setting ablation

The quick configuration (`delta_order` ∈ {0, 2}, Ridge, 4 folds × 2 repeats):

```bash
voice-severity-bench ablate --config configs/quick_demo.yaml --out outputs/demo/ablate
```

| Setting | Feature dimensions | RMSE mean ± std | CCC mean ± std |
| --- | --- | --- | --- |
| `delta_order=0` (baseline, 13-dimensional frame features) | 26 | 14.009 ± 4.670 | 0.531 ± 0.413 |
| `delta_order=2` (39-dimensional frame features) | 78 | 16.212 ± 2.972 | 0.437 ± 0.453 |

For 39 versus 13 dimensions, the mean RMSE difference is +2.204, with corrected t-test p = 0.547 and Wilcoxon p = 0.188 (a single comparison, so Holm correction leaves them unchanged): 24 recordings are not enough to tell the two apart.

The default configuration (60 recordings of 1 s, 20 speakers, `n_ceps` ∈ {12, 6, 18} × `delta_order` ∈ {0, 1, 2} × {Ridge, SVR}, 5 folds × 3 repeats, 18 settings in all):

```bash
voice-severity-bench synthesize --out data/synthetic_full
voice-severity-bench ablate --dataset data/synthetic_full/dataset.csv --out outputs/demo/ablate_full
```

RMSE (mean ± std over 15 folds):

| Model | `n_ceps` | Static (13-dimensional frame features are 12 + c0) | + Δ | + Δ + ΔΔ |
| --- | --- | --- | --- | --- |
| Ridge | 6 | 13.056 ± 1.836 | 14.575 ± 2.004 | 16.547 ± 2.583 |
| Ridge | 12 | 8.624 ± 1.799 (baseline) | 11.994 ± 2.143 | 13.431 ± 2.524 |
| Ridge | 18 | **7.463 ± 1.818** | 11.416 ± 2.798 | 12.155 ± 2.953 |
| SVR | 6 | 13.217 ± 4.143 | 15.447 ± 5.225 | 18.594 ± 5.551 |
| SVR | 12 | 11.501 ± 3.613 | 15.826 ± 4.114 | 18.421 ± 4.060 |
| SVR | 18 | 10.367 ± 3.536 | 14.430 ± 3.239 | 16.113 ± 3.211 |

The top-ranked `n_ceps=18|delta_order=0|model=ridge` (38 columns) is 1.162 below the baseline, with CCC 0.920 ± 0.080. After Holm correction, though, the corrected t-test gives p = 0.171 while Wilcoxon gives p = 0.003. That gap is exactly what the Nadeau–Bengio correction is for: the training sets of the folds overlap heavily, and a test that treats the 15 folds as independent samples is overconfident. Under the corrected t-test, 8 settings stay significant (p < 0.05) after correction, all of them **worse** than the baseline and all with Δ added: for Ridge, `n_ceps=6` with Δ (p < 0.001) and with Δ and ΔΔ (p = 0.001), and `n_ceps=12` with Δ and ΔΔ (p = 0.044); for SVR, `n_ceps=6` with Δ and ΔΔ (p = 0.044), `n_ceps=12` with Δ (p = 0.012) and with Δ and ΔΔ (p = 0.001), and `n_ceps=18` with Δ (p = 0.006) and with Δ and ΔΔ (p = 0.001). In this synthetic data a sustained vowel barely changes over time, so Δ and ΔΔ add dimensions without adding information, and with 48 training recordings for 78 or 114 feature columns the error goes up. The result only shows that the framework can find a difference and test it; it is not a conclusion about real voices.

---

## Known limitations

- Synthetic data does not represent real performance. Severity in the synthetic vowels shows up only through jitter, shimmer and noise, which is far simpler than a real pathological voice; every number in this README is there only to show that the software works correctly.
- This is research code. It has not been validated on clinical data and must not be used for diagnosis.
- Load model files only from sources you trust: `model.joblib` is saved with joblib (pickle), and `model.pt` also holds its settings in pickle format, so loading a file of unknown origin can execute arbitrary code.
- The cepstral front end and the pooling statistics are designed for sustained vowels. `start_s`/`end_s` can point at sentences and `compare` accepts features from any front end, but voiced-segment detection and prosodic features for connected speech are not part of the package.
- With a few dozen speakers the percentile bootstrap intervals are too narrow. In `tests/test_bootstrap.py`, with 40 speakers × 3 recordings, the 95% interval covered the true RMSE in 179 of 200 simulated samples.
- With only a few dozen speakers the fold scores vary a lot, and after multiple-comparison correction the corrected t-test is often not significant. Read "no significant difference" as "not enough data to tell them apart", not as "the two are the same".
- The p-values and $R^2$ reported by `stepwise` are in-sample numbers after selection and overstate how good the model is; measure its real performance with the cross-validation in `evaluate`.
- With very few speakers, or recordings spread very unevenly among them, a test fold can still end up with only one class. Such a fold yields the recall of that class only: specificity is NaN when the fold holds only positives, sensitivity is NaN when it holds only negatives, and ROC-AUC is always NaN. The fold's UAR averages over the classes actually present in that fold, while macro F1 averages over the classes that are either present or predicted, so the two disagree on such a fold. Look at the pooled metrics first, or add speakers, or use fewer folds.

---

## Design notes

### Feature extraction

- Frame length, filters, number of cepstra, energy term, Δ order, liftering and pooling are all configuration keys, and 13/26/39 dimensions are just `frontend.delta_order` 0, 1 and 2. `ablate` compares the whole grid in one run and writes the results together with the settings.
- Frame length and hop are set in milliseconds and converted at each file's sample rate. Hard-coding them as sample counts at one rate would give files at other rates the wrong frame length and the wrong mel mapping, without any error message; when a common rate is needed, resample with `audio.sample_rate`.
- Head and tail trimming is also in seconds (`audio.trim_s`), because the same number of samples is a different length at a different sample rate.
- The filter energies have a configurable floor before the log, so an all-zero frame does not turn into $-\infty$.
- All frames of a recording are stacked into one matrix and go through the FFT in a single call, and mel filtering is a single matrix product.
- HTK files are written big-endian, with the frame period in 100 ns units and `parmKind` computed from the actual layout (8966 for the 39-dimensional `c0` layout, for example), so other tools that read the HTK format can use them directly; npz and text formats are available as well.
- `extract` handles only `.wav` (the extension is case-insensitive) and sorts by relative path, so the same folder is always processed in the same order. Every file leaves a row in `extraction.csv` with its status and error message, so files that were too short or could not be read can still be found.

### Evaluation

- Scaling, reduction and tuning are all inside the `Pipeline` and are fitted on the training folds only, with an inner cross-validation for tuning when needed. Choosing the model, deciding when to stop training and learning the LDA projection never touch the test fold; otherwise the scores would be too optimistic.
- 5 folds by default, grouped by the `group` column and also stratified for classification. The splits depend only on the target, the groups and the seed, and the same seed gives the same folds on different NumPy versions.
- UAR is the main classification metric, along with sensitivity, specificity, macro F1, ROC-AUC and the confusion matrix; accuracy alone gets inflated by the majority class.
- When `dataset.csv` is read, any blank in the target column, or any non-numeric value in a regression task, is an error that lists the offending rows; those rows are not skipped.
- Every setting is compared with the baseline by the corrected resampled t-test and the Wilcoxon test, with multiple-comparison correction.
- Whether predictions agree with the clinical ratings is judged by CCC and Bland–Altman, not by correlation alone.

### Regression analysis

- A least-squares fit reports the standard errors, t, p and confidence intervals of the coefficients, adjusted $R^2$, AIC, BIC and the condition number, along with normality, heteroscedasticity, autocorrelation, leverage, Cook's distance and VIF. The t and F tests and the confidence intervals are valid only when the residuals are normal, have equal variance and are independent, so the diagnostics are written out together with the coefficient table.
- When the residual degrees of freedom are 0 (fitting 7 parameters to 7 points, for example), the $R^2$ of 1 means nothing and neither do F and p, so they are output as NaN and marked not estimable; more parameters than observations, or collinear columns, raise an error.
- Orthogonal polynomials and centered response surfaces are the default, so that high powers and interaction terms do not make the design matrix nearly singular.
- The polynomial order sweep puts the in-sample metrics next to LOOCV and k-fold RMSE, AIC and BIC, and recommends an order by the configured criterion; it is the out-of-sample error that shows how much a fit overfits.
- Stepwise selection records the action and statistics of every step and adds the VIF, so you can see how collinearity affects the selection (on the cement data, starting from the empty model and from the full model selects different variables).
- Gradient descent standardizes the variables first and uses the vectorized simultaneous update. It judges convergence by the relative change in cost under an iteration cap, detects divergence, and matches the closed-form solution to $10^{-6}$.

### Engineering

I set each dependency's lower bound to a version that installs on Python 3.10. CI runs 3.10–3.13 on Linux and Windows, and a separate job runs the same tests with the oldest dependency versions.

The default ablation is 18 settings with 15 folds each, so one run fits 270 models, and every result has to trace back to the settings that produced it. That is why all settings live in frozen dataclasses read from and written to YAML, a mistyped key is an error, and the data and output locations come only from configuration keys or command-line options. Every command except `index`, which only writes `dataset.csv`, leaves `config.yaml` and `manifest.json` in its output folder, and every figure has a CSV with the same data next to it, so checking a number or redrawing a figure in another tool never needs a rerun. On an input or configuration error the CLI prints one line and exits with code 2; progress and warnings go through logging to stderr.

The code style conventions (type hints, numpy-style docstrings in English, ruff) are in CONTRIBUTING.md, which is written in Traditional Chinese. Tests use only synthetic data and the bundled public cement data. `tests/test_examples.py` actually runs the six examples with `--quick`, so an API change that breaks them fails in the tests first.

Pre-emphasis, framing, the mel filterbank, Δ, CMN/CMVN, endpoint detection, HTK and WAV I/O, and UAR come from `speechdsp`; the combination of window, power spectrum, DCT scaling and energy term stays in this project, for the reason given in section 1 of [Method](#method).

---

## Development and testing

```bash
pip install -e ".[dev]"
python -m pytest -q                 # all tests, doctests included
python -m ruff check .
python -m ruff format --check .
# Python 3.10 syntax gate (runs on any Python version)
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not {'.venv', 'build', 'dist', '__pycache__'} & set(p.parts)]"
```

`tests/test_python310_compat.py` parses every file with the Python 3.10 grammar and scans the code for standard-library names that arrived in 3.11 or later, names that are incompatible between NumPy 1.x and 2.x, and APIs newer than the declared lower bounds of the dependencies; it also checks that `import voice_severity_bench` and the command line do not load the optional PyTorch. `tests/test_documentation.py` rejects simplified characters and mainland Chinese wording, checks the heading order of the READMEs, and makes sure no local path or e-mail address was left behind. Tests that need PyTorch are skipped when it is not installed and run in an environment with the `dl` extra; the `dl` job in CI uses the CPU build of PyTorch.

---

## References

### Feature extraction

- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980.
- S. S. Stevens, J. Volkmann, and E. B. Newman, "A scale for the measurement of the psychological magnitude pitch," *The Journal of the Acoustical Society of America*, vol. 8, no. 3, pp. 185–190, 1937.
- D. O'Shaughnessy, *Speech Communication: Human and Machine*. Reading, MA, USA: Addison-Wesley, 1987.
- S. Young et al., *The HTK Book (for HTK Version 3.4)*. Cambridge, U.K.: Cambridge University Engineering Department, 2006.
- ETSI ES 201 108 V1.1.3, "Speech Processing, Transmission and Quality Aspects (STQ); Distributed speech recognition; Front-end feature extraction algorithm; Compression algorithms," ETSI, 2003.
- S. Furui, "Speaker-independent isolated word recognition using dynamic features of speech spectrum," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 34, no. 1, pp. 52–59, 1986.
- B.-H. Juang, L. R. Rabiner, and J. G. Wilpon, "On the use of bandpass liftering in speech recognition," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 35, no. 7, pp. 947–954, 1987.

### Regression and statistical inference

- N. R. Draper and H. Smith, *Applied Regression Analysis*, 3rd ed. New York, NY, USA: Wiley, 1998.
- G. E. P. Box and K. B. Wilson, "On the experimental attainment of optimum conditions," *Journal of the Royal Statistical Society, Series B*, vol. 13, no. 1, pp. 1–45, 1951.
- M. A. Efroymson, "Multiple regression analysis," in *Mathematical Methods for Digital Computers*, A. Ralston and H. S. Wilf, Eds. New York, NY, USA: Wiley, 1960, pp. 191–203.
- A. Cauchy, "Méthode générale pour la résolution des systèmes d'équations simultanées," *Comptes Rendus de l'Académie des Sciences*, vol. 25, pp. 536–538, 1847.
- H. Akaike, "A new look at the statistical model identification," *IEEE Transactions on Automatic Control*, vol. 19, no. 6, pp. 716–723, 1974.
- G. Schwarz, "Estimating the dimension of a model," *The Annals of Statistics*, vol. 6, no. 2, pp. 461–464, 1978.
- S. S. Shapiro and M. B. Wilk, "An analysis of variance test for normality (complete samples)," *Biometrika*, vol. 52, no. 3–4, pp. 591–611, 1965.
- C. M. Jarque and A. K. Bera, "Efficient tests for normality, homoscedasticity and serial independence of regression residuals," *Economics Letters*, vol. 6, no. 3, pp. 255–259, 1980.
- T. S. Breusch and A. R. Pagan, "A simple test for heteroscedasticity and random coefficient variation," *Econometrica*, vol. 47, no. 5, pp. 1287–1294, 1979.
- R. Koenker, "A note on studentizing a test for heteroscedasticity," *Journal of Econometrics*, vol. 17, no. 1, pp. 107–112, 1981.
- J. Durbin and G. S. Watson, "Testing for serial correlation in least squares regression. I," *Biometrika*, vol. 37, no. 3–4, pp. 409–428, 1950.
- T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York, NY, USA: Springer, 2009.
- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.
- A. C. Davison and D. V. Hinkley, *Bootstrap Methods and Their Application*. Cambridge, U.K.: Cambridge University Press, 1997.
- D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *Journal of Pharmacokinetics and Biopharmaceutics*, vol. 15, no. 6, pp. 657–680, 1987.
- R. D. Cook, "Detection of influential observation in linear regression," *Technometrics*, vol. 19, no. 1, pp. 15–18, 1977.
- D. W. Marquardt, "Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation," *Technometrics*, vol. 12, no. 3, pp. 591–612, 1970.

### Machine learning and model evaluation

- A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.
- H. Drucker, C. J. C. Burges, L. Kaufman, A. Smola, and V. Vapnik, "Support vector regression machines," in *Advances in Neural Information Processing Systems*, vol. 9, 1997, pp. 155–161.
- L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
- R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics*, vol. 7, no. 2, pp. 179–188, 1936.
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in *Proc. 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 9, 2010, pp. 249–256.
- D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd International Conference on Learning Representations (ICLR)*, 2015, arXiv:1412.6980.
- F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
- M. Stone, "Cross-validatory choice and assessment of statistical predictions," *Journal of the Royal Statistical Society, Series B*, vol. 36, no. 2, pp. 111–147, 1974.
- S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, "The need to approximate the use-case in clinical machine learning," *GigaScience*, vol. 6, no. 5, pp. 1–9, 2017.
- S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC Bioinformatics*, vol. 7, p. 91, 2006.
- B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion Challenge," in *Proc. Interspeech 2009*, 2009, pp. 312–315.
- C. Nadeau and Y. Bengio, "Inference for the generalization error," *Machine Learning*, vol. 52, no. 3, pp. 239–281, 2003.
- M. Ojala and G. C. Garriga, "Permutation tests for studying classifier performance," *Journal of Machine Learning Research*, vol. 11, pp. 1833–1863, 2010.
- B. Phipson and G. K. Smyth, "Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn," *Statistical Applications in Genetics and Molecular Biology*, vol. 9, no. 1, art. 39, 2010.
- J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology*, vol. 143, no. 1, pp. 29–36, 1982.
- J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *Journal of Machine Learning Research*, vol. 7, pp. 1–30, 2006.
- S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.
- Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: A practical and powerful approach to multiple testing," *Journal of the Royal Statistical Society, Series B*, vol. 57, no. 1, pp. 289–300, 1995.

### Association and agreement

- C. Spearman, "The proof and measurement of association between two things," *The American Journal of Psychology*, vol. 15, no. 1, pp. 72–101, 1904.
- R. A. Fisher, "Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population," *Biometrika*, vol. 10, no. 4, pp. 507–521, 1915.
- L. I.-K. Lin, "A concordance correlation coefficient to evaluate reproducibility," *Biometrics*, vol. 45, no. 1, pp. 255–268, 1989.
- J. M. Bland and D. G. Altman, "Statistical methods for assessing agreement between two methods of clinical measurement," *The Lancet*, vol. 327, no. 8476, pp. 307–310, 1986.
- P. E. Shrout and J. L. Fleiss, "Intraclass correlations: Uses in assessing rater reliability," *Psychological Bulletin*, vol. 86, no. 2, pp. 420–428, 1979.
- K. O. McGraw and S. P. Wong, "Forming inferences about some intraclass correlation coefficients," *Psychological Methods*, vol. 1, no. 1, pp. 30–46, 1996.

### Voice assessment and synthetic voices

- M. Hirano, *Clinical Examination of Voice*. Vienna, Austria: Springer-Verlag, 1981.
- G. B. Kempster, B. R. Gerratt, K. Verdolini Abbott, J. Barkmeier-Kraemer, and R. E. Hillman, "Consensus Auditory-Perceptual Evaluation of Voice: Development of a standardized clinical protocol," *American Journal of Speech-Language Pathology*, vol. 18, no. 2, pp. 124–132, 2009.
- Y. Maryn, P. Corthals, P. Van Cauwenberge, N. Roy, and M. De Bodt, "Toward improved ecological validity in the acoustic measurement of overall voice quality: Combining continuous speech and sustained vowels," *Journal of Voice*, vol. 24, no. 5, pp. 540–555, 2010.
- A. E. Rosenberg, "Effect of glottal pulse shape on the quality of natural vowels," *The Journal of the Acoustical Society of America*, vol. 49, no. 2B, pp. 583–590, 1971.
- G. E. Peterson and H. L. Barney, "Control methods used in a study of the vowels," *The Journal of the Acoustical Society of America*, vol. 24, no. 2, pp. 175–184, 1952.

### Datasets

- H. Woods, H. H. Steinour, and H. R. Starke, "Effect of composition of Portland cement on heat evolved during hardening," *Industrial & Engineering Chemistry*, vol. 24, no. 11, pp. 1207–1214, 1932.
- A. Hald, *Statistical Theory with Engineering Applications*. New York, NY, USA: Wiley, 1952.
- P. R. Walden, "Perceptual Voice Qualities Database (PVQD): Database characteristics," *Journal of Voice*, vol. 36, no. 6, pp. 875.e15–875.e23, 2022, doi: 10.1016/j.jvoice.2020.10.001.
- P. Walden, "Perceptual Voice Qualities Database (PVQD)," Mendeley Data, version 4, 2022, doi: 10.17632/9dz247gnyb.4. CC BY 4.0.

---

## License

MIT License, Copyright (c) RL. See [LICENSE](LICENSE).
