# 參與開發

## 開發環境

需要 Python 3.10 以上。建議使用虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 共用套件 speechdsp

預強調、分框、梅爾濾波器組、Δ、HTK 檔案讀寫、端點偵測與 UAR 等共用的訊號處理來自 `speechdsp`，它發布在 PyPI 上，安裝本專案時 pip 會一併裝好。

- PyPI：<https://pypi.org/project/speechdsp/>
- 原始碼：<https://github.com/recklight/SpeechDsp>

### 安裝本專案

```bash
pip install -e ".[dev]"
voice-severity-bench --help
```

需要 `torch_mlp` 模型時再加裝 `pip install -e ".[dl]"`（PyTorch）。

---

## 提交前的檢查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI 會執行相同的檢查，並在 Linux 與 Windows 上跑 Python 3.10–3.13、最舊的相依版本，以及安裝 CPU 版 PyTorch 的測試。

### Python 3.10 相容性

本專案宣告 `requires-python = ">=3.10"`，請避開 3.11 以後才有的語法與標準函式庫：

| 不要用 | 替代做法 |
| --- | --- |
| `tomllib` | 本專案的設定檔是 YAML，不需要 TOML |
| `typing.Self` | 類別名稱字串 |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| `except*`、`ExceptionGroup` | 一般的 `except` 與單一例外 |
| `asyncio.TaskGroup` | `asyncio.gather` |
| PEP 695（`class Foo[T]`、`type X = ...`） | `TypeVar` 與一般的型別註記 |
| `itertools.batched` | 明確的切片迴圈 |
| `np.float_`、`np.trapz`、`np.in1d`、`np.trapezoid`、`np.concat` 等 NumPy 2.0 才有或已移除的名稱 | `np.float64`、`scipy.integrate.trapezoid`、`np.isin`、`np.concatenate` |

任何 Python 版本都能執行的 3.10 語法閘門：

```bash
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not {'.venv', 'build', 'dist', '__pycache__'} & set(p.parts)]"
```

`tests/test_python310_compat.py` 會自動掃描上述語法、標準函式庫、NumPy 名稱與相依版本下限，並確認匯入套件時不會載入選用套件。

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式都要有 type hints 與英文 numpy 風格 docstring；實作文獻方法的公開函式要有 `References` 段。
- 識別字與 docstring 一律用美式拼法（normalize、analyze、color）。
- 用 `logging` 輸出進度與警告，不要用 `print`。
- 陣列運算盡量向量化，不寫逐元素的迴圈。
- 所有參數都放在 `config.py` 的 dataclass，不要在程式裡寫死路徑或數值。

---

## 資料

**本 repository 不包含任何實驗資料。** 測試使用 `synthetic.py` 產生的資料，以及 `reference_data.py` 內附的公開水泥硬化熱資料（`tests/test_reference_data.py`）。請不要提交錄音、影像、`.mat`、`.npy`、模型檔，或任何含個人資訊的檔案。

---

## 提交訊息

請使用 Conventional Commits（`feat`／`fix`／`docs`／`test`／`refactor`）。有功能變動時，請同步更新 `CHANGELOG.md` 的 `[Unreleased]` 段。
