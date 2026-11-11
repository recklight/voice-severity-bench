# 報告

[English](README.md) | 繁體中文

Fu-Kai Chuang，*Automatic Rating of Dysphonia Severity From the Sustained Vowel: A Frozen-Plan Comparison With Clinical Measures and a Single Rater*。

| 檔案 | 內容 |
| --- | --- |
| `paper.pdf` | 報告本身，IEEE 期刊版面，四頁 |
| `main.tex`、`release.tex`、`numbers.tex`、`refs.bib`、`tables/` | LaTeX 原始檔；內文和表格的每個數字都在 `numbers.tex`，由分析輸出產生 |
| `thesis.toml` | [thesis-template](https://github.com/recklight/thesis-template) 的 `thesis-cli` 設定，PDF 用它以 pdfLaTeX 加 BibTeX 編譯 |

報告在凍結的分析計畫下，用 PVQD 的持續 /a/ 比較這個套件的各組特徵。研究的計畫和腳本在 [`../studies/voice-severity/`](../studies/voice-severity/)。沒有 thesis-cli 的話，依序跑 `pdflatex main`、`bibtex main`，再跑兩次 `pdflatex main`，得到的是同一份報告；IEEEtran 類別 TeX Live 和 MiKTeX 都有附。

報告以創用 CC 姓名標示 4.0 國際授權條款（CC BY 4.0，<https://creativecommons.org/licenses/by/4.0/>）授權。repo 的程式碼以 [`../LICENSE`](../LICENSE) 的 MIT 授權釋出。
