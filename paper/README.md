# Report

English | [繁體中文](README.zh-TW.md)

Fu-Kai Chuang, *Automatic Rating of Dysphonia Severity From the Sustained Vowel: A Frozen-Plan Comparison With Clinical Measures and a Single Rater*.

| File | Content |
| --- | --- |
| `paper.pdf` | the report, four pages in the IEEE journal layout |
| `main.tex`, `release.tex`, `numbers.tex`, `refs.bib`, `tables/` | its LaTeX source; `numbers.tex` holds every number of the text and the tables, generated from the analysis outputs |
| `thesis.toml` | settings for `thesis-cli` of [thesis-template](https://github.com/recklight/thesis-template), which built the PDF with pdfLaTeX and BibTeX |

The report compares the feature sets of this package on the sustained /a/ of PVQD under a frozen analysis plan. The plan and the scripts of the study are in [`../studies/voice-severity/`](../studies/voice-severity/). Without thesis-cli, `pdflatex main`, `bibtex main` and `pdflatex main` twice build the same report; the IEEEtran class comes with TeX Live and MiKTeX.

The report is licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>). The code of the repository is under the MIT licence in [`../LICENSE`](../LICENSE).
