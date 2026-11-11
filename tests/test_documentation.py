"""Documentation checks: Traditional Chinese with Taiwan usage, the layout of the
English README and its Traditional Chinese version, and no path or e-mail address
from a personal machine.

The mainland terms are kept as escapes so that a text search of the repository
does not find them here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
README_ZH = (PROJECT_ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
THIS_FILE = Path(__file__).resolve()

#: Every level-2 heading of README.md, in order.
EXPECTED_H2 = [
    "Contents",
    "Overview",
    "Method",
    "Installation",
    "Quick start",
    "Usage",
    "CLI reference",
    "Project layout",
    "Preparing data",
    "Configuration",
    "Results and metrics",
    "Known limitations",
    "Design notes",
    "Development and testing",
    "References",
    "License",
]

#: The level-2 headings of README.zh-TW.md, which follows README.md section by section.
EXPECTED_H2_ZH = [
    "目錄",
    "專案簡介",
    "方法說明",
    "安裝步驟",
    "快速開始",
    "使用範例",
    "CLI 指令對照表",
    "專案結構",
    "資料準備",
    "設定檔",
    "結果與評估指標",
    "已知限制",
    "設計重點",
    "開發與測試",
    "References",
    "License",
]

#: Mainland wording and the Taiwan wording to use instead. The mathematical
#: sense of "function" keeps its usual word, so it is deliberately absent.
MAINLAND_TERMS: dict[str, str] = {
    "\u6587\u4ef6\u593e": "資料夾",
    "\u5167\u5b58": "記憶體",
    "\u6578\u7d44": "陣列",
    "\u512a\u5316": "最佳化",
    "\u9ed8\u8a8d": "預設",
    "\u7f3a\u7701": "預設",
    "\u5b57\u7b26\u4e32": "字串",
    "\u5206\u8fa8\u7387": "解析度",
    "\u7db2\u7d61": "網路",
    "\u4fe1\u606f": "資訊",
    "\u6a19\u7c3d": "標籤",
    "\u8996\u983b": "影片",
    "\u97f3\u983b": "音訊",
    "\u8edf\u4ef6": "軟體",
    "\u786c\u4ef6": "硬體",
    "\u904b\u884c": "執行",
    "\u8abf\u7528": "呼叫",
    "\u5c4f\u5e55": "螢幕",
    "\u754c\u9762": "介面",
    "\u7dda\u7a0b": "執行緒",
    "\u6253\u5370": "列印",
    "\u6a21\u584a": "模組",
    "\u6587\u6a94": "文件",
    "\u8b8a\u91cf": "變數",
    "\u5e38\u91cf": "常數",
    "\u6578\u64da\u5eab": "資料庫",
    "\u670d\u52d9\u5668": "伺服器",
    "\u5be6\u73fe": "實作",
    "\u63a1\u6a23": "取樣",
    "\u9b6f\u68d2": "穩健",
    "\u7a0b\u5e8f": "程式",
}

#: Local details that must not appear anywhere in the repository.
LOCAL_DETAILS: dict[str, str] = {
    "drive-letter path": r"\b[A-Za-z]:[\\/]{1,2}\w[\w .-]*[\\/]",
    "home folder path": r"/(?:home|Users)/[^/\s]+/",
    "e-mail address": r"[\w.+-]+@[\w-]+\.[a-z]{2,}\b",
}

_CJK = re.compile(r"[\u4e00-\u9fff]")
#: Generated or user-owned locations: data, results and environments are not source.
_SKIP_TOP = {".git", ".venv", "venv", "build", "dist", "data", "outputs"}
_SKIP_ANY = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".cff", ".txt", ".csv", ".cfg"}


def text_files() -> list[Path]:
    files = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == THIS_FILE:
            continue
        parts = path.relative_to(PROJECT_ROOT).parts
        if parts[0] in _SKIP_TOP or _SKIP_ANY & set(parts) or parts[:2] == ("examples", "output"):
            continue
        # pip install -e leaves metadata (with a copy of the README) under src/.
        if any(part.endswith(".egg-info") for part in parts):
            continue
        if path.suffix in _TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            files.append(path)
    return files


def test_there_is_something_to_check():
    assert len(text_files()) > 10


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_no_mainland_wording(path):
    text = path.read_text(encoding="utf-8")
    found = [f"{term} -> {fix}" for term, fix in MAINLAND_TERMS.items() if term in text]
    assert not found, f"{path.name}: " + "; ".join(found)


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_every_chinese_character_is_traditional(path):
    """Simplified-only characters have no Big5 (cp950) encoding."""
    text = path.read_text(encoding="utf-8")
    bad = []
    for char in sorted(set(_CJK.findall(text))):
        try:
            char.encode("cp950")
        except UnicodeEncodeError:
            bad.append(char)
    assert not bad, f"{path.name}: {''.join(bad)}"


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_no_local_path_or_email_address(path):
    text = path.read_text(encoding="utf-8")
    found = [label for label, pattern in LOCAL_DETAILS.items() if re.search(pattern, text)]
    assert not found, f"{path.name}: {found}"


def test_the_patterns_catch_what_they_are_meant_to():
    samples = {
        "drive-letter path": "C:\\Users\\someone\\data",
        "home folder path": "/home/someone/data",
        "e-mail address": "someone@example.org",
    }
    for label, sample in samples.items():
        assert re.search(LOCAL_DETAILS[label], sample), label
    assert not re.search(LOCAL_DETAILS["drive-letter path"], "$f:\\mathbb{R}$")


def test_readme_heading_order():
    headings = re.findall(r"^## (.+?)\s*$", README, re.MULTILINE)
    assert headings == EXPECTED_H2


def test_chinese_readme_heading_order():
    headings = re.findall(r"^## (.+?)\s*$", README_ZH, re.MULTILINE)
    assert headings == EXPECTED_H2_ZH


@pytest.mark.parametrize(
    ("text", "author", "license_line"),
    [
        pytest.param(
            README,
            "Author: RL",
            "MIT License, Copyright (c) RL. See [LICENSE](LICENSE).",
            id="README.md",
        ),
        pytest.param(
            README_ZH,
            "作者：RL",
            "MIT License，Copyright (c) RL。詳見 [LICENSE](LICENSE)。",
            id="README.zh-TW.md",
        ),
    ],
)
def test_readme_opens_with_badges_and_author_and_closes_with_the_license(
    text, author, license_line
):
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    name = re.search(r'^name = "([^"]+)"$', pyproject, re.MULTILINE).group(1)
    lines = text.splitlines()
    assert lines[0] == f"# {name}"
    assert lines[2].startswith("[![CI](") and lines[3].startswith("[![Python](")
    assert lines[4].startswith("[![License: MIT](")
    assert f"\n{author}\n" in text
    assert license_line in text


def test_each_readme_links_to_the_other_under_the_badges():
    assert README.splitlines()[:6] == README_ZH.splitlines()[:6]
    assert README.splitlines()[6] == "English | [繁體中文](README.zh-TW.md)"
    assert README_ZH.splitlines()[6] == "[English](README.md) | 繁體中文"


@pytest.mark.parametrize(
    ("text", "stem"),
    [
        pytest.param(README, "workflow", id="README.md"),
        pytest.param(README_ZH, "workflow.zh-TW", id="README.zh-TW.md"),
    ],
)
def test_readme_shows_the_workflow_diagram_in_its_own_language(text, stem):
    online = (
        f"https://raw.githack.com/recklight/voice-severity-bench/master/docs/{stem}.html?theme=dark"
    )
    # The figure links to the online page, and so does the sentence under it.
    assert f"(docs/images/{stem}.png)]({online})" in text
    assert text.count(f"]({online})") == 2
    assert f"](docs/{stem}.html)" in text


def test_readme_is_mostly_english_prose():
    # The link to the Chinese version is the only Chinese in README.md.
    assert not _CJK.findall(README.replace("[繁體中文](README.zh-TW.md)", ""))
    assert len(re.findall(r"\b[A-Za-z]{3,}\b", README)) > 5000


def test_chinese_readme_is_mostly_chinese_prose():
    assert len(_CJK.findall(README_ZH)) > 2000


@pytest.mark.parametrize(
    "text",
    [pytest.param(README, id="README.md"), pytest.param(README_ZH, id="README.zh-TW.md")],
)
def test_readme_runs_pytest_without_pythonpath(text):
    # pyproject.toml already puts src on pytest's path (pythonpath = ["src"]).
    assert "PYTHONPATH=src python -m pytest" not in text
