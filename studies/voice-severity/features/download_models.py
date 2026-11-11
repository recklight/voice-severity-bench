"""Download the three SSL models at a fixed revision into models/ (S4).

    .venv/Scripts/python.exe features/download_models.py

Only the configuration, the feature-extractor settings, the weights and the
model card are fetched. The revision (commit hash), the licence from the
model card and the SHA-256 of every file go to models/manifest.json and to a
REVISION file beside the weights; extract_ssl.py then loads offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODELS = common.ROOT / "models"
WANTED = ("config.json", "preprocessor_config.json", "README.md")


def main() -> int:
    api = HfApi()
    manifest = {}
    for name in ("A4", "A5", "A6"):
        repo = common.plan()["feature_sets"][name]["model"]
        info = api.model_info(repo, files_metadata=True)
        files = {sibling.rfilename: sibling for sibling in info.siblings}
        weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
        patterns = [weights, *(name for name in WANTED if name in files)]
        target = MODELS / repo.replace("/", "__")
        snapshot_download(repo, revision=info.sha, allow_patterns=patterns, local_dir=target)
        (target / "REVISION").write_text(info.sha + "\n", encoding="utf-8")
        card = getattr(info, "card_data", None) or {}
        licence = card.get("license") if hasattr(card, "get") else None
        manifest[name] = {
            "repo": repo,
            "revision": info.sha,
            "license": licence,
            "files": {
                pattern: {
                    "bytes": (target / pattern).stat().st_size,
                    "sha256": common.sha256(target / pattern),
                    "hub_sha256": getattr(getattr(files[pattern], "lfs", None), "sha256", None),
                }
                for pattern in patterns
            },
        }
        print(name, repo, info.sha, licence, weights)
    common.write_json(MODELS / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
