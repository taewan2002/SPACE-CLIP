"""Download the exact study CLIP revision into the repository-local HF cache."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download("openai/clip-vit-base-patch16", revision=REVISION))
    # The original model requests main. Resolve it to the study revision locally;
    # run.py uses offline mode so the default ref cannot move during the study.
    reference = snapshot.parents[1] / "refs/main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(REVISION)
    print(f"Cached and pinned openai/clip-vit-base-patch16 at {REVISION}")


if __name__ == "__main__":
    main()
