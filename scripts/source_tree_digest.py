#!/usr/bin/env python3
"""Hash every tracked and non-ignored untracked workspace file deterministically."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunfish.source_tree import source_tree_digest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--with-count", action="store_true")
    args = parser.parse_args()
    digest, count = source_tree_digest(args.root)
    print(f"{digest} {count}" if args.with_count else digest)


if __name__ == "__main__":
    main()
