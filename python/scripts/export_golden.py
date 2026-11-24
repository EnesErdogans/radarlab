#!/usr/bin/env python3
"""Write cross-language golden vectors to ``<repo>/golden/*.json``.

Usage: python python/scripts/export_golden.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from radarlab.golden import build_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "golden")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = build_all()
    for name, case in cases.items():
        p = args.out / f"{name}.json"
        p.write_text(json.dumps(case, indent=1) + "\n")
        print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KiB)")
    (args.out / "index.json").write_text(json.dumps({"files": [f"{n}.json" for n in cases]}, indent=1) + "\n")


if __name__ == "__main__":
    main()
