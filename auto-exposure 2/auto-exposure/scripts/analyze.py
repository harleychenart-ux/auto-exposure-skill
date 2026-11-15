#!/usr/bin/env python3
"""Diagnose exposure only - no pixels are modified. Prints JSON by default."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exposure_lib as lib  # noqa: E402


def analyze_one(path: str) -> dict:
    rgb, meta = lib.load_image(path)
    metrics = lib.compute_metrics(rgb)
    verdict, severity, notes = lib.classify(metrics)
    plan = lib.plan_correction(metrics, verdict, severity)
    return {
        "input": os.path.abspath(path),
        "size": f"{meta['width']}x{meta['height']}",
        "mode": meta["mode"],
        "verdict": verdict,
        "severity": severity,
        "diagnosis": notes,
        "metrics": metrics,
        "plan": plan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="曝光诊断（不改图）")
    ap.add_argument("input", help="图片路径或目录")
    ap.add_argument("--pretty", action="store_true", help="打印人读摘要而不是 JSON")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        paths = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        ]
    else:
        paths = [args.input]

    out = [analyze_one(p) for p in paths]

    if args.pretty:
        for r in out:
            m = r["metrics"]
            print(f"\n{r['input']}  ({r['size']})")
            print(f"  判定: {r['verdict']} / {r['severity']}")
            print(
                f"  平均亮度 {m['mean']:.1%} | 中位 {m['median']:.1%} | "
                f"p5-p95 {m['contrast']:.1%}"
            )
            print(
                f"  死白 {m['clip_high']:.2%} | 死黑 {m['clip_low']:.2%} | "
                f"暗部占比 {m['shadow_share']:.1%} | 高亮占比 {m['highlight_share']:.1%}"
            )
            for n in r["diagnosis"]:
                print(f"  - {n}")
            p = r["plan"]
            print(
                f"  建议: 目标亮度 {p['target_mean']:.0%}, EV {p['ev']:+.2f}, "
                f"高光肩部 {p['shoulder']:.2f}"
            )
    else:
        print(json.dumps(out if len(out) > 1 else out[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
