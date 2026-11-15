#!/usr/bin/env python3
"""Deterministic, content-preserving exposure correction (no model call).

Use it as: fallback when Ark is unavailable, a fast preview, or a batch pass on
images where a generative repaint is unacceptable (documents, products, faces
that must stay pixel-accurate).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exposure_lib as lib  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="本地确定性曝光校正")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None, help="默认 <名字>_fixed.png")
    ap.add_argument("--target", type=float, default=None, help="目标平均亮度 0-1")
    ap.add_argument("--ev", type=float, default=None, help="强制曝光补偿（EV），会跳过自动目标")
    ap.add_argument("--no-shadow-lift", action="store_true")
    args = ap.parse_args()

    rgb, _ = lib.load_image(args.input)
    m = lib.compute_metrics(rgb)
    verdict, severity, notes = lib.classify(m)
    plan = lib.plan_correction(m, verdict, severity)

    if args.target is not None:
        plan["target_mean"] = args.target
    if args.ev is not None:
        plan["ev"] = args.ev
        plan["target_mean"] = None  # skip the feedback loop's auto target
    if args.no_shadow_lift:
        plan["shadow_lift"] = 0.0

    if plan.get("target_mean") is None:
        out, params = lib.apply_params(rgb, plan), plan
        after = lib.compute_metrics(out)
    else:
        out, params, after = lib.auto_correct(rgb, plan)

    out_path = args.output or f"{os.path.splitext(args.input)[0]}_fixed.png"
    lib.save_image(out, out_path)

    print(
        json.dumps(
            {
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(out_path),
                "verdict": verdict,
                "diagnosis": notes,
                "params": params,
                "before": {k: m[k] for k in ("mean", "median", "contrast", "clip_high", "clip_low")},
                "after": {k: after[k] for k in ("mean", "median", "contrast", "clip_high", "clip_low")},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
