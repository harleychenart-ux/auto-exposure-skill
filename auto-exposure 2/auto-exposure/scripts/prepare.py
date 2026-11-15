#!/usr/bin/env python3
"""Step 1 of the built-in-tool path (Doubao account quota, no Ark key).

Diagnoses the photo and writes everything the agent needs to run its own
image-generation tool: a numeric correction brief (prompt.txt), the exact
image to feed the model (input_for_model.png, pre-normalized when the
exposure is extreme), and a plan.json. Prints one JSON object.

Then the agent calls its image tool, and step 2 is `finish.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exposure_lib as lib  # noqa: E402


def needs_pre_normalize(metrics: dict, verdict: str) -> bool:
    return (metrics["mean"] <= 0.14 and verdict == "underexposed") or (
        metrics["clip_high"] >= 0.20 and verdict in ("overexposed", "backlit")
    )


def ratio_string(w: int, h: int) -> str:
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def main() -> None:
    ap = argparse.ArgumentParser(description="生成式校正第 1 步：诊断 + 生成提示词")
    ap.add_argument("-i", "--input", required=True, help="原图路径")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument("--target", type=float, default=None, help="手动指定目标平均亮度 0-1")
    ap.add_argument("--force", action="store_true", help="曝光正常时也强制处理")
    ap.add_argument("--no-pre", action="store_true", help="关闭极端曝光的送图前归一化")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    rgb, meta = lib.load_image(args.input)
    metrics = lib.compute_metrics(rgb)
    verdict, severity, notes = lib.classify(metrics)
    plan = lib.plan_correction(metrics, verdict, severity)
    if not args.target:
        # Tell the model a target this frame can actually reach; asking for a
        # number the highlights can't afford is what pushes windows to pure white.
        plan = lib.calibrate_plan(rgb, plan)
    if args.target:
        plan["target_mean"] = float(args.target)

    result = {
        "input": os.path.abspath(args.input),
        "stem": stem,
        "size": {"width": meta.get("width"), "height": meta.get("height"),
                 "ratio": ratio_string(meta.get("width") or 1, meta.get("height") or 1)},
        "verdict": verdict,
        "severity": severity,
        "diagnosis": notes,
        "plan": plan,
        "before": metrics,
    }

    if verdict == "ok" and not args.force:
        result["status"] = "ok_no_change"
        result["message"] = "曝光正常，未做校正（如需强制处理加 --force）"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    pre_normalized = False
    source_rgb = rgb
    if not args.no_pre and needs_pre_normalize(metrics, verdict):
        pre_plan = dict(plan, target_mean=0.33 if verdict == "underexposed" else 0.62)
        source_rgb = lib.auto_correct(rgb, pre_plan)[0]
        pre_normalized = True
        result["pre_normalized"] = True

    model_input = os.path.join(args.output, f"{stem}_input_for_model.png")
    lib.save_image(source_rgb, model_input)

    prompt = lib.build_prompt(metrics, verdict, plan, pre_normalized)
    result["prompt"] = prompt
    result["model_input"] = os.path.abspath(model_input)

    prompt_path = os.path.join(args.output, f"{stem}_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(prompt)
    result["prompt_file"] = os.path.abspath(prompt_path)

    w0, h0 = result["size"]["width"] or 1, result["size"]["height"] or 1
    long_edge = max(w0, h0)
    # Keep a sane working resolution for every input: downscale huge files,
    # upscale tiny ones (a 400px output from a 400px input just looks smeared).
    target_long = 2048 if long_edge > 2048 else max(1024, long_edge)
    scale = target_long / long_edge
    tw, th = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    result["size_hint"] = {
        "target_width": tw,
        "target_height": th,
        "note": (
            f"原图 {w0}x{h0}（{result['size']['ratio']}）。"
            + ("原图长边已超过 2048，按长边 2048 等比输出即可。" if long_edge > 2048 else
               "按原尺寸或长边 ≤2048 等比输出，保持同一比例。")
        ),
    }

    result["status"] = "ready"
    result["next_step"] = (
        f"调用内置生图工具：参考图={result['model_input']}，提示词={prompt_path}，"
        f"尺寸≈{result['size_hint']['target_width']}x{result['size_hint']['target_height']}；"
        "拿到结果后运行 finish.py 验收。"
    )

    plan_path = os.path.join(args.output, f"{stem}_plan.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    result["plan_file"] = os.path.abspath(plan_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
