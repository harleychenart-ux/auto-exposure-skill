#!/usr/bin/env python3
"""Step 2 of the built-in-tool path: verify the generated image and deliver.

Takes the original photo and the image produced by the agent's own generation
tool, re-computes metrics, runs the same acceptance checks as the Ark path
(lightness-independent structure score so a big exposure shift is not mistaken
for content drift), writes *_fixed.png, *_compare.jpg and report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exposure_lib as lib  # noqa: E402


def _match_size(rgb: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Resize to the reference canvas when the ratio is the same but pixels differ."""
    if rgb.shape[:2] == ref.shape[:2]:
        return rgb
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    im = im.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser(description="生成式校正第 2 步：验收 + 出对比图与报告")
    ap.add_argument("--before", required=True, help="原图路径")
    ap.add_argument("--after", required=True, help="生成结果图路径")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument("--source", default="doubao", help="生成来源标注，如 doubao / ark / 其他")
    ap.add_argument("--model", default="", help="模型名，写进报告")
    ap.add_argument("--quota", default="", help="额度说明，如「豆包每日免费次数」")
    ap.add_argument("--target", type=float, default=None, help="手动指定目标平均亮度 0-1")
    ap.add_argument("--no-local-fallback", action="store_true",
                    help="生成结果验收不通过时不自动给本地版（默认会给）")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.before))[0]

    before, _ = lib.load_image(args.before)
    after_raw, _ = lib.load_image(args.after)
    after = _match_size(after_raw, before)

    m_before = lib.compute_metrics(before)
    m_after = lib.compute_metrics(after)
    verdict, severity, notes = lib.classify(m_before)
    plan = lib.plan_correction(m_before, verdict, severity)
    if args.target:
        plan["target_mean"] = float(args.target)

    ver = lib.verify(before, after, plan, strict_sharpness=True)

    fixed_path = lib.save_image(after, os.path.join(args.output, f"{stem}_fixed.png"))
    cmp_path = lib.save_image(
        lib.stitch_compare(before, after), os.path.join(args.output, f"{stem}_compare.jpg"),
        quality=92,
    )

    # Any image must end up with a usable deliverable: when the generative pass
    # drifts (text re-written, face swapped, content changed), fall back to the
    # deterministic correction instead of handing over a broken picture.
    fallback = None
    if ver["status"] == "fail" and not args.no_local_fallback:
        local_rgb, local_params, _ = lib.auto_correct(before, plan)
        local_path = lib.save_image(local_rgb, os.path.join(args.output, f"{stem}_local.png"))
        local_cmp = lib.save_image(
            lib.stitch_compare(before, local_rgb),
            os.path.join(args.output, f"{stem}_compare_local.jpg"), quality=92,
        )
        fallback = {
            "path": local_path,
            "compare": local_cmp,
            "params": local_params,
            "verify": lib.verify(before, local_rgb, plan, strict_sharpness=False),
            "reason": "生成结果结构漂移，已自动改用本地确定性校正",
        }
        fixed_path, cmp_path = local_path, local_cmp

    result = {
        "input": os.path.abspath(args.before),
        "generated": os.path.abspath(args.after),
        "verdict": verdict,
        "severity": severity,
        "diagnosis": notes,
        "source": args.source,
        "model": args.model,
        "quota_note": args.quota,
        "plan": plan,
        "before": m_before,
        "after": m_after if not fallback else lib.compute_metrics(lib.load_image(fixed_path)[0]),
        "verify": ver,
        "fallback": fallback,
        "paths": {"fixed": fixed_path, "compare": cmp_path},
        "status": f"done_{args.source}" if not fallback else "done_local_after_reject",
    }

    lines = ["# 曝光校正报告", "", f"## {os.path.basename(args.before)}", ""]
    lines.append(f"- 判定：**{verdict}**（{severity}）")
    lines.append(f"- 诊断：{'；'.join(notes)}")
    src_text = f"{args.source} 生图模型生成式重打光"
    if args.model:
        src_text += f"（`{args.model}`）"
    if fallback:
        src_text += " → 验收不通过，已自动改用本地确定性校正"
    lines.append(f"- 策略：{src_text}")
    if args.quota and not fallback:
        lines.append(f"- 额度：{args.quota}")
    lines.append("")
    lines.append(lib.metrics_md_table(m_before, result["after"]))
    lines.append("")
    if fallback:
        lv = fallback["verify"]
        lines.append(
            f"**生成结果验收：不通过**（结构相似度 {ver['structure']:.2f}）→ "
            f"最终交付本地校正版（结构相似度 {lv['structure']:.2f}）"
        )
    elif ver["status"] == "fail":
        lines.append(f"**验收：不通过**（结构相似度 {ver['structure']:.2f}）")
    else:
        lines.append(f"**验收：{ver['status']}**（结构相似度 {ver['structure']:.2f}）")
    for n in (fallback["verify"]["notes"] if fallback else ver["notes"]):
        lines.append(f"- {n}")
    lines.append("")
    lines.append(f"产物：`{os.path.basename(fixed_path)}`、`{os.path.basename(cmp_path)}`")
    lines.append("")

    md_path = os.path.join(args.output, "report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(os.path.join(args.output, f"{stem}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    result["report"] = md_path
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
