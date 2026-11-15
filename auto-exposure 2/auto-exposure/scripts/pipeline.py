#!/usr/bin/env python3
"""End-to-end exposure correction: diagnose -> pre-normalize -> Ark generate ->
verify -> deliver. Also usable with --mode local when no Ark key is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ark_client  # noqa: E402
import exposure_lib as lib  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_inputs(path: str) -> list[str]:
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if os.path.splitext(f)[1].lower() in IMAGE_EXT
        ]
        if not files:
            raise SystemExit(f"目录里没有可处理的图片：{path}")
        return files
    return [path]


def needs_pre_normalize(metrics: dict, verdict: str) -> bool:
    return (metrics["mean"] <= 0.14 and verdict == "underexposed") or (
        metrics["clip_high"] >= 0.20 and verdict in ("overexposed", "backlit")
    )


def _stitch(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    h = 720
    ims = []
    for arr in (before, after):
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
        w = max(1, round(im.width * h / im.height))
        ims.append(im.resize((w, h), Image.LANCZOS))
    gap = 10
    total_w = ims[0].width + ims[1].width + gap
    canvas = Image.new("RGB", (total_w, h + 34), (24, 24, 24))
    canvas.paste(ims[0], (0, 34))
    canvas.paste(ims[1], (ims[0].width + gap, 34))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), "BEFORE", fill=(240, 240, 240))
    d.text((ims[0].width + gap + 8, 10), "AFTER", fill=(240, 240, 240))
    return np.asarray(canvas).astype(np.float32) / 255.0


def _md_table(before: dict, after: dict) -> str:
    rows = [
        ("平均亮度", "mean", "{:.1%}"),
        ("中位亮度", "median", "{:.1%}"),
        ("动态范围 p95-p05", "contrast", "{:.1%}"),
        ("三通道死白占比", "clip_high", "{:.2%}"),
        ("死黑占比", "clip_low", "{:.2%}"),
    ]
    lines = ["| 指标 | 校正前 | 校正后 |", "| --- | --- | --- |"]
    for label, key, fmt in rows:
        lines.append(f"| {label} | {fmt.format(before[key])} | {fmt.format(after[key])} |")
    return "\n".join(lines)


def process_one(path: str, out_dir: str, args) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()

    rgb, meta = lib.load_image(path)
    metrics = lib.compute_metrics(rgb)
    verdict, severity, notes = lib.classify(metrics)
    plan = lib.plan_correction(metrics, verdict, severity)

    result: dict[str, Any] = {
        "input": os.path.abspath(path),
        "verdict": verdict,
        "severity": severity,
        "diagnosis": notes,
        "plan": plan,
        "before": metrics,
        "paths": {},
        "status": "skipped",
    }

    if args.target:
        plan["target_mean"] = float(args.target)

    if verdict == "ok" and not args.force:
        result["status"] = "ok_no_change"
        result["message"] = "曝光正常，未做校正（如需强制处理加 --force）"
        return result

    # ------------------------------------------------------ local deterministic
    local_rgb, local_params, local_metrics = lib.auto_correct(rgb, plan)
    local_path = lib.save_image(local_rgb, os.path.join(out_dir, f"{stem}_local.png"))
    result["paths"]["local"] = local_path
    result["local_params"] = local_params
    result["local_metrics"] = local_metrics

    if args.mode == "local":
        final_rgb, final_metrics, source = local_rgb, local_metrics, "local"
    else:
        # -------------------------------------------------- ark generative pass
        pre = None
        pre_normalized = False
        source_rgb = rgb
        has_key = bool((ark_client.load_config() or {}).get("api_key"))
        if has_key and not args.no_pre and needs_pre_normalize(metrics, verdict):
            pre_plan = dict(plan, target_mean=0.33 if verdict == "underexposed" else 0.62)
            pre = lib.auto_correct(rgb, pre_plan)[0]
            source_rgb, pre_normalized = pre, True
            result["pre_normalized"] = True

        prompt = lib.build_prompt(metrics, verdict, plan, pre_normalized)
        result["prompt"] = prompt
        try:
            gen_rgb, gen_meta = ark_client.generate(
                source_rgb, prompt, model=args.model, size=args.size, timeout=args.timeout
            )
            result["ark"] = gen_meta
            final_rgb, final_metrics, source = gen_rgb, lib.compute_metrics(gen_rgb), "ark"
        except ark_client.ArkError as exc:
            result["ark_error"] = {"message": str(exc), "hint": exc.hint}
            if args.mode == "ark":          # explicit request -> never fake success
                raise
            final_rgb, final_metrics, source = local_rgb, local_metrics, "local_fallback"
            result["fallback_reason"] = "ark unavailable"

    # ---------------------------------------------------------------- verify
    ver = lib.verify(rgb, final_rgb, plan, strict_sharpness=(source == "ark"))
    result["verify"] = ver

    if source == "ark" and ver["status"] == "fail":
        # Generative result drifted -> prefer the content-safe deterministic one.
        result["ark_rejected"] = True
        final_rgb, final_metrics, source = local_rgb, local_metrics, "local_after_reject"
        result["verify_after_fallback"] = lib.verify(rgb, final_rgb, plan)

    fixed_path = lib.save_image(final_rgb, os.path.join(out_dir, f"{stem}_fixed.png"))
    cmp_path = lib.save_image(
        _stitch(rgb, final_rgb), os.path.join(out_dir, f"{stem}_compare.jpg"), quality=92
    )
    result["paths"]["fixed"] = fixed_path
    result["paths"]["compare"] = cmp_path
    result["source"] = source
    result["after"] = final_metrics
    result["ark_used"] = source == "ark"
    result["status"] = (
        "done_ark" if source == "ark" else
        "done_local" if source == "local" else
        "done_fallback"
    )
    result["elapsed_sec"] = round(time.time() - t0, 1)

    with open(os.path.join(out_dir, f"{stem}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def write_md(results: list[dict], out_dir: str) -> str:
    lines = ["# 曝光校正报告", ""]
    for r in results:
        name = os.path.basename(r["input"])
        lines.append(f"## {name}")
        lines.append("")
        if r["status"] == "failed":
            lines.append(f"- 处理失败：{r['ark_error']['message']}")
            if r["ark_error"].get("hint"):
                lines.append(f"- 下一步：{r['ark_error']['hint']}")
            lines.append("")
            continue
        if r["status"] == "ok_no_change":
            lines.append(f"- 判定：**曝光正常**（平均亮度 {r['before']['mean']:.1%}）")
            lines.append(f"- 处理：{r.get('message','')}")
            lines.append("")
            continue

        src = {
            "ark": "方舟生图模型生成式重打光",
            "local": "本地确定性校正",
            "local_fallback": "方舟不可用，退回本地确定性校正",
            "local_after_reject": "方舟结果验收不合格，退回本地确定性校正",
        }.get(r.get("source", ""), r.get("source", ""))

        lines.append(f"- 判定：**{r['verdict']}**（{r['severity']}）")
        lines.append(f"- 诊断：{'；'.join(r['diagnosis'])}")
        lines.append(f"- 策略：{src}")
        if r.get("ark"):
            lines.append(
                f"- 模型：`{r['ark'].get('model')}`，输出 {r['ark'].get('size_returned') or r['ark'].get('size_requested')}"
            )
        if r.get("ark_error"):
            lines.append(f"- 方舟调用失败：{r['ark_error']['message']} → 已自动退回本地校正")
        if r.get("pre_normalized"):
            lines.append("- 预处理：极端曝光，已先做轻度归一化再送模型")
        lines.append("")
        lines.append(_md_table(r["before"], r["after"]))
        lines.append("")
        v = r.get("verify_after_fallback") or r.get("verify")
        if v:
            lines.append(
                f"**验收：{v['status']}**（结构相似度 {v['structure']:.2f}）"
                if v["status"] != "fail"
                else f"**验收：不通过**（结构相似度 {v['structure']:.2f}）"
            )
            for n in v["notes"]:
                lines.append(f"- {n}")
        lines.append("")
        lines.append(f"产物：`{os.path.basename(r['paths']['fixed'])}`、"
                     f"`{os.path.basename(r['paths']['compare'])}`")
        lines.append("")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="自动曝光校正（方舟生成式 + 本地确定性兜底）")
    ap.add_argument("-i", "--input", required=True, help="图片路径或图片目录")
    ap.add_argument("-o", "--output", default=None, help="输出目录，默认 <输入同级>/exposure_fixed")
    ap.add_argument("--mode", choices=["auto", "ark", "local"], default="auto",
                    help="auto=优先方舟、失败退回本地；ark=只用方舟；local=只用本地")
    ap.add_argument("--model", default=None, help="指定方舟生图模型，如 doubao-seedream-5.0-flash")
    ap.add_argument("--size", default=None, help="输出尺寸档位，默认取配置 2K")
    ap.add_argument("--target", type=float, default=None, help="手动指定目标平均亮度 0-1")
    ap.add_argument("--force", action="store_true", help="曝光正常时也强制校正")
    ap.add_argument("--no-pre", action="store_true", help="关闭极端曝光的送图前归一化")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json", action="store_true", help="仅打印 JSON 结果")
    args = ap.parse_args()

    inputs = collect_inputs(args.input)
    out_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(inputs[0])), "exposure_fixed"
    )
    os.makedirs(out_dir, exist_ok=True)

    results = []
    failed = False
    for p in inputs:
        if not args.json:
            print(f"[处理] {os.path.basename(p)}", file=sys.stderr)
        try:
            results.append(process_one(p, out_dir, args))
        except ark_client.ArkError as exc:
            failed = True
            results.append({
                "input": os.path.abspath(p),
                "status": "failed",
                "ark_error": {"message": str(exc), "hint": exc.hint},
            })
        time.sleep(0.2)

    md = write_md(results, out_dir)
    payload = {
        "output_dir": out_dir,
        "report": md,
        "count": len(results),
        "items": [
            {
                k: r[k]
                for k in (
                    "input", "verdict", "severity", "status", "source",
                    "paths", "before", "after", "verify", "ark", "ark_error", "ark_used",
                    "plan", "diagnosis", "message",
                )
                if k in r
            }
            for r in results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
