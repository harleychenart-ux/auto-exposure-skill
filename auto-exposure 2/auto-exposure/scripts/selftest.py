#!/usr/bin/env python3
"""Self-check: verify this skill will run on THIS machine before blaming the skill.

Run it right after installing:

    python3 scripts/selftest.py

It checks the Python version and required packages, then runs a full local
correction + verification on a synthetic image so you know the math and the
image IO both work. Nothing is written outside a temp directory.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    print("=" * 56)
    print("auto-exposure 自检 / selftest")
    print("=" * 56)
    print(f"Python : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"Script : {HERE}")

    problems = 0

    if sys.version_info < (3, 9):
        fail("需要 Python 3.9 及以上（类型注解语法）")
        problems += 1
    else:
        ok("Python 版本满足要求（>=3.9）")

    try:
        import numpy as np
        ok(f"numpy {np.__version__}")
    except Exception as exc:  # noqa: BLE001
        fail(f"缺少 numpy：{exc} → pip install numpy")
        return 1

    try:
        import PIL
        from PIL import Image
        ok(f"Pillow {PIL.__version__}")
    except Exception as exc:  # noqa: BLE001
        fail(f"缺少 Pillow：{exc} → pip install pillow")
        return 1

    try:
        import requests  # noqa: F401
        ok("requests 可用（火山方舟批量路径就绪）")
    except Exception:  # noqa: BLE001
        warn("缺少 requests → 只在走火山方舟批量路径时需要：pip install requests")

    # ---- functional round trip: diagnose -> local correct -> verify --------
    try:
        import exposure_lib as lib

        rng = np.random.default_rng(7)
        h, w = 360, 540
        base = rng.random((h, w, 3)).astype("float32") * 0.12
        yy = np.linspace(0, 1, h, dtype="float32")[:, None, None]
        xx = np.linspace(0, 1, w, dtype="float32")[None, :, None]
        smooth = 0.5 + 0.5 * np.sin(xx * 6.0) * np.cos(yy * 4.0)
        img = np.clip((base + 0.55 * smooth) * 2.2, 0, 1).astype("float32")

        metrics = lib.compute_metrics(img)
        verdict, severity, _ = lib.classify(metrics)
        plan = lib.plan_correction(metrics, verdict, severity)
        fixed, params, m2 = lib.auto_correct(img, plan)
        ver = lib.verify(img, fixed, plan)

        ok(f"诊断正常：{verdict}/{severity}，平均亮度 {metrics['mean']:.1%}")
        ok(f"校正正常：{metrics['mean']:.1%} → {m2['mean']:.1%}，EV {params.get('ev', 0):+.2f}")
        if ver["status"] == "fail":
            fail(f"验收异常：{ver['status']} / {ver['notes']}")
            problems += 1
        else:
            ok(f"验收正常：{ver['status']}，结构相似度 {ver['structure']:.3f}")

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "roundtrip.png")
            lib.save_image(fixed, p)
            back, meta = lib.load_image(p)
            ok(f"读写正常：写出 {os.path.getsize(p)} 字节，读回 {meta['width']}x{meta['height']}")
            if back.shape != fixed.shape:
                fail("读写尺寸不一致")
                problems += 1

        cmp_rgb = lib.stitch_compare(img, fixed)
        ok(f"对比图正常：{cmp_rgb.shape[1]}x{cmp_rgb.shape[0]}")
    except Exception as exc:  # noqa: BLE001
        fail(f"功能自检失败：{type(exc).__name__}: {exc}")
        problems += 1

    print("-" * 56)
    if problems:
        print(f"自检未通过（{problems} 项问题）。请先修好上面标 [FAIL] 的项。")
        return 1
    print("自检通过。可以直接用了：输入「执行AEV」+ 图片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
