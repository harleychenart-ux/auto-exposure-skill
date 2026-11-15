"""Shared library for the auto-exposure skill.

Everything here is deterministic and dependency-light (numpy + Pillow only):
  * image IO that preserves EXIF orientation and 16-bit depth
  * display-referred + linear exposure metrics
  * verdict classification and correction planning
  * parametric tone mapping (EV gain -> highlight shoulder -> contrast)
  * structural similarity used to catch generative content drift
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Rec.709 luma weights.
_WR, _WG, _WB = 0.2126, 0.7152, 0.0722


# --------------------------------------------------------------------------- io


def load_image(path: str) -> tuple[np.ndarray, dict]:
    """Return (rgb float32 HxWx3 in 0..1, meta)."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    meta: dict[str, Any] = {
        "width": im.width,
        "height": im.height,
        "mode": im.mode,
        "icc_profile": bool(im.info.get("icc_profile")),
    }

    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im).astype(np.float32)
        peak = 65535.0 if arr.max() > 255 else 255.0
        arr = arr / peak
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif im.mode == "F":
        arr = np.asarray(im, dtype=np.float32)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    else:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        rgb = np.asarray(im).astype(np.float32) / 255.0

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, meta


def save_image(rgb: np.ndarray, path: str, quality: int = 95) -> str:
    out = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(out, "RGB")
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        im.save(path, quality=quality, subsampling=0, optimize=True)
    else:
        im.save(path, optimize=True)
    return path


def rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 92, max_side: int = 2048) -> bytes:
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def to_data_url(rgb: np.ndarray, quality: int = 92, max_side: int = 2048) -> str:
    b = rgb_to_jpeg_bytes(rgb, quality=quality, max_side=max_side)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


# ------------------------------------------------------------------- color math


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(
        np.float32
    )


def luma(rgb: np.ndarray) -> np.ndarray:
    return _WR * rgb[..., 0] + _WG * rgb[..., 1] + _WB * rgb[..., 2]


# --------------------------------------------------------------------- analysis


def compute_metrics(rgb: np.ndarray) -> dict:
    y = luma(rgb)
    lin = luma(srgb_to_linear(rgb))
    p01, p05, p50, p95, p99 = (float(v) for v in np.percentile(y, [1, 5, 50, 95, 99]))

    # Dead-white: *all* channels pinned at the ceiling -> detail physically gone.
    clip_high = float(np.mean(rgb.min(axis=2) >= 0.985))
    clip_any = float(np.mean(rgb.max(axis=2) >= 0.995))
    # Dead-black: every channel at the floor -> shadow detail gone.
    clip_low = float(np.mean(rgb.max(axis=2) <= 0.015))

    grid = _grid_stats(rgb)

    return {
        "mean": round(float(y.mean()), 4),
        "median": round(p50, 4),
        "mean_linear": round(float(lin.mean()), 6),
        "p01": round(p01, 4),
        "p05": round(p05, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
        "contrast": round(p95 - p05, 4),
        "clip_high": round(clip_high, 4),
        "clip_any": round(clip_any, 4),
        "clip_low": round(clip_low, 4),
        "shadow_share": round(float(np.mean(y < 0.15)), 4),
        "highlight_share": round(float(np.mean(y > 0.90)), 4),
        "grid": grid,
        "worst_blocks": _worst_blocks(grid),
    }


def _grid_stats(rgb: np.ndarray, n: int = 3) -> list[dict]:
    h, w = rgb.shape[:2]
    out = []
    for r in range(n):
        for c in range(n):
            blk = rgb[r * h // n : (r + 1) * h // n, c * w // n : (c + 1) * w // n]
            if blk.size == 0:
                continue
            out.append(
                {
                    "row": r,
                    "col": c,
                    "mean": round(float(luma(blk).mean()), 3),
                    "clip": round(float(np.mean(blk.min(axis=2) >= 0.985)), 3),
                }
            )
    return out


def _worst_blocks(grid: list[dict], limit: int = 3) -> list[dict]:
    """Blocks that are blown or crushed relative to the frame average."""
    if not grid:
        return []
    frame_mean = sum(g["mean"] for g in grid) / len(grid)
    flagged = [
        g
        for g in grid
        if (g["clip"] >= 0.10) or (g["mean"] <= 0.10 and frame_mean > 0.30)
    ]
    flagged.sort(key=lambda g: (-g["clip"], g["mean"]))
    return flagged[:limit]


def classify(metrics: dict) -> tuple[str, str, list[str]]:
    """Return (verdict, severity, notes)."""
    mean = metrics["mean"]
    clip_high = metrics["clip_high"]
    clip_any = metrics["clip_any"]
    clip_low = metrics["clip_low"]
    contrast = metrics["contrast"]
    hl_share = metrics["highlight_share"]
    notes: list[str] = []

    if clip_high >= 0.05 and mean >= 0.60:
        verdict = "overexposed"
        severity = "severe" if clip_high >= 0.12 or mean >= 0.75 else "moderate"
        notes.append(f"画面 {clip_high:.1%} 面积为三通道死白，高光细节已丢失")
    elif mean < 0.45 and (clip_high >= 0.02 or hl_share >= 0.05):
        # A blown window / sky + a dark subject is a contrast problem, not a
        # "too dark" problem: lifting the whole frame just blows the window
        # wider. 2% clipped is enough evidence of a real light source.
        verdict = "backlit"
        severity = "severe" if clip_high >= 0.15 or mean <= 0.15 else "moderate"
        notes.append(
            f"亮部已达 {hl_share:.0%} 面积（死白 {clip_high:.1%}），而画面均值仅 {mean:.1%}，"
            "属于逆光/大光比场景：整体提亮会把亮部推向纯白，需按亮部保形处理"
        )
    elif mean >= 0.72 and hl_share >= 0.30:
        # No hard clipping, yet half the frame sits above 90%: glare/haze has
        # pushed everything up and the black floor never reaches zero.
        verdict = "overexposed"
        severity = "moderate"
        notes.append(
            f"平均亮度 {mean:.0%}、{hl_share:.0%} 像素落在高光区，整体泛白；"
            f"最暗处也被抬到 {metrics['p01']:.0%}，说明有眩光/雾化，画面无黑场"
        )
    elif mean <= 0.30 or metrics["median"] <= 0.22:
        verdict = "underexposed"
        severity = "severe" if mean <= 0.16 else "moderate"
        notes.append(f"平均亮度仅 {mean:.1%}，暗部占比 {metrics['shadow_share']:.1%}")
        if clip_low >= 0.10:
            notes.append(f"其中 {clip_low:.1%} 为死黑，最深处信息不可恢复")
    elif contrast < 0.30 and 0.30 < mean < 0.78:
        verdict = "low_contrast"
        severity = "moderate"
        notes.append("动态范围利用率低，画面灰、缺乏层次")
    elif mean >= 0.55 and clip_any >= 0.05:
        verdict = "slightly_over"
        severity = "light"
        notes.append("高光接近临界，尚可局部挽回")
    else:
        verdict = "ok"
        severity = "none"
        notes.append("曝光处于正常范围")
        if clip_high >= 0.01:
            notes.append(f"有 {clip_high:.1%} 小面积高光溢出（太阳/灯光类），通常可接受")

    if hl_share > 0.015 and mean < 0.35:
        if clip_high < 0.02:
            # Small bright spots and no clipped mass = likely a real night
            # scene; warn against turning it into daylight.
            notes.append("暗部占绝大多数、亮点零散，像夜景/舞台光，提亮幅度需保守")
    return verdict, severity, notes


def plan_correction(metrics: dict, verdict: str, severity: str) -> dict:
    mean = metrics["mean"]
    # A night scene has small bright spots but no fully clipped mass. If 2%+
    # of the frame is pure white there is a real light source (window, sky,
    # lamp close-up) and "keep it dark" would be the wrong instruction.
    night = (
        0.015 < metrics["highlight_share"] < 0.05
        and mean < 0.35
        and metrics["clip_high"] < 0.02
    )

    if verdict == "overexposed":
        target = 0.50 if severity == "severe" else 0.52
        shoulder = 0.55 if severity == "severe" else 0.40
        contrast = 0.0
    elif verdict == "backlit":
        # A high-DR frame: lifting shadows globally would just push the window
        # into pure white, so cap the highlight end below 1.0 and give the gain
        # room to work on the dark half. Only when real clipping exists.
        target = 0.40
        shoulder = 0.75
        contrast = 0.10
    elif verdict == "underexposed":
        target = 0.34 if night else 0.47
        shoulder = 0.20
        contrast = 0.05
    elif verdict == "low_contrast":
        target = 0.47
        shoulder = 0.25
        contrast = 0.16
    elif verdict == "slightly_over":
        target = 0.55
        shoulder = 0.35
        contrast = 0.0
    else:
        target = mean
        shoulder = 0.20
        contrast = 0.0

    # Veiling glare / haze: the black floor sits far above zero, so the frame
    # reads as a milky wall no matter how you set the exposure. Strip it first.
    p01 = metrics.get("p01", 0.0)
    black_point = 0.0
    washed = bool(mean > 0.60 and p01 > 0.15)
    if washed:
        black_point = round(min(0.35, max(0.0, p01 - 0.03)), 3)
        contrast = max(contrast, 0.10)

    ev = _ev_to_target(metrics["mean_linear"], target)
    # Global EV is clamped: beyond ~3.3 stops a global multiply just crushes or
    # blows the frame, the rest has to come from local tone mapping / generation.
    ev = max(-3.2, min(3.3, ev))

    bright_mass = metrics["clip_high"] >= 0.02 or metrics["highlight_share"] >= 0.05
    shoulder_to = 0.90 if (verdict == "backlit" and bright_mass) else 1.0

    return {
        "target_mean": round(target, 3),
        "shoulder_to": shoulder_to,
        "ev": round(ev, 2),
        "shoulder": round(shoulder, 3),
        "knee": 0.72,
        "contrast": round(contrast, 3),
        "black_point": black_point,
        "shadow_lift": 1.0 if verdict in ("underexposed", "backlit") else 0.0,
        "night_guard": bool(night),
        "washed": washed,
    }


def _ev_to_target(mean_linear: float, target_display: float) -> float:
    if mean_linear <= 1e-5:
        return 3.3
    target_lin = float(srgb_to_linear(np.float32(target_display)))
    return float(math.log2(max(target_lin, 1e-5) / mean_linear))


# ----------------------------------------------------------------- tone mapping


def _soft_shoulder(y: np.ndarray, knee: float, strength: float) -> np.ndarray:
    """Monotonic highlight rolloff above `knee`; strength 0 => identity."""
    if strength <= 0:
        return y
    k = 1e-4 + 3.5 * strength
    t = np.clip((y - knee) / (1.0 - knee), 0.0, None)
    rolled = knee + (1.0 - knee) * (np.tanh(t * k) / np.tanh(k))
    # Above knee blend in the rolled curve so the knee point stays continuous.
    out = np.where(y > knee, rolled, y)
    return np.clip(out, 0.0, 1.0)


def apply_params(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Veiling-glare strip + exposure gain + highlight shoulder + shadow lift + S-curve."""
    lin = srgb_to_linear(rgb)

    # Veiling glare is an additive neutral offset in *linear* radiance, so remove
    # it there. Subtracting it in sRGB instead shifts hue and turns haze purple.
    bp = params.get("black_point", 0.0)
    if bp > 0:
        bp_lin = float(srgb_to_linear(np.float32(bp)))
        lin = np.clip(lin - bp_lin, 0.0, None)

    gain = 2.0 ** params.get("ev", 0.0)
    lin = lin * gain

    lum_lin = luma(lin)
    if params.get("shoulder", 0) > 0:
        # Compress in LINEAR light against a headroom ceiling, so highlights that
        # the gain pushed past 1.0 still land on distinct values instead of all
        # collapsing into one clipped white. Doing this in display space after
        # the conversion is too late: everything above 1.0 already reads as 1.0.
        knee_disp = float(params.get("knee", 0.72))
        knee_lin = float(srgb_to_linear(np.float32(knee_disp)))
        ceiling = max(1.0, float(np.percentile(lum_lin, 99.5)))
        if ceiling > knee_lin + 1e-6:
            k = 1e-4 + 3.5 * float(params["shoulder"])
            t = np.clip((lum_lin - knee_lin) / (ceiling - knee_lin), 0.0, None)
            top_disp = float(params.get("shoulder_to", 1.0))
            top_lin = float(srgb_to_linear(np.float32(top_disp)))
            comp = knee_lin + (top_lin - knee_lin) * (np.tanh(t * k) / np.tanh(k))
            comp = np.where(lum_lin > knee_lin, comp, lum_lin)
            ratio = comp / np.maximum(lum_lin, 1e-6)
            lin = lin * ratio[..., None]

    disp = linear_to_srgb(np.clip(lin, 0.0, 1.0))

    # Shadow lift: pull up low mids only, keep the black anchor (avoids milky fog).
    lift = params.get("shadow_lift", 0.0)
    if lift > 0:
        y = luma(disp)
        w = np.clip((0.62 - y) / 0.62, 0.0, 1.0) ** 1.6
        gamma_up = 0.55
        y_new = y ** gamma_up
        y_out = y + (y_new - y) * w * lift
        ratio = np.clip(y_out / np.maximum(y, 1e-6), 0.0, 4.0)
        disp = np.clip(disp * ratio[..., None], 0.0, 1.0)

    c = params.get("contrast", 0.0)
    if c > 0:
        disp = np.clip(0.5 + (disp - 0.5) * (1.0 + c), 0.0, 1.0)

    return disp.astype(np.float32)


def _downscale(rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max(rgb.shape[:2]) <= max_side:
        return rgb
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    scale = max_side / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def calibrate_plan(rgb: np.ndarray, plan: dict) -> dict:
    """Lower the target to what this frame can actually reach.

    A fixed target is a wish, not a fact: a room shot against a blown window
    cannot reach 40% mean without turning the window into pure white. Probe the
    proxy image with rising EV, stop where clipping starts to grow, and cap the
    target there. Only ever lowers the target, never raises it, with a floor so
    the result still has to be a real lift.
    """
    target = float(plan.get("target_mean", 0.0))
    if target <= 0.0:
        return plan

    proxy = _downscale(rgb, 520)
    m0 = compute_metrics(proxy)
    allowed_clip = max(m0["clip_high"] + 0.01, 0.02)

    best_mean = float(m0["mean"])
    ev = float(plan.get("ev", 0.0))
    for _ in range(10):
        ev = min(3.3, ev + 0.4)
        m = compute_metrics(apply_params(proxy, dict(plan, ev=ev)))
        if m["clip_high"] > allowed_clip:
            break
        best_mean = float(m["mean"])

    reach = max(best_mean, 0.30)
    if reach < target - 0.02:
        plan = dict(plan)
        plan["target_mean"] = round(min(target, reach), 3)
        plan["target_limited"] = True
        plan["reachable_mean"] = round(reach, 3)
    return plan


def auto_correct(rgb: np.ndarray, plan: dict, max_iter: int = 6, tol: float = 0.015) -> tuple[np.ndarray, dict, dict]:
    """Secant search on the *display* mean luminance.

    The EV -> mean(map) relation is non-linear (and Jensen-biased), so a closed
    form log2(target/mean) overshoots. Measuring the local slope is cheap and
    lands on target within a pixel-fraction of a stop.
    """
    # Parameter search runs on a proxy when the image is large; the final pass
    # is always applied at full resolution.
    plan = calibrate_plan(rgb, plan)
    search = _downscale(rgb, 900)
    params = dict(plan)
    out = apply_params(search, params)
    m = compute_metrics(out)
    target = plan["target_mean"]

    for _ in range(max_iter):
        err = target - m["mean"]
        if abs(err) <= tol:
            break
        h = 0.6
        probe_ev = max(-3.4, min(3.4, params["ev"] + h))
        probe = apply_params(search, dict(params, ev=probe_ev))
        slope = (compute_metrics(probe)["mean"] - m["mean"]) / (probe_ev - params["ev"] + 1e-9)
        if slope < 1e-4:
            break
        step = max(-0.8, min(0.8, err / slope))
        new_ev = max(-3.4, min(3.4, params["ev"] + step))
        if abs(new_ev - params["ev"]) < 1e-3:
            break
        params["ev"] = round(new_ev, 3)
        out = apply_params(search, params)
        m = compute_metrics(out)

    full = apply_params(rgb, params)
    return full, params, compute_metrics(full)


# ----------------------------------------------------------------- verification


def _prep_gray(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on 8x8 pooled luminance. Brightness-sensitive by design, so it
    is reported for reference only - use structure_score() for pass/fail."""
    ga, gb = _prep_gray(a), _prep_gray(b)
    n = 8
    pa = ga.reshape(n, 32, n, 32).mean(axis=(1, 3))
    pb = gb.reshape(n, 32, n, 32).mean(axis=(1, 3))
    mu_a, mu_b = pa.mean(), pb.mean()
    va, vb = pa.var(), pb.var()
    cov = ((pa - mu_a) * (pb - mu_b)).mean()
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    )


def _highpass(rgb: np.ndarray, size: int = 256, k: int = 9) -> np.ndarray:
    """Zero-mean, unit-variance high-pass map: exposure and gamma become irrelevant,
    only layout/edges remain."""
    g = _prep_gray(rgb, size)
    pad = np.pad(g, k // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
    blur = win.mean(axis=(2, 3))
    hp = g - blur
    return hp / (hp.std() + 1e-6)


def structure_score(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of high-pass maps -> 1.0 means the layout is untouched."""
    ha, hb = _highpass(a), _highpass(b)
    return float(np.clip((ha * hb).mean(), -1.0, 1.0))


def aspect_delta(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.shape[1] / max(a.shape[0], 1)
    rb = b.shape[1] / max(b.shape[0], 1)
    return abs(rb - ra) / ra


def verify(before: np.ndarray, after: np.ndarray, plan: dict, strict_sharpness: bool = False) -> dict:
    mb, ma = compute_metrics(before), compute_metrics(after)
    ssim = ssim_global(before, after)
    struct = structure_score(before, after)
    target = plan.get("target_mean", 0.47)

    reached = abs(ma["mean"] - target) <= 0.10
    no_new_clip = ma["clip_high"] <= max(mb["clip_high"] + 0.02, 0.06)
    structure_ok = struct >= 0.62
    ratio_delta = aspect_delta(before, after)
    aspect_ok = ratio_delta <= 0.03
    sharp_ok = True
    if strict_sharpness:
        sharp_ok = _sharpness(after) >= 0.70 * _sharpness(before)

    notes = []
    if not reached:
        notes.append(f"亮度未落到目标区间：{ma['mean']:.0%} vs 目标 {target:.0%}")
    if not no_new_clip:
        notes.append(
            f"高光死白由 {mb['clip_high']:.1%} 增加到 {ma['clip_high']:.1%}，出现过曝加剧"
        )
    if not structure_ok:
        notes.append(f"结构相似度 {struct:.2f}（阈值 0.62），构图或主体可能被改动")
    if not aspect_ok:
        notes.append(f"宽高比变化 {ratio_delta:.1%}，输出被裁切或补边")
    if not sharp_ok:
        notes.append("输出清晰度明显低于原图，可能被过度平滑")
    if ma["clip_high"] >= 0.03:
        notes.append(f"仍有 {ma['clip_high']:.1%} 死白，纯白区域信息已丢失，只能靠生成重绘补回")
    if ma["clip_low"] >= 0.05:
        notes.append(f"仍有 {ma['clip_low']:.1%} 死黑，最暗部数据不可恢复")

    hard_fail = (not structure_ok) or (not aspect_ok) or (not sharp_ok)
    return {
        "pass": bool(reached and no_new_clip and not hard_fail),
        "status": "fail" if hard_fail else ("pass" if reached and no_new_clip else "partial"),
        "structure": round(struct, 3),
        "ssim": round(ssim, 3),
        "before": mb,
        "after": ma,
        "notes": notes,
    }


def _sharpness(rgb: np.ndarray) -> float:
    g = _prep_gray(rgb, 512)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return float(gx + gy)


# ---------------------------------------------------------------------- prompts

_VERDICT_TEXT = {
    "overexposed": "整体过曝，高光一片死白",
    "backlit": "逆光/大光比，高光过曝且中间调偏暗",
    "underexposed": "曝光不足，整体发暗、暗部发黑",
    "low_contrast": "曝光偏灰、反差不足",
    "slightly_over": "高光接近临界、略有过曝",
    "ok": "曝光基本正常，仅需轻微优化",
}


def build_prompt(metrics: dict, verdict: str, plan: dict, pre_normalized: bool = False) -> str:
    clip = metrics["clip_high"]
    low = metrics["clip_low"]
    target = plan["target_mean"]

    if plan.get("washed"):
        goal = (
            f"压回正常曝光（平均亮度约 {target:.0%}），去掉蒙在画面上的灰雾与眩光，"
            "把被抬起的黑场压回真正的深黑，恢复通透度与层次；不要简单整体死压变脏"
        )
    elif verdict == "overexposed":
        goal = (
            f"把整体压回到正常中性曝光（平均亮度约 {target:.0%}），"
            "重点找回被烧白区域（天空/高光/皮肤高光）的层次与渐变，不要整体发灰"
        )
    elif verdict == "backlit":
        goal = (
            f"恢复成正常日光曝光（平均亮度约 {target:.0%}）：高光收回到有细节的范围，"
            "同时把背光主体（人物或物体暗部）提亮到可辨，保留真实阴影方向"
        )
    elif verdict == "underexposed":
        goal = (
            f"整体提亮到正常曝光（平均亮度约 {target:.0%}），"
            "暗部提亮后要有层次、不能糊成一片灰，噪点不要被放大成彩色噪斑"
        )
    elif verdict == "low_contrast":
        goal = (
            f"把平均亮度落在 {target:.0%} 附近，拉开黑白场、增加清晰层次，"
            "去除灰雾感但保留高光不过曝"
        )
    else:
        goal = f"把平均亮度落在 {target:.0%} 附近，只做轻微曝光优化"

    parts = [
        "对这张照片做【曝光校正】，只调整光照与明暗，绝不重绘画面内容。",
        f"当前问题：{_VERDICT_TEXT.get(verdict, '曝光异常')}"
        f"（平均亮度 {metrics['mean']:.0%}，三通道死白 {clip:.1%}，死黑 {low:.1%}）。",
        f"目标：{goal}。",
    ]
    if plan.get("night_guard"):
        parts.append("这是夜景/弱光场景，允许保留夜色氛围，不要提亮成白天。")
    if pre_normalized:
        parts.append(
            "输入图已做轻度自动归一化处理（直方图被压缩过），请在此基础上还原自然、真实的光比。"
        )
    parts.append(
        "必须原样保留：构图与裁切、视角与透视、人物身份与五官、物体形状与位置、"
        "文字内容、白平衡与色温倾向、景深与镜头畸变、颗粒与胶片质感。"
    )
    parts.append(
        "禁止：增删任何元素、替换或美化人物、改变肤色与材质、叠加滤镜或 HDR 光晕、"
        "重新添加锐化、改变画面比例、输出 AI 字样水印。"
    )
    parts.append("画面比例与输入图一致，输出为自然真实的摄影照片质感。")
    return "\n".join(parts)


# -------------------------------------------------------------------- reporting


def stitch_compare(before: np.ndarray, after: np.ndarray, height: int = 720) -> np.ndarray:
    """BEFORE | AFTER side-by-side strip."""
    ims = []
    for arr in (before, after):
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
        w = max(1, round(im.width * height / im.height))
        ims.append(im.resize((w, height), Image.LANCZOS))
    gap = 10
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + gap, height + 34), (24, 24, 24))
    canvas.paste(ims[0], (0, 34))
    canvas.paste(ims[1], (ims[0].width + gap, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "BEFORE", fill=(240, 240, 240))
    draw.text((ims[0].width + gap + 8, 10), "AFTER", fill=(240, 240, 240))
    return np.asarray(canvas).astype(np.float32) / 255.0


def metrics_md_table(before: dict, after: dict) -> str:
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
