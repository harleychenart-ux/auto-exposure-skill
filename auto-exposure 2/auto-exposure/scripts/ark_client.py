"""Minimal Ark (火山方舟) image-generation client used for exposure correction.

Auth is read from, in order:
  1. a config file:  ~/.ark/exposure_correction.json
  2. env ARK_API_KEY  (standard Ark endpoint)
  3. env AGENT_API_KEY (Agent Plan endpoint, base URL .../api/plan/v3)

Only the standard library + numpy/Pillow + requests are used so the skill runs
in a bare Python environment.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

import exposure_lib as lib

CONFIG_PATH = os.path.expanduser("~/.ark/exposure_correction.json")
DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"
PLAN_BASE = "https://ark.cn-beijing.volces.com/api/plan/v3"

# Ark grants free *image* quota, counted in sheets - not tokens. Order below is
# "burn the free quota first": 5.0-lite 50 sheets -> 4.5 200 sheets -> 4.0
# 200 sheets, then the cheapest paid model. Alias entries are kept because
# different consoles/examples spell the same model with '.' or '-'.
MODEL_INFO = {
    "doubao-seedream-5-0-lite-260128": {"free": 50, "price": 0.22, "note": "5.0 lite"},
    "doubao-seedream-4-5-251128": {"free": 200, "price": 0.25, "note": "4.5（控制台标注即将下线）"},
    "doubao-seedream-4-0-250828": {"free": 200, "price": 0.20, "note": "4.0（控制台标注即将下线）"},
    "doubao-seedream-5-0-flash-260915": {"free": 0, "price": 0.12, "note": "5.0 flash，最便宜的付费档"},
    "doubao-seedream-5-0-pro-260628": {"free": 0, "price": 0.30, "note": "5.0 pro"},
    "doubao-seedream-5-5-261028": {"free": 0, "price": 0.30, "note": "5.5，编辑能力最强"},
    # aliases without the date suffix
    "doubao-seedream-5-0-lite": {"free": 50, "price": 0.22, "note": "5.0 lite"},
    "doubao-seedream-4-5": {"free": 200, "price": 0.25, "note": "4.5"},
    "doubao-seedream-4-0": {"free": 200, "price": 0.20, "note": "4.0"},
    "doubao-seedream-5-0-flash": {"free": 0, "price": 0.12, "note": "5.0 flash"},
}

DEFAULT_MODELS = [
    "doubao-seedream-5-0-lite-260128",   # 50 张免费
    "doubao-seedream-5-0-lite",          # 同上，别名
    "doubao-seedream-4-5-251128",        # 200 张免费
    "doubao-seedream-4-0-250828",        # 200 张免费
    "doubao-seedream-5-0-flash-260915",  # 无免费额度，0.12 元/张
]


def model_info(model: str) -> dict:
    return MODEL_INFO.get(model, {"free": None, "price": None, "note": ""})


class ArkError(RuntimeError):
    def __init__(self, message: str, hint: str = "", http: int = 0, code: str = ""):
        super().__init__(message)
        self.hint = hint
        self.http = http
        self.code = code


def load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {"base_url": DEFAULT_BASE, "size": "2K", "models": DEFAULT_MODELS}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg.update(json.load(fh))
        except (OSError, ValueError):
            pass

    if os.environ.get("ARK_API_KEY"):
        cfg["api_key"] = os.environ["ARK_API_KEY"]
        cfg["auth_source"] = "ARK_API_KEY"
    elif os.environ.get("AGENT_API_KEY"):
        cfg["api_key"] = os.environ["AGENT_API_KEY"]
        cfg["auth_source"] = "AGENT_API_KEY"
        if cfg.get("base_url") in (None, DEFAULT_BASE):
            cfg["base_url"] = PLAN_BASE
    elif cfg.get("api_key"):
        cfg["auth_source"] = "config"

    if os.environ.get("ARK_BASE_URL"):
        cfg["base_url"] = os.environ["ARK_BASE_URL"]
    if os.environ.get("ARK_IMAGE_MODEL"):
        cfg["models"] = [os.environ["ARK_IMAGE_MODEL"]]

    return cfg


def config_status() -> dict:
    cfg = load_config()
    return {
        "has_api_key": bool(cfg.get("api_key")),
        "auth_source": cfg.get("auth_source", "missing"),
        "base_url": cfg.get("base_url"),
        "config_path": CONFIG_PATH,
        "models": cfg.get("models"),
        "size": cfg.get("size"),
        "endpoint_reachable": _probe(cfg.get("base_url", DEFAULT_BASE)),
    }


def _probe(base_url: str) -> Any:
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/images/generations",
            json={},
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
        return f"ok(http {r.status_code})"
    except requests.RequestException as exc:  # network blocked / DNS / TLS
        return f"unreachable({type(exc).__name__})"


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _classify_error(status: int, body: dict) -> ArkError:
    err = body.get("error") or {}
    code = str(err.get("code") or body.get("code") or "")
    msg = str(err.get("message") or body.get("message") or "").strip()
    low = (code + " " + msg).lower()

    if status == 401 or "authentication" in low or "api key" in low:
        return ArkError(
            f"方舟鉴权失败（{code or status}）：{msg}",
            "API Key 无效或已删除。到火山方舟控制台重新复制 API Key，写入 "
            "~/.ark/exposure_correction.json 或环境变量 ARK_API_KEY。",
            status, code,
        )
    if status == 403 or "accessdenied" in low or "not activated" in low or "未开通" in msg:
        return ArkError(
            f"方舟拒绝访问（{code or status}）：{msg}",
            "该模型未开通或账号无权限。在方舟控制台「开通管理」里开通对应生图模型，"
            "开通前可先用免费额度在体验中心试用。",
            status, code,
        )
    if status == 429 or "ratelimit" in low or "setlimit" in low or "quota" in low:
        return ArkError(
            f"限流/额度用尽（{code or status}）：{msg}",
            "免费额度或并发限流。等 1 分钟后重试，或换价格更低的 doubao-seedream-5.0-flash。",
            status, code,
        )
    if status == 404 or "invalidendpoint" in low or "modelnotfound" in low or "model not" in low:
        return ArkError(
            f"模型不可用（{code or status}）：{msg}", "", status, code
        )
    if "sensitive" in low or "risk" in low or "审核" in msg:
        return ArkError(
            f"内容审核未通过（{code or status}）：{msg}",
            "图片被判为敏感内容。可改用本地确定性校正（correct_local.py），或换图重试。",
            status, code,
        )
    return ArkError(f"方舟接口报错（{code or status}）：{msg or '无详细信息'}", "", status, code)


def _is_model_error(exc: ArkError) -> bool:
    low = (exc.code + " " + str(exc)).lower()
    return any(k in low for k in ("invalidendpoint", "modelnotfound", "model not", "not found"))


def _is_size_error(exc: ArkError) -> bool:
    low = str(exc).lower()
    return "size" in low or "resolution" in low or "像素" in low


def generate(
    rgb: np.ndarray,
    prompt: str,
    cfg: dict | None = None,
    model: str | None = None,
    size: str | None = None,
    timeout: int = 180,
    retries: int = 2,
) -> tuple[np.ndarray, dict]:
    """Call Ark, return (image, meta). Raises ArkError with an actionable hint."""
    cfg = cfg or load_config()
    api_key = cfg.get("api_key")
    if not api_key:
        raise ArkError(
            "缺少方舟 API Key",
            "先按 references/ark-setup.md 开通方舟并拿到 API Key，再运行："
            'python3 scripts/ark_client.py --save-key "<你的 ARK_API_KEY>"',
            0, "NoApiKey",
        )

    base = str(cfg.get("base_url", DEFAULT_BASE)).rstrip("/")
    url = f"{base}/images/generations"
    models = [model] if model else list(cfg.get("models") or DEFAULT_MODELS)
    sizes = [size or cfg.get("size") or "2K", "1.5K", "auto"]

    image_url = lib.to_data_url(rgb, quality=92, max_side=2048)
    payload_base = {
        "prompt": prompt,
        "image": [image_url],
        "watermark": False,
        "response_format": "b64_json",
    }

    last_err: ArkError | None = None
    tried_models: list[str] = []
    for m in models:
        tried_models.append(m)
        model_dead = False
        for s in sizes:
            body = dict(payload_base, model=m, size=s)
            for attempt in range(retries + 1):
                try:
                    resp = requests.post(
                        url, headers=_headers(api_key), json=body, timeout=timeout
                    )
                except requests.RequestException as exc:
                    if attempt < retries:
                        time.sleep(2 + attempt * 3)
                        continue
                    raise ArkError(
                        f"网络请求失败：{type(exc).__name__}",
                        "当前环境可能无法访问 ark.cn-beijing.volces.com。请检查代理/网络，"
                        "或改用本地确定性校正。",
                    ) from exc

                if resp.status_code == 200:
                    data = resp.json()
                    return _decode(data, m, s, timeout)

                try:
                    err_body = resp.json()
                except ValueError:
                    err_body = {"message": resp.text[:200]}
                err = _classify_error(resp.status_code, err_body)

                if resp.status_code in (429, 500, 502, 503) and attempt < retries:
                    time.sleep(3 + attempt * 4)
                    continue

                last_err = err
                if _is_size_error(err) and s != sizes[-1]:
                    break          # only this size is rejected -> try next size
                model_dead = True  # model name invalid / auth / quota -> stop this model
                break
            if model_dead:
                break

        if not model_dead:
            continue  # all size fallbacks rejected; let another model try
        if _is_model_error(last_err):
            continue  # this model id is not available on this account -> next candidate
        break

    assert last_err is not None
    if _is_model_error(last_err):
        raise ArkError(
            f"没有可用的生图模型（已尝试：{', '.join(tried_models)}）：{last_err}",
            "到方舟控制台确认已开通/可试用的生图模型名，然后写入配置的 models 字段，"
            "或用 --model 指定。",
            last_err.http, last_err.code,
        )
    raise last_err


def _decode(data: dict, model: str, size: str, timeout: int) -> tuple[np.ndarray, dict]:
    item = (data.get("data") or [{}])[0]
    if item.get("error"):
        e = item["error"]
        raise ArkError(f"生成失败：{e.get('message', e)}", "换图重试或改用本地校正。")

    if item.get("b64_json"):
        raw = item["b64_json"]
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        rgb = np.asarray(img).astype(np.float32) / 255.0
    elif item.get("url"):
        resp = requests.get(item["url"], timeout=min(timeout, 60))
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        rgb = np.asarray(img).astype(np.float32) / 255.0
    else:
        raise ArkError("接口未返回图片数据", "检查 response_format 参数。")

    usage = data.get("usage") or {}
    used = data.get("model", model)
    info = model_info(used)
    return rgb, {
        "model": used,
        "free_quota": info["free"],
        "unit_price": info["price"],
        "model_note": info["note"],
        "counts_against_free_quota": bool(info["free"]),
        "size_requested": size,
        "size_returned": item.get("size"),
        "images_generated": usage.get("generated_images", 1),
        "output_tokens": usage.get("output_tokens"),
    }


def save_key(api_key: str, base_url: str = DEFAULT_BASE, models: list[str] | None = None) -> str:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    cfg = {"api_key": api_key, "base_url": base_url, "size": "2K"}
    if models:
        cfg["models"] = models
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    os.chmod(CONFIG_PATH, 0o600)
    return CONFIG_PATH


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ark image client for exposure correction")
    ap.add_argument("--check", action="store_true", help="打印配置与连通性")
    ap.add_argument("--save-key", metavar="KEY", help="保存 API Key 到用户配置目录")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    args = ap.parse_args()

    if args.save_key:
        print(f"saved -> {save_key(args.save_key, args.base_url)}")
    else:
        print(json.dumps(config_status(), ensure_ascii=False, indent=2))
