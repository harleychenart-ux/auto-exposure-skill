#!/usr/bin/env python3
"""Copy the skill folder into a skills root directory.

    python3 install.py --target "<技能根目录>"

Without --target it only prints the usual locations and does nothing.
It never deletes anything: if the target already exists it reports the
conflict and stops, so you can decide whether to replace it.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.join(HERE, "auto-exposure")
SKILL_NAME = "auto-exposure"


def hints() -> list[str]:
    home = os.path.expanduser("~")
    system = platform.system()
    if system == "Darwin":
        return [
            f"{home}/Library/Application Support/<客户端名>/skills",
            f"{home}/.doubao/agent_mode/workspace/.user_skills",
        ]
    if system == "Windows":
        appdata = os.environ.get("APPDATA", f"{home}\\AppData\\Roaming")
        return [
            f"{appdata}\\<客户端名>\\skills",
            f"{home}\\.doubao\\agent_mode\\workspace\\.user_skills",
        ]
    return [
        f"{home}/.config/<客户端名>/skills",
        f"{home}/.doubao/agent_mode/workspace/.user_skills",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="把 auto-exposure 复制到技能根目录")
    ap.add_argument("--target", help="技能根目录的绝对路径")
    ap.add_argument("--force", action="store_true", help="目标已存在时覆盖")
    args = ap.parse_args()

    if not os.path.isdir(SKILL_DIR):
        print(f"[错误] 找不到技能目录：{SKILL_DIR}", file=sys.stderr)
        return 1

    if not args.target:
        print("没有指定 --target。常见技能根目录（把 <客户端名> 换成你的客户端）：")
        for h in hints():
            print(f"  {h}")
        print("\n用法：python3 install.py --target \"<技能根目录>\"")
        print(f"装好后路径应为：<技能根目录>/{SKILL_NAME}/SKILL.md")
        return 0

    target = os.path.abspath(os.path.expanduser(args.target))
    dst = os.path.join(target, SKILL_NAME)

    if not os.path.isdir(target):
        print(f"[错误] 目标目录不存在：{target}", file=sys.stderr)
        print("       请先确认路径（客户端里通常有「技能目录」设置项）。", file=sys.stderr)
        return 1

    if os.path.exists(dst):
        if not args.force:
            print(f"[跳过] 已存在：{dst}")
            print("       确认要替换就加 --force；或者先把旧目录改名备份。")
            return 2
        shutil.rmtree(dst)

    shutil.copytree(
        SKILL_DIR, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    )

    if not os.path.isfile(os.path.join(dst, "SKILL.md")):
        print("[错误] 复制后缺少 SKILL.md，安装失败", file=sys.stderr)
        return 1

    print(f"[完成] 已安装到：{dst}")
    print("下一步：")
    print(f"  1) pip install -r requirements.txt")
    print(f"  2) python3 \"{os.path.join(dst, 'scripts', 'selftest.py')}\"")
    print("  3) 对豆包说「执行AEV」并附上图片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
