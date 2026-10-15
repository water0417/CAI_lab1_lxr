# -*- coding: utf-8 -*-
"""等阶段 1（12 组筛选）跑完 → 自动挑最优 → 跑阶段 2（25 折确认）。

为什么要单独挂一个守护进程：阶段 1 要跑 2~3 小时，阶段 2 还要 1~2 小时。
用一个进程串起来，就不用人工盯着阶段 1 结束后再手动启动阶段 2。

注意：本进程**不设** OMP_NUM_THREADS，所以它拉起阶段 2 时用的是 numpy 的默认
线程设置——与 `main.py` 主表完全同口径。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import main as M

SCREEN = M.RESULT_DIR / "mlp_screen_5fold.csv"
N_COMBOS = 12


def main_cli() -> None:
    print(f"[守护] 等待阶段 1 完成（{N_COMBOS} 组）…", flush=True)
    while True:
        if SCREEN.exists():
            try:
                done = pd.read_csv(SCREEN)
                if len(done) >= N_COMBOS:
                    break
                print(f"[守护] 已完成 {len(done)}/{N_COMBOS}", flush=True)
            except Exception as e:
                print(f"[守护] 读取失败（{e}），重试", flush=True)
        time.sleep(120)

    d = pd.read_csv(SCREEN).sort_values("Macro-F1", ascending=False)
    print("\n[守护] 阶段 1 最终排名：", flush=True)
    print(d[["配置", "Macro-F1", "标准差", "泛化间隙"]].to_string(index=False), flush=True)
    best = d.iloc[0]
    hidden = str(best["hidden"]).replace(" ", "")
    print(f"\n[守护] 最优：{best['配置']}  → 启动阶段 2（25 折）", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", "mlp_tune.py",
                    "confirm", hidden, f"{best['alpha']:g}"], check=False)
    print("[守护] 两阶段全部结束", flush=True)


if __name__ == "__main__":
    main_cli()
