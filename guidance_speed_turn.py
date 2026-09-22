"""Speed and turn margin, without a new warhead.

The closed drone flies at 50 m/s with an 80° bank.  Lateral accel is
then g·tan(80°) ≈ 5.7 g, and heading rate falls as 1/v if only the
speed cap is raised.  This study separates those two levers on the
cells that failed for lack of time or turn: tail-chase timeout, and
beam lag.  Head-on is the control.

Thrust is raised only so drag does not eat the commanded speed.  That
is not a motor design.  Both warheads are scored at the same closest
approach.  Higher speed also raises kinetic energy, so kill rate is
not the geometry result — miss and timeout are.

Output: ``results/guidance_speed_turn/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aircraft import G, RHO
from interceptor import (
    EngagementConfig, InterceptorDrone, PNGuidance, PredictivePursuit,
    SensorNoise, WarheadType, run_engagement, scenario_beam,
    scenario_head_on, scenario_tail_chase, score_at_cpa,
)
from guidance_law_study import FRAG, KIN, stable_seed

OUT = "results/guidance_speed_turn"

LAWS = (
    ("Predictive", "pred"),
    ("PN raw", "pn"),
)
SCENARIOS = ("head_on", "tail_chase", "beam")
SPEEDS = (50.0, 100.0, 200.0)
# g6 is the current 80° bank.  g30 holds a higher lateral accel so
# heading rate does not collapse as 1/v.
TURNS = (("g6", 5.67), ("g30", 30.0))
DISTANCES = (500, 1000, 2000)
TARGET_SPEEDS = (15.0, 20.0)
N_TRIALS = 30
SEED_BASE = 19000
ALT = 50.0
SIGMA = 0.5


def _airframe(speed: float, n_g: float) -> InterceptorDrone:
    bank = min(math.atan(n_g), math.radians(89.0))
    drag = 0.5 * RHO * speed * speed * 0.20 * 0.035
    # Hold the commanded speed.  The 150 N drone cannot sustain 200 m/s.
    thrust = max(150.0, 1.2 * drag)
    return InterceptorDrone(
        v_cruise=speed,
        max_bank=bank,
        max_accel=n_g * G,
        T_max=thrust,
        bank_tau=0.15,
    )


def _law(law_id: str):
    if law_id == "pred":
        return PredictivePursuit()
    if law_id == "pn":
        return PNGuidance(N=4.0, los_tau=0.0)
    raise ValueError(law_id)


def _geometry(name, distance, tgt_speed, intr_speed):
    if name == "head_on":
        return scenario_head_on(distance, tgt_speed, intr_speed, ALT)
    if name == "tail_chase":
        return scenario_tail_chase(distance, tgt_speed, intr_speed, ALT)
    if name == "beam":
        return scenario_beam(distance, tgt_speed, intr_speed, ALT)
    raise ValueError(name)


def _run_cell(task: dict) -> list[dict]:
    records = []
    intr = _airframe(task["speed"], task["n_g"])
    for trial in range(task["n_trials"]):
        seed = stable_seed(
            "spd", task["scenario"], task["distance"], task["tgt_speed"],
            task["speed"], task["turn"], task["law_id"], trial,
            base=task["seed_base"])
        tp, tv, ip, iv = _geometry(
            task["scenario"], task["distance"], task["tgt_speed"], task["speed"])
        cfg = EngagementConfig(
            interceptor=intr,
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"]),
            sensor=SensorNoise(SIGMA, SIGMA * 0.4),
            store_trajectory=False,
        )
        res = run_engagement(tp, tv, ip, iv, cfg, seed=seed)
        miss = float(res.miss_distance)
        _, kin_pk = score_at_cpa(miss, res.approach_speed,
                                 WarheadType.KINETIC_IMPACT, FRAG, KIN)
        timed_out = res.engagement_time >= cfg.max_time - 0.05 and miss > 50.0
        records.append({
            "law": task["law"], "turn": task["turn"],
            "speed": task["speed"], "scenario": task["scenario"],
            "distance": task["distance"], "tgt_speed": task["tgt_speed"],
            "trial": trial,
            "miss": miss if math.isfinite(miss) else 1e9,
            "time": float(res.engagement_time),
            "timeout": timed_out,
            "inside3": bool(miss < 3.0),
            "inside6": bool(miss < 6.0),
            "kin_pk": float(kin_pk),
        })
    return records


def _tasks(n_trials: int, seed_base: int):
    tasks = []
    for law, law_id in LAWS:
        for turn, n_g in TURNS:
            for speed in SPEEDS:
                for scenario in SCENARIOS:
                    for dist in DISTANCES:
                        for tgt in TARGET_SPEEDS:
                            tasks.append({
                                "law": law, "law_id": law_id,
                                "turn": turn, "n_g": n_g, "speed": speed,
                                "scenario": scenario, "distance": dist,
                                "tgt_speed": tgt,
                                "n_trials": n_trials, "seed_base": seed_base,
                            })
    return tasks


def _rows(records, **kw):
    out = records
    for key, val in kw.items():
        out = [r for r in out if r[key] == val]
    return out


def _stats(rows):
    n = len(rows)
    misses = [r["miss"] for r in rows if r["miss"] < 500]
    return {
        "n": n,
        "timeout": float(np.mean([1.0 if r["timeout"] else 0.0 for r in rows])) if n else float("nan"),
        "median_miss": float(np.median(misses)) if misses else float("nan"),
        "p_inside3": float(np.mean([1.0 if r["inside3"] else 0.0 for r in rows])) if n else float("nan"),
        "p_inside6": float(np.mean([1.0 if r["inside6"] else 0.0 for r in rows])) if n else float("nan"),
        "kin_pk": float(np.mean([r["kin_pk"] for r in rows])) if n else float("nan"),
        "mean_time": float(np.mean([r["time"] for r in rows])) if n else float("nan"),
    }


def summarize(records):
    out = {}
    for law, _ in LAWS:
        out[law] = {}
        for turn, _ in TURNS:
            out[law][turn] = {}
            for speed in SPEEDS:
                out[law][turn][str(speed)] = {}
                for scenario in SCENARIOS:
                    rows = _rows(records, law=law, turn=turn, speed=speed, scenario=scenario)
                    out[law][turn][str(speed)][scenario] = _stats(rows)
                    # The timeout cell, kept separate.
                    cell = _rows(records, law=law, turn=turn, speed=speed,
                                 scenario="tail_chase", distance=2000, tgt_speed=20.0)
                    out[law][turn][str(speed)]["tail2000_20"] = _stats(cell)
    return out


def _fmt(x, nd=2):
    if x != x:
        return "—"
    return f"{x:.{nd}f}"


def generate_report(summ, n, elapsed) -> str:
    lines = [
        "# 速度と旋回余裕 — 時間切れとビームの遅れ",
        "",
        "## 何を確かめたか",
        "",
        "50 m/s・バンク 80° の機体では、追尾 2000 m / 標的 20 m/s は閉鎖に 67 秒かかり、"
        "制限時間 60 秒の外に出る。ビームは旋回が遅れるとミスが開く。",
        "",
        "速度だけを上げると、旋回率は 1/v で落ちる。横加速度は g·tan(バンク) で、"
        "速度にはよらない。だから速度と旋回余裕は別のレバーとして振る。",
        "",
        "- 速度: 50 / 100 / 200 m/s",
        "- g6: いまの 80° バンク（約 5.7 g）。高速では旋回率が落ちる",
        "- g30: 横加速度 30 g。200 m/s でも旋回率が 50 m/s・g6 を下回らない",
        "",
        "推力は、指令速度で抗力に食われないところまで上げただけである。"
        "モーターの設計ではない。弾頭曲線は動かしていない。",
        "衝撃の期待殺傷は速度とともに衝突エネルギーも上がるので、"
        "幾何の結果はミスと時間切れで読む。",
        "",
        "## 設定",
        "",
        f"- 誘導則: Predictive、PN raw（N=4、フィルタなし）",
        f"- シナリオ: {', '.join(SCENARIOS)}",
        f"- 距離 {list(DISTANCES)} m × 標的速度 {list(TARGET_SPEEDS)} m/s",
        f"- センサ σ={SIGMA} m、風なし、n={N_TRIALS}",
        f"- 合計 {n} 交戦、{elapsed:.0f} s",
        "",
        "表はミス中央 (m) / 3 m 以内の割合 / 時間切れの割合。",
        "",
    ]
    for law, _ in LAWS:
        for scenario in SCENARIOS:
            lines.append(f"## {law} / {scenario}")
            lines.append("")
            lines.append("| 旋回 | 50 m/s | 100 m/s | 200 m/s |")
            lines.append("|---|---|---|---|")
            for turn, _ in TURNS:
                cells = []
                for speed in SPEEDS:
                    s = summ[law][turn][str(speed)][scenario]
                    cells.append(
                        f"{_fmt(s['median_miss'], 1)} / {s['p_inside3']:.0%} / {s['timeout']:.0%}"
                    )
                lines.append(f"| {turn} | " + " | ".join(cells) + " |")
            lines.append("")
    lines.append("## 追尾 2000 m / 標的 20 m/s")
    lines.append("")
    lines.append("| 誘導則 | 旋回 | 50 m/s | 100 m/s | 200 m/s |")
    lines.append("|---|---|---|---|---|")
    for law, _ in LAWS:
        for turn, _ in TURNS:
            cells = []
            for speed in SPEEDS:
                s = summ[law][turn][str(speed)]["tail2000_20"]
                cells.append(
                    f"t/o {s['timeout']:.0%}、miss {_fmt(s['median_miss'], 1)}"
                )
            lines.append(f"| {law} | {turn} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend(_read(summ))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 風は入れてない。横風で crab しない Predictive が死ぬ結果は、速度では消えない。",
        "- 30 g は旋回余裕の比較値である。機体構造の設計値ではない。",
        "- 推力は指令速度を維持するためだけに合わせた。",
        "- 衝撃 Pk は高速でエネルギーが閾値を超えるので、同じミスでも上がる。幾何は 3 m 以内の割合で読む。",
        "- センサは σ=0.5 m のまま。測距機は足していない。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/summary.json` / `{OUT}/trials.jsonl` / `{OUT}/miss.png`",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _read(summ):
    lines = ["## 読み", ""]
    tail = summ["Predictive"]["g6"]
    lines.append(
        "追尾 2000 m / 20 m/s、Predictive・g6 の時間切れは "
        + "、".join(
            f"{int(v)} m/s で {tail[str(v)]['tail2000_20']['timeout']:.0%}"
            for v in SPEEDS
        )
        + "。"
    )
    lines.append("")
    beam_pn = summ["PN raw"]
    lines.append(
        "ビームの PN・g6 のミス中央は "
        + "、".join(
            f"{int(v)} m/s で {_fmt(beam_pn['g6'][str(v)]['beam']['median_miss'], 1)} m"
            for v in SPEEDS
        )
        + "。g30 では "
        + "、".join(
            f"{int(v)} m/s で {_fmt(beam_pn['g30'][str(v)]['beam']['median_miss'], 1)} m"
            for v in SPEEDS
        )
        + "。"
    )
    lines.append("")
    lines.append(
        "時間切れが速度だけで消え、ビームのミスが g6 でも g30 でも開くなら、"
        "余った横加速度はビームでは使われていない。"
        "高速化は追尾の時間を買い、ビームのリード角を小さくする。"
    )
    return lines


def plot(summ):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for turn, style in (("g6", "-o"), ("g30", "--s")):
        ys = [summ["PN raw"][turn][str(v)]["beam"]["median_miss"] for v in SPEEDS]
        axes[0].plot(SPEEDS, ys, style, lw=2, label=f"PN {turn}")
        ys = [summ["Predictive"][turn][str(v)]["beam"]["median_miss"] for v in SPEEDS]
        axes[0].plot(SPEEDS, ys, style, lw=2, label=f"Pred {turn}")
    axes[0].axhline(3.0, color="black", ls=":", lw=1)
    axes[0].set_xlabel("speed (m/s)")
    axes[0].set_ylabel("median miss (m)")
    axes[0].set_title("beam")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    for turn, style in (("g6", "-o"), ("g30", "--s")):
        ys = [summ["Predictive"][turn][str(v)]["tail2000_20"]["timeout"] for v in SPEEDS]
        axes[1].plot(SPEEDS, ys, style, lw=2, label=f"Pred {turn}")
    axes[1].set_xlabel("speed (m/s)")
    axes[1].set_ylabel("timeout fraction")
    axes[1].set_title("tail-chase 2000 m / 20 m/s")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/miss.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Speed and turn-margin study")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    tasks = _tasks(args.trials, SEED_BASE)
    print(f"{len(tasks)} cells × {args.trials} trials  workers={args.workers}", flush=True)
    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_cell, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            records.extend(fut.result())
            done += 1
            if done % 40 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} cells  ({time.time() - t0:.0f} s)", flush=True)
    elapsed = time.time() - t0
    print(f"Total {elapsed:.1f} s, {len(records)} engagements", flush=True)
    with open(f"{OUT}/trials.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    summ = summarize(records)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"elapsed_s": elapsed, "n": len(records), "summary": summ}, f)
    plot(summ)
    text = generate_report(summ, len(records), elapsed)
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")
    cell = summ["Predictive"]["g6"]
    print("  tail 2000/20 g6",
          " ".join(f"{int(v)}={cell[str(v)]['tail2000_20']['timeout']:.0%}" for v in SPEEDS))


if __name__ == "__main__":
    main()
