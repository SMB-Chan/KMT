"""Third craft: a ranger that holds course and does not kill.

The two warhead types already span the miss-to-kill map.  What the
tables actually moved was the observation.  This study puts a
non-expending sensor craft on a held ground track and fuses its
range/bearing fix into the interceptor's own GPS-grade target fix.

Modes
    alone            interceptor σ = 0.5 m, no craft
    oracle           interceptor σ = 0.1 m, no craft (ceiling, no geometry)
    along50          craft 50 m off the interceptor's track, same ground velocity
    along200         same, 200 m off
    along200_d150    along200, ranging delayed 150 ms with the rest of the fix
    station200       craft fixed at the lane midpoint, 200 m off the target track

Laws are the closed pair: Predictive, and crab with the true wind
(not the air-data estimate).  The craft is not scored.

Output: ``results/guidance_sensor_craft/``.
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

from interceptor import (
    CrabPredictive, EngagementConfig, PNGuidance, PredictivePursuit,
    SensorCraft, SensorNoise, WarheadType, WindField,
    evasive_heading_fn, run_engagement, scenario_beam, scenario_head_on,
    score_at_cpa,
)
from guidance_law_study import FRAG, KIN, stable_seed

OUT = "results/guidance_sensor_craft"

LAWS = (
    ("Predictive", "pred"),
    ("Crab oracle", "crab_or"),
)
MODES = (
    "alone",
    "oracle",
    "along50",
    "along200",
    "along200_d150",
    "station200",
)
SCENARIOS = ("head_on", "beam", "evasive")
WINDS = (0.0, 10.0)
DISTANCES = (500, 1000, 2000)
SPEED = 15.0
N_TRIALS = 40
SEED_BASE = 18000
ALT = 50.0
INTR_SPEED = 50.0


def _law(law_id: str, wind_v: float):
    if law_id == "pred":
        return PredictivePursuit()
    if law_id == "crab_or":
        return CrabPredictive(wind=(0.0, wind_v, 0.0), estimate_wind=False)
    raise ValueError(law_id)


def _geometry(name, distance, speed):
    if name == "beam":
        return scenario_beam(distance, speed, INTR_SPEED, ALT)
    return scenario_head_on(distance, speed, INTR_SPEED, ALT)


def _craft(mode: str, ip, iv, tp, tv) -> SensorCraft | None:
    if mode in ("alone", "oracle"):
        return None
    speed = max(math.hypot(float(iv[0]), float(iv[1])), 1.0)
    # Left of the interceptor's ground track.
    nx, ny = -float(iv[1]) / speed, float(iv[0]) / speed
    if mode.startswith("along"):
        offset = 50.0 if mode == "along50" else 200.0
        pos = np.asarray(ip, dtype=float) + offset * np.array([nx, ny, 0.0])
        return SensorCraft(tuple(pos), tuple(np.asarray(iv, dtype=float)))
    # Station: midpoint of the initial lane, 200 m off the target track.
    tspeed = max(math.hypot(float(tv[0]), float(tv[1])), 1.0)
    tx, ty = -float(tv[1]) / tspeed, float(tv[0]) / tspeed
    mid = 0.5 * (np.asarray(ip, dtype=float) + np.asarray(tp, dtype=float))
    pos = mid + 200.0 * np.array([tx, ty, 0.0])
    return SensorCraft(tuple(float(x) for x in pos), (0.0, 0.0, 0.0))


def _run_cell(task: dict) -> list[dict]:
    records = []
    wind_v = task["wind"]
    for trial in range(task["n_trials"]):
        # Mode and law stay out of the seed so the jink and the
        # own-ship noise are the same fix the craft is fused onto.
        seed = stable_seed("ranger", task["scenario"], task["distance"],
                           task["speed"], wind_v, trial, base=task["seed_base"])
        tp, tv, ip, iv = _geometry(task["scenario"], task["distance"], task["speed"])
        mode = task["mode"]
        sigma = 0.1 if mode == "oracle" else 0.5
        delay = 0.15 if mode.endswith("d150") else 0.0
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"], wind_v),
            sensor=SensorNoise(sigma, sigma * 0.4),
            wind=WindField(mean=(0.0, wind_v, 0.0)),
            sensor_craft=_craft(mode, ip, iv, tp, tv),
            comm_delay=delay,
            store_trajectory=False,
        )
        hdg = evasive_heading_fn(seed=seed) if task["scenario"] == "evasive" else None
        res = run_engagement(tp, tv, ip, iv, cfg, seed=seed, target_heading_fn=hdg)
        _, kin_pk = score_at_cpa(res.miss_distance, res.approach_speed,
                                 WarheadType.KINETIC_IMPACT, FRAG, KIN)
        _, frag_pk = score_at_cpa(res.miss_distance, res.approach_speed,
                                  WarheadType.FRAGMENTATION, FRAG, KIN)
        records.append({
            "law": task["law"], "mode": mode,
            "scenario": task["scenario"], "wind": wind_v,
            "distance": task["distance"], "speed": task["speed"],
            "trial": trial,
            "miss": float(res.miss_distance),
            "kin_pk": float(kin_pk),
            "frag_pk": float(frag_pk),
        })
    return records


def _tasks(n_trials: int, seed_base: int):
    tasks = []
    for law, law_id in LAWS:
        for mode in MODES:
            for scenario in SCENARIOS:
                for wind in WINDS:
                    for dist in DISTANCES:
                        tasks.append({
                            "law": law, "law_id": law_id, "mode": mode,
                            "scenario": scenario, "wind": wind,
                            "distance": dist, "speed": SPEED,
                            "n_trials": n_trials, "seed_base": seed_base,
                        })
    return tasks


def summarize(records):
    out = {}
    for law, _ in LAWS:
        out[law] = {}
        for scenario in SCENARIOS:
            out[law][scenario] = {}
            for wind in WINDS:
                out[law][scenario][str(wind)] = {}
                for mode in MODES:
                    rows = [r for r in records
                            if r["law"] == law and r["scenario"] == scenario
                            and r["wind"] == wind and r["mode"] == mode]
                    misses = [r["miss"] for r in rows if r["miss"] < 500]
                    out[law][scenario][str(wind)][mode] = {
                        "n": len(rows),
                        "kin_pk": float(np.mean([r["kin_pk"] for r in rows])) if rows else float("nan"),
                        "frag_pk": float(np.mean([r["frag_pk"] for r in rows])) if rows else float("nan"),
                        "median_miss": float(np.median(misses)) if misses else float("nan"),
                    }
    return out


def _fmt(x, nd=2):
    if x != x:
        return "—"
    return f"{x:.{nd}f}"


def generate_report(summ, n, elapsed) -> str:
    lines = [
        "# 測距機 — 殺傷しない第三機",
        "",
        "## 何を確かめたか",
        "",
        "衝撃型と炸裂型は、最接近距離を半径 3 m か 6 m の曲線に通すかどうかで分かれている。"
        "平均を一番動かしたのは誘導則ではなく観測だった。"
        "第三機は自分では誘導も殺傷もせず、与えられた対地航跡を保って距離と方位を渡す。",
        "",
        "比べるのは次の 6 つである。",
        "",
        "- alone: 迎撃機自身の σ=0.5 m",
        "- oracle: 迎撃機自身の σ=0.1 m。幾何のない上限",
        "- along50 / along200: 迎撃機の航跡から 50 m / 200 m 横、同じ対地速度で直進",
        "- along200_d150: along200 の測距を、他の観測と一緒に 150 ms 遅らせる",
        "- station200: 初期レーンの中点から、標的航跡の横 200 m に停止",
        "",
        "渡すのは相対ベクトルである。測距機の GPS 座標に足した世界位置は、"
        "位置誤差 0.5 m が下限になって迎撃機自身の観測を越えられない。"
        "基線の誤差は link σ=0.2 m。方位は 1 mrad × 距離、距離方向は 1 m。"
        "速度は迎撃機自身の観測のまま。"
        "誘導則は閉じた組だけを使う。Predictive と、真値の風を使う crab。"
        "空力推定は入れない。",
        "",
        "## 設定",
        "",
        f"- シナリオ: {', '.join(SCENARIOS)}",
        f"- 風: {list(WINDS)} m/s（定常、gust なし）。測距機は対地航跡を保ち、風では流されない",
        f"- 距離 {list(DISTANCES)} m、標的 {SPEED:.0f} m/s、迎撃 {INTR_SPEED:.0f} m/s、n={N_TRIALS}",
        f"- 合計 {n} 交戦、{elapsed:.0f} s",
        "",
        "数値は衝撃 mean Pk / ミス中央 (m)。",
        "",
    ]
    for law, _ in LAWS:
        for scenario in SCENARIOS:
            lines.append(f"## {law} / {scenario}")
            lines.append("")
            header = "| 風 | " + " | ".join(MODES) + " |"
            lines.append(header)
            lines.append("|---|" + "---|" * len(MODES))
            for wind in WINDS:
                cells = []
                for mode in MODES:
                    s = summ[law][scenario][str(wind)][mode]
                    cells.append(f"{s['kin_pk']:.2f} / {_fmt(s['median_miss'], 1)}")
                lines.append(f"| {wind:.0f} m/s | " + " | ".join(cells) + " |")
            lines.append("")
    lines.extend(_read(summ))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 方位 1 mrad・距離 1 m は、この比較のための測距モデルである。機材の仕様ではない。",
        "- 速度は融合しない。測距機が渡すのは位置だけである。",
        "- 測距機は対地航跡を保つ。風の中で crab する第二の誘導問題は解いていない。",
        "- 測距機自身は採点しない。along50 のオフセットは両弾頭の半径の外である。",
        "- 弾頭曲線は動かしていない。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/summary.json` / `{OUT}/trials.jsonl` / `{OUT}/pk.png`",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _read(summ):
    """Comparisons that are just differences of the table."""
    lines = ["## 読み", ""]
    # Head-on calm, Predictive: does a held craft move Pk toward the oracle?
    base = summ["Predictive"]["head_on"]["0.0"]
    lines.append(
        "平静・正面・Predictive の衝撃 Pk は "
        f"alone {_fmt(base['alone']['kin_pk'])}、"
        f"along50 {_fmt(base['along50']['kin_pk'])}、"
        f"along200 {_fmt(base['along200']['kin_pk'])}、"
        f"station200 {_fmt(base['station200']['kin_pk'])}、"
        f"oracle {_fmt(base['oracle']['kin_pk'])}。"
        f"ミス中央は alone {_fmt(base['alone']['median_miss'], 1)} m、"
        f"along50 {_fmt(base['along50']['median_miss'], 1)} m、"
        f"oracle {_fmt(base['oracle']['median_miss'], 1)} m。"
    )
    lines.append("")
    wind = summ["Predictive"]["head_on"]["10.0"]
    crab = summ["Crab oracle"]["head_on"]["10.0"]
    lines.append(
        "横風 10 m/s・正面では、Predictive の alone が "
        f"{_fmt(wind['alone']['kin_pk'])}、along50 が {_fmt(wind['along50']['kin_pk'])}、"
        f"oracle が {_fmt(wind['oracle']['kin_pk'])}。"
        "同じ風の crab 真値は "
        f"alone {_fmt(crab['alone']['kin_pk'])}、"
        f"along50 {_fmt(crab['along50']['kin_pk'])}、"
        f"oracle {_fmt(crab['oracle']['kin_pk'])}。"
    )
    lines.append("")
    beam = summ["Crab oracle"]["beam"]["0.0"]
    lines.append(
        "ビーム・平静・crab 真値のミス中央は "
        f"alone {_fmt(beam['alone']['median_miss'], 1)} m、"
        f"along50 {_fmt(beam['along50']['median_miss'], 1)} m、"
        f"along200 {_fmt(beam['along200']['median_miss'], 1)} m、"
        f"oracle {_fmt(beam['oracle']['median_miss'], 1)} m。"
        f"衝撃 Pk は alone {_fmt(beam['alone']['kin_pk'])}、"
        f"along50 {_fmt(beam['along50']['kin_pk'])}、"
        f"oracle {_fmt(beam['oracle']['kin_pk'])}。"
    )
    lines.append("")
    dly = summ["Crab oracle"]["head_on"]["10.0"]
    lines.append(
        "along200 に 150 ms を足すと、横風・正面・crab 真値の衝撃 Pk は "
        f"{_fmt(dly['along200']['kin_pk'])} から {_fmt(dly['along200_d150']['kin_pk'])} になる。"
    )
    lines.append("")
    lines.append(
        "航跡を保って位置だけを渡す測距機が alone と同じなら、"
        "その第三機は観測の上限を買っていない。"
        "oracle の行だけが動くなら、動いたのは迎撃機自身の位置と速度である。"
        "横風で Predictive が oracle でも 0 近辺なら、測距は crab の代わりにならない。"
    )
    return lines


def plot(summ):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    modes = ["alone", "station200", "along200", "along50", "oracle"]
    labels = ["alone", "station", "along200", "along50", "oracle"]
    for ax, wind, title in zip(axes, (0.0, 10.0), ("calm", "cross 10 m/s")):
        x = np.arange(len(modes))
        for law, marker in (("Predictive", "o"), ("Crab oracle", "s")):
            vals = [summ[law]["head_on"][str(wind)][m]["kin_pk"] for m in modes]
            ax.plot(x, vals, marker=marker, lw=2, label=law)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"head-on, {title}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("kinetic mean Pk")
    axes[1].legend(fontsize=8)
    fig.suptitle("Ranger on a held track (not a third warhead)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pk.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Non-expending ranger craft")
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
    base = summ["Predictive"]["head_on"]["0.0"]
    print("  calm head-on Predictive",
          " ".join(f"{m}={base[m]['kin_pk']:.2f}" for m in MODES))


if __name__ == "__main__":
    main()
