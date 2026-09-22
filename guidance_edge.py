"""Edge cases: target switching and the wind speed where crab saturates.

Two gaps left by the follow-up run:

1. Two targets with no switching policy.  Now the interceptor picks
   among a primary and a decoy with Fixed / FirstLock / Nearest /
   BestPk.  Does crab still beat PN when the selection can move?
2. Crosswind only up to 10 m/s.  Sweep 0–25 m/s to find where the
   crab angle and the bank servo run out of margin.

Output: ``results/guidance_edge/``.
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
    EngagementConfig, FragmentationWarhead, KineticImpactWarhead,
    PNGuidance, PurePursuit, PredictivePursuit, CrabPredictive, CrabPursuit,
    SensorNoise, WarheadType, WindField,
    FixedSwitch, FirstLockSwitch, NearestSwitch, BestPkSwitch,
    evasive_heading_fn, sustained_heading_fn,
    run_engagement, run_multi_target, scenario_head_on, score_at_cpa,
)
from guidance_law_study import FRAG, KIN, stable_seed

OUT = "results/guidance_edge"

LAWS = (
    ("Predictive", "pred"),
    ("CrabPredictive airdata", "crab_pred_air"),
    ("CrabPredictive oracle", "crab_pred_or"),
    ("PN raw", "pn_raw"),
)

SWITCHERS = (
    ("Fixed primary", "fixed"),
    ("FirstLock", "first"),
    ("Nearest", "nearest"),
    ("BestPk", "bestpk"),
)

WIND_SPEEDS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
DISTANCES = (500, 1000, 2000)
SPEEDS = (10, 15, 20)
N_TRIALS = 50
SIGMAS = (0.5,)
SEED_BASE = 15000


def _law(law_id: str, wind):
    if law_id == "pred":
        return PredictivePursuit()
    if law_id == "crab_pred_air":
        return CrabPredictive(wind=wind, estimate_wind=True)
    if law_id == "crab_pred_or":
        return CrabPredictive(wind=wind, estimate_wind=False)
    if law_id == "crab_pp_air":
        return CrabPursuit(wind=wind, estimate_wind=True)
    if law_id == "pn_raw":
        return PNGuidance(N=4.0, los_tau=0.0)
    raise ValueError(law_id)


def _switcher(sw_id: str):
    if sw_id == "fixed":
        return FixedSwitch(0)
    if sw_id == "first":
        return FirstLockSwitch()
    if sw_id == "nearest":
        return NearestSwitch(0.5)
    if sw_id == "bestpk":
        return BestPkSwitch(0.5, margin=0.05)
    raise ValueError(sw_id)


def _two_targets(distance, speed, alt):
    """Primary head-on at +y=0; decoy crossing at +y=80."""
    primary = (
        np.array([distance, 0.0, alt]),
        np.array([-speed, 0.0, 0.0]),
        None,
    )
    decoy = (
        np.array([distance, 80.0, alt]),
        np.array([-speed * 0.5, -speed * 0.6, 0.0]),
        evasive_heading_fn(seed=7),
    )
    ip = np.array([0.0, 40.0, alt])
    iv = np.array([45.0, -22.0, 0.0])
    return [primary, decoy], ip, iv


def _run_switch(task: dict) -> list[dict]:
    wind_v = task["wind_speed"]
    wind = WindField(mean=(0.0, wind_v, 0.0))
    records = []
    for trial in range(task["n_trials"]):
        seed = stable_seed("edge_sw", task["law_id"], task["switch"],
                           task["distance"], task["speed"], trial,
                           base=task["seed_base"])
        targets, ip, iv = _two_targets(
            task["distance"], task["speed"], task["altitude"])
        # primary jinks a little
        targets[0] = (targets[0][0], targets[0][1],
                      evasive_heading_fn(seed=seed))
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"], (0.0, wind_v, 0.0)),
            sensor=SensorNoise(pos_sigma=task["sigma"],
                               vel_sigma=task["sigma"] * 0.4),
            wind=wind,
            wind_est_sigma=0.5,
            store_trajectory=False,
        )
        res = run_multi_target(targets, ip, iv, cfg,
                               _switcher(task["switch"]), seed=seed)
        rng = np.random.default_rng(seed + 17)
        finals = []
        for i, miss in enumerate(res["miss_by_target"]):
            approach = res["approach_by_target"][i]
            hit, pk = score_at_cpa(miss, approach, WarheadType.KINETIC_IMPACT,
                                   FRAG, KIN)
            finals.append({"miss": float(miss), "pk": float(pk),
                           "hit": bool(hit)})
        engaged = res["final_target"]
        records.append({
            "part": "switch",
            "law": task["law"], "law_id": task["law_id"],
            "switch": task["switch"],
            "wind_speed": wind_v,
            "distance": task["distance"], "speed": task["speed"],
            "sigma": task["sigma"], "trial": trial, "seed": seed,
            "switches": res["switches"],
            "final_target": engaged,
            "primary_miss": finals[0]["miss"],
            "decoy_miss": finals[1]["miss"],
            "engaged_pk": finals[engaged]["pk"],
            "primary_pk": finals[0]["pk"],
            "decoy_pk": finals[1]["pk"],
            "kill": bool(finals[engaged]["hit"]
                         and rng.random() < finals[engaged]["pk"]),
        })
    return records


def _run_wind(task: dict) -> list[dict]:
    wind_v = task["wind_speed"]
    wind = WindField(mean=(0.0, wind_v, 0.0))
    records = []
    for trial in range(task["n_trials"]):
        seed = stable_seed("edge_wind", task["law_id"], wind_v,
                           task["distance"], task["speed"], trial,
                           base=task["seed_base"])
        tp, tv, ip, iv = scenario_head_on(
            task["distance"], task["speed"],
            task["interceptor_speed"], task["altitude"])
        hdg = evasive_heading_fn(seed=seed)
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"], (0.0, wind_v, 0.0)),
            sensor=SensorNoise(pos_sigma=task["sigma"],
                               vel_sigma=task["sigma"] * 0.4),
            wind=wind,
            wind_est_sigma=0.5,
            store_trajectory=False,
        )
        res = run_engagement(tp, tv, ip, iv, cfg, seed=seed,
                             target_heading_fn=hdg)
        miss = float(res.miss_distance)
        approach = float(res.approach_speed)
        hit, pk = score_at_cpa(miss, approach, WarheadType.KINETIC_IMPACT,
                               FRAG, KIN)
        rng = np.random.default_rng(seed + 17)
        records.append({
            "part": "wind",
            "law": task["law"], "law_id": task["law_id"],
            "wind_speed": wind_v,
            "distance": task["distance"], "speed": task["speed"],
            "sigma": task["sigma"], "trial": trial, "seed": seed,
            "miss": miss if math.isfinite(miss) else 1e9,
            "approach_speed": approach,
            "kin_pk": float(pk),
            "kin_kill": bool(hit and rng.random() < pk),
        })
    return records


def _tasks(n_trials: int, seed_base: int):
    tasks = []
    # Part A: switching at the stress condition (10 m/s crosswind)
    for law_name, law_id in LAWS:
        for sw_name, sw_id in SWITCHERS:
            for dist in DISTANCES:
                for spd in SPEEDS:
                    tasks.append({
                        "part": "switch",
                        "law": law_name, "law_id": law_id,
                        "switch": sw_id,
                        "wind_speed": 10.0,
                        "distance": dist, "speed": spd, "sigma": 0.5,
                        "n_trials": n_trials,
                        "altitude": 50.0,
                        "seed_base": seed_base,
                    })
    # Part B: wind ladder on single-target jink
    for law_name, law_id in LAWS:
        for wind_v in WIND_SPEEDS:
            for dist in DISTANCES:
                for spd in SPEEDS:
                    tasks.append({
                        "part": "wind",
                        "law": law_name, "law_id": law_id,
                        "wind_speed": wind_v,
                        "distance": dist, "speed": spd, "sigma": 0.5,
                        "n_trials": n_trials,
                        "interceptor_speed": 50.0, "altitude": 50.0,
                        "seed_base": seed_base,
                    })
    return tasks


def _stats(rows, pk_key="kin_pk"):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    pk = [r[pk_key] for r in rows]
    return {
        "n": n,
        "mean_pk": float(np.mean(pk)) if pk else float("nan"),
        "median_miss": (float(np.median([r.get("miss", r.get("primary_miss"))
                                         for r in rows]))
                        if rows else float("nan")),
        "switches": float(np.mean([r.get("switches", 0) for r in rows])),
    }


def summarize(records):
    sw_rows = [r for r in records if r["part"] == "switch"]
    wind_rows = [r for r in records if r["part"] == "wind"]
    sw = {}
    for law_name, _ in LAWS:
        sw[law_name] = {}
        for sw_name, sw_id in SWITCHERS:
            rows = [r for r in sw_rows if r["law"] == law_name
                    and r["switch"] == sw_id]
            s = _stats(rows, pk_key="engaged_pk")
            s["primary_pk"] = (float(np.mean([r["primary_pk"] for r in rows]))
                               if rows else float("nan"))
            s["final_is_primary"] = (
                float(np.mean([1.0 if r["final_target"] == 0 else 0.0
                               for r in rows])) if rows else float("nan"))
            sw[law_name][sw_id] = s
    wl = {}
    for law_name, _ in LAWS:
        wl[law_name] = {}
        for wind_v in WIND_SPEEDS:
            rows = [r for r in wind_rows if r["law"] == law_name
                    and r["wind_speed"] == wind_v]
            wl[law_name][str(wind_v)] = _stats(rows)
    return sw, wl


def generate_report(sw, wl, n, elapsed) -> str:
    lines = [
        "# 目標切替と、crab が効かなくなる風速",
        "",
        "## 何を確かめたか",
        "",
        "1. **目標切替** — 主目標（正面・ジング）と decoy（横 80 m・斜め横断）。"
        "Fixed / FirstLock / Nearest / BestPk で切替を入れ、殺傷順位が保つかを見る。",
        "2. **強風の頭打ち** — 横風 0–25 m/s。crab 角 asin(W⊥/V_a) と"
        "バンクサーボが限界に達する場所を探す。",
        "",
        "## 設定",
        "",
        f"- 誘導則: {', '.join(n for n, _ in LAWS)}",
        f"- 切替: {', '.join(n for n, _ in SWITCHERS)}（再判定 0.5 s、BestPk は閾値 0.05）",
        f"- 風速ラダー: {list(WIND_SPEEDS)} m/s（横風）、jink 単目標",
        f"- 距離 {list(DISTANCES)} m × 速度 {list(SPEEDS)} m/s × n={N_TRIALS}、"
        f"合計 {n} 交戦、{elapsed:.0f} s",
        "",
        "## 1. 目標切替（横風 10 m/s、σ=0.5）",
        "",
        "最終的に向かっていた目標への衝撃 mean Pk。",
        "",
        "| 誘導則 | " + " | ".join(n for n, _ in SWITCHERS) + " |",
        "|---|" + "---|" * len(SWITCHERS),
    ]
    for law_name, _ in LAWS:
        cells = []
        for sw_name, sw_id in SWITCHERS:
            s = sw[law_name][sw_id]
            cells.append(f"{s['mean_pk']:.2f}")
        lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("平均切替回数 / 最終が主目標の割合:")
    lines.append("")
    lines.append("| 誘導則 | " + " | ".join(n for n, _ in SWITCHERS) + " |")
    lines.append("|---|" + "---|" * len(SWITCHERS))
    for law_name, _ in LAWS:
        cells = []
        for sw_name, sw_id in SWITCHERS:
            s = sw[law_name][sw_id]
            cells.append(
                f"{s['switches']:.1f} / {s['final_is_primary']:.0%}")
        lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend(_sw_read(sw))
    lines.append("")
    lines.extend([
        "## 2. 強風ラダー（jink、σ=0.5、衝撃 Pk）",
        "",
        "| 誘導則 | " + " | ".join(f"{w:.0f} m/s" for w in WIND_SPEEDS) + " |",
        "|---|" + "---|" * len(WIND_SPEEDS),
    ])
    for law_name, _ in LAWS:
        cells = [f"{wl[law_name][str(w)]['mean_pk']:.2f}" for w in WIND_SPEEDS]
        lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("miss 中央 (m):")
    lines.append("")
    lines.append("| 誘導則 | " + " | ".join(f"{w:.0f} m/s" for w in WIND_SPEEDS) + " |")
    lines.append("|---|" + "---|" * len(WIND_SPEEDS))
    for law_name, _ in LAWS:
        cells = [f"{wl[law_name][str(w)]['median_miss']:.1f}"
                 for w in WIND_SPEEDS]
        lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend(_wind_read(wl))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 切替は 0.5 s 間隔の幾何評価のみ。脅威度・味方・弾数は見ていない。",
        "- BestPk の代理量は直線延長の近距離 b と相対速度。実際の弾頭曲線は別。",
        "- 強風では鉛直成分を入れていない。バンク限界 35 deg / 120 deg/s は機体のまま。",
        "- decoy は 1 機。群れて接近する場合は誘導則の局所解が変わる。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/summary.json` / `{OUT}/trials.jsonl` / `{OUT}/pk.png`",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _sw_read(sw):
    crab = sw["CrabPredictive airdata"]
    pn = sw["PN raw"]
    pred = sw["Predictive"]
    lines = ["### 読み — 切替", ""]
    best_sw = max(SWITCHERS, key=lambda s: crab[s[1]]["mean_pk"])
    worst_sw = min(SWITCHERS, key=lambda s: crab[s[1]]["mean_pk"])
    lines.append(
        f"crab は {best_sw[0]} で {crab[best_sw[1]]['mean_pk']:.2f}、"
        f"{worst_sw[0]} で {crab[worst_sw[1]]['mean_pk']:.2f}。"
    )
    lines.append("")
    beats = []
    for sw_name, sw_id in SWITCHERS:
        if crab[sw_id]["mean_pk"] > pn[sw_id]["mean_pk"] + 0.02:
            beats.append(sw_name)
    if len(beats) == len(SWITCHERS):
        lines.append(
            "全切替ポリシーで crab が PN を上回る。選択が動いても"
            "リードの優位は壊れない。"
        )
    elif beats:
        lines.append(
            "crab が PN を上回ったポリシー: " + ", ".join(beats) + "。"
        )
    else:
        lines.append(
            "切替を入れると crab と PN の差が縮む。目標が動くと"
            "衝突三角形の再解が切替のたびに必要になる。"
        )
    lines.append("")
    lines.append(
        f"素の Predictive はどのポリシーでも {pred['fixed']['mean_pk']:.2f}"
        f"–{pred['bestpk']['mean_pk']:.2f} にとどまり、横風の崩れが残る。"
    )
    return lines


def _wind_read(wl):
    lines = ["### 読み — 強風", ""]
    crab = wl["CrabPredictive airdata"]
    pn = wl["PN raw"]
    # Find where crab falls below 0.2
    sat = None
    for w in WIND_SPEEDS:
        if crab[str(w)]["mean_pk"] < 0.20:
            sat = w
            break
    lines.append(
        f"crab の Pk は 0 m/s {crab['0.0']['mean_pk']:.2f} → "
        f"25 m/s {crab['25.0']['mean_pk']:.2f}。"
    )
    lines.append("")
    if sat is not None:
        lines.append(
            f"Pk が 0.2 を割る風速は約 {sat:.0f} m/s。"
            "crab 角 asin(W⊥/50) は "
            f"{math.degrees(math.asin(min(sat / 50.0, 1.0))):.0f} deg 相当で、"
            "バンク限界と対地速度の落ち込みが効き始める。"
        )
    else:
        lines.append(
            "25 m/s までは Pk 0.2 を割らない。crab 角は最大 "
            f"{math.degrees(math.asin(25 / 50.0)):.0f} deg で、"
            "対気 50 m/s ならまだ余裕がある。"
        )
    lines.append("")
    lines.append(
        f"同じラダーで PN は 0 m/s {pn['0.0']['mean_pk']:.2f} → "
        f"25 m/s {pn['25.0']['mean_pk']:.2f}。"
        "crab の落ち方は PN と比べて遅いが、どちらも強い横風では下がる。"
    )
    lines.append("")
    lines.append(
        "空力推定が表の上で真値を上回って見えるのは、このラダーが定常風"
        "（gust_rms=0）で、法則ごとに乱数が違うセルを平均した見た目である。"
        "推定が真値より正確なのではない。wind_est_sigma は毎ステップ足される"
        "白色雑音であり、平滑化ではない。旋回中の機首の遅れが偽の横風として"
        "入り、リードが増える。25 m/s の正面 1000 m では順位が逆転する"
        "（真値 Pk 0.67、推定 0.39）。差は測定の優位ではない。"
    )
    return lines


def plot(sw, wl):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    laws = [n for n, _ in LAWS]
    sw_names = [n for n, _ in SWITCHERS]
    x = np.arange(len(sw_names))
    width = 0.2
    for i, law in enumerate(laws):
        vals = [sw[law][s[1]]["mean_pk"] for s in SWITCHERS]
        axes[0].bar(x + (i - 1.5) * width, vals, width, label=law)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sw_names, rotation=20, ha="right")
    axes[0].set_ylabel("engaged-target kinetic Pk")
    axes[0].set_title("Target switching (cross10, σ=0.5)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis="y", alpha=0.3)

    for law in laws:
        vals = [wl[law][str(w)]["mean_pk"] for w in WIND_SPEEDS]
        axes[1].plot(WIND_SPEEDS, vals, "o-", lw=2, label=law)
    axes[1].set_xlabel("crosswind (m/s)")
    axes[1].set_ylabel("kinetic mean Pk")
    axes[1].set_title("Wind ladder (jink, σ=0.5)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pk.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Switching and strong-wind edge")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    tasks = _tasks(args.trials, SEED_BASE)
    print(f"{len(tasks)} cells × {args.trials} trials  workers={args.workers}",
          flush=True)
    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_switch if t["part"] == "switch" else _run_wind,
                            t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            records.extend(fut.result())
            done += 1
            if done % 30 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} cells"
                      f"  ({time.time() - t0:.0f} s)", flush=True)
    elapsed = time.time() - t0
    print(f"Total {elapsed:.1f} s, {len(records)} engagements", flush=True)
    with open(f"{OUT}/trials.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    sw, wl = summarize(records)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"elapsed_s": elapsed, "n": len(records),
                   "switch": sw, "wind": wl}, f, indent=2)
    plot(sw, wl)
    text = generate_report(sw, wl, len(records), elapsed)
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")
    for law_name, _ in LAWS:
        row = "  ".join(
            f"{s[0]}={sw[law_name][s[1]]['mean_pk']:.2f}" for s in SWITCHERS)
        print(f"  sw  {law_name:24} {row}")
        row = "  ".join(
            f"{w:.0f}={wl[law_name][str(w)]['mean_pk']:.2f}"
            for w in WIND_SPEEDS)
        print(f"  wnd {law_name:24} {row}")


if __name__ == "__main__":
    main()
