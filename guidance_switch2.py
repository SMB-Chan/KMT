"""Target switching with both targets on a reachable collision course.

The edge run put the interceptor at y=40 on velocity (45, -22) — neither
target on a collision course — so every policy scored ~0 and the switching
question never loaded.

Here the interceptor starts on the primary's head-on collision course.
A decoy crosses 80 m to the side on a steady track.  The primary jinks.
If switching is useful, BestPk should abandon the jinking primary for the
steady decoy; Fixed / FirstLock should stay.  If the policies tie, the
answer is "switching does not matter in this geometry" and the guidance
story does not move either way.

Output: ``results/guidance_switch2/``.
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

OUT = "results/guidance_switch2"

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

# Steady crosswind is the condition that split the ranking before.
# Calm is the control: if switching helps anywhere it should show here too.
CONDITIONS = (
    ("calm",  0.0),
    ("cross10", 10.0),
)

DISTANCES = (500, 1000, 2000)
SPEEDS = (10, 15, 20)
N_TRIALS = 50
SEED_BASE = 17000
ALT = 50.0


def _law(law_id: str, wind_vec):
    if law_id == "pred":
        return PredictivePursuit()
    if law_id == "crab_pred_air":
        return CrabPredictive(wind=wind_vec, estimate_wind=True)
    if law_id == "crab_pred_or":
        return CrabPredictive(wind=wind_vec, estimate_wind=False)
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


def _collision_geometry(distance, target_speed, alt):
    """Interceptor on the primary's head-on collision course.

    Primary at (D, 0, alt) flying -x at ``target_speed``.  Interceptor at
    (0, 0, alt) flying +x at 50 m/s — that is the straight collision
    course.  Decoy crosses 80 m to the side at a steady velocity that
    would also allow an intercept if the interceptor turned.
    """
    primary_pos = np.array([distance, 0.0, alt])
    primary_vel = np.array([-target_speed, 0.0, 0.0])
    # Decoy: offset in y, closing in x, drifting toward the interceptor lane.
    decoy_pos = np.array([distance, 80.0, alt])
    decoy_vel = np.array([-target_speed * 0.7, -target_speed * 0.4, 0.0])
    ip = np.array([0.0, 0.0, alt])
    iv = np.array([50.0, 0.0, 0.0])
    return primary_pos, primary_vel, decoy_pos, decoy_vel, ip, iv


def _run_cell(task: dict) -> list[dict]:
    wind_v = task["wind_speed"]
    wind = WindField(mean=(0.0, wind_v, 0.0))
    wind_vec = (0.0, wind_v, 0.0)
    records = []
    for trial in range(task["n_trials"]):
        seed = stable_seed("sw2", task["law_id"], task["switch"],
                           task["distance"], task["speed"], trial,
                           base=task["seed_base"])
        pp, pv, dp, dv, ip, iv = _collision_geometry(
            task["distance"], task["speed"], ALT)
        # Primary jinks; decoy holds a straight track (the easy kill).
        primary = (pp, pv, evasive_heading_fn(seed=seed))
        decoy = (dp, dv, None)
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"], wind_vec),
            sensor=SensorNoise(pos_sigma=0.5, vel_sigma=0.2),
            wind=wind,
            wind_est_sigma=0.5,
            store_trajectory=False,
        )
        res = run_multi_target([primary, decoy], ip, iv, cfg,
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
            "law": task["law"], "law_id": task["law_id"],
            "switch": task["switch"],
            "cond": task["cond"], "wind_speed": wind_v,
            "distance": task["distance"], "speed": task["speed"],
            "trial": trial, "seed": seed,
            "switches": res["switches"],
            "final_target": engaged,
            "primary_miss": finals[0]["miss"],
            "decoy_miss": finals[1]["miss"],
            "primary_pk": finals[0]["pk"],
            "decoy_pk": finals[1]["pk"],
            "engaged_pk": finals[engaged]["pk"],
            "kill": bool(finals[engaged]["hit"]
                         and rng.random() < finals[engaged]["pk"]),
        })
    return records


def _tasks(n_trials: int, seed_base: int):
    tasks = []
    for law_name, law_id in LAWS:
        for sw_name, sw_id in SWITCHERS:
            for cond_name, wind_v in CONDITIONS:
                for dist in DISTANCES:
                    for spd in SPEEDS:
                        tasks.append({
                            "law": law_name, "law_id": law_id,
                            "switch": sw_id,
                            "cond": cond_name, "wind_speed": wind_v,
                            "distance": dist, "speed": spd,
                            "n_trials": n_trials,
                            "seed_base": seed_base,
                        })
    return tasks


def summarize(records):
    out = {}
    for law_name, _ in LAWS:
        out[law_name] = {}
        for cond_name, _ in CONDITIONS:
            out[law_name][cond_name] = {}
            for sw_name, sw_id in SWITCHERS:
                rows = [r for r in records
                        if r["law"] == law_name and r["cond"] == cond_name
                        and r["switch"] == sw_id]
                n = len(rows)
                pk = [r["engaged_pk"] for r in rows]
                p0 = [r["primary_pk"] for r in rows]
                d0 = [r["decoy_pk"] for r in rows]
                out[law_name][cond_name][sw_id] = {
                    "n": n,
                    "engaged_pk": float(np.mean(pk)) if pk else float("nan"),
                    "primary_pk": float(np.mean(p0)) if p0 else float("nan"),
                    "decoy_pk": float(np.mean(d0)) if d0 else float("nan"),
                    "median_primary_miss": float(np.median(
                        [r["primary_miss"] for r in rows])) if rows else float("nan"),
                    "median_engaged_miss": float(np.median(
                        [r["primary_miss"] if r["final_target"] == 0
                         else r["decoy_miss"] for r in rows])) if rows else float("nan"),
                    "switches": float(np.mean([r["switches"] for r in rows])) if rows else 0.0,
                    "final_primary": float(np.mean(
                        [1.0 if r["final_target"] == 0 else 0.0
                         for r in rows])) if rows else float("nan"),
                    "kill_rate": float(np.mean([1.0 if r["kill"] else 0.0
                                                for r in rows])) if rows else float("nan"),
                }
    return out


def generate_report(summ, n, elapsed) -> str:
    lines = [
        "# 衝突コースに載せた目標切替",
        "",
        "## 何を確かめたか",
        "",
        "前回（guidance_edge）の 2 目標は、迎撃機が y=40・速度 (45, −22) で"
        "どちらの衝突コースにも乗っておらず、全ポリシーの Pk が 0.03 以下になって"
        "切替の問いが成立しなかった。",
        "",
        "今回は迎撃機を主目標の正面衝突コースに載せる。主目標はジング、"
        "decoy は横 80 m を定常横断。切替が効くなら BestPk はジングする主目標を"
        "捨てて decoy に切り、Fixed / FirstLock は残るはずである。",
        "",
        "最初の 14400 交戦は高度保持がなく、重力で迎撃機が落ちて"
        "衝突コースでもミス中央が 35 m 前後だった。その表は切替を測っていない。"
        "空力推定だけミス中央 12–17 m だったのも同じ落下の副産物で、"
        "風補償の新結果ではない（本結果は guidance_windcomp）。"
        "本報告は、単目標と同じ高度保持を入れた再実行である。",
        "",
        "## 設定",
        "",
        f"- 誘導則: {', '.join(n for n, _ in LAWS)}",
        f"- 切替: {', '.join(n for n, _ in SWITCHERS)}（再判定 0.5 s）",
        f"- 条件: calm / cross10（定常、gust_rms=0）、σ=0.5",
        f"- 距離 {list(DISTANCES)} m × 速度 {list(SPEEDS)} m/s × n={N_TRIALS}",
        f"- 合計 {n} 交戦、{elapsed:.0f} s",
        "",
        "採点は「最終的に向かっていた目標」への衝撃 Pk。primary_pk / decoy_pk は"
        "切替の有無にかかわらず、各目標までの最接近で評価した参考値。",
        "",
    ]
    for cond_name, _ in CONDITIONS:
        lines.append(f"## 条件 {cond_name}")
        lines.append("")
        lines.append("| 誘導則 | 切替 | engaged Pk | primary Pk | decoy Pk | 主ミス中央 | 切替回数 | 最終が主 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for law_name, _ in LAWS:
            for sw_name, sw_id in SWITCHERS:
                s = summ[law_name][cond_name][sw_id]
                lines.append(
                    f"| {law_name} | {sw_name} | {s['engaged_pk']:.3f} | "
                    f"{s['primary_pk']:.3f} | {s['decoy_pk']:.3f} | "
                    f"{s['median_primary_miss']:.1f} m | "
                    f"{s['switches']:.1f} | {s['final_primary']:.0%} |"
                )
        lines.append("")

    lines.extend(_read(summ))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 主目標ジング・decoy 定常という非対称のみ。両方ジング、両方定常は見ていない。",
        "- BestPk の代理量は直線延長の近距離 b と相対速度。",
        "- decoy は 1 機。群接近では誘導の局所解が変わる。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/summary.json` / `{OUT}/trials.jsonl` / `{OUT}/pk.png`",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _read(summ):
    lines = ["## 読み", ""]
    for cond_name, _ in CONDITIONS:
        fixed = {}
        best = {}
        for law_name, _ in LAWS:
            fixed[law_name] = summ[law_name][cond_name]["fixed"]["engaged_pk"]
            best[law_name] = summ[law_name][cond_name]["bestpk"]["engaged_pk"]
        lines.append(f"**{cond_name}** — Fixed → BestPk の engaged Pk:")
        lines.append("")
        for law_name, _ in LAWS:
            d = best[law_name] - fixed[law_name]
            lines.append(
                f"- {law_name}: {fixed[law_name]:.3f} → {best[law_name]:.3f} ({d:+.3f})"
            )
        lines.append("")
        # Does BestPk beat Fixed by more than noise (~0.02)?
        alive = [n for n, _ in LAWS if fixed[n] >= 0.05]
        helped = [n for n in alive if best[n] - fixed[n] >= 0.02]
        hurt = [n for n in alive if fixed[n] - best[n] >= 0.02]
        dead = [n for n, _ in LAWS if fixed[n] < 0.05]
        if hurt and not helped:
            lines.append(
                "土俵に乗っている則では BestPk が Fixed を 0.02 以上下回る。"
                "この幾何では切ると下がる。"
            )
        elif helped and not hurt:
            lines.append(
                "BestPk が Fixed を 0.02 以上上回った則: "
                + ", ".join(helped) + "。この幾何では切替が効く。"
            )
        elif not alive:
            lines.append(
                "Fixed の engaged Pk が全則で 0.05 未満。"
                "切替の差を読む土俵に乗っていない。"
            )
        else:
            lines.append(
                "BestPk と Fixed の差は 0.02 以内の則が残る。"
                "切替で得をするとは読めない。"
            )
        if dead:
            lines.append(
                "Fixed が 0.05 未満で土俵外: " + ", ".join(dead) + "。"
            )
        lines.append("")
    # Always note the primary-decoy asymmetry
    s0 = summ["CrabPredictive airdata"]["calm"]
    lines.append(
        f"参考（Crab airdata / calm）: primary Pk {s0['fixed']['primary_pk']:.3f}、"
        f"decoy Pk {s0['fixed']['decoy_pk']:.3f}。"
        "主目標がジングし decoy が定常なので、理想の切替は decoy に付くことである。"
    )
    return lines


def plot(summ):
    laws = [n for n, _ in LAWS]
    sw_names = [n for n, _ in SWITCHERS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (cond_name, _) in zip(axes, CONDITIONS):
        x = np.arange(len(sw_names))
        width = 0.2
        for i, law in enumerate(laws):
            vals = [summ[law][cond_name][s[1]]["engaged_pk"]
                    for s in SWITCHERS]
            ax.bar(x + (i - 1.5) * width, vals, width, label=law)
        ax.set_xticks(x)
        ax.set_xticklabels(sw_names, rotation=20, ha="right")
        ax.set_title(cond_name)
        ax.set_ylabel("engaged-target kinetic Pk")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Collision-course target switching (σ = 0.5 m)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pk.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collision-course target switching")
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
        futs = [pool.submit(_run_cell, t) for t in tasks]
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
    summ = summarize(records)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"elapsed_s": elapsed, "n": len(records), "summary": summ},
                  f, indent=2)
    plot(summ)
    text = generate_report(summ, len(records), elapsed)
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")
    for law_name, _ in LAWS:
        for cond_name, _ in CONDITIONS:
            row = "  ".join(
                f"{s[0]}={summ[law_name][cond_name][s[1]]['engaged_pk']:.3f}"
                for s in SWITCHERS)
            print(f"  {cond_name:8} {law_name:24} {row}")


if __name__ == "__main__":
    main()
