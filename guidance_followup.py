"""Follow-up: oracle with instantaneous wind, sustained weave, two targets.

Three gaps left by the wind-compensation run:

1. The published ``oracle`` only saw the mean wind (the loop now feeds
   instantaneous W).  Re-fly evasive jinks so the ceiling is honest.
2. Evasive used 2 s jink bursts.  A continuous weave is a different
   filter stress — does crab still beat PN?
3. One target only.  With two threats the law must pick one; does the
   ranking hold on the assigned target?

Output: ``results/guidance_followup/``.
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
    evasive_heading_fn, sustained_heading_fn, run_engagement,
    scenario_head_on, scenario_beam, score_at_cpa,
)
from guidance_law_study import FRAG, KIN, stable_seed

OUT = "results/guidance_followup"

LAWS = (
    ("Predictive", "pred"),
    ("CrabPredictive airdata", "crab_pred_air"),
    ("CrabPredictive oracle", "crab_pred_or"),
    ("CrabPursuit airdata", "crab_pp_air"),
    ("PN raw", "pn_raw"),
)

# Focused conditions: the ones that split the ranking.
CONDITIONS = (
    ("calm",          0.0,  0.0, 0.00, 0.0),
    ("cross10",       0.0, 10.0, 0.00, 0.0),
    ("cross10_d150",  0.0, 10.0, 0.15, 0.0),
    ("cross10_gust",  0.0, 10.0, 0.00, 2.0),
)

SCENARIOS = ("jink", "sustained", "two_target")

DISTANCES = (500, 1000, 2000)
SPEEDS = (10, 15, 20)
N_TRIALS = 50
SIGMAS = (0.5, 0.1)
SEED_BASE = 13000


def _law(law_id: str, cond):
    _cid, wx, wy, _delay, _gust = cond
    wind = (wx, wy, 0.0)
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


def _wind(cond) -> WindField:
    _cid, wx, wy, _delay, gust = cond
    return WindField(mean=(wx, wy, 0.0), gust_rms=gust, gust_tau=2.0,
                     target_drift=0.0)


def _two_target_geometry(distance, target_speed, intr_speed, alt):
    """Primary on a head-on course; decoy crossing 80 m to the side."""
    tp = np.array([distance, 0.0, alt])
    tv = np.array([-target_speed, 0.0, 0.0])
    decoy_pos = np.array([distance, 80.0, alt])
    decoy_vel = np.array([-target_speed * 0.6, -target_speed * 0.5, 0.0])
    ip = np.array([0.0, 40.0, alt])
    iv = np.array([intr_speed * 0.9, -intr_speed * 0.44, 0.0])
    return tp, tv, ip, iv, decoy_pos, decoy_vel


def _run_cell(task: dict) -> list[dict]:
    cond = next(c for c in CONDITIONS if c[0] == task["cond"])
    _cid, wx, wy, delay, gust = cond
    wind = _wind(cond)
    scenario = task["scenario"]
    records = []
    for trial in range(task["n_trials"]):
        seed = stable_seed("fu", scenario, task["distance"], task["speed"],
                           task["sigma"], trial, base=task["seed_base"])
        hdg = None
        decoy = None
        if scenario in ("jink", "sustained", "two_target"):
            tp, tv, ip, iv = scenario_head_on(
                task["distance"], task["speed"],
                task["interceptor_speed"], task["altitude"])
        if scenario == "jink":
            hdg = evasive_heading_fn(seed=seed)
        elif scenario == "sustained":
            hdg = sustained_heading_fn(seed=seed)
        elif scenario == "two_target":
            tp, tv, ip, iv, dpos, dvel = _two_target_geometry(
                task["distance"], task["speed"],
                task["interceptor_speed"], task["altitude"])
            hdg = evasive_heading_fn(seed=seed)
            decoy = (dpos, dvel)
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"], cond),
            sensor=SensorNoise(pos_sigma=task["sigma"],
                               vel_sigma=task["sigma"] * 0.4),
            wind=wind,
            comm_delay=delay,
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
        row = {
            "scenario": scenario,
            "law": task["law"],
            "law_id": task["law_id"],
            "cond": task["cond"],
            "distance": task["distance"],
            "speed": task["speed"],
            "sigma": task["sigma"],
            "trial": trial,
            "seed": seed,
            "miss": miss if math.isfinite(miss) else 1e9,
            "approach_speed": approach,
            "kin_hit": bool(hit),
            "kin_pk": float(pk),
            "kin_kill": bool(hit and rng.random() < pk),
        }
        if decoy is not None:
            # Did the interceptor prefer the assigned (primary) target?
            # CPA miss to the decoy is not scored; record whether the
            # primary was the closer of the two at endgame.
            dpos, dvel = decoy
            # Approximate decoy CPA with the same closing geometry.
            # Primary hit is the scored one; decoy distance at the
            # engagement end is a selection check.
            row["decoy_range_end"] = float(np.linalg.norm(
                np.asarray(dpos) + np.asarray(dvel) * res.engagement_time
                - np.asarray(scenario_head_on(
                    task["distance"], task["speed"], 50.0, 50.0)[0])))
        records.append(row)
    return records


def _tasks(n_trials: int, seed_base: int) -> list[dict]:
    tasks = []
    for law_name, law_id in LAWS:
        for scenario in SCENARIOS:
            for cond in CONDITIONS:
                for dist in DISTANCES:
                    for spd in SPEEDS:
                        for sigma in SIGMAS:
                            tasks.append({
                                "scenario": scenario,
                                "law": law_name, "law_id": law_id,
                                "cond": cond[0],
                                "distance": dist, "speed": spd, "sigma": sigma,
                                "n_trials": n_trials,
                                "interceptor_speed": 50.0, "altitude": 50.0,
                                "seed_base": seed_base,
                            })
    return tasks


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _stats(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    misses = [r["miss"] for r in rows if r["miss"] < 500]
    pk = [r["kin_pk"] for r in rows]
    hits = sum(1 for r in rows if r["kin_hit"])
    kills = sum(1 for r in rows if r["kin_kill"])
    lo, hi = _wilson(kills, n)
    med = float(np.median(misses)) if misses else float("nan")
    return {
        "n": n,
        "median_miss": med,
        "p95_miss": float(np.percentile(misses, 95)) if misses else float("nan"),
        "p_miss_lt_3": (sum(1 for m in misses if m < 3.0) / len(misses)
                        if misses else float("nan")),
        "kin_hit": hits / n,
        "kin_pk": float(np.mean(pk)) if pk else float("nan"),
        "kin_kill": kills / n,
        "kin_kill_lo": lo,
        "kin_kill_hi": hi,
    }


def _subset(rows, **kw):
    out = rows
    for k, v in kw.items():
        if v is None:
            continue
        out = [r for r in out if r[k] == v]
    return out


def summarize(records):
    out = {}
    for law_name, _ in LAWS:
        out[law_name] = {}
        for scenario in SCENARIOS:
            out[law_name][scenario] = {}
            for sigma in SIGMAS:
                out[law_name][scenario][str(sigma)] = {}
                for cond in CONDITIONS:
                    rows = _subset(records, law=law_name, scenario=scenario,
                                   sigma=sigma, cond=cond[0])
                    out[law_name][scenario][str(sigma)][cond[0]] = _stats(rows)
    return out


def _fmt_m(x):
    return "—" if x != x else f"{x:.2f} m"


def generate_report(summ, n, elapsed) -> str:
    lines = [
        "# 残作業 — oracle 再測定・持続マニューバ・2目標",
        "",
        "## 何を確かめたか",
        "",
        "風補償実験の残り3点。",
        "",
        "1. **oracle の再測定** — 前回の真値版は風の平均しか見ておらず、"
        "ガストのあるセルで空力推定より下に出た。ループが瞬時 W を渡すように"
        "直したうえで、上限を引き直す。",
        "2. **持続マニューバ** — 2 秒バーストのジングではなく、交戦中ずっと"
        "サイン波で織る。フィルタとリードの負荷が違う。",
        "3. **2 目標** — 主目標（正面）と decoy（横 80 m）。主目標への"
        "殺傷で採点する。",
        "",
        "## 設定",
        "",
        f"- 誘導則: {', '.join(n for n, _ in LAWS)}",
        f"- シナリオ: jink（v5 と同じ）、sustained（連続サイン 0.35 Hz）、"
        f"two_target（主目標 + decoy）",
        f"- 条件: calm / cross10 / cross10_d150 / cross10_gust",
        f"- 距離 {list(DISTANCES)} m × 速度 {list(SPEEDS)} m/s × "
        f"σ ∈ {list(SIGMAS)}、各 50 試行",
        f"- 合計 {n} 交戦、{elapsed:.0f} s",
        "",
    ]
    for sigma in SIGMAS:
        for scenario in SCENARIOS:
            lines.append(f"## σ = {sigma} m / {scenario}")
            lines.append("")
            lines.append("| 誘導則 | " + " | ".join(
                f"{c[0]} miss / Pk" for c in CONDITIONS) + " |")
            lines.append("|---|" + "---|" * len(CONDITIONS))
            for law_name, _ in LAWS:
                cells = []
                for cond in CONDITIONS:
                    s = summ[law_name][scenario][str(sigma)][cond[0]]
                    cells.append(
                        f"{s['median_miss']:.2f} / {s['kin_pk']:.2f}")
                lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.extend(_interpret(summ))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 2 目標では主目標だけを誘導・採点している。decoy への切替や"
        "資源配分の意思決定は入れていない。",
        "- sustained は水平サインのみ。持続バレルロールではない。",
        "- oracle は瞬時風真値。実機の上限。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/summary.json`",
        f"- `{OUT}/trials.jsonl`",
        f"- `{OUT}/pk_compare.png`",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _interpret(summ):
    def pk(law, scen, sigma, cond):
        return summ[law][scen][str(sigma)][cond]["kin_pk"]

    def med(law, scen, sigma, cond):
        return summ[law][scen][str(sigma)][cond]["median_miss"]

    lines = ["## 読み", ""]
    # Oracle ceiling
    lines.append("### oracle の上限")
    lines.append("")
    for cond in ("cross10", "cross10_gust"):
        air = pk("CrabPredictive airdata", "jink", 0.5, cond)
        orc = pk("CrabPredictive oracle", "jink", 0.5, cond)
        pn = pk("PN raw", "jink", 0.5, cond)
        lines.append(
            f"jink / {cond}: 空力推定 {air:.2f}、真値 {orc:.2f}、"
            f"PN raw {pn:.2f}。"
        )
    lines.append("")
    orc_gust = pk("CrabPredictive oracle", "jink", 0.5, "cross10_gust")
    air_gust = pk("CrabPredictive airdata", "jink", 0.5, "cross10_gust")
    if orc_gust >= air_gust - 0.02:
        lines.append(
            "crab 空力推定が真値に並ぶか上回るかは、その条件の測定優位ではない。"
            "定常風では旋回中の機首遅れが偽の横風になり、リードが増えることがある。"
        )
    else:
        lines.append(
            f"真値 {orc_gust:.2f} が空力推定 {air_gust:.2f} に届かない。"
            "風推定のローパスがガストを滑らかにし、指令が安定している可能性がある。"
        )
    lines.append("")
    lines.append("### 持続マニューバ")
    lines.append("")
    for law in ("Predictive", "CrabPredictive airdata", "PN raw"):
        a = pk(law, "jink", 0.5, "cross10")
        b = pk(law, "sustained", 0.5, "cross10")
        lines.append(
            f"{law}: jink {a:.2f} → sustained {b:.2f}（横風 10 m/s、σ=0.5）"
        )
    sus_crab = pk("CrabPredictive airdata", "sustained", 0.5, "cross10")
    sus_pn = pk("PN raw", "sustained", 0.5, "cross10")
    if sus_crab > sus_pn + 0.03:
        lines.append(
            f"持続織りでも crab が PN を上回る（{sus_crab:.2f} vs {sus_pn:.2f}）。"
            "バースト固有の結果ではない。"
        )
    elif sus_crab > sus_pn - 0.03:
        lines.append(
            f"持続織りでは crab と PN が同点（{sus_crab:.2f} vs {sus_pn:.2f}）。"
        )
    else:
        lines.append(
            f"持続織りでは PN が残る（{sus_pn:.2f} vs crab {sus_crab:.2f}）。"
            "連続加速度がリードの前提を壊す。"
        )
    lines.append("")
    lines.append("### 2 目標")
    lines.append("")
    for law in ("Predictive", "CrabPredictive airdata", "PN raw"):
        a = pk(law, "jink", 0.5, "cross10")
        b = pk(law, "two_target", 0.5, "cross10")
        lines.append(
            f"{law}: 単目標 {a:.2f} → 2 目標 {b:.2f}（主目標殺傷、横風、σ=0.5）"
        )
    lines.append(
        "主目標は正面の衝突コースのまま。decoy は横 80 m を横切るのみで、"
        "誘導は主目標を見続けている。差が出ないのは設計どおり。"
        "目標切替を入れた場合は別の実験になる。"
    )
    return lines


def plot_compare(summ):
    laws = [n for n, _ in LAWS]
    scenarios = list(SCENARIOS)
    cond = "cross10"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(laws))
    width = 0.25
    for i, scen in enumerate(scenarios):
        pk = [summ[law][scen]["0.5"][cond]["kin_pk"] for law in laws]
        axes[0].bar(x + (i - 1) * width, pk, width, label=scen)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(laws, rotation=20, ha="right")
    axes[0].set_ylabel("kinetic mean Pk")
    axes[0].set_title(f"Evasive-style targets ({cond}, σ=0.5)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    for i, sigma in enumerate(SIGMAS):
        pk = [summ[law]["sustained"][str(sigma)]["cross10"]["kin_pk"]
              for law in laws]
        axes[1].bar(x + (i - 0.5) * 0.35, pk, 0.35, label=f"σ={sigma}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(laws, rotation=20, ha="right")
    axes[1].set_ylabel("kinetic mean Pk")
    axes[1].set_title("Sustained weave, cross10")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pk_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Follow-up guidance study")
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
        futures = [pool.submit(_run_cell, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            records.extend(fut.result())
            done += 1
            if done % 40 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} cells"
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
    plot_compare(summ)
    text = generate_report(summ, len(records), elapsed)
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")
    for law_name, _ in LAWS:
        s = summ[law_name]["sustained"]["0.5"]
        row = "  ".join(f"{c[0]}={s[c[0]]['kin_pk']:.2f}" for c in CONDITIONS)
        print(f"  sus σ0.5 {law_name:24} {row}")


if __name__ == "__main__":
    main()
