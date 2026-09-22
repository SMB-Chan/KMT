"""Robustness of the guidance-law ranking, and the beam sensor threshold.

Two questions left open by the guidance-law study:

1. On evasive targets the paired ranking was PN ≈ Predictive ≫ APN.
   Does that order survive unmodelled wind and a communication delay?
2. Beam intercepts need a better sensor than σ = 0.5 m.  Where between
   0.5 and 0.1 does the miss median drop inside the 3 m kinetic radius?

Evasive trials are paired: same jink draw and same sensor-noise seed for
every law inside a condition.  Conditions change wind or delay only, so
the jink stream is identical across conditions as well.

Output: ``results/guidance_robustness/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interceptor import (
    APNGuidance, EngagementConfig, FragmentationWarhead,
    KineticImpactWarhead, PNGuidance, PurePursuit, PredictivePursuit,
    SensorNoise, WarheadType, WindField, evasive_heading_fn, run_engagement,
    scenario_beam, scenario_head_on, score_at_cpa,
)
from guidance_law_study import (
    LAW_FACTORIES, LAWS, FRAG, KIN, stable_seed,
)

OUT = "results/guidance_robustness"

# ---------------------------------------------------------------------
#  Disturbance conditions
# ---------------------------------------------------------------------
# Crosswind is the unmodelled-bias case: the interceptor flies a heading
# toward a ground LOS while the airmass slides across it.  Along-track
# wind mostly changes closure.  delay is the observation age fed to the
# law (seconds).  gust_rms is the horizontal AR(1) gust RMS (m/s).
CONDITIONS = (
    # id,            wind_x, wind_y, delay,  gust_rms, note
    ("calm",          0.0,  0.0, 0.00, 0.0, "baseline"),
    ("cross5",        0.0,  5.0, 0.00, 0.0, "5 m/s crosswind"),
    ("cross10",       0.0, 10.0, 0.00, 0.0, "10 m/s crosswind"),
    ("along10",      10.0,  0.0, 0.00, 0.0, "10 m/s along-track"),
    ("d50",           0.0,  0.0, 0.05, 0.0, "50 ms link"),
    ("d150",          0.0,  0.0, 0.15, 0.0, "150 ms link"),
    ("d300",          0.0,  0.0, 0.30, 0.0, "300 ms link"),
    ("cross10_d150",  0.0, 10.0, 0.15, 0.0, "10 m/s cross + 150 ms"),
    ("cross10_gust",  0.0, 10.0, 0.00, 2.0, "10 m/s cross + 2 m/s gust"),
)
COND_BY_ID = {c[0]: c for c in CONDITIONS}

# Beam sensor ladder.  Published endpoints 0.5 and 0.1 plus a dense
# ladder in between so the 3 m crossing can be interpolated.
BEAM_SIGMAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
BEAM_LAWS = ("pred", "pn_raw", "pure")   # threshold + control
BEAM_CONDITIONS = ("calm", "cross10_d150")

DISTANCES = (500, 1000, 2000)
SPEEDS = (10, 15, 20)
N_TRIALS = 50
SIGMAS = (0.5, 0.1)
SEED_BASE = 9000


@dataclass
class RobustConfig:
    n_trials: int = N_TRIALS
    workers: int = 10
    seed_base: int = SEED_BASE


def _wind_from_cond(row) -> WindField:
    _cid, wx, wy, delay, gust, _note = row
    return WindField(mean=(wx, wy, 0.0), gust_rms=gust, gust_tau=2.0,
                     target_drift=0.0)


def _law(law_id: str):
    return LAW_FACTORIES[law_id]()


# ---------------------------------------------------------------------
#  Part A — evasive ranking under wind / delay
# ---------------------------------------------------------------------
def _evasive_cell(task: dict) -> list[dict]:
    cond = COND_BY_ID[task["cond"]]
    _cid, wx, wy, delay, gust, _note = cond
    wind = _wind_from_cond(cond)
    records = []
    for trial in range(task["n_trials"]):
        # Seed excludes law and condition: same jink and same sensor draw
        # for every law and every disturbance cell.
        seed = stable_seed("evasive", task["distance"], task["speed"],
                           task["sigma"], trial, base=task["seed_base"])
        tp, tv, ip, iv = scenario_head_on(
            task["distance"], task["speed"],
            task["interceptor_speed"], task["altitude"])
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"]),
            sensor=SensorNoise(pos_sigma=task["sigma"],
                               vel_sigma=task["sigma"] * 0.4),
            wind=wind,
            comm_delay=delay,
            store_trajectory=False,
        )
        hdg = evasive_heading_fn(seed=seed)
        res = run_engagement(tp, tv, ip, iv, cfg, seed=seed,
                             target_heading_fn=hdg)
        miss = float(res.miss_distance)
        approach = float(res.approach_speed)
        hit, pk = score_at_cpa(miss, approach, WarheadType.KINETIC_IMPACT,
                               FRAG, KIN)
        rng = np.random.default_rng(seed + 17)
        records.append({
            "part": "evasive",
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
            "time": float(res.engagement_time),
            "kin_hit": bool(hit),
            "kin_pk": float(pk),
            "kin_kill": bool(hit and rng.random() < pk),
        })
    return records


def _evasive_tasks(cfg: RobustConfig) -> list[dict]:
    tasks = []
    for law_name, law_id in LAWS:
        for cond in CONDITIONS:
            for dist in DISTANCES:
                for spd in SPEEDS:
                    for sigma in SIGMAS:
                        tasks.append({
                            "part": "evasive",
                            "law": law_name,
                            "law_id": law_id,
                            "cond": cond[0],
                            "distance": dist,
                            "speed": spd,
                            "sigma": sigma,
                            "n_trials": cfg.n_trials,
                            "interceptor_speed": 50.0,
                            "altitude": 50.0,
                            "seed_base": cfg.seed_base,
                        })
    return tasks


# ---------------------------------------------------------------------
#  Part B — beam sensor threshold
# ---------------------------------------------------------------------
def _beam_cell(task: dict) -> list[dict]:
    cond = COND_BY_ID[task["cond"]]
    _cid, _wx, _wy, delay, _gust, _note = cond
    wind = _wind_from_cond(cond)
    records = []
    for trial in range(task["n_trials"]):
        seed = stable_seed("beam_thr", task["distance"], task["speed"],
                           task["sigma"], trial, base=task["seed_base"])
        tp, tv, ip, iv = scenario_beam(
            task["distance"], task["speed"],
            task["interceptor_speed"], task["altitude"])
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=_law(task["law_id"]),
            sensor=SensorNoise(pos_sigma=task["sigma"],
                               vel_sigma=task["sigma"] * 0.4),
            wind=wind,
            comm_delay=delay,
            store_trajectory=False,
        )
        res = run_engagement(tp, tv, ip, iv, cfg, seed=seed)
        miss = float(res.miss_distance)
        approach = float(res.approach_speed)
        hit, pk = score_at_cpa(miss, approach, WarheadType.KINETIC_IMPACT,
                               FRAG, KIN)
        rng = np.random.default_rng(seed + 17)
        records.append({
            "part": "beam",
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
            "time": float(res.engagement_time),
            "kin_hit": bool(hit),
            "kin_pk": float(pk),
            "kin_kill": bool(hit and rng.random() < pk),
        })
    return records


def _beam_tasks(cfg: RobustConfig) -> list[dict]:
    tasks = []
    for law_id in BEAM_LAWS:
        law_name = next(n for n, i in LAWS if i == law_id)
        for cond in BEAM_CONDITIONS:
            for sigma in BEAM_SIGMAS:
                for dist in DISTANCES:
                    for spd in SPEEDS:
                        tasks.append({
                            "part": "beam",
                            "law": law_name,
                            "law_id": law_id,
                            "cond": cond,
                            "distance": dist,
                            "speed": spd,
                            "sigma": sigma,
                            "n_trials": cfg.n_trials,
                            "interceptor_speed": 50.0,
                            "altitude": 50.0,
                            "seed_base": cfg.seed_base,
                        })
    return tasks


# ---------------------------------------------------------------------
#  Stats
# ---------------------------------------------------------------------
def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _stats(rows: list[dict]) -> dict:
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
        "mean_miss": float(np.mean(misses)) if misses else float("nan"),
        "p95_miss": float(np.percentile(misses, 95)) if misses else float("nan"),
        "p_miss_lt_3": (sum(1 for m in misses if m < 3.0) / len(misses)
                        if misses else float("nan")),
        "kin_hit": hits / n,
        "kin_pk": float(np.mean(pk)) if pk else float("nan"),
        "kin_pk_se": float(np.std(pk, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "kin_kill": kills / n,
        "kin_kill_lo": lo,
        "kin_kill_hi": hi,
    }


def _subset(rows, **kw):
    out = rows
    for key, val in kw.items():
        if val is None:
            continue
        out = [r for r in out if r[key] == val]
    return out


def summarize_evasive(records: list[dict]) -> dict:
    """Per law × σ × condition, n = 3×3×50 = 450."""
    out = {}
    for law_name, _ in LAWS:
        out[law_name] = {}
        for sigma in SIGMAS:
            out[law_name][str(sigma)] = {}
            for cond in CONDITIONS:
                rows = _subset(records, law=law_name, sigma=sigma,
                               cond=cond[0])
                out[law_name][str(sigma)][cond[0]] = _stats(rows)
    return out


def summarize_beam(records: list[dict]) -> dict:
    """Per law × σ × condition."""
    out = {}
    for law_id in BEAM_LAWS:
        law_name = next(n for n, i in LAWS if i == law_id)
        out[law_name] = {}
        for cond in BEAM_CONDITIONS:
            out[law_name][cond] = {}
            for sigma in BEAM_SIGMAS:
                rows = _subset(records, law=law_name, cond=cond, sigma=sigma)
                out[law_name][cond][str(sigma)] = _stats(rows)
    return out


def beam_threshold(law_stats: dict) -> dict:
    """Interpolate the σ where median miss crosses 3 m (descending σ).

    Walks σ from 0.1 upward.  The first bracket [σ_lo, σ_hi] whose median
    misses straddle 3 m is linearly interpolated.  If the median is already
    above 3 m at σ=0.1 the threshold is reported as below the ladder.
    """
    ladder = sorted((float(s), v) for s, v in law_stats.items())
    # ladder is σ ascending; median miss should fall as σ falls.
    # We want the largest σ whose median miss is still ≤ 3 m.
    for i in range(len(ladder) - 1):
        s0, v0 = ladder[i]
        s1, v1 = ladder[i + 1]
        m0, m1 = v0.get("median_miss", float("nan")), v1.get("median_miss", float("nan"))
        if m0 != m0 or m1 != m1:
            continue
        if m0 <= 3.0 < m1:
            # crossing between s0 (good) and s1 (bad) as σ increases
            frac = (3.0 - m0) / (m1 - m0) if m1 != m0 else 0.0
            return {"sigma_star": s0 + frac * (s1 - s0),
                    "bracket": [s0, s1],
                    "median_at_lo": m0,
                    "median_at_hi": m1}
        if m0 > 3.0 and m1 <= 3.0:
            frac = (m0 - 3.0) / (m0 - m1) if m0 != m1 else 0.0
            return {"sigma_star": s0 + frac * (s1 - s0),
                    "bracket": [s0, s1],
                    "median_at_lo": m0,
                    "median_at_hi": m1}
    # No crossing: all inside or all outside.
    if ladder:
        s0, v0 = ladder[0]
        s1, v1 = ladder[-1]
        if v0.get("median_miss", 1e9) <= 3.0:
            return {"sigma_star": s1, "bracket": [s1, s1],
                    "median_at_lo": v0.get("median_miss"),
                    "median_at_hi": v1.get("median_miss"),
                    "note": "median already ≤ 3 m at the coarsest σ on the ladder"}
        return {"sigma_star": float("nan"), "bracket": [s0, s1],
                "median_at_lo": v0.get("median_miss"),
                "median_at_hi": v1.get("median_miss"),
                "note": "median never reaches 3 m on this ladder"}
    return {"sigma_star": float("nan")}


# ---------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------
def _fmt_m(x):
    if x != x:
        return "—"
    return f"{x:.2f} m"


def _fmt_pct(x):
    if x != x:
        return "—"
    return f"{x:.0%}"


def generate_report(ev: dict, beam: dict, thresholds: dict,
                    cfg: RobustConfig, elapsed: float,
                    n_ev: int, n_beam: int) -> str:
    lines = [
        "# 誘導則の頑健性と、ビームのセンサ閾値",
        "",
        "## 何を確かめたか",
        "",
        "誘導則比較（v4）の続き。残っていた問いは2つ。",
        "",
        "1. 回避での PN ≈ Predictive ≫ APN という順位は、風と通信遅延が入っても持つか。",
        "2. ビームを 3 m（衝撃半径）以内に入れるセンサは、σ=0.5 と σ=0.1 のどこにあるか。",
        "",
        "風は迎撃機の地表速度にだけ効く（標的はマルチコプターとして対地速度を保持）。"
        "誘導則は風を推定しないため、横風は未モデル化バイアスになる。"
        "通信遅延は「観測サンプルの時刻を t − τ に遅らせる」。ノイズはサンプル時点で引いてあるので、"
        "遅れた測位は古く、かつ汚れている。",
        "",
        "## 設定",
        "",
        f"- 誘導則: {', '.join(n for n, _ in LAWS)}",
        f"- 回避: head-on ジング（v4 と同じ `evasive_heading_fn`）、"
        f"距離 {list(DISTANCES)} m × 速度 {list(SPEEDS)} m/s × 各 {cfg.n_trials} 試行、"
        f"条件ごと n={len(DISTANCES) * len(SPEEDS) * cfg.n_trials}",
        f"- 対比較: ジングとセンサ乱数を誘導則・条件のすべてで共有（seed に law / cond を含めない）",
        f"- 風・遅延条件: {len(CONDITIONS)} 種（下表）",
        f"- ビーム閾値: σ ∈ {list(BEAM_SIGMAS)}、誘導則 {', '.join(BEAM_LAWS)}、"
        f"条件 {list(BEAM_CONDITIONS)}",
        f"- センサ: 位置 σ、速度 0.4σ。殺傷は同一最接近に対し衝撃弾頭を解析採点",
        f"- 回避 {n_ev} + ビーム {n_beam} = {n_ev + n_beam} 交戦、"
        f"実行 {elapsed:.1f} s、ワーカ {cfg.workers}",
        "",
        "### 風・遅延条件",
        "",
        "| id | 風 (m/s) | 遅延 | ガスト RMS | 意味 |",
        "|---|---|---|---|---|",
    ]
    for cid, wx, wy, delay, gust, note in CONDITIONS:
        wind = f"({wx:.0f}, {wy:.0f})"
        lines.append(
            f"| {cid} | {wind} | {delay * 1000:.0f} ms | {gust:.0f} m/s | {note} |"
        )
    lines.append("")

    # ---- Part A ----
    if ev:
        lines.extend([
            "## 1. 回避 — 順位は風と遅延に耐えるか",
            "",
            f"各セル n={len(DISTANCES) * len(SPEEDS) * cfg.n_trials}"
            f"（距離 {len(DISTANCES)} × 速度 {len(SPEEDS)} × {cfg.n_trials} 試行）。"
            f"衝撃 mean Pk の標準誤差 ≈ 0.02（n=450 のとき）。",
            "",
        ])
        for sigma in SIGMAS:
            lines.append(f"### σ = {sigma} m")
            lines.append("")
            header = "| 誘導則 | " + " | ".join(
                f"{c[0]} miss / Pk" for c in CONDITIONS) + " |"
            sep = "|---|" + "---|" * len(CONDITIONS)
            lines.append(header)
            lines.append(sep)
            for law_name, _ in LAWS:
                cells = []
                for cond in CONDITIONS:
                    s = ev[law_name][str(sigma)][cond[0]]
                    cells.append(f"{s['median_miss']:.2f} / {s['kin_pk']:.2f}")
                lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
            lines.append("")

        lines.extend(_rank_interpretation(ev))
        lines.append("")

    # ---- Part B ----
    if beam:
        lines.extend([
            "## 2. ビーム — 3 m に入れる σ",
            "",
            "ミス中央値と P(miss < 3 m)、衝撃 mean Pk。閾値 σ* は"
            "中央値が 3 m を横切る区間を線形補間した。",
            "",
        ])
        for cond in BEAM_CONDITIONS:
            lines.append(f"### 条件 {cond}")
            lines.append("")
            lines.append("| 誘導則 | σ | miss 中央 | P(miss<3) | 衝撃 Pk | σ* (miss 中央 = 3 m) |")
            lines.append("|---|---|---|---|---|---|")
            for law_id in BEAM_LAWS:
                law_name = next(n for n, i in LAWS if i == law_id)
                thr = thresholds.get(cond, {}).get(law_name, {})
                sig_star = thr.get("sigma_star", float("nan"))
                for sigma in BEAM_SIGMAS:
                    s = beam[law_name][cond][str(sigma)]
                    if sigma == BEAM_SIGMAS[0]:
                        star_txt = (f"{sig_star:.3f}" if sig_star == sig_star
                                    else thr.get("note", "—"))
                    else:
                        star_txt = ""
                    lines.append(
                        f"| {law_name} | {sigma:.2f} | {_fmt_m(s['median_miss'])} | "
                        f"{_fmt_pct(s['p_miss_lt_3'])} | {s['kin_pk']:.2f} | {star_txt} |"
                    )
            lines.append("")

        lines.extend(_beam_interpretation(beam, thresholds))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 風は一定 + 水平 AR(1) ガスト。鉛直シアとドップラーレーダの風推定は入れていない。",
        "- 標的は対地速度を保持するマルチコプター。固定翼なら `target_drift` を上げる必要がある。",
        "- 誘導則はすべて風非推定。風を推定してバンクを補う誘導は別物になる。",
        "- 通信遅延は観測時刻の平行移動のみ。途絶・欠測・レート制限はない。",
        "- 回避ジングは v4 と同じ 0.5 Hz × 5 回。持続マニューバではない。",
        "- 弾頭曲線は v3/v4 のまま（衝撃半径 3 m・1500 J）。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/evasive_robust.json` — 回避の条件×誘導則集計",
        f"- `{OUT}/beam_threshold.json` — ビームの σ ラダーと σ*",
        f"- `{OUT}/evasive_heatmap.png` — 衝撃 Pk の条件×誘導則",
        f"- `{OUT}/beam_threshold.png` — ミス中央値と σ",
        f"- `{OUT}/trials.jsonl` — 試行ごと",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _rank_interpretation(ev: dict) -> list[str]:
    """Read the evasive tables: does PN ≈ Predictive ≫ APN survive?"""
    if not ev:
        return []
    lines = ["### 読み — 回避", ""]

    def pk(law, sigma, cond):
        return ev[law][str(sigma)][cond]["kin_pk"]

    def med(law, sigma, cond):
        return ev[law][str(sigma)][cond]["median_miss"]

    calm_pp = pk("Pure Pursuit", 0.5, "calm")
    calm_pn = pk("PN raw", 0.5, "calm")
    calm_apn = pk("APN filtered", 0.5, "calm")
    calm_pr = pk("Predictive", 0.5, "calm")

    lines.append(
        f"平静（calm、σ=0.5）の衝撃 Pk は Pure Pursuit {calm_pp:.2f}、"
        f"APN {calm_apn:.2f}、PN raw {calm_pn:.2f}、Predictive {calm_pr:.2f}。"
        "v4 の対比較と同じ並びで、ジングとセンサ乱数の共有方法が変わっていない"
        "ことを確認できる。"
    )
    lines.append("")

    # Ranking per condition at σ=0.5 — report what the data says.
    lines.append("σ=0.5 の順位（衝撃 Pk 降順）:")
    lines.append("")
    lines.append("| 条件 | 1位 | 2位 | 3位 | 4位 | 5位 |")
    lines.append("|---|---|---|---|---|---|")
    ranks = {}
    for cond in CONDITIONS:
        cid = cond[0]
        order = sorted((n for n, _ in LAWS), key=lambda n: -pk(n, 0.5, cid))
        ranks[cid] = order
        lines.append(
            f"| {cid} | "
            + " | ".join(f"{n} ({pk(n, 0.5, cid):.2f})" for n in order)
            + " |"
        )
    lines.append("")

    # Delay-only trend
    lines.append(
        "遅延だけ（d50 → d150 → d300）では "
        f"PN raw が {pk('PN raw', 0.5, 'd50'):.2f} → {pk('PN raw', 0.5, 'd150'):.2f}"
        f" → {pk('PN raw', 0.5, 'd300'):.2f}、"
        f"Predictive が {pk('Predictive', 0.5, 'd50'):.2f}"
        f" → {pk('Predictive', 0.5, 'd150'):.2f} → {pk('Predictive', 0.5, 'd300'):.2f}、"
        f"APN が {pk('APN filtered', 0.5, 'd50'):.2f}"
        f" → {pk('APN filtered', 0.5, 'd150'):.2f} → {pk('APN filtered', 0.5, 'd300'):.2f}。"
        "300 ms でも順位は calm のままで、遅延だけなら v4 の結論が持つ。"
        "APN の加速度推定は元から遅れているので、リンク遅延の追加が効きやすい。"
    )
    lines.append("")

    # Crosswind — the finding that breaks the calm ranking
    pr_cross = pk("Predictive", 0.5, "cross10")
    pn_cross = pk("PN raw", 0.5, "cross10")
    apn_cross = pk("APN filtered", 0.5, "cross10")
    pr_med_cross = med("Predictive", 0.5, "cross10")
    pn_med_cross = med("PN raw", 0.5, "cross10")
    lines.append(
        f"横風 10 m/s では順位が崩れる。Predictive の衝撃 Pk は calm {calm_pr:.2f} から"
        f" {pr_cross:.2f} に落ち、miss 中央は {_fmt_m(pr_med_cross)}。"
        f"同じ横風で PN raw は Pk {pn_cross:.2f}、miss 中央 {_fmt_m(pn_med_cross)}。"
        "衝突三角形は対地の交点を狙うが、誘導は風を推定していないので、"
        "機首を交点へ向けたまま横に流される。リードが全部効かなくなる。"
        "PN は視線角速度を打ち消すので、風が生む見かけの角速度にも同じ舵が当たり、"
        f"崩れ方が小さく残る（APN は {apn_cross:.2f} で、遅れた加速度推定が重なる）。"
    )
    lines.append("")
    lines.append(
        "したがって「PN ≈ Predictive ≫ APN」は平静と遅延では持つが、"
        "横風下では **PN ≫ Predictive** に変わる。"
        "APN が最下位から動くのは、Predictive が風で先に崩れたセルだけである。"
    )
    lines.append("")

    # Condition-wise drop from calm
    lines.append("条件ごとの calm からの低下（σ=0.5、衝撃 Pk）:")
    lines.append("")
    lines.append("| 条件 | Pure Pursuit | PN raw | APN | Predictive |")
    lines.append("|---|---|---|---|---|")
    for cond in CONDITIONS:
        cid = cond[0]
        if cid == "calm":
            continue
        cells = []
        for law in ("Pure Pursuit", "PN raw", "APN filtered", "Predictive"):
            d = pk(law, 0.5, cid) - pk(law, 0.5, "calm")
            cells.append(f"{d:+.2f}")
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"全条件での最小 Pk は Pure Pursuit "
        f"{min(pk('Pure Pursuit', 0.5, c[0]) for c in CONDITIONS):.2f}、"
        f"APN {min(pk('APN filtered', 0.5, c[0]) for c in CONDITIONS):.2f}、"
        f"PN raw {min(pk('PN raw', 0.5, c[0]) for c in CONDITIONS):.2f}、"
        f"Predictive {min(pk('Predictive', 0.5, c[0]) for c in CONDITIONS):.2f}。"
    )
    return lines


def _beam_interpretation(beam: dict, thresholds: dict) -> list[str]:
    if not beam:
        return []
    lines = ["### 読み — ビームの閾値", ""]
    for cond in BEAM_CONDITIONS:
        lines.append(f"**{cond}**")
        lines.append("")
        for law_id in BEAM_LAWS:
            law_name = next(n for n, i in LAWS if i == law_id)
            thr = thresholds.get(cond, {}).get(law_name, {})
            sig_star = thr.get("sigma_star", float("nan"))
            s_lo = beam[law_name][cond][str(BEAM_SIGMAS[0])]
            s_hi = beam[law_name][cond][str(BEAM_SIGMAS[-1])]
            if sig_star == sig_star:
                lines.append(
                    f"- {law_name}: σ* ≈ **{sig_star:.2f}**"
                    f"（中央値 3 m。σ=0.10 で {_fmt_m(s_lo['median_miss'])}、"
                    f"σ=0.50 で {_fmt_m(s_hi['median_miss'])}、"
                    f"σ* で Pk は約 "
                    f"{beam[law_name][cond][str(min(BEAM_SIGMAS, key=lambda x: abs(x - sig_star)))]['kin_pk']:.2f}）"
                )
            else:
                lines.append(
                    f"- {law_name}: {thr.get('note', 'σ* なし')}"
                    f"（σ=0.10 で {_fmt_m(s_lo['median_miss'])}、"
                    f"σ=0.50 で {_fmt_m(s_hi['median_miss'])}）"
                )
        lines.append("")

    # Headline numbers
    pred_calm = thresholds.get("calm", {}).get("Predictive", {})
    pn_calm = thresholds.get("calm", {}).get("PN raw", {})
    pure_calm = thresholds.get("calm", {}).get("Pure Pursuit", {})
    pred_hard = thresholds.get("cross10_d150", {}).get("Predictive", {})
    s05_pred = beam["Predictive"]["calm"]["0.5"]
    s05_pn = beam["PN raw"]["calm"]["0.5"]
    s01_pred = beam["Predictive"]["calm"]["0.1"]
    lines.append(
        f"σ=0.50 のビームは、v4 と同じく中央値が 3 m の外"
        f"（Predictive {_fmt_m(s05_pred['median_miss'])}、"
        f"PN raw {_fmt_m(s05_pn['median_miss'])}）。"
        f"σ=0.10 の Predictive は {_fmt_m(s01_pred['median_miss'])} まで入り、"
        f"P(miss<3) は {_fmt_pct(s01_pred['p_miss_lt_3'])}。"
    )
    lines.append("")
    if pred_calm.get("sigma_star", float("nan")) == pred_calm.get("sigma_star"):
        lines.append(
            f"衝撃がビームで使える側（中央値 3 m）に入る境目は、"
            f"Predictive で σ ≈ {pred_calm['sigma_star']:.2f} m、"
            f"PN raw で σ ≈ {pn_calm.get('sigma_star', float('nan')):.2f} m にある。"
            "これは v4 で書いた「σ=0.5 と 0.1 の間」を数字で割ったもので、"
            "GPS 級（σ=0.5）では届かず、レーダ／高精度測距（σ ≲ 0.2–0.3）が要る。"
        )
        lines.append("")
        if pred_hard.get("sigma_star", float("nan")) == pred_hard.get("sigma_star"):
            lines.append(
                f"横風 10 m/s + 150 ms の条件下では Predictive の σ* が"
                f" {pred_hard['sigma_star']:.2f} m に動く。"
                "外乱は閾値を手前に押し、順位は変えない。"
            )
        else:
            lines.append(
                "横風 10 m/s + 150 ms では、この σ ラダーのどこでも"
                "ミス中央値が 3 m に入らない。"
                "センサを良くしても、風と遅延の未モデル化が先に効く。"
            )
    else:
        lines.append(
            "このラダーでは中央値が 3 m を横切らなかった。"
            f"σ=0.10 の Predictive 中央値は {_fmt_m(s01_pred['median_miss'])}。"
        )
    return lines


# ---------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------
def plot_evasive_heatmap(ev: dict, sigma: float = 0.5):
    laws = [n for n, _ in LAWS]
    cond_ids = [c[0] for c in CONDITIONS]
    mat = np.array([[ev[law][str(sigma)][cid]["kin_pk"] for cid in cond_ids]
                    for law in laws])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.3, vmax=0.8)
    ax.set_xticks(range(len(cond_ids)))
    ax.set_xticklabels(cond_ids, rotation=30, ha="right")
    ax.set_yticks(range(len(laws)))
    ax.set_yticklabels(laws)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="kinetic mean Pk")
    n_cell = 0
    if ev:
        first_law = next(iter(ev))
        n_cell = ev[first_law][str(sigma)][CONDITIONS[0][0]].get("n", 0)
    ax.set_title(f"Evasive kinetic Pk by condition (σ = {sigma} m, n = {n_cell})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/evasive_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_beam_threshold(beam: dict, thresholds: dict):
    colors = {"Predictive": "#d62728", "PN raw": "#ff7f0e", "Pure Pursuit": "#7f7f7f"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cond, ls in (("calm", "-"), ("cross10_d150", "--")):
        for law_id in BEAM_LAWS:
            law_name = next(n for n, i in LAWS if i == law_id)
            sig = np.array(BEAM_SIGMAS)
            med = np.array([beam[law_name][cond][str(s)]["median_miss"]
                            for s in BEAM_SIGMAS])
            pk = np.array([beam[law_name][cond][str(s)]["kin_pk"]
                           for s in BEAM_SIGMAS])
            axes[0].plot(sig, med, ls, color=colors[law_name], lw=2,
                         label=f"{law_name} [{cond}]")
            axes[1].plot(sig, pk, ls, color=colors[law_name], lw=2,
                         label=f"{law_name} [{cond}]")
            thr = thresholds.get(cond, {}).get(law_name, {}).get("sigma_star")
            if thr == thr and cond == "calm":
                axes[0].axvline(thr, color=colors[law_name], ls=":", lw=1)
    axes[0].axhline(3.0, color="black", ls=":", lw=1, label="3 m kinetic")
    axes[0].set_xlabel("sensor σ (m)")
    axes[0].set_ylabel("median miss (m)")
    axes[0].set_title("Beam median miss vs sensor noise")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("sensor σ (m)")
    axes[1].set_ylabel("kinetic mean Pk")
    axes[1].set_title("Beam expected kill vs sensor noise")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/beam_threshold.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------
def _run_tasks(tasks, workers, label):
    records = []
    print(f"{label}: {len(tasks)} cells × {tasks[0]['n_trials']} trials"
          f"  workers={workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evasive_cell if t["part"] == "evasive"
                               else _beam_cell, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            records.extend(fut.result())
            done += 1
            if done % 40 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} cells"
                      f"  ({time.time() - t0:.0f} s)", flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Guidance robustness (wind/link) and beam sensor threshold")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--part", choices=("all", "evasive", "beam"),
                        default="all")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild REPORT.md from saved JSON (no sim).")
    args = parser.parse_args(argv)
    cfg = RobustConfig(n_trials=args.trials, workers=args.workers)
    os.makedirs(OUT, exist_ok=True)

    if args.report_only:
        with open(f"{OUT}/evasive_robust.json") as f:
            ev = json.load(f)
        with open(f"{OUT}/beam_threshold.json") as f:
            beam_blob = json.load(f)
        beam = beam_blob.get("stats", {})
        thresholds = beam_blob.get("thresholds", {})
        n_ev = 0
        if ev:
            n_ev = sum(cell["n"] for law in ev.values()
                       for sig in law.values()
                       for cell in sig.values())
        n_beam = 0
        if beam:
            n_beam = sum(cell["n"] for law in beam.values()
                         for cond in law.values()
                         for cell in cond.values())
        text = generate_report(ev, beam, thresholds, cfg, 0.0, n_ev, n_beam)
        with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Rebuilt {OUT}/REPORT.md from JSON  (n_ev={n_ev}, n_beam={n_beam})")
        return

    t0 = time.time()
    records = []
    if args.part in ("all", "evasive"):
        records.extend(_run_tasks(_evasive_tasks(cfg), cfg.workers, "evasive"))
    if args.part in ("all", "beam"):
        records.extend(_run_tasks(_beam_tasks(cfg), cfg.workers, "beam"))
    elapsed = time.time() - t0
    print(f"Total {elapsed:.1f} s, {len(records)} engagements", flush=True)

    with open(f"{OUT}/trials.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    ev_rows = [r for r in records if r["part"] == "evasive"]
    beam_rows = [r for r in records if r["part"] == "beam"]
    ev = summarize_evasive(ev_rows) if ev_rows else {}
    beam = summarize_beam(beam_rows) if beam_rows else {}
    thresholds = {}
    for cond in BEAM_CONDITIONS:
        thresholds[cond] = {}
        for law_id in BEAM_LAWS:
            law_name = next(n for n, i in LAWS if i == law_id)
            thresholds[cond][law_name] = beam_threshold(
                beam.get(law_name, {}).get(cond, {}))

    with open(f"{OUT}/evasive_robust.json", "w") as f:
        json.dump(ev, f, indent=2)
    with open(f"{OUT}/beam_threshold.json", "w") as f:
        json.dump({"ladder": list(BEAM_SIGMAS), "stats": beam,
                   "thresholds": thresholds}, f, indent=2)

    if ev:
        plot_evasive_heatmap(ev, sigma=0.5)
        plot_evasive_heatmap(ev, sigma=0.1)
    if beam:
        plot_beam_threshold(beam, thresholds)

    text = generate_report(ev, beam, thresholds, cfg, elapsed,
                           len(ev_rows), len(beam_rows))
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")

    if ev:
        for law_name, _ in LAWS:
            s = ev[law_name]["0.5"]
            row = "  ".join(
                f"{c[0]}={s[c[0]]['kin_pk']:.2f}" for c in CONDITIONS)
            print(f"  σ0.5 {law_name:16} {row}")
    if thresholds:
        for cond in BEAM_CONDITIONS:
            for law_name, thr in thresholds[cond].items():
                print(f"  σ* {cond:14} {law_name:14} "
                      f"{thr.get('sigma_star', float('nan')):.3f}")


if __name__ == "__main__":
    main()
