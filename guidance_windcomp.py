"""Does crab compensation restore Predictive under wind, and move beam σ*?

v5 found that unmodelled crosswind collapses Predictive (Pk 0.00) while
PN holds on (0.31).  The collision triangle is a ground-track aim flown
as an air heading.  ``CrabPredictive`` / ``CrabPursuit`` offset the
heading into the wind so the ground track holds the aim point.

Two questions:

1. On evasive targets, does crab put Predictive back beside PN?
2. On a beam, does crab let a coarser sensor reach miss < 3 m?

Wind is either reconstructed from ground velocity (``est``) or known
perfectly (``oracle``) — the ceiling of compensation.

Output: ``results/guidance_windcomp/``.
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
    SensorNoise, WarheadType, WindField, evasive_heading_fn, run_engagement,
    scenario_beam, scenario_head_on, score_at_cpa,
)
from guidance_law_study import FRAG, KIN, stable_seed

OUT = "results/guidance_windcomp"

# cond_id, wind_x, wind_y, delay, gust
CONDITIONS = (
    ("calm",          0.0,  0.0, 0.00, 0.0),
    ("cross5",        0.0,  5.0, 0.00, 0.0),
    ("cross10",       0.0, 10.0, 0.00, 0.0),
    ("cross10_d150",  0.0, 10.0, 0.15, 0.0),
    ("cross10_gust",  0.0, 10.0, 0.00, 2.0),
)

BEAM_SIGMAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
BEAM_LAWS = ("pred", "crab_pred_air", "crab_pred_or", "pn_raw")
BEAM_CONDITIONS = ("calm", "cross10_d150")

DISTANCES = (500, 1000, 2000)
SPEEDS = (10, 15, 20)
N_TRIALS = 50
SIGMAS = (0.5, 0.1)
SEED_BASE = 11000

# id, factory kwargs
LAWS = (
    ("Predictive", "pred"),
    ("CrabPredictive airdata", "crab_pred_air"),
    ("CrabPredictive oracle", "crab_pred_or"),
    ("CrabPursuit airdata", "crab_pp_air"),
    ("PN raw", "pn_raw"),
)


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


def _run_cell(task: dict) -> list[dict]:
    cond = next(c for c in CONDITIONS if c[0] == task["cond"])
    _cid, wx, wy, delay, gust = cond
    wind = _wind(cond)
    scenario = task["scenario"]
    records = []
    for trial in range(task["n_trials"]):
        # Law-independent seed: same jink and sensor draw for every law.
        seed = stable_seed("wc", scenario, task["distance"], task["speed"],
                           task["sigma"], trial, base=task["seed_base"])
        if scenario == "evasive":
            tp, tv, ip, iv = scenario_head_on(
                task["distance"], task["speed"],
                task["interceptor_speed"], task["altitude"])
            hdg = evasive_heading_fn(seed=seed)
        else:
            tp, tv, ip, iv = scenario_beam(
                task["distance"], task["speed"],
                task["interceptor_speed"], task["altitude"])
            hdg = None
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
        records.append({
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
        })
    return records


def _evasive_tasks(n_trials: int, seed_base: int):
    tasks = []
    for law_name, law_id in LAWS:
        for cond in CONDITIONS:
            for dist in DISTANCES:
                for spd in SPEEDS:
                    for sigma in SIGMAS:
                        tasks.append({
                            "scenario": "evasive",
                            "law": law_name, "law_id": law_id,
                            "cond": cond[0],
                            "distance": dist, "speed": spd, "sigma": sigma,
                            "n_trials": n_trials,
                            "interceptor_speed": 50.0, "altitude": 50.0,
                            "seed_base": seed_base,
                        })
    return tasks


def _beam_tasks(n_trials: int, seed_base: int):
    tasks = []
    name_of = dict((i, n) for n, i in LAWS)
    for law_id in BEAM_LAWS:
        for cond in BEAM_CONDITIONS:
            for sigma in BEAM_SIGMAS:
                for dist in DISTANCES:
                    for spd in SPEEDS:
                        tasks.append({
                            "scenario": "beam",
                            "law": name_of[law_id], "law_id": law_id,
                            "cond": cond,
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
        "mean_miss": float(np.mean(misses)) if misses else float("nan"),
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


def summarize_evasive(records):
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


def summarize_beam(records):
    out = {}
    name_of = dict((i, n) for n, i in LAWS)
    for law_id in BEAM_LAWS:
        law_name = name_of[law_id]
        out[law_name] = {}
        for cond in BEAM_CONDITIONS:
            out[law_name][cond] = {}
            for sigma in BEAM_SIGMAS:
                rows = _subset(records, law=law_name, cond=cond, sigma=sigma)
                out[law_name][cond][str(sigma)] = _stats(rows)
    return out


def beam_threshold(law_stats: dict) -> dict:
    ladder = sorted((float(s), v) for s, v in law_stats.items())
    for i in range(len(ladder) - 1):
        s0, v0 = ladder[i]
        s1, v1 = ladder[i + 1]
        m0 = v0.get("median_miss", float("nan"))
        m1 = v1.get("median_miss", float("nan"))
        if m0 != m0 or m1 != m1:
            continue
        if m0 <= 3.0 < m1:
            frac = (3.0 - m0) / (m1 - m0) if m1 != m0 else 0.0
            return {"sigma_star": s0 + frac * (s1 - s0), "bracket": [s0, s1]}
        if m0 > 3.0 and m1 <= 3.0:
            frac = (m0 - 3.0) / (m0 - m1) if m0 != m1 else 0.0
            return {"sigma_star": s0 + frac * (s1 - s0), "bracket": [s0, s1]}
    if ladder:
        s0, v0 = ladder[0]
        s1, v1 = ladder[-1]
        if v0.get("median_miss", 1e9) <= 3.0:
            return {"sigma_star": s1, "bracket": [s1, s1],
                    "note": "already ≤ 3 m at the coarsest σ"}
        return {"sigma_star": float("nan"), "bracket": [s0, s1],
                "note": "median never reaches 3 m"}
    return {"sigma_star": float("nan")}


def _fmt_m(x):
    return "—" if x != x else f"{x:.2f} m"


def generate_report(ev, beam, thresholds, n_ev, n_beam, elapsed) -> str:
    lines = [
        "# 風補償で Predictive を戻せるか",
        "",
        "## 何を確かめたか",
        "",
        "v5 は、横風 10 m/s で Predictive の衝撃 Pk が 0.67 → 0.00 に崩れ、"
        "PN だけが 0.31 で残ることを示した。衝突三角形は対地の交点を狙うが、"
        "指令は対気機首なので風で流れる。",
        "",
        "続きとして、**機首を風に向け直す（crab）**ことで、そのリードを"
        "取り戻せるかを見る。衝突三角形は気団基準で解き、"
        "ψ = 方向(必要な対気速度) とする。風は (a) 空速計+INS から毎ステップ"
        "推定する（airdata）か、(b) 真値を与える（oracle、補償の上限）。",
        "",
        "## 設定",
        "",
        f"- 誘導則: {', '.join(n for n, _ in LAWS)}",
        f"- crab: 気団基準の衝突三角形 |r+(v_t−W)t| = V_a t を解き、"
        f"対気速度 (aim−r_i)/t_go − W の向きを機首にする",
        f"- 風推定の誤差 σ_W = 0.5 m/s（空速計+INS の代表値）",
        f"- 回避: v5 と同じジング対比較、条件ごと n=450",
        f"- ビーム: σ ∈ {list(BEAM_SIGMAS)}、条件 {list(BEAM_CONDITIONS)}",
        f"- 回避 {n_ev} + ビーム {n_beam} = {n_ev + n_beam} 交戦、"
        f"{elapsed:.0f} s",
        "",
        "### 条件",
        "",
        "| id | 風 (m/s) | 遅延 | ガスト |",
        "|---|---|---|---|",
    ]
    for cid, wx, wy, delay, gust in CONDITIONS:
        lines.append(
            f"| {cid} | ({wx:.0f}, {wy:.0f}) | {delay * 1000:.0f} ms | {gust:.0f} m/s |"
        )
    lines.append("")

    # ---- Evasive ----
    lines.extend([
        "## 1. 回避 — crab は Predictive を戻すか",
        "",
        "miss 中央 / 衝撃 mean Pk（n=450）。",
        "",
    ])
    for sigma in SIGMAS:
        lines.append(f"### σ = {sigma} m")
        lines.append("")
        lines.append("| 誘導則 | " + " | ".join(
            f"{c[0]} miss / Pk" for c in CONDITIONS) + " |")
        lines.append("|---|" + "---|" * len(CONDITIONS))
        for law_name, _ in LAWS:
            cells = []
            for cond in CONDITIONS:
                s = ev[law_name][str(sigma)][cond[0]]
                cells.append(f"{s['median_miss']:.2f} / {s['kin_pk']:.2f}")
            lines.append(f"| {law_name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(_ev_read(ev))
    lines.append("")

    # ---- Beam ----
    lines.extend([
        "## 2. ビーム — crab で σ* は動くか",
        "",
    ])
    for cond in BEAM_CONDITIONS:
        lines.append(f"### 条件 {cond}")
        lines.append("")
        lines.append("| 誘導則 | σ | miss 中央 | P(miss<3) | 衝撃 Pk | σ* |")
        lines.append("|---|---|---|---|---|---|")
        name_of = dict((i, n) for n, i in LAWS)
        for law_id in BEAM_LAWS:
            law_name = name_of[law_id]
            thr = thresholds.get(cond, {}).get(law_name, {})
            sig_star = thr.get("sigma_star", float("nan"))
            for sigma in BEAM_SIGMAS:
                s = beam[law_name][cond][str(sigma)]
                if sigma == BEAM_SIGMAS[0]:
                    star = (f"{sig_star:.3f}" if sig_star == sig_star
                            else thr.get("note", "—"))
                else:
                    star = ""
                lines.append(
                    f"| {law_name} | {sigma:.2f} | {_fmt_m(s['median_miss'])} | "
                    f"{s['p_miss_lt_3']:.0%} | {s['kin_pk']:.2f} | {star} |"
                )
        lines.append("")
    lines.extend(_beam_read(beam, thresholds))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 風推定は機体の対気姿勢と対地速度から作る（空速計+INS）。旋回中の機体-風速のずれが残る。",
        "- oracle は真値の風。実機では届かない上限。",
        "- crab は水平面の 1 自由度。バンク限界を超える風では頭打ち。",
        "- 弾頭・ジング・センサモデルは v3–v5 のまま。",
        "",
        "## 出力",
        "",
        f"- `{OUT}/evasive_windcomp.json`",
        f"- `{OUT}/beam_windcomp.json`",
        f"- `{OUT}/evasive_pk.png` / `{OUT}/beam_threshold.png`",
        f"- `{OUT}/trials.jsonl`",
        f"- `{OUT}/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _ev_read(ev):
    def pk(law, sigma, cond):
        return ev[law][str(sigma)][cond]["kin_pk"]

    def med(law, sigma, cond):
        return ev[law][str(sigma)][cond]["median_miss"]

    raw_cross = pk("Predictive", 0.5, "cross10")
    crab_cross = pk("CrabPredictive airdata", 0.5, "cross10")
    oracle_cross = pk("CrabPredictive oracle", 0.5, "cross10")
    pn_cross = pk("PN raw", 0.5, "cross10")
    raw_med = med("Predictive", 0.5, "cross10")
    crab_med = med("CrabPredictive airdata", 0.5, "cross10")
    oracle_med = med("CrabPredictive oracle", 0.5, "cross10")
    calm_crab = pk("CrabPredictive airdata", 0.5, "calm")
    calm_raw = pk("Predictive", 0.5, "calm")

    lines = [
        "### 読み — 回避",
        "",
        f"横風 10 m/s（σ=0.5）で、素の Predictive は Pk {raw_cross:.2f}"
        f"（miss 中央 {_fmt_m(raw_med)}）。crab を入れると"
        f"空力推定版が {crab_cross:.2f}（{_fmt_m(crab_med)}）、"
        f"真値版が {oracle_cross:.2f}（{_fmt_m(oracle_med)}）。"
        f"同じ条件の PN raw は {pn_cross:.2f}。",
        "",
        f"平静では crab 空力推定版 {calm_crab:.2f}、素の Predictive {calm_raw:.2f}。"
        "crab は無風では過補償にならず、風下ではリードを取り戻す。",
        "",
    ]
    # Does crab beat PN under wind?
    if crab_cross > pn_cross + 0.03:
        lines.append(
            f"crab 版は横風で PN raw（{pn_cross:.2f}）を上回る。"
            "風が既知に近ければ、衝突三角形は PN より優位に戻る。"
        )
    elif crab_cross > pn_cross - 0.03:
        lines.append(
            f"crab 版は横風で PN raw（{pn_cross:.2f}）と同点。"
            "リードの復活は、少なくとも PN の持つロバスト性と釣り合う。"
        )
    else:
        lines.append(
            f"crab 版（{crab_cross:.2f}）は横風でも PN raw（{pn_cross:.2f}）"
            "に届かない。風推定の遅れか、ジング中の過渡が残る。"
        )
    # Hard combined condition
    hard_crab = pk("CrabPredictive airdata", 0.5, "cross10_d150")
    hard_oracle = pk("CrabPredictive oracle", 0.5, "cross10_d150")
    hard_pn = pk("PN raw", 0.5, "cross10_d150")
    hard_raw = pk("Predictive", 0.5, "cross10_d150")
    lines.extend([
        "",
        f"横風 + 150 ms では素の Predictive {hard_raw:.2f}、"
        f"crab 空力推定 {hard_crab:.2f}、crab 真値 {hard_oracle:.2f}、"
        f"PN raw {hard_pn:.2f}。遅延が重なると風推定も遅れる。",
    ])
    return lines


def _beam_read(beam, thresholds):
    name_of = dict((i, n) for n, i in LAWS)
    lines = ["### 読み — ビーム", ""]
    for cond in BEAM_CONDITIONS:
        parts = []
        for law_id in BEAM_LAWS:
            law_name = name_of[law_id]
            thr = thresholds.get(cond, {}).get(law_name, {})
            sig_star = thr.get("sigma_star", float("nan"))
            s_lo = beam[law_name][cond][str(BEAM_SIGMAS[0])]
            s_hi = beam[law_name][cond][str(BEAM_SIGMAS[-1])]
            if sig_star == sig_star:
                parts.append(
                    f"- {law_name}: σ* ≈ **{sig_star:.2f}**"
                    f"（σ=0.10 で {_fmt_m(s_lo['median_miss'])}、"
                    f"σ=0.50 で {_fmt_m(s_hi['median_miss'])}）"
                )
            else:
                parts.append(
                    f"- {law_name}: 3 m に入らない"
                    f"（σ=0.10 で {_fmt_m(s_lo['median_miss'])}）"
                )
        lines.append(f"**{cond}**")
        lines.append("")
        lines.extend(parts)
        lines.append("")

    # Headline: did crab move the hard-condition threshold?
    thr_hard_pred = thresholds.get("cross10_d150", {}).get("Predictive", {})
    thr_hard_crab = thresholds.get("cross10_d150", {}).get(
        "CrabPredictive airdata", {})
    thr_hard_or = thresholds.get("cross10_d150", {}).get(
        "CrabPredictive oracle", {})
    thr_calm_crab = thresholds.get("calm", {}).get("CrabPredictive airdata", {})
    sp_hard_pred = thr_hard_pred.get("sigma_star", float("nan"))
    sp_hard_crab = thr_hard_crab.get("sigma_star", float("nan"))
    sp_hard_or = thr_hard_or.get("sigma_star", float("nan"))
    sp_calm = thr_calm_crab.get("sigma_star", float("nan"))
    lines.append(
        "v5 では横風+150 ms の σ ラダー上どこでも miss 中央が 3 m に入らなかった。"
    )
    lines.append("")
    if sp_hard_or == sp_hard_or:
        lines.append(
            f"crab 真値では σ* ≈ {sp_hard_or:.2f} m まで動く。"
            "風が正しければ、ビームでもセンサの要求が緩む。"
        )
    else:
        lines.append(
            "crab 真値でも横風+150 ms の 3 m は達成できない。"
            "残る遅延か、crab の過渡が miss を決める。"
        )
    if sp_hard_crab == sp_hard_crab:
        lines.append(
            f"風推定版の σ* は {sp_hard_crab:.2f} m。"
        )
    if sp_calm == sp_calm:
        lines.append(
            f"参考: 平静の crab 空力推定版 σ* は {sp_calm:.2f} m。"
        )
    return lines


def plot_evasive(ev):
    laws = [n for n, _ in LAWS]
    cond_ids = [c[0] for c in CONDITIONS]
    mat = np.array([[ev[law]["0.5"][cid]["kin_pk"] for cid in cond_ids]
                    for law in laws])
    fig, ax = plt.subplots(figsize=(9, 4.0))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=0.8)
    ax.set_xticks(range(len(cond_ids)))
    ax.set_xticklabels(cond_ids, rotation=30, ha="right")
    ax.set_yticks(range(len(laws)))
    ax.set_yticklabels(laws)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="kinetic mean Pk")
    ax.set_title("Evasive kinetic Pk — crab compensation (σ = 0.5 m)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/evasive_pk.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_beam(beam, thresholds):
    colors = {
        "Predictive": "#d62728",
        "CrabPredictive airdata": "#ff7f0e",
        "CrabPredictive oracle": "#2ca02c",
        "CrabPursuit airdata": "#9467bd",
        "PN raw": "#1f77b4",
    }
    name_of = dict((i, n) for n, i in LAWS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cond, ls in (("calm", "-"), ("cross10_d150", "--")):
        for law_id in BEAM_LAWS:
            law_name = name_of[law_id]
            sig = np.array(BEAM_SIGMAS)
            med = np.array([beam[law_name][cond][str(s)]["median_miss"]
                            for s in BEAM_SIGMAS])
            pk = np.array([beam[law_name][cond][str(s)]["kin_pk"]
                           for s in BEAM_SIGMAS])
            axes[0].plot(sig, med, ls, color=colors[law_name], lw=2,
                         label=f"{law_name} [{cond}]")
            axes[1].plot(sig, pk, ls, color=colors[law_name], lw=2,
                         label=f"{law_name} [{cond}]")
    axes[0].axhline(3.0, color="black", ls=":", lw=1)
    axes[0].set_xlabel("sensor σ (m)")
    axes[0].set_ylabel("median miss (m)")
    axes[0].set_title("Beam median miss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, loc="upper left")
    axes[1].set_xlabel("sensor σ (m)")
    axes[1].set_ylabel("kinetic mean Pk")
    axes[1].set_title("Beam expected kill")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/beam_threshold.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _run_tasks(tasks, workers, label):
    records = []
    print(f"{label}: {len(tasks)} cells × {tasks[0]['n_trials']} trials"
          f"  workers={workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_cell, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            records.extend(fut.result())
            done += 1
            if done % 40 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} cells"
                      f"  ({time.time() - t0:.0f} s)", flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wind-crab compensation study")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--part", choices=("all", "evasive", "beam"),
                        default="all")
    args = parser.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    records = []
    if args.part in ("all", "evasive"):
        records.extend(_run_tasks(_evasive_tasks(args.trials, SEED_BASE),
                                  args.workers, "evasive"))
    if args.part in ("all", "beam"):
        records.extend(_run_tasks(_beam_tasks(args.trials, SEED_BASE),
                                  args.workers, "beam"))
    elapsed = time.time() - t0
    print(f"Total {elapsed:.1f} s, {len(records)} engagements", flush=True)

    with open(f"{OUT}/trials.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    ev_rows = [r for r in records if r["scenario"] == "evasive"]
    beam_rows = [r for r in records if r["scenario"] == "beam"]
    ev = summarize_evasive(ev_rows) if ev_rows else {}
    beam = summarize_beam(beam_rows) if beam_rows else {}
    thresholds = {}
    name_of = dict((i, n) for n, i in LAWS)
    for cond in BEAM_CONDITIONS:
        thresholds[cond] = {}
        for law_id in BEAM_LAWS:
            law_name = name_of[law_id]
            thresholds[cond][law_name] = beam_threshold(
                beam.get(law_name, {}).get(cond, {}))

    with open(f"{OUT}/evasive_windcomp.json", "w") as f:
        json.dump(ev, f, indent=2)
    with open(f"{OUT}/beam_windcomp.json", "w") as f:
        json.dump({"ladder": list(BEAM_SIGMAS), "stats": beam,
                   "thresholds": thresholds}, f, indent=2)

    if ev:
        plot_evasive(ev)
    if beam:
        plot_beam(beam, thresholds)

    text = generate_report(ev, beam, thresholds,
                           len(ev_rows), len(beam_rows), elapsed)
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")

    if ev:
        for law_name, _ in LAWS:
            s = ev[law_name]["0.5"]
            row = "  ".join(
                f"{c[0]}={s[c[0]]['kin_pk']:.2f}" for c in CONDITIONS)
            print(f"  σ0.5 {law_name:24} {row}")
    for cond in BEAM_CONDITIONS:
        for law_name, thr in thresholds.get(cond, {}).items():
            print(f"  σ* {cond:14} {law_name:24} "
                  f"{thr.get('sigma_star', float('nan')):.3f}")


if __name__ == "__main__":
    main()
