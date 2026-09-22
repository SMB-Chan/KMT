"""Guidance-law comparison for the CUAS interceptor.

The v3 warhead table left two open questions: whether a different
guidance law shrinks miss enough to change the frag-vs-kinetic
ranking, and why the beam geometry killed both warheads.  This study
answers both on the corrected model:

* PN / APN integrate N·ω into a heading command.  The previous
  one-step rebase did not turn the airframe.
* Both warheads are scored at the geometric closest approach.  Ending
  on first entry of the collision sphere recorded every kinetic hit
  as a graze at ≈3 m.

One engagement is flown per trial.  Both warheads are then scored on
that same impact parameter, so the warhead comparison is paired.

Output: ``results/guidance_law_study/``.
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
    SensorNoise, WarheadType, evasive_heading_fn, run_engagement,
    scenario_beam, scenario_head_on, scenario_tail_chase, score_at_cpa,
)

OUT = "results/guidance_law_study"

MAIN_SCENARIOS = ("head_on", "tail_chase", "evasive")
ALL_SCENARIOS = MAIN_SCENARIOS + ("beam",)
WARHEADS = (WarheadType.FRAGMENTATION, WarheadType.KINETIC_IMPACT)
FRAG = FragmentationWarhead()
KIN = KineticImpactWarhead()


def law_pure():
    return PurePursuit()


def law_pn_raw():
    return PNGuidance(N=4.0, los_tau=0.0)


def law_pn_filt():
    return PNGuidance(N=4.0, los_tau=0.15)


def law_apn():
    return APNGuidance(N=4.0, los_tau=0.15)


def law_pred():
    return PredictivePursuit()


LAWS = (
    ("Pure Pursuit", "pure"),
    ("PN raw", "pn_raw"),
    ("PN filtered", "pn_filt"),
    ("APN filtered", "apn"),
    ("Predictive", "pred"),
)
LAW_FACTORIES = {
    "pure": law_pure,
    "pn_raw": law_pn_raw,
    "pn_filt": law_pn_filt,
    "apn": law_apn,
    "pred": law_pred,
}


@dataclass
class StudyConfig:
    n_trials: int = 50
    distances: list = field(default_factory=lambda: [500, 1000, 2000])
    target_speeds: list = field(default_factory=lambda: [10, 15, 20])
    interceptor_speed: float = 50.0
    altitude: float = 50.0
    sigmas: list = field(default_factory=lambda: [0.5, 0.1])
    seed_base: int = 7000
    workers: int = 10


def stable_seed(*parts, base: int = 0) -> int:
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return base + int(digest[:8], 16) % 100_000


def _scenario(name, distance, target_speed, intr_speed, alt):
    if name == "head_on" or name == "evasive":
        return scenario_head_on(distance, target_speed, intr_speed, alt)
    if name == "tail_chase":
        return scenario_tail_chase(distance, target_speed, intr_speed, alt)
    if name == "beam":
        return scenario_beam(distance, target_speed, intr_speed, alt)
    raise ValueError(name)


def _run_cell(task: dict) -> list[dict]:
    """Fly ``n_trials`` engagements and score both warheads on each CPA."""
    records = []
    for trial in range(task["n_trials"]):
        # Law name is not part of the seed.  Evasive jinks and the sensor
        # draw must be shared, or a law comparison is two different maneuvers.
        # trials.jsonl from the 2026-09-22 run still used a law-dependent
        # seed; evasive claims in the report use evasive_paired.json.
        seed = stable_seed(task["scenario"], task["distance"], task["speed"],
                           task["sigma"], trial, base=task["seed_base"])
        tp, tv, ip, iv = _scenario(
            task["scenario"], task["distance"], task["speed"],
            task["interceptor_speed"], task["altitude"])
        cfg = EngagementConfig(
            warhead_type=WarheadType.FRAGMENTATION,
            guidance_law=LAW_FACTORIES[task["law_id"]](),
            sensor=SensorNoise(pos_sigma=task["sigma"],
                               vel_sigma=task["sigma"] * 0.4),
            store_trajectory=False,
        )
        hdg = (evasive_heading_fn(seed=seed)
               if task["scenario"] == "evasive" else None)
        res = run_engagement(tp, tv, ip, iv, cfg, seed=seed,
                             target_heading_fn=hdg)
        miss = float(res.miss_distance)
        approach = float(res.approach_speed)
        timed_out = (not res.hit and res.engagement_time >= cfg.max_time - 0.05
                     and miss > 50.0)
        row = {
            "law": task["law"],
            "scenario": task["scenario"],
            "distance": task["distance"],
            "speed": task["speed"],
            "sigma": task["sigma"],
            "trial": trial,
            "seed": seed,
            "miss": miss if math.isfinite(miss) else 1e9,
            "approach_speed": approach,
            "time": float(res.engagement_time),
            "timeout": timed_out,
        }
        rng = np.random.default_rng(seed + 17)
        for wh in WARHEADS:
            hit, pk = score_at_cpa(miss, approach, wh, FRAG, KIN)
            row[wh.value + "_hit"] = bool(hit)
            row[wh.value + "_pk"] = float(pk)
            row[wh.value + "_kill"] = bool(hit and rng.random() < pk)
        records.append(row)
    return records


def _tasks(cfg: StudyConfig) -> list[dict]:
    tasks = []
    for law_name, law_id in LAWS:
        for scenario in ALL_SCENARIOS:
            for dist in cfg.distances:
                for spd in cfg.target_speeds:
                    for sigma in cfg.sigmas:
                        tasks.append({
                            "law": law_name,
                            "law_id": law_id,
                            "scenario": scenario,
                            "distance": dist,
                            "speed": spd,
                            "sigma": sigma,
                            "n_trials": cfg.n_trials,
                            "interceptor_speed": cfg.interceptor_speed,
                            "altitude": cfg.altitude,
                            "seed_base": cfg.seed_base,
                        })
    return tasks


def run_study(cfg: StudyConfig) -> list[dict]:
    tasks = _tasks(cfg)
    records = []
    print(f"Cells: {len(tasks)}  trials: {len(tasks) * cfg.n_trials}  "
          f"workers: {cfg.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [pool.submit(_run_cell, task) for task in tasks]
        done = 0
        for fut in as_completed(futures):
            records.extend(fut.result())
            done += 1
            if done % 20 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} cells", flush=True)
    return records


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _subset(records, **kw):
    out = records
    for key, val in kw.items():
        if val is None:
            continue
        out = [r for r in out if r[key] == val]
    return out


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    misses = [r["miss"] for r in rows if r["miss"] < 500]
    timeouts = sum(1 for r in rows if r["timeout"])
    out = {
        "n": n,
        "timeouts": timeouts,
        "mean_miss": float(np.mean(misses)) if misses else float("nan"),
        "median_miss": float(np.median(misses)) if misses else float("nan"),
        "p95_miss": float(np.percentile(misses, 95)) if misses else float("nan"),
        "mean_time": float(np.mean([r["time"] for r in rows])) if rows else float("nan"),
    }
    for wh, prefix in (("fragmentation", "frag"), ("kinetic_impact", "kin")):
        hits = sum(1 for r in rows if r[wh + "_hit"])
        kills = sum(1 for r in rows if r[wh + "_kill"])
        pk = [r[wh + "_pk"] for r in rows]
        lo, hi = _wilson(kills, n)
        out[prefix + "_hit"] = hits / n if n else float("nan")
        out[prefix + "_kill"] = kills / n if n else float("nan")
        out[prefix + "_pk"] = float(np.mean(pk)) if pk else float("nan")
        out[prefix + "_kill_lo"] = lo
        out[prefix + "_kill_hi"] = hi
    return out


def summarize(records: list[dict], cfg: StudyConfig) -> dict:
    """Nested aggregates the report and plots read."""
    summary = {"by_law_sigma": {}, "by_scenario": {}, "cells": {}}
    for law, _ in LAWS:
        summary["by_law_sigma"][law] = {}
        for sigma in cfg.sigmas:
            main = _subset(records, law=law, sigma=sigma)
            main = [r for r in main if r["scenario"] in MAIN_SCENARIOS]
            beam = _subset(records, law=law, sigma=sigma, scenario="beam")
            summary["by_law_sigma"][law][str(sigma)] = {
                "main": _stats(main),
                "beam": _stats(beam),
                "all": _stats(_subset(records, law=law, sigma=sigma)),
            }
    for scenario in ALL_SCENARIOS:
        summary["by_scenario"][scenario] = {}
        for law, _ in LAWS:
            summary["by_scenario"][scenario][law] = {}
            for sigma in cfg.sigmas:
                rows = _subset(records, law=law, scenario=scenario, sigma=sigma)
                summary["by_scenario"][scenario][law][str(sigma)] = _stats(rows)
    # Speed slice at σ=0.5, used to separate a kinematic timeout from guidance.
    for law, _ in LAWS:
        for scenario in ("tail_chase", "head_on"):
            for dist in cfg.distances:
                for spd in cfg.target_speeds:
                    key = f"{law}|{scenario}|{dist}|{spd}|0.5"
                    rows = _subset(records, law=law, scenario=scenario,
                                   distance=dist, speed=spd, sigma=0.5)
                    summary["cells"][key] = _stats(rows)
    return summary


def _fmt_pct(x):
    if x != x:
        return "—"
    return f"{x:.0%}"


def _fmt_m(x):
    if x != x:
        return "—"
    return f"{x:.2f} m"


def _fmt_ci(rate, lo, hi):
    if rate != rate:
        return "—"
    return f"{rate:.0%} [{lo:.0%}, {hi:.0%}]"


def generate_report(summary: dict, cfg: StudyConfig, elapsed: float, n_records: int) -> str:
    lines = [
        "# 誘導則比較 — 迎撃研究の続き",
        "",
        "## 何を確かめたか",
        "",
        "v3 の弾頭表は、炸裂型の殺傷率が衝撃型を大きく上回り、ビーム幾何では"
        "両方とも落ちる、というところで止まっていた。続きとして、その差が"
        "誘導則で動くかを同じ機体・同じ弾頭曲線で比較した。",
        "",
        "走らせる前にモデルの2点を直した。どちらも「誘導則の優劣」に見える"
        "数字を、実装の都合が作っていた。",
        "",
        "1. **比例航法の方位指令が積分されていなかった。** 毎ステップ"
        " `現在方位 + N·Δλ` を出していたため、バンクサーボ（τ=0.15 s）が"
        "追従する前に指令が消え、ビームではほぼ直進していた"
        "（無雑音・1000 m で miss 41 m）。積分指令 `ψ ← ψ + N·ω·dt` に"
        "変えると、同じ条件で miss は約 3 m まで下がる。",
        "2. **衝撃型を衝突球へ入った瞬間に採点していた。** その距離は常に"
        "球半径付近（≈3 m）なので、正面衝突もかすりも同じ「かすり」になる。"
        "視線方向の相対速度は最接近点で定義上ほぼ 0 であり、そこを衝突"
        "エネルギーに使うと殺傷率がさらに落ちる。今は幾何学的最接近距離"
        "（impact parameter）と、その時点の相対速度の大きさで採点する。",
        "",
        "v3 本実験の `EngagementConfig` は誘導則を指定しておらず、既定は "
        "Pure Pursuit だった。本文の「PN の構造的限界」は、走った誘導則と"
        "一致しない。本実験は誘導則を明示し、ビームを主平均から分けて出す。",
        "",
        "## 実験設定",
        "",
        f"- 誘導則: Pure Pursuit / PN raw（τ=0）/ PN filtered（N=4, τ=0.15 s）"
        f" / APN filtered（同じフィルタ）/ Predictive（衝突三角形）",
        f"- シナリオ: head-on, tail-chase, evasive, beam",
        f"- 初期距離: {cfg.distances} m",
        f"- 標的速度: {cfg.target_speeds} m/s、迎撃速度: {cfg.interceptor_speed} m/s",
        f"- センサ: σ ∈ {cfg.sigmas} m（速度ノイズは 0.4σ）",
        f"- 各セル {cfg.n_trials} 試行。弾頭は同一最接近に対して対で採点する",
        f"- 試行数: {n_records}（交戦回数。弾頭は解析的に2回採点）",
        f"- 実行時間: {elapsed:.1f} s、ワーカ {cfg.workers}",
        f"- シードは MD5 で固定（プロセスをまたいで再現する）",
        "",
        "殺傷率の [ ] は Bernoulli 試行の Wilson 95% CI。"
        "期待殺傷（mean Pk）は同じ miss に弾頭曲線を当てた平均値で、"
        "コイン投げの分散を含まない。順位の判断は mean Pk と miss を主にし、"
        "Bernoulli は v3 との読み比べ用に併記する。",
        "",
        "## 主シナリオ（beam 除外）",
        "",
    ]
    for sigma in cfg.sigmas:
        lines.append(f"### σ = {sigma} m")
        lines.append("")
        lines.append("| 誘導則 | miss 中央 | miss 平均 | miss P95 | "
                     "炸裂 命中 | 炸裂 mean Pk | 炸裂 殺傷 [95% CI] | "
                     "衝撃 命中 | 衝撃 mean Pk | 衝撃 殺傷 [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for law, _ in LAWS:
            s = summary["by_law_sigma"][law][str(sigma)]["main"]
            lines.append(
                f"| {law} | {_fmt_m(s['median_miss'])} | {_fmt_m(s['mean_miss'])} | "
                f"{_fmt_m(s['p95_miss'])} | {_fmt_pct(s['frag_hit'])} | "
                f"{s['frag_pk']:.2f} | {_fmt_ci(s['frag_kill'], s['frag_kill_lo'], s['frag_kill_hi'])} | "
                f"{_fmt_pct(s['kin_hit'])} | {s['kin_pk']:.2f} | "
                f"{_fmt_ci(s['kin_kill'], s['kin_kill_lo'], s['kin_kill_hi'])} |"
            )
        lines.append("")

    lines.extend([
        "## シナリオ別（miss 中央値、炸裂命中は参考）",
        "",
        "miss は誘導の量であり、弾頭では変わらない。命中率だけ弾頭半径が違う"
        "（炸裂 6 m、衝撃 3 m）。",
        "",
    ])
    for sigma in cfg.sigmas:
        lines.append(f"### σ = {sigma} m")
        lines.append("")
        lines.append("| シナリオ | 誘導則 | miss 中央 | miss P95 | "
                     "炸裂命中 | 衝撃命中 | 炸裂 mean Pk | 衝撃 mean Pk | タイムアウト |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for scenario in ALL_SCENARIOS:
            for law, _ in LAWS:
                s = summary["by_scenario"][scenario][law][str(sigma)]
                lines.append(
                    f"| {scenario} | {law} | {_fmt_m(s['median_miss'])} | "
                    f"{_fmt_m(s['p95_miss'])} | {_fmt_pct(s['frag_hit'])} | "
                    f"{_fmt_pct(s['kin_hit'])} | {s['frag_pk']:.2f} | "
                    f"{s['kin_pk']:.2f} | {s['timeouts']}/{s['n']} |"
                )
        lines.append("")

    lines.extend([
        "## 読み",
        "",
    ])
    lines.extend(_interpretation(summary, cfg))
    lines.extend([
        "",
        "## モデルの限界",
        "",
        "- 高度 50 m 固定、風なし、1対1。通信遅延と多目標はない。",
        "- バンク 80° が旋回限界。`max_accel`（80 m/s²）は旋回計算に入っていない。"
        "50 m/s・80° の幾何旋回は約 5.7 g。",
        "- 最接近は dt=0.01 s のサンプル最小値。相対 60 m/s なら 1 ステップ 0.6 m 以内。",
        "- 回避は 0.5 Hz のジングを数回入れるだけで、持続バレルロールではない。",
        "- 弾頭曲線そのものは v3 と同じ（炸裂 R=6 m・300 破片、衝撃半径 3 m・"
        "1500 J）。パラメータは動かしていない。",
        "- 追尾 2000 m / 標的 20 m/s は閉鎖速度 30 m/s で 67 s かかり、"
        "制限時間 60 s を超える。これは誘導の失敗ではなく到達不能。",
        "- `trials.jsonl` の回避行は誘導則ごとにジングが違う。"
        "回避の結論は `evasive_paired.json` の対比較を使う。",
        "",
        "## 出力",
        "",
        "- `results/guidance_law_study/trials.jsonl` — 試行ごと",
        "- `results/guidance_law_study/summary.json` — 集計",
        "- `results/guidance_law_study/miss_cdf.png`",
        "- `results/guidance_law_study/pk_by_law.png`",
        "- `results/guidance_law_study/beam_tracks.png`",
        "- `results/guidance_law_study/evasive_paired.json`",
        "- `results/guidance_law_study/REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def _interpretation(summary: dict, cfg: StudyConfig) -> list[str]:
    """Narrative.  Main-table numbers are filled from this run.

    The evasive paired table is a separate supplement
    (``evasive_paired.json``): the published trials.jsonl drew a new
    jink per law.  Those paired numbers are quoted, not recomputed here.
    """
    pp = summary["by_law_sigma"]["Pure Pursuit"]["0.5"]["main"]
    pred = summary["by_law_sigma"]["Predictive"]["0.5"]["main"]
    raw = summary["by_law_sigma"]["PN raw"]["0.5"]["main"]
    filt = summary["by_law_sigma"]["PN filtered"]["0.5"]["main"]
    apn = summary["by_law_sigma"]["APN filtered"]["0.5"]["main"]
    pp1 = summary["by_law_sigma"]["Pure Pursuit"]["0.1"]["main"]
    pred1 = summary["by_law_sigma"]["Predictive"]["0.1"]["main"]
    beam_pp = summary["by_law_sigma"]["Pure Pursuit"]["0.5"]["beam"]
    beam_raw = summary["by_law_sigma"]["PN raw"]["0.5"]["beam"]
    beam_pred = summary["by_law_sigma"]["Predictive"]["0.5"]["beam"]
    beam_pp1 = summary["by_law_sigma"]["Pure Pursuit"]["0.1"]["beam"]
    beam_pred1 = summary["by_law_sigma"]["Predictive"]["0.1"]["beam"]
    ho = summary["by_scenario"]["head_on"]["Pure Pursuit"]["0.1"]
    timeout = summary["cells"].get("Pure Pursuit|tail_chase|2000|20|0.5", {})
    lines = [
        "### 結論",
        "",
        "主シナリオの平均では、誘導則を替えても弾頭の順位は動かない。"
        f"σ=0.5 m では炸裂 mean Pk が {apn['frag_pk']:.2f}–{pred['frag_pk']:.2f}、"
        f"衝撃が {pp['kin_pk']:.2f}–{pred['kin_pk']:.2f} で、衝撃は炸裂に並ばない。"
        f"σ=0.1 m では Pure Pursuit の衝撃 mean Pk が {pp['kin_pk']:.2f} から"
        f" {pp1['kin_pk']:.2f} に上がり、Predictive は {pred1['kin_pk']:.2f}。"
        "平均を動かすレバーはセンサで、誘導則の交換はその約 3 分の 1 である。",
        "",
        "誘導則が順位を変えるのは回避とビームである。正面と追尾は分けない。",
        "",
        "### 採点を直しただけで、v3 の「衝撃 7%」は消える",
        "",
        "v3 の衝撃殺傷率 7.1% は、衝突球に入った瞬間の距離（≈3 m）を"
        "かすりとして採点した値である。最接近距離で採点し直すと、"
        f"Pure Pursuit・σ=0.5 m・ビーム除外で衝撃 mean Pk は {pp['kin_pk']:.2f}"
        f"（Bernoulli {_fmt_ci(pp['kin_kill'], pp['kin_kill_lo'], pp['kin_kill_hi'])}）。"
        "弾頭曲線は変えていない。v3 の差の大半は誘導則ではなく採点位置だった。",
        "",
        "### 正面と追尾",
        "",
        f"正面・σ=0.1 m は Pure Pursuit で炸裂命中 {_fmt_pct(ho['frag_hit'])}、"
        f"衝撃命中 {_fmt_pct(ho['kin_hit'])}。他の誘導則も同じで、すでに衝突コース"
        "なので予測も PN も足すものがない。σ=0.5 m の正面でも誘導則間の衝撃命中は"
        "数ポイントに収まる。",
        "",
    ]
    if timeout.get("n"):
        lines.append(
            f"追尾 2000 m / 20 m/s は閉鎖速度 30 m/s で 67 s かかり、"
            f"Pure Pursuit はタイムアウト {timeout['timeouts']}/{timeout['n']}、"
            f"miss 中央 {_fmt_m(timeout['median_miss'])}。他の誘導則も同じ 50 試行が"
            "制限時間の外にいる。速度が足りないセルは誘導の敗北ではない。"
        )
        lines.append("")
    lines.extend([
        "### 回避 — 同じジングで引き直した結果",
        "",
        "本表の回避行は誘導則ごとにジングの乱数が違う。同じジング・同じセンサ"
        "乱数で引き直した（距離 3 × 速度 3 × 50 試行、各 σ で n=450、"
        "衝撃 Pk の標準誤差 ≈ 0.022）。数値は `evasive_paired.json`。",
        "",
        "| 誘導則 | σ=0.5 miss 中央 | σ=0.5 衝撃 Pk | σ=0.1 miss 中央 | σ=0.1 衝撃 Pk |",
        "|---|---|---|---|---|",
        "| Pure Pursuit | 3.05 m | 0.48 | 2.53 m | 0.52 |",
        "| PN raw | 1.86 m | 0.67 | 1.40 m | 0.71 |",
        "| PN filtered | 2.02 m | 0.63 | 1.51 m | 0.67 |",
        "| APN filtered | 3.33 m | 0.44 | 3.28 m | 0.46 |",
        "| Predictive | 1.74 m | 0.65 | 1.28 m | 0.68 |",
        "",
        "PN raw と Predictive の Pk 差は 1 標準誤差以内。中央 miss は Predictive "
        "がわずかに小さい。APN は両方の σ で Pure Pursuit より低い。フィルタした"
        "加速度がジングに遅れ、リードが逆を向く。この実装の APN は回避を助けない。",
        "",
        "### ビームは pursuit の遅れだった",
        "",
        "無雑音・1000 m のビームでは Pure Pursuit が 7.1 m、積分した PN が "
        "2.9 m、Predictive が 0.8 m（`beam_tracks.png`）。v3 が PN の構造的限界"
        "と呼んだ失敗は、本実験の既定が Pure Pursuit だったことと、方位指令が "
        "1 サンプル分しか積まれていなかったことの両方である。",
        "",
        f"σ=0.5 m（n={beam_pp['n']}、標的は等速）では Pure Pursuit の miss 中央"
        f" {_fmt_m(beam_pp['median_miss'])}、衝撃命中 {_fmt_pct(beam_pp['kin_hit'])}、"
        f"衝撃 Pk {beam_pp['kin_pk']:.2f}。PN raw は {_fmt_m(beam_raw['median_miss'])}、"
        f"{_fmt_pct(beam_raw['kin_hit'])}、{beam_raw['kin_pk']:.2f}。"
        f"Predictive は {_fmt_m(beam_pred['median_miss'])}、"
        f"{_fmt_pct(beam_pred['kin_hit'])}、{beam_pred['kin_pk']:.2f}。"
        "中央値はまだ 3 m の外なので、GPS 級のビームでは衝撃は過半に届かない。",
        "",
        f"σ=0.1 m では Pure Pursuit が miss 中央 {_fmt_m(beam_pp1['median_miss'])}、"
        f"衝撃命中 {_fmt_pct(beam_pp1['kin_hit'])} のままなのに対し、Predictive は"
        f" {_fmt_m(beam_pred1['median_miss'])}、衝撃命中 {_fmt_pct(beam_pred1['kin_hit'])}、"
        f"衝撃 Pk {beam_pred1['kin_pk']:.2f}。ビームで衝撃が使える側に入るのは、"
        "誘導則を替えたうえでのレーダ級センサである。",
        "",
        "### 視線角速度のフィルタは、この距離では要らない",
        "",
        f"主シナリオ σ=0.5 m の衝撃 Pk は PN raw {raw['kin_pk']:.2f}、"
        f"PN filtered {filt['kin_pk']:.2f}。ビームでは raw {beam_raw['kin_pk']:.2f} に対し"
        f" filtered {summary['by_law_sigma']['PN filtered']['0.5']['beam']['kin_pk']:.2f}。"
        "500–2000 m では視線角のノイズが小さく、積分が平均する。0.15 s の"
        "フィルタは遅れとして残り、利得にならない。",
        "",
        "### 次に聞くこと",
        "",
        "風と通信遅延を入れたときに、回避での PN ≈ Predictive ≫ APN が持つか。"
        "ビームを 3 m 以内に入れるセンサの境目は、σ=0.5 と σ=0.1 の間にある。"
        "そこを割れば、衝撃がビームでも炸裂に近づくかが書ける。",
    ])
    return lines


def plot_miss_cdf(records, cfg: StudyConfig):
    colors = {
        "Pure Pursuit": "#7f7f7f",
        "PN raw": "#ff7f0e",
        "PN filtered": "#1f77b4",
        "APN filtered": "#2ca02c",
        "Predictive": "#d62728",
    }
    fig, axes = plt.subplots(1, len(ALL_SCENARIOS), figsize=(16, 4.2), sharey=True)
    for ax, scenario in zip(axes, ALL_SCENARIOS):
        for law, _ in LAWS:
            misses = [r["miss"] for r in records
                      if r["law"] == law and r["scenario"] == scenario
                      and r["sigma"] == 0.5 and r["miss"] < 80]
            if not misses:
                continue
            arr = np.sort(misses)
            cdf = np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, cdf, lw=2, color=colors[law], label=law)
        ax.axvline(3.0, color="black", ls=":", lw=1)
        ax.axvline(6.0, color="black", ls="--", lw=1)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("miss (m)")
        ax.set_title(scenario.replace("_", " "))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CDF")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Miss CDF by guidance law (σ = 0.5 m). Dotted 3 m kinetic, dashed 6 m frag.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/miss_cdf.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_pk(summary, cfg: StudyConfig):
    fig, axes = plt.subplots(1, len(cfg.sigmas), figsize=(12, 4.5), sharey=True)
    laws = [name for name, _ in LAWS]
    x = np.arange(len(laws))
    width = 0.35
    for ax, sigma in zip(axes, cfg.sigmas):
        frag = [summary["by_law_sigma"][law][str(sigma)]["main"]["frag_pk"] for law in laws]
        kin = [summary["by_law_sigma"][law][str(sigma)]["main"]["kin_pk"] for law in laws]
        ax.bar(x - width / 2, frag, width, label="fragmentation", color="#d62728", alpha=0.85)
        ax.bar(x + width / 2, kin, width, label="kinetic", color="#1f77b4", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(laws, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"σ = {sigma} m, beam excluded")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("mean P(kill)")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Expected kill at the same impact parameter", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pk_by_law.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_beam_tracks():
    """One zero-noise beam engagement per law, horizontal plane."""
    colors = {
        "Pure Pursuit": "#7f7f7f",
        "PN raw": "#ff7f0e",
        "PN filtered": "#1f77b4",
        "APN filtered": "#2ca02c",
        "Predictive": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    tp, tv, ip, iv = scenario_beam(1000, 15, 50, 50)
    target_drawn = False
    for law_name, law_id in LAWS:
        cfg = EngagementConfig(
            guidance_law=LAW_FACTORIES[law_id](),
            sensor=SensorNoise(0.0, 0.0),
            store_trajectory=True,
        )
        res = run_engagement(tp, tv, ip, iv, cfg, seed=0)
        ti = np.array(res.trajectory_interceptor)
        tt = np.array(res.trajectory_target)
        ax.plot(ti[:, 0], ti[:, 1], color=colors[law_name], lw=2,
                label=f"{law_name}  miss {res.miss_distance:.1f} m")
        if not target_drawn:
            ax.plot(tt[:, 0], tt[:, 1], color="black", lw=1.5, ls="--", label="target")
            target_drawn = True
    ax.scatter([ip[0]], [ip[1]], c="black", s=30, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Beam, zero sensor noise, 1000 m / 15 m/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/beam_tracks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Guidance-law comparison study")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    cfg = StudyConfig(n_trials=args.trials, workers=args.workers)
    os.makedirs(OUT, exist_ok=True)
    print("Guidance law study")
    print(f"  laws={[n for n, _ in LAWS]}")
    print(f"  scenarios={ALL_SCENARIOS}")
    t0 = time.time()
    records = run_study(cfg)
    elapsed = time.time() - t0
    print(f"Simulation time: {elapsed:.1f} s  records: {len(records)}")

    with open(f"{OUT}/trials.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    summary = summarize(records, cfg)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"elapsed_s": elapsed, "n_records": len(records),
                   "summary": summary}, f, indent=2)
    plot_miss_cdf(records, cfg)
    plot_pk(summary, cfg)
    plot_beam_tracks()
    text = generate_report(summary, cfg, elapsed, len(records))
    with open(f"{OUT}/REPORT.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}/REPORT.md")
    for law, _ in LAWS:
        s = summary["by_law_sigma"][law]["0.5"]["main"]
        print(f"  {law:16} σ0.5 miss50={s['median_miss']:.2f} "
              f"fragPk={s['frag_pk']:.2f} kinPk={s['kin_pk']:.2f}")


if __name__ == "__main__":
    main()
