# 残作業 — oracle 再測定・持続マニューバ・2目標

## 何を確かめたか

風補償実験の残り3点。

1. **oracle の再測定** — 前回の真値版は風の平均しか見ておらず、ガストのあるセルで空力推定より下に出た。ループが瞬時 W を渡すように直したうえで、上限を引き直す。
2. **持続マニューバ** — 2 秒バーストのジングではなく、交戦中ずっとサイン波で織る。フィルタとリードの負荷が違う。
3. **2 目標** — 主目標（正面）と decoy（横 80 m）。主目標への殺傷で採点する。

## 設定

- 誘導則: Predictive, CrabPredictive airdata, CrabPredictive oracle, CrabPursuit airdata, PN raw
- シナリオ: jink（v5 と同じ）、sustained（連続サイン 0.35 Hz）、two_target（主目標 + decoy）
- 条件: calm / cross10 / cross10_d150 / cross10_gust
- 距離 [500, 1000, 2000] m × 速度 [10, 15, 20] m/s × σ ∈ [0.5, 0.1]、各 50 試行
- 合計 54000 交戦、1390 s

## σ = 0.5 m / jink

| 誘導則 | calm miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|
| Predictive | 1.66 / 0.65 | 40.41 / 0.00 | 44.02 / 0.00 | 39.63 / 0.00 |
| CrabPredictive airdata | 1.69 / 0.66 | 4.29 / 0.40 | 4.29 / 0.37 | 4.96 / 0.29 |
| CrabPredictive oracle | 1.65 / 0.65 | 5.36 / 0.39 | 6.12 / 0.38 | 6.06 / 0.24 |
| CrabPursuit airdata | 2.71 / 0.51 | 4.05 / 0.35 | 4.24 / 0.34 | 5.31 / 0.27 |
| PN raw | 1.84 / 0.62 | 6.75 / 0.25 | 8.60 / 0.21 | 7.09 / 0.20 |

## σ = 0.5 m / sustained

| 誘導則 | calm miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|
| Predictive | 1.41 / 0.91 | 41.39 / 0.00 | 44.98 / 0.00 | 40.66 / 0.00 |
| CrabPredictive airdata | 1.52 / 0.89 | 2.12 / 0.60 | 2.17 / 0.58 | 3.57 / 0.39 |
| CrabPredictive oracle | 1.42 / 0.91 | 2.14 / 0.60 | 2.20 / 0.57 | 3.38 / 0.38 |
| CrabPursuit airdata | 3.75 / 0.37 | 4.56 / 0.33 | 4.81 / 0.32 | 4.88 / 0.26 |
| PN raw | 1.61 / 0.82 | 4.44 / 0.36 | 5.06 / 0.37 | 5.92 / 0.26 |

## σ = 0.5 m / two_target

| 誘導則 | calm miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|
| Predictive | 6.96 / 0.34 | 34.51 / 0.00 | 38.43 / 0.00 | 33.42 / 0.00 |
| CrabPredictive airdata | 5.56 / 0.34 | 2.98 / 0.47 | 3.13 / 0.47 | 4.34 / 0.35 |
| CrabPredictive oracle | 6.97 / 0.34 | 5.37 / 0.36 | 6.12 / 0.34 | 6.00 / 0.25 |
| CrabPursuit airdata | 12.18 / 0.24 | 6.02 / 0.28 | 6.01 / 0.27 | 6.15 / 0.23 |
| PN raw | 6.70 / 0.24 | 4.49 / 0.32 | 5.65 / 0.26 | 6.06 / 0.22 |

## σ = 0.1 m / jink

| 誘導則 | calm miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|
| Predictive | 1.26 / 0.67 | 40.53 / 0.00 | 44.01 / 0.00 | 38.82 / 0.00 |
| CrabPredictive airdata | 1.27 / 0.71 | 3.80 / 0.42 | 4.09 / 0.41 | 4.60 / 0.32 |
| CrabPredictive oracle | 1.27 / 0.67 | 4.90 / 0.40 | 5.67 / 0.37 | 5.94 / 0.27 |
| CrabPursuit airdata | 2.68 / 0.51 | 3.98 / 0.38 | 3.87 / 0.38 | 5.31 / 0.31 |
| PN raw | 1.45 / 0.69 | 6.20 / 0.29 | 7.66 / 0.23 | 6.60 / 0.21 |

## σ = 0.1 m / sustained

| 誘導則 | calm miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|
| Predictive | 1.23 / 0.91 | 41.18 / 0.00 | 44.63 / 0.00 | 40.57 / 0.00 |
| CrabPredictive airdata | 1.23 / 0.93 | 1.96 / 0.61 | 2.05 / 0.60 | 3.40 / 0.42 |
| CrabPredictive oracle | 1.22 / 0.92 | 1.89 / 0.64 | 2.01 / 0.62 | 3.34 / 0.41 |
| CrabPursuit airdata | 3.57 / 0.38 | 3.96 / 0.36 | 4.33 / 0.34 | 4.75 / 0.28 |
| PN raw | 1.36 / 0.85 | 4.29 / 0.39 | 4.69 / 0.38 | 5.94 / 0.27 |

## σ = 0.1 m / two_target

| 誘導則 | calm miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|
| Predictive | 5.02 / 0.38 | 33.52 / 0.00 | 36.75 / 0.00 | 33.23 / 0.00 |
| CrabPredictive airdata | 3.99 / 0.41 | 2.35 / 0.57 | 2.18 / 0.56 | 4.04 / 0.37 |
| CrabPredictive oracle | 5.02 / 0.38 | 4.50 / 0.40 | 5.10 / 0.37 | 5.29 / 0.26 |
| CrabPursuit airdata | 10.18 / 0.26 | 5.76 / 0.28 | 5.77 / 0.27 | 5.93 / 0.21 |
| PN raw | 6.47 / 0.25 | 4.47 / 0.34 | 5.54 / 0.28 | 5.40 / 0.29 |

## 読み

### oracle の上限

jink / cross10: 空力推定 0.40、真値 0.39、PN raw 0.25。
jink / cross10_gust: 空力推定 0.29、真値 0.24、PN raw 0.20。

真値 0.24 が空力推定 0.29 に届かない。風推定のローパスがガストを滑らかにし、指令が安定している可能性がある。

### 持続マニューバ

Predictive: jink 0.00 → sustained 0.00（横風 10 m/s、σ=0.5）
CrabPredictive airdata: jink 0.40 → sustained 0.60（横風 10 m/s、σ=0.5）
PN raw: jink 0.25 → sustained 0.36（横風 10 m/s、σ=0.5）
持続織りでも crab が PN を上回る（0.60 vs 0.36）。バースト固有の結果ではない。

### 2 目標

Predictive: 単目標 0.00 → 2 目標 0.00（主目標殺傷、横風、σ=0.5）
CrabPredictive airdata: 単目標 0.40 → 2 目標 0.47（主目標殺傷、横風、σ=0.5）
PN raw: 単目標 0.25 → 2 目標 0.32（主目標殺傷、横風、σ=0.5）
主目標は正面の衝突コースのまま。decoy は横 80 m を横切るのみで、誘導は主目標を見続けている。差が出ないのは設計どおり。目標切替を入れた場合は別の実験になる。

## モデルの限界

- 2 目標では主目標だけを誘導・採点している。decoy への切替や資源配分の意思決定は入れていない。
- sustained は水平サインのみ。持続バレルロールではない。
- oracle は瞬時風真値。実機の上限。

## 出力

- `results/guidance_followup/summary.json`
- `results/guidance_followup/trials.jsonl`
- `results/guidance_followup/pk_compare.png`
- `results/guidance_followup/REPORT.md`
