# 風補償で Predictive を戻せるか

## 何を確かめたか

v5 は、横風 10 m/s で Predictive の衝撃 Pk が 0.67 → 0.00 に崩れ、PN だけが 0.31 で残ることを示した。衝突三角形は対地の交点を狙うが、指令は対気機首なので風で流れる。

続きとして、**機首を風に向け直す（crab）**ことで、そのリードを取り戻せるかを見る。衝突三角形は気団基準で解き、ψ = 方向(必要な対気速度) とする。風は (a) 空速計+INS から毎ステップ推定する（airdata）か、(b) 真値を与える（oracle、補償の上限）。

## 設定

- 誘導則: Predictive, CrabPredictive airdata, CrabPredictive oracle, CrabPursuit airdata, PN raw
- crab: 気団基準の衝突三角形 |r+(v_t−W)t| = V_a t を解き、対気速度 (aim−r_i)/t_go − W の向きを機首にする
- 風推定の誤差 σ_W = 0.5 m/s（空速計+INS の代表値）
- 回避: v5 と同じジング対比較、条件ごと n=450
- ビーム: σ ∈ [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]、条件 ['calm', 'cross10_d150']
- 回避 22500 + ビーム 28800 = 51300 交戦、1122 s

### 条件

| id | 風 (m/s) | 遅延 | ガスト |
|---|---|---|---|
| calm | (0, 0) | 0 ms | 0 m/s |
| cross5 | (0, 5) | 0 ms | 0 m/s |
| cross10 | (0, 10) | 0 ms | 0 m/s |
| cross10_d150 | (0, 10) | 150 ms | 0 m/s |
| cross10_gust | (0, 10) | 0 ms | 2 m/s |

## 1. 回避 — crab は Predictive を戻すか

miss 中央 / 衝撃 mean Pk（n=450）。

### σ = 0.5 m

| 誘導則 | calm miss / Pk | cross5 miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|---|
| Predictive | 1.63 / 0.67 | 22.14 / 0.01 | 40.50 / 0.00 | 44.27 / 0.00 | 39.05 / 0.00 |
| CrabPredictive airdata | 1.68 / 0.69 | 2.41 / 0.58 | 3.94 / 0.40 | 4.16 / 0.39 | 4.86 / 0.31 |
| CrabPredictive oracle | 1.63 / 0.67 | 4.29 / 0.41 | 5.47 / 0.40 | 6.11 / 0.38 | 7.64 / 0.19 |
| CrabPursuit airdata | 2.89 / 0.48 | 3.21 / 0.45 | 4.02 / 0.35 | 4.14 / 0.34 | 6.00 / 0.24 |
| PN raw | 1.85 / 0.66 | 4.43 / 0.34 | 6.36 / 0.26 | 8.29 / 0.22 | 6.88 / 0.19 |

### σ = 0.1 m

| 誘導則 | calm miss / Pk | cross5 miss / Pk | cross10 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|---|
| Predictive | 1.38 / 0.67 | 22.16 / 0.01 | 40.62 / 0.00 | 44.44 / 0.00 | 39.77 / 0.00 |
| CrabPredictive airdata | 1.31 / 0.69 | 2.17 / 0.61 | 3.62 / 0.45 | 3.88 / 0.43 | 4.66 / 0.30 |
| CrabPredictive oracle | 1.38 / 0.67 | 3.75 / 0.43 | 4.71 / 0.42 | 5.39 / 0.40 | 6.99 / 0.22 |
| CrabPursuit airdata | 2.36 / 0.52 | 2.73 / 0.50 | 3.94 / 0.39 | 3.86 / 0.39 | 5.72 / 0.26 |
| PN raw | 1.45 / 0.69 | 4.09 / 0.36 | 6.00 / 0.27 | 8.10 / 0.23 | 6.60 / 0.22 |

### 読み — 回避

横風 10 m/s（σ=0.5）で、素の Predictive は Pk 0.00（miss 中央 40.50 m）。crab を入れると空力推定版が 0.40（3.94 m）、真値版が 0.40（5.47 m）。同じ条件の PN raw は 0.26。

平静では crab 空力推定版 0.69、素の Predictive 0.67。crab は無風では過補償にならず、風下ではリードを取り戻す。

crab 版は横風で PN raw（0.26）を上回る。風が既知に近ければ、衝突三角形は PN より優位に戻る。

横風 + 150 ms では素の Predictive 0.00、crab 空力推定 0.39、crab 真値 0.38、PN raw 0.22。遅延が重なると風推定も遅れる。

## 2. ビーム — crab で σ* は動くか

### 条件 calm

| 誘導則 | σ | miss 中央 | P(miss<3) | 衝撃 Pk | σ* |
|---|---|---|---|---|---|
| Predictive | 0.10 | 1.34 m | 66% | 0.64 | 0.292 |
| Predictive | 0.15 | 1.68 m | 63% | 0.59 |  |
| Predictive | 0.20 | 2.23 m | 57% | 0.53 |  |
| Predictive | 0.25 | 2.46 m | 56% | 0.49 |  |
| Predictive | 0.30 | 3.11 m | 49% | 0.44 |  |
| Predictive | 0.35 | 3.62 m | 46% | 0.40 |  |
| Predictive | 0.40 | 4.16 m | 41% | 0.34 |  |
| Predictive | 0.50 | 5.07 m | 33% | 0.28 |  |
| CrabPredictive airdata | 0.10 | 1.06 m | 88% | 0.84 | 0.500 |
| CrabPredictive airdata | 0.15 | 1.25 m | 85% | 0.79 |  |
| CrabPredictive airdata | 0.20 | 1.52 m | 79% | 0.72 |  |
| CrabPredictive airdata | 0.25 | 1.80 m | 78% | 0.68 |  |
| CrabPredictive airdata | 0.30 | 1.97 m | 70% | 0.63 |  |
| CrabPredictive airdata | 0.35 | 2.22 m | 67% | 0.57 |  |
| CrabPredictive airdata | 0.40 | 2.42 m | 61% | 0.50 |  |
| CrabPredictive airdata | 0.50 | 2.77 m | 53% | 0.44 |  |
| CrabPredictive oracle | 0.10 | 1.34 m | 66% | 0.64 | 0.292 |
| CrabPredictive oracle | 0.15 | 1.68 m | 63% | 0.59 |  |
| CrabPredictive oracle | 0.20 | 2.23 m | 57% | 0.53 |  |
| CrabPredictive oracle | 0.25 | 2.46 m | 56% | 0.49 |  |
| CrabPredictive oracle | 0.30 | 3.11 m | 49% | 0.44 |  |
| CrabPredictive oracle | 0.35 | 3.62 m | 46% | 0.40 |  |
| CrabPredictive oracle | 0.40 | 4.16 m | 41% | 0.34 |  |
| CrabPredictive oracle | 0.50 | 5.07 m | 33% | 0.28 |  |
| PN raw | 0.10 | 1.96 m | 71% | 0.53 | 0.300 |
| PN raw | 0.15 | 2.60 m | 66% | 0.48 |  |
| PN raw | 0.20 | 2.75 m | 57% | 0.43 |  |
| PN raw | 0.25 | 2.81 m | 56% | 0.41 |  |
| PN raw | 0.30 | 3.00 m | 50% | 0.35 |  |
| PN raw | 0.35 | 3.23 m | 46% | 0.33 |  |
| PN raw | 0.40 | 3.38 m | 42% | 0.29 |  |
| PN raw | 0.50 | 3.97 m | 36% | 0.26 |  |

### 条件 cross10_d150

| 誘導則 | σ | miss 中央 | P(miss<3) | 衝撃 Pk | σ* |
|---|---|---|---|---|---|
| Predictive | 0.10 | 35.20 m | 0% | 0.00 | median never reaches 3 m |
| Predictive | 0.15 | 35.21 m | 0% | 0.00 |  |
| Predictive | 0.20 | 35.21 m | 0% | 0.00 |  |
| Predictive | 0.25 | 35.25 m | 0% | 0.00 |  |
| Predictive | 0.30 | 35.22 m | 0% | 0.00 |  |
| Predictive | 0.35 | 35.20 m | 0% | 0.00 |  |
| Predictive | 0.40 | 35.24 m | 0% | 0.00 |  |
| Predictive | 0.50 | 35.30 m | 0% | 0.00 |  |
| CrabPredictive airdata | 0.10 | 1.25 m | 99% | 0.95 | 0.500 |
| CrabPredictive airdata | 0.15 | 1.39 m | 95% | 0.88 |  |
| CrabPredictive airdata | 0.20 | 1.60 m | 90% | 0.78 |  |
| CrabPredictive airdata | 0.25 | 1.75 m | 88% | 0.73 |  |
| CrabPredictive airdata | 0.30 | 1.81 m | 80% | 0.68 |  |
| CrabPredictive airdata | 0.35 | 1.95 m | 74% | 0.61 |  |
| CrabPredictive airdata | 0.40 | 2.10 m | 68% | 0.54 |  |
| CrabPredictive airdata | 0.50 | 2.39 m | 61% | 0.48 |  |
| CrabPredictive oracle | 0.10 | 1.68 m | 77% | 0.69 | 0.373 |
| CrabPredictive oracle | 0.15 | 1.90 m | 72% | 0.62 |  |
| CrabPredictive oracle | 0.20 | 2.20 m | 67% | 0.54 |  |
| CrabPredictive oracle | 0.25 | 2.39 m | 64% | 0.48 |  |
| CrabPredictive oracle | 0.30 | 2.48 m | 57% | 0.44 |  |
| CrabPredictive oracle | 0.35 | 2.79 m | 53% | 0.39 |  |
| CrabPredictive oracle | 0.40 | 3.24 m | 47% | 0.34 |  |
| CrabPredictive oracle | 0.50 | 3.96 m | 40% | 0.28 |  |
| PN raw | 0.10 | 8.09 m | 29% | 0.22 | median never reaches 3 m |
| PN raw | 0.15 | 8.15 m | 27% | 0.20 |  |
| PN raw | 0.20 | 8.05 m | 22% | 0.17 |  |
| PN raw | 0.25 | 8.11 m | 21% | 0.15 |  |
| PN raw | 0.30 | 8.23 m | 18% | 0.12 |  |
| PN raw | 0.35 | 8.38 m | 16% | 0.12 |  |
| PN raw | 0.40 | 8.51 m | 14% | 0.09 |  |
| PN raw | 0.50 | 9.11 m | 12% | 0.08 |  |

### 読み — ビーム

**calm**

- Predictive: σ* ≈ **0.29**（σ=0.10 で 1.34 m、σ=0.50 で 5.07 m）
- CrabPredictive airdata: σ* ≈ **0.50**（σ=0.10 で 1.06 m、σ=0.50 で 2.77 m）
- CrabPredictive oracle: σ* ≈ **0.29**（σ=0.10 で 1.34 m、σ=0.50 で 5.07 m）
- PN raw: σ* ≈ **0.30**（σ=0.10 で 1.96 m、σ=0.50 で 3.97 m）

**cross10_d150**

- Predictive: 3 m に入らない（σ=0.10 で 35.20 m）
- CrabPredictive airdata: σ* ≈ **0.50**（σ=0.10 で 1.25 m、σ=0.50 で 2.39 m）
- CrabPredictive oracle: σ* ≈ **0.37**（σ=0.10 で 1.68 m、σ=0.50 で 3.96 m）
- PN raw: 3 m に入らない（σ=0.10 で 8.09 m）

v5 では横風+150 ms の σ ラダー上どこでも miss 中央が 3 m に入らなかった。

crab 真値では σ* ≈ 0.37 m まで動く。風補償そのものの利得はここにある。
0.50 は空力推定経路の値であり、風補償の利得としては読まない。
風推定版の σ* は 0.50 m。
参考: 平静の crab 空力推定版 σ* は 0.50 m。

## モデルの限界

- 風推定は機体の対気姿勢と対地速度から作る（空速計+INS）。旋回中の機体-風速のずれが残る。
- oracle は真値の風。実機では届かない上限。
- crab は水平面の 1 自由度。バンク限界を超える風では頭打ち。
- 弾頭・ジング・センサモデルは v3–v5 のまま。

## 出力

- `results/guidance_windcomp/evasive_windcomp.json`
- `results/guidance_windcomp/beam_windcomp.json`
- `results/guidance_windcomp/evasive_pk.png` / `results/guidance_windcomp/beam_threshold.png`
- `results/guidance_windcomp/trials.jsonl`
- `results/guidance_windcomp/REPORT.md`
