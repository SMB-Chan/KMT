# 誘導則の頑健性と、ビームのセンサ閾値

## 何を確かめたか

誘導則比較（v4）の続き。残っていた問いは2つ。

1. 回避での PN ≈ Predictive ≫ APN という順位は、風と通信遅延が入っても持つか。
2. ビームを 3 m（衝撃半径）以内に入れるセンサは、σ=0.5 と σ=0.1 のどこにあるか。

風は迎撃機の地表速度にだけ効く（標的はマルチコプターとして対地速度を保持）。誘導則は風を推定しないため、横風は未モデル化バイアスになる。通信遅延は「観測サンプルの時刻を t − τ に遅らせる」。ノイズはサンプル時点で引いてあるので、遅れた測位は古く、かつ汚れている。

## 設定

- 誘導則: Pure Pursuit, PN raw, PN filtered, APN filtered, Predictive
- 回避: head-on ジング（v4 と同じ `evasive_heading_fn`）、距離 [500, 1000, 2000] m × 速度 [10, 15, 20] m/s × 各 50 試行、条件ごと n=450
- 対比較: ジングとセンサ乱数を誘導則・条件のすべてで共有（seed に law / cond を含めない）
- 風・遅延条件: 9 種（下表）
- ビーム閾値: σ ∈ [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]、誘導則 pred, pn_raw, pure、条件 ['calm', 'cross10_d150']
- センサ: 位置 σ、速度 0.4σ。殺傷は同一最接近に対し衝撃弾頭を解析採点
- 回避 40500 + ビーム 21600 = 62100 交戦、実行 0.0 s、ワーカ 10

### 風・遅延条件

| id | 風 (m/s) | 遅延 | ガスト RMS | 意味 |
|---|---|---|---|---|
| calm | (0, 0) | 0 ms | 0 m/s | baseline |
| cross5 | (0, 5) | 0 ms | 0 m/s | 5 m/s crosswind |
| cross10 | (0, 10) | 0 ms | 0 m/s | 10 m/s crosswind |
| along10 | (10, 0) | 0 ms | 0 m/s | 10 m/s along-track |
| d50 | (0, 0) | 50 ms | 0 m/s | 50 ms link |
| d150 | (0, 0) | 150 ms | 0 m/s | 150 ms link |
| d300 | (0, 0) | 300 ms | 0 m/s | 300 ms link |
| cross10_d150 | (0, 10) | 150 ms | 0 m/s | 10 m/s cross + 150 ms |
| cross10_gust | (0, 10) | 0 ms | 2 m/s | 10 m/s cross + 2 m/s gust |

## 1. 回避 — 順位は風と遅延に耐えるか

各セル n=450（距離 3 × 速度 3 × 50 試行）。衝撃 mean Pk の標準誤差 ≈ 0.02（n=450 のとき）。

### σ = 0.5 m

| 誘導則 | calm miss / Pk | cross5 miss / Pk | cross10 miss / Pk | along10 miss / Pk | d50 miss / Pk | d150 miss / Pk | d300 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|---|---|---|---|---|
| Pure Pursuit | 2.95 / 0.48 | 25.52 / 0.03 | 48.19 / 0.03 | 2.80 / 0.50 | 3.00 / 0.48 | 3.10 / 0.47 | 3.23 / 0.47 | 51.20 / 0.03 | 45.54 / 0.02 |
| PN raw | 1.68 / 0.67 | 4.10 / 0.34 | 5.76 / 0.31 | 1.83 / 0.66 | 1.76 / 0.66 | 1.85 / 0.65 | 2.03 / 0.61 | 7.59 / 0.24 | 6.82 / 0.20 |
| PN filtered | 1.85 / 0.65 | 5.21 / 0.28 | 7.18 / 0.26 | 1.89 / 0.65 | 1.86 / 0.65 | 1.98 / 0.62 | 2.24 / 0.56 | 9.08 / 0.16 | 8.04 / 0.14 |
| APN filtered | 3.26 / 0.45 | 5.49 / 0.26 | 8.40 / 0.19 | 2.49 / 0.54 | 3.37 / 0.43 | 3.75 / 0.40 | 4.21 / 0.37 | 10.64 / 0.12 | 9.05 / 0.11 |
| Predictive | 1.61 / 0.67 | 21.96 / 0.01 | 40.46 / 0.00 | 1.89 / 0.60 | 1.64 / 0.67 | 1.68 / 0.65 | 1.73 / 0.65 | 44.05 / 0.00 | 38.77 / 0.00 |

### σ = 0.1 m

| 誘導則 | calm miss / Pk | cross5 miss / Pk | cross10 miss / Pk | along10 miss / Pk | d50 miss / Pk | d150 miss / Pk | d300 miss / Pk | cross10_d150 miss / Pk | cross10_gust miss / Pk |
|---|---|---|---|---|---|---|---|---|---|
| Pure Pursuit | 2.74 / 0.49 | 25.79 / 0.02 | 49.23 / 0.04 | 2.48 / 0.54 | 2.81 / 0.49 | 2.92 / 0.49 | 3.06 / 0.48 | 52.21 / 0.03 | 47.70 / 0.02 |
| PN raw | 1.44 / 0.68 | 4.34 / 0.36 | 6.14 / 0.28 | 1.38 / 0.72 | 1.55 / 0.67 | 1.66 / 0.64 | 1.90 / 0.60 | 7.37 / 0.22 | 7.05 / 0.21 |
| PN filtered | 1.62 / 0.64 | 5.15 / 0.29 | 7.11 / 0.22 | 1.51 / 0.70 | 1.68 / 0.63 | 1.86 / 0.61 | 2.14 / 0.57 | 8.80 / 0.14 | 8.36 / 0.16 |
| APN filtered | 3.14 / 0.46 | 5.11 / 0.31 | 8.95 / 0.18 | 2.21 / 0.59 | 3.27 / 0.45 | 3.45 / 0.45 | 3.80 / 0.41 | 10.88 / 0.13 | 8.67 / 0.17 |
| Predictive | 1.22 / 0.71 | 22.06 / 0.01 | 40.57 / 0.00 | 1.56 / 0.65 | 1.25 / 0.70 | 1.30 / 0.69 | 1.40 / 0.68 | 44.19 / 0.00 | 38.85 / 0.00 |

### 読み — 回避

平静（calm、σ=0.5）の衝撃 Pk は Pure Pursuit 0.48、APN 0.45、PN raw 0.67、Predictive 0.67。v4 の対比較と同じ並びで、ジングとセンサ乱数の共有方法が変わっていないことを確認できる。

σ=0.5 の順位（衝撃 Pk 降順）:

| 条件 | 1位 | 2位 | 3位 | 4位 | 5位 |
|---|---|---|---|---|---|
| calm | PN raw (0.67) | Predictive (0.67) | PN filtered (0.65) | Pure Pursuit (0.48) | APN filtered (0.45) |
| cross5 | PN raw (0.34) | PN filtered (0.28) | APN filtered (0.26) | Pure Pursuit (0.03) | Predictive (0.01) |
| cross10 | PN raw (0.31) | PN filtered (0.26) | APN filtered (0.19) | Pure Pursuit (0.03) | Predictive (0.00) |
| along10 | PN raw (0.66) | PN filtered (0.65) | Predictive (0.60) | APN filtered (0.54) | Pure Pursuit (0.50) |
| d50 | Predictive (0.67) | PN raw (0.66) | PN filtered (0.65) | Pure Pursuit (0.48) | APN filtered (0.43) |
| d150 | Predictive (0.65) | PN raw (0.65) | PN filtered (0.62) | Pure Pursuit (0.47) | APN filtered (0.40) |
| d300 | Predictive (0.65) | PN raw (0.61) | PN filtered (0.56) | Pure Pursuit (0.47) | APN filtered (0.37) |
| cross10_d150 | PN raw (0.24) | PN filtered (0.16) | APN filtered (0.12) | Pure Pursuit (0.03) | Predictive (0.00) |
| cross10_gust | PN raw (0.20) | PN filtered (0.14) | APN filtered (0.11) | Pure Pursuit (0.02) | Predictive (0.00) |

遅延だけ（d50 → d150 → d300）では PN raw が 0.66 → 0.65 → 0.61、Predictive が 0.67 → 0.65 → 0.65、APN が 0.43 → 0.40 → 0.37。300 ms でも順位は calm のままで、遅延だけなら v4 の結論が持つ。APN の加速度推定は元から遅れているので、リンク遅延の追加が効きやすい。

横風 10 m/s では順位が崩れる。Predictive の衝撃 Pk は calm 0.67 から 0.00 に落ち、miss 中央は 40.46 m。同じ横風で PN raw は Pk 0.31、miss 中央 5.76 m。衝突三角形は対地の交点を狙うが、誘導は風を推定していないので、機首を交点へ向けたまま横に流される。リードが全部効かなくなる。PN は視線角速度を打ち消すので、風が生む見かけの角速度にも同じ舵が当たり、崩れ方が小さく残る（APN は 0.19 で、遅れた加速度推定が重なる）。

したがって「PN ≈ Predictive ≫ APN」は平静と遅延では持つが、横風下では **PN ≫ Predictive** に変わる。APN が最下位から動くのは、Predictive が風で先に崩れたセルだけである。

条件ごとの calm からの低下（σ=0.5、衝撃 Pk）:

| 条件 | Pure Pursuit | PN raw | APN | Predictive |
|---|---|---|---|---|
| cross5 | -0.45 | -0.33 | -0.20 | -0.67 |
| cross10 | -0.45 | -0.37 | -0.26 | -0.67 |
| along10 | +0.02 | -0.02 | +0.09 | -0.07 |
| d50 | -0.00 | -0.01 | -0.02 | -0.00 |
| d150 | -0.01 | -0.03 | -0.06 | -0.02 |
| d300 | -0.01 | -0.07 | -0.08 | -0.03 |
| cross10_d150 | -0.46 | -0.43 | -0.33 | -0.67 |
| cross10_gust | -0.46 | -0.48 | -0.34 | -0.67 |

全条件での最小 Pk は Pure Pursuit 0.02、APN 0.11、PN raw 0.20、Predictive 0.00。

## 2. ビーム — 3 m に入れる σ

ミス中央値と P(miss < 3 m)、衝撃 mean Pk。閾値 σ* は中央値が 3 m を横切る区間を線形補間した。

### 条件 calm

| 誘導則 | σ | miss 中央 | P(miss<3) | 衝撃 Pk | σ* (miss 中央 = 3 m) |
|---|---|---|---|---|---|
| Predictive | 0.10 | 1.44 m | 66% | 0.64 | 0.305 |
| Predictive | 0.15 | 1.67 m | 63% | 0.59 |  |
| Predictive | 0.20 | 2.18 m | 58% | 0.53 |  |
| Predictive | 0.25 | 2.39 m | 58% | 0.50 |  |
| Predictive | 0.30 | 2.94 m | 50% | 0.42 |  |
| Predictive | 0.35 | 3.51 m | 45% | 0.40 |  |
| Predictive | 0.40 | 4.02 m | 43% | 0.37 |  |
| Predictive | 0.50 | 5.06 m | 33% | 0.28 |  |
| PN raw | 0.10 | 2.11 m | 69% | 0.52 | 0.308 |
| PN raw | 0.15 | 2.54 m | 66% | 0.48 |  |
| PN raw | 0.20 | 2.78 m | 57% | 0.42 |  |
| PN raw | 0.25 | 2.81 m | 58% | 0.41 |  |
| PN raw | 0.30 | 2.96 m | 51% | 0.34 |  |
| PN raw | 0.35 | 3.20 m | 44% | 0.34 |  |
| PN raw | 0.40 | 3.35 m | 43% | 0.32 |  |
| PN raw | 0.50 | 4.00 m | 34% | 0.24 |  |
| Pure Pursuit | 0.10 | 6.85 m | 26% | 0.12 | median never reaches 3 m on this ladder |
| Pure Pursuit | 0.15 | 6.86 m | 22% | 0.11 |  |
| Pure Pursuit | 0.20 | 6.89 m | 17% | 0.10 |  |
| Pure Pursuit | 0.25 | 6.81 m | 15% | 0.07 |  |
| Pure Pursuit | 0.30 | 6.89 m | 13% | 0.06 |  |
| Pure Pursuit | 0.35 | 7.00 m | 12% | 0.05 |  |
| Pure Pursuit | 0.40 | 7.27 m | 12% | 0.06 |  |
| Pure Pursuit | 0.50 | 7.59 m | 8% | 0.05 |  |

### 条件 cross10_d150

| 誘導則 | σ | miss 中央 | P(miss<3) | 衝撃 Pk | σ* (miss 中央 = 3 m) |
|---|---|---|---|---|---|
| Predictive | 0.10 | 35.20 m | 0% | 0.00 | median never reaches 3 m on this ladder |
| Predictive | 0.15 | 35.21 m | 0% | 0.00 |  |
| Predictive | 0.20 | 35.16 m | 0% | 0.00 |  |
| Predictive | 0.25 | 35.19 m | 0% | 0.00 |  |
| Predictive | 0.30 | 35.23 m | 0% | 0.00 |  |
| Predictive | 0.35 | 35.25 m | 0% | 0.00 |  |
| Predictive | 0.40 | 35.21 m | 0% | 0.00 |  |
| Predictive | 0.50 | 35.24 m | 0% | 0.00 |  |
| PN raw | 0.10 | 8.07 m | 29% | 0.22 | median never reaches 3 m on this ladder |
| PN raw | 0.15 | 8.08 m | 27% | 0.20 |  |
| PN raw | 0.20 | 8.09 m | 24% | 0.17 |  |
| PN raw | 0.25 | 8.18 m | 22% | 0.16 |  |
| PN raw | 0.30 | 8.18 m | 17% | 0.10 |  |
| PN raw | 0.35 | 8.47 m | 16% | 0.12 |  |
| PN raw | 0.40 | 8.58 m | 15% | 0.10 |  |
| PN raw | 0.50 | 9.04 m | 10% | 0.06 |  |
| Pure Pursuit | 0.10 | 11.48 m | 8% | 0.02 | median never reaches 3 m on this ladder |
| Pure Pursuit | 0.15 | 11.49 m | 7% | 0.01 |  |
| Pure Pursuit | 0.20 | 11.51 m | 6% | 0.01 |  |
| Pure Pursuit | 0.25 | 11.60 m | 4% | 0.01 |  |
| Pure Pursuit | 0.30 | 11.56 m | 4% | 0.01 |  |
| Pure Pursuit | 0.35 | 11.74 m | 2% | 0.00 |  |
| Pure Pursuit | 0.40 | 11.90 m | 3% | 0.01 |  |
| Pure Pursuit | 0.50 | 12.03 m | 3% | 0.00 |  |

### 読み — ビームの閾値

**calm**

- Predictive: σ* ≈ **0.31**（中央値 3 m。σ=0.10 で 1.44 m、σ=0.50 で 5.06 m、σ* で Pk は約 0.42）
- PN raw: σ* ≈ **0.31**（中央値 3 m。σ=0.10 で 2.11 m、σ=0.50 で 4.00 m、σ* で Pk は約 0.34）
- Pure Pursuit: median never reaches 3 m on this ladder（σ=0.10 で 6.85 m、σ=0.50 で 7.59 m）

**cross10_d150**

- Predictive: median never reaches 3 m on this ladder（σ=0.10 で 35.20 m、σ=0.50 で 35.24 m）
- PN raw: median never reaches 3 m on this ladder（σ=0.10 で 8.07 m、σ=0.50 で 9.04 m）
- Pure Pursuit: median never reaches 3 m on this ladder（σ=0.10 で 11.48 m、σ=0.50 で 12.03 m）

σ=0.50 のビームは、v4 と同じく中央値が 3 m の外（Predictive 5.06 m、PN raw 4.00 m）。σ=0.10 の Predictive は 1.44 m まで入り、P(miss<3) は 66%。

衝撃がビームで使える側（中央値 3 m）に入る境目は、Predictive で σ ≈ 0.31 m、PN raw で σ ≈ 0.31 m にある。これは v4 で書いた「σ=0.5 と 0.1 の間」を数字で割ったもので、GPS 級（σ=0.5）では届かず、レーダ／高精度測距（σ ≲ 0.2–0.3）が要る。

横風 10 m/s + 150 ms では、この σ ラダーのどこでもミス中央値が 3 m に入らない。センサを良くしても、風と遅延の未モデル化が先に効く。

## モデルの限界

- 風は一定 + 水平 AR(1) ガスト。鉛直シアとドップラーレーダの風推定は入れていない。
- 標的は対地速度を保持するマルチコプター。固定翼なら `target_drift` を上げる必要がある。
- 誘導則はすべて風非推定。風を推定してバンクを補う誘導は別物になる。
- 通信遅延は観測時刻の平行移動のみ。途絶・欠測・レート制限はない。
- 回避ジングは v4 と同じ 0.5 Hz × 5 回。持続マニューバではない。
- 弾頭曲線は v3/v4 のまま（衝撃半径 3 m・1500 J）。

## 出力

- `results/guidance_robustness/evasive_robust.json` — 回避の条件×誘導則集計
- `results/guidance_robustness/beam_threshold.json` — ビームの σ ラダーと σ*
- `results/guidance_robustness/evasive_heatmap.png` — 衝撃 Pk の条件×誘導則
- `results/guidance_robustness/beam_threshold.png` — ミス中央値と σ
- `results/guidance_robustness/trials.jsonl` — 試行ごと
- `results/guidance_robustness/REPORT.md`
