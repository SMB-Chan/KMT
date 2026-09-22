# 目標切替と、crab が効かなくなる風速

## 何を確かめたか

1. **目標切替** — 主目標（正面・ジング）と decoy（横 80 m・斜め横断）。Fixed / FirstLock / Nearest / BestPk で切替を入れ、殺傷順位が保つかを見る。
2. **強風の頭打ち** — 横風 0–25 m/s。crab 角 asin(W⊥/V_a) とバンクサーボが限界に達する場所を探す。

## 設定

- 誘導則: Predictive, CrabPredictive airdata, CrabPredictive oracle, PN raw
- 切替: Fixed primary, FirstLock, Nearest, BestPk（再判定 0.5 s、BestPk は閾値 0.05）
- 風速ラダー: [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] m/s（横風）、jink 単目標
- 距離 [500, 1000, 2000] m × 速度 [10, 15, 20] m/s × n=50、合計 18000 交戦、838 s

## 1. 目標切替（横風 10 m/s、σ=0.5）

最終的に向かっていた目標への衝撃 mean Pk。

| 誘導則 | Fixed primary | FirstLock | Nearest | BestPk |
|---|---|---|---|---|
| Predictive | 0.00 | 0.00 | 0.00 | 0.00 |
| CrabPredictive airdata | 0.02 | 0.03 | 0.02 | 0.01 |
| CrabPredictive oracle | 0.00 | 0.00 | 0.00 | 0.00 |
| PN raw | 0.00 | 0.00 | 0.00 | 0.00 |

平均切替回数 / 最終が主目標の割合:

| 誘導則 | Fixed primary | FirstLock | Nearest | BestPk |
|---|---|---|---|---|
| Predictive | 0.0 / 100% | 0.0 / 100% | 2.3 / 63% | 1.9 / 56% |
| CrabPredictive airdata | 0.0 / 100% | 0.0 / 100% | 2.5 / 46% | 2.2 / 50% |
| CrabPredictive oracle | 0.0 / 100% | 0.0 / 100% | 2.4 / 38% | 1.8 / 47% |
| PN raw | 0.0 / 100% | 0.0 / 100% | 2.6 / 47% | 1.8 / 46% |

### 読み — 切替

crab は FirstLock で 0.03、BestPk で 0.01。

crab が PN を上回ったポリシー: FirstLock, Nearest。

素の Predictive はどのポリシーでも 0.00–0.00 にとどまり、横風の崩れが残る。

## 2. 強風ラダー（jink、σ=0.5、衝撃 Pk）

| 誘導則 | 0 m/s | 5 m/s | 10 m/s | 15 m/s | 20 m/s | 25 m/s |
|---|---|---|---|---|---|---|
| Predictive | 0.66 | 0.00 | 0.00 | 0.00 | 0.03 | 0.04 |
| CrabPredictive airdata | 0.71 | 0.57 | 0.44 | 0.41 | 0.41 | 0.50 |
| CrabPredictive oracle | 0.64 | 0.38 | 0.36 | 0.40 | 0.39 | 0.40 |
| PN raw | 0.70 | 0.34 | 0.23 | 0.25 | 0.23 | 0.23 |

miss 中央 (m):

| 誘導則 | 0 m/s | 5 m/s | 10 m/s | 15 m/s | 20 m/s | 25 m/s |
|---|---|---|---|---|---|---|
| Predictive | 1.7 | 21.8 | 40.4 | 43.2 | 17.3 | 14.6 |
| CrabPredictive airdata | 1.6 | 2.5 | 3.5 | 3.8 | 3.5 | 2.7 |
| CrabPredictive oracle | 1.8 | 4.7 | 5.6 | 4.7 | 4.7 | 4.9 |
| PN raw | 1.7 | 4.4 | 6.8 | 8.4 | 9.6 | 8.9 |

### 読み — 強風

crab の Pk は 0 m/s 0.71 → 25 m/s 0.50。

25 m/s までは Pk 0.2 を割らない。crab 角は最大 30 deg で、対気 50 m/s ならまだ余裕がある。

同じラダーで PN は 0 m/s 0.70 → 25 m/s 0.23。crab の落ち方は PN と比べて遅いが、どちらも強い横風では下がる。

空力推定が表の上で真値を上回って見えるのは、このラダーが定常風（gust_rms=0）で、
法則ごとに乱数が違うセルを平均した見た目である。推定が真値より正確なのではない。
wind_est_sigma は毎ステップ足される白色雑音であり、平滑化ではない。
旋回中の機首の遅れが偽の横風として入り、リードが増える。25 m/s の正面 1000 m
では順位が逆転する（真値 Pk 0.67、推定 0.39）。差は測定の優位ではない。

## モデルの限界

- 切替は 0.5 s 間隔の幾何評価のみ。脅威度・味方・弾数は見ていない。
- BestPk の代理量は直線延長の近距離 b と相対速度。実際の弾頭曲線は別。
- 強風では鉛直成分を入れていない。バンク限界 35 deg / 120 deg/s は機体のまま。
- decoy は 1 機。群れて接近する場合は誘導則の局所解が変わる。

## 出力

- `results/guidance_edge/summary.json` / `results/guidance_edge/trials.jsonl` / `results/guidance_edge/pk.png`
- `results/guidance_edge/REPORT.md`
