# 衝突コースに載せた目標切替

## 何を確かめたか

前回（guidance_edge）の 2 目標は、迎撃機が y=40・速度 (45, −22) でどちらの衝突コースにも乗っておらず、全ポリシーの Pk が 0.03 以下になって切替の問いが成立しなかった。

今回は迎撃機を主目標の正面衝突コースに載せる。主目標はジング、decoy は横 80 m を定常横断。切替が効くなら BestPk はジングする主目標を捨てて decoy に切り、Fixed / FirstLock は残るはずである。

最初の 14400 交戦は高度保持がなく、重力で迎撃機が落ちて衝突コースでもミス中央が 35 m 前後だった。その表は切替を測っていない。本報告は、単目標と同じ高度保持を入れた再実行である。

## 設定

- 誘導則: Predictive, CrabPredictive airdata, CrabPredictive oracle, PN raw
- 切替: Fixed primary, FirstLock, Nearest, BestPk（再判定 0.5 s）
- 条件: calm / cross10（定常、gust_rms=0）、σ=0.5
- 距離 [500, 1000, 2000] m × 速度 [10, 15, 20] m/s × n=50
- 合計 14400 交戦、1106 s

採点は「最終的に向かっていた目標」への衝撃 Pk。primary_pk / decoy_pk は切替の有無にかかわらず、各目標までの最接近で評価した参考値。

## 条件 calm

| 誘導則 | 切替 | engaged Pk | primary Pk | decoy Pk | 主ミス中央 | 切替回数 | 最終が主 |
|---|---|---|---|---|---|---|---|
| Predictive | Fixed primary | 0.704 | 0.704 | 0.020 | 1.5 m | 0.0 | 100% |
| Predictive | FirstLock | 0.675 | 0.675 | 0.013 | 1.6 m | 0.0 | 100% |
| Predictive | Nearest | 0.277 | 0.425 | 0.082 | 4.3 m | 3.3 | 39% |
| Predictive | BestPk | 0.416 | 0.638 | 0.056 | 1.9 m | 1.2 | 58% |
| CrabPredictive airdata | Fixed primary | 0.812 | 0.812 | 0.022 | 1.4 m | 0.0 | 100% |
| CrabPredictive airdata | FirstLock | 0.822 | 0.822 | 0.027 | 1.4 m | 0.0 | 100% |
| CrabPredictive airdata | Nearest | 0.416 | 0.579 | 0.125 | 2.0 m | 3.3 | 51% |
| CrabPredictive airdata | BestPk | 0.535 | 0.721 | 0.079 | 1.6 m | 1.2 | 62% |
| CrabPredictive oracle | Fixed primary | 0.666 | 0.666 | 0.008 | 1.6 m | 0.0 | 100% |
| CrabPredictive oracle | FirstLock | 0.639 | 0.639 | 0.013 | 1.9 m | 0.0 | 100% |
| CrabPredictive oracle | Nearest | 0.278 | 0.480 | 0.090 | 3.2 m | 3.4 | 40% |
| CrabPredictive oracle | BestPk | 0.416 | 0.675 | 0.055 | 1.7 m | 1.1 | 54% |
| PN raw | Fixed primary | 0.663 | 0.663 | 0.010 | 1.8 m | 0.0 | 100% |
| PN raw | FirstLock | 0.654 | 0.654 | 0.007 | 1.8 m | 0.0 | 100% |
| PN raw | Nearest | 0.284 | 0.425 | 0.060 | 4.9 m | 3.1 | 59% |
| PN raw | BestPk | 0.368 | 0.627 | 0.009 | 1.9 m | 0.9 | 59% |

## 条件 cross10

| 誘導則 | 切替 | engaged Pk | primary Pk | decoy Pk | 主ミス中央 | 切替回数 | 最終が主 |
|---|---|---|---|---|---|---|---|
| Predictive | Fixed primary | 0.049 | 0.049 | 0.060 | 20.5 m | 0.0 | 100% |
| Predictive | FirstLock | 0.034 | 0.034 | 0.075 | 21.3 m | 0.0 | 100% |
| Predictive | Nearest | 0.078 | 0.087 | 0.050 | 22.9 m | 3.2 | 53% |
| Predictive | BestPk | 0.077 | 0.085 | 0.048 | 19.0 m | 1.6 | 59% |
| CrabPredictive airdata | Fixed primary | 0.704 | 0.704 | 0.018 | 1.7 m | 0.0 | 100% |
| CrabPredictive airdata | FirstLock | 0.730 | 0.730 | 0.017 | 1.7 m | 0.0 | 100% |
| CrabPredictive airdata | Nearest | 0.265 | 0.373 | 0.107 | 3.7 m | 3.1 | 42% |
| CrabPredictive airdata | BestPk | 0.393 | 0.396 | 0.141 | 3.8 m | 2.0 | 51% |
| CrabPredictive oracle | Fixed primary | 0.452 | 0.452 | 0.023 | 3.0 m | 0.0 | 100% |
| CrabPredictive oracle | FirstLock | 0.416 | 0.416 | 0.021 | 3.9 m | 0.0 | 100% |
| CrabPredictive oracle | Nearest | 0.185 | 0.227 | 0.102 | 9.3 m | 3.3 | 46% |
| CrabPredictive oracle | BestPk | 0.215 | 0.201 | 0.142 | 10.6 m | 1.9 | 46% |
| PN raw | Fixed primary | 0.266 | 0.266 | 0.011 | 5.6 m | 0.0 | 100% |
| PN raw | FirstLock | 0.229 | 0.229 | 0.026 | 6.1 m | 0.0 | 100% |
| PN raw | Nearest | 0.089 | 0.118 | 0.053 | 12.5 m | 3.2 | 68% |
| PN raw | BestPk | 0.155 | 0.144 | 0.071 | 11.7 m | 2.1 | 45% |

## 読み

高度を保った再実行では、主目標の衝突コースは生きている。calm の Fixed は engaged Pk 0.66–0.81、主ミス中央 1.4–1.8 m。切替の問いはこの表で読める。

切ると下がる。calm の Fixed → BestPk は Predictive 0.704 → 0.416、crab 空力推定 0.812 → 0.535、真値 0.666 → 0.416、PN 0.663 → 0.368。Nearest はさらに低く 0.28–0.42。FirstLock は切替回数 0 で Fixed とほぼ同じ。decoy の Pk は Fixed のまま 0.01–0.03 で、横 80 m の定常横断は「易しい殺し」ではなかった。ジング中にそちらへ舵を切ると、当たっていた主目標を捨てる。

cross10 でも、土俵に乗っている則では同じ向きである。crab 空力推定 0.704 → 0.393、真値 0.452 → 0.215、PN 0.266 → 0.155。素の Predictive は Fixed がすでに 0.049 で、切替の差を読む側ではない。

高度保持前の表で見えた従属シグナル（空力推定だけ Pk 0.012–0.018、ミス中央 12–17 m、他は 30 m 台）は新結果ではない。迎撃機が落下しており、真値 crab は他則と同じ 30 m 台だった。高度を保った風補償の本結果は `results/guidance_windcomp/REPORT.md`（横風 10 m/s で空力推定 Pk 0.40、ミス 3.9 m）。この再実行の cross10 Fixed（空力推定 0.70、素の Predictive 0.05）はそちらの追認で、別実験の種にはしない。

## モデルの限界

- 主目標ジング・decoy 定常という非対称のみ。両方ジング、両方定常は見ていない。
- BestPk の代理量は直線延長の近距離 b と相対速度。
- decoy は 1 機。群接近では誘導の局所解が変わる。

## 出力

- `results/guidance_switch2/summary.json` / `results/guidance_switch2/trials.jsonl` / `results/guidance_switch2/pk.png`
- `results/guidance_switch2/REPORT.md`
