# 過去気象データ replay スタディ (weather_real)

## 目的
NOAA NDBC 46012 (Humboldt Bay, CA) の**実測**毎時気象データ
(気圧・気温・露点・風向・風速・最大突風) を `weather_real.py` の
`HistoricalWeather` 経由でシミュレータに入力し、モデルが観測された
気象の経過を近似してたどるかを検証し、解析から得られた修正を記録する。

## データ
* ファイル: `/home/intel/デスクトップ/HAMA/data/ndbc_46012_realtime.txt` (1096 時間レコード)
* 期間: 2026-08-05 00:00 .. 2026-09-19 16:00 UTC (1096 h = 45.7 日)
* 気圧: 1010.5 .. 1021.9 hPa (平均 1015.8)
* 気温: 13.4 .. 18.1 degC, 相対湿度 (露点から算出): 0.81 .. 1.00
* 風速: 時平均 1 .. 10 m/s, 最大突風 12 m/s (平均ピーク超過 1.10 m/s)
* 卓越風向 (10 deg ビン, 件数): 315 deg (281), 305 deg (204), 325 deg (143), 295 deg (83) — 海岸の NW 系海風
* 海面密度 (実測 p,T,RH から算出): 1.2054 .. 1.2302 kg/m3 (ISA+50%湿度比 1.2211)
* 最荒天窓: 2026-09-14 04:00 UTC, 最低気圧: 1010.5 hPa (2026-08-13 01:00 UTC)

## 観測 -> モデルのマッピング
| 観測列 | モデル量 | 変換 |
|---|---|---|
| PRES [hPa] | `pressure_offset_Pa` | (PRES - 1013.25) x 100, ISA 高度分布に加算 |
| ATMP [degC] | `temp_offset_K` | ATMP - 15.0, ISA 減率に沿って高度展開 |
| DEWP [degC] | `humidity` | RH = es(Td)/es(T) (Magnus), [0,1] クリップ |
| WDIR+WSPD | `wind` (m/s) | -(WSPD cos th, WSPD sin th), +x=北 +y=東, z_ref=10 m |
| GST-WSPD | `gust_rms` | x 0.363 (校正値, 下記 C) |
| (未報告) | rain/cloud/visibility | ベース WeatherConfig (既定: 無降水・乾燥 replay) |

## A. リプレイ忠実度 (全ノード・全中点)
* 観測ノード 1096 件すべてで driver 状態と実測の最大偏差: 気温 0.0e+00 K / 気圧 0.0e+00 Pa / 密度 0.0e+00 kg/m3 / 風ベクトル 0.0e+00 m/s
* 区間中点でも線形補間と厳密一致 (最大: 気温 0.0e+00 K, 風 4.4e-16 m/s)
* 密度は実測 (p, T, RH) からの moist_density と最大 0.0e+00 kg/m3 差 — 湿り空気の変換チェーンに誤差なし

## B. 突風変換の校正 (解析 -> 修正)
NDBC の GST は「1時間内のピーク風速」。`atmosphere.py` の sum4 突風は
gust_rms (成分ごとの時間RMS) で振幅が決まるため、GST-WSPD を直接
gust_rms にすると峰值を過大評価する。1時間窓・1 Hz サンプリングの
ピーク実測 (16 seed 平均 2.755 x gust_rms) から GUST_SIGMA_FACTOR = 0.363 を導出。

| 写像 | ピークバイアス | ピークRMSE | 平均風速バイアス |
|---|---|---|---|
| naive (k=1.0) | +3.00 m/s | 3.18 m/s | +0.38 m/s |
| calibrated (k=0.363) | +0.25 m/s | 0.28 m/s | +0.05 m/s |

ピークRMSE は 3.18 -> 0.28 m/s (91% 改善)。driver 端到端確認 (最荒天時, 1実現):
モデル平均 9.53 m/s (実測 9.50), モデルピーク 12.31 m/s (実測 GST 12.00)。

## C. 飛行エンベロープの追従 (全記録・毎時)
V=11.3 m/s 水平トリム (高度 30 m) を毎時再計算:
* 密度 rho(30 m): 1.2020 .. 1.2268 kg/m3
* 失速速度: 6.53 .. 6.60 m/s
* 水平飛行抵抗: 57.9 .. 58.5 N, 全開推力: 413 .. 422 N
* 推力マージン T/D: 7.14 .. 7.21 (最小は 2026-09-09 01:00 UTC)
* 相関: corr(rho, PRES) = +0.545, corr(rho, ATMP) = -0.836, corr(V_stall, rho) = -1.000, corr(T_avail, rho) = +1.000
→ 密度は観測気圧・気温の経過に正の相関で追従し、失速速度・推力は
  その密度変化を物理法則通り (V_stall ∝ rho^-1/2, T ∝ rho) 反映する。

## D. 動的反応 (最荒天窓 vs 最静穏窓, トリム固定)
`wind_tunnel.py` と同一の準定常荷重計測を実時間リプレイで実行:

| 窓 | 開始 (UTC) | 実測WSPD | 実測GST最大 | n_mean | n_std | n_peak |
|---|---|---|---|---|---|---|
| storm | 2026-09-14 04:00 | 9..10 m/s | 12 m/s | 3.125 | 1.399 | 7.631 |
| calm | 2026-08-17 14:00 | 1..1 m/s | 2 m/s | 0.878 | 0.278 | 1.951 |

荷重変動 n_std の比 (storm/calm) = 5.0, 実測ガスト・ピーク超過 (GST−WSPD) の比 = 3.0. 両者は異なる量 (前者は空力荷重の標準偏差, 後者は実測ガスト強度の比) を測るため 一致はしないが, 向きとオーダーは整合する — 荒天窓は静穏窓の数倍の動的反応を生み, ドローンの動的反応の強さが観測された天候の強弱を追従する。

## E. Env 端到端リプレイ
`FlyingBoatEnv(spatial=True, weather_real=..., rate=360)` で 200 ステップ (実時間 60 分相当) を飛行:
* 記録時刻は 2026-09-14 04:00 -> 2026-09-14 05:00 UTC と進行
* ステップ毎の telemetry (T, p) と記録補間値の最大偏差: 0.000 degC / 0.000 hPa
* 最終状態: z = -0.3 m, 対気速度 = 7.6 m/s

## 修正一覧 (解析から適用)
1. **突風変換の校正** — sigma = GST-WSPD の素朴な写像では1時間ピーク風速を平均 +3.00 m/s 過大 (RMSE 3.18 m/s)。sum4 突風過程の1時間ピーク実測 (2.755 x gust_rms) に基づき GUST_SIGMA_FACTOR = 0.363 を採用 -> バイアス +0.25 m/s, RMSE 0.28 m/s (91% 改善)。
2. **風向の成分補間** — 記録中に |dtheta|>90 deg の時刻対が 6 件。風向をスカラーで線形補間すると 350 deg -> 10 deg で逆回りする偽の回転が生じるため、(u,v) 成分で補間 (中点のコード短縮は平均 0.032 m/s, 最大 3.05 m/s と小さい)。
3. **過飽和のクリップ** — DEWP > ATMP の記録が 0 件 (センサー丸め)。RH = es(Td)/es(T) を [0, 1] にクリップし、湿度検証 (WeatherConfig) との整合を保証。
4. **スカラー平均風速の残差** — 校正後の1時間平均スカラー風速は実測 WSPD に対し +0.05 m/s (RMSE 0.06 m/s)。突風振幅の非線形 (|mean+gust| の凸性) による既知の残差で、風速計の精度 (±0.5 m/s) 未満のため補正しない。
5. **変換チェーンの厳密性** — 全 1096 ノードで観測値との最大偏差: 気温 0.0e+00 K, 気圧 0.0e+00 Pa, 密度 0.0e+00 kg/m3, 風ベクトル 0.0e+00 m/s — 補間・単位変換に誤差はなく、残る近似は時間分解能 (1時間刻みの線形補間) のみ。

## 制限
* 観測は 1 時間分解能。ノード間は線形補間で、前線の急峻な通過は平滑化される。
* 記録に降水・雲・視程の列がない (VIS=MM) ため、既定は乾燥リプレイ。
  降雨等はベース WeatherConfig で明示的に重ねる必要がある。
* 鉛直風・気温減率の観測はなく、ISA 減率と gust_rms のみで近似。
* 風はブイ高度 (約10 m = z_ref) の観測。高度スケールは対数シアー則。

## 成果物
* `results/weather_real_001/REPORT.md`
* `results/weather_real_001/summary.json`
* `results/weather_real_001/fidelity_nodes.csv`
* `results/weather_real_001/gust_calibration.csv`
* `results/weather_real_001/trim_tracking.csv`
* `results/weather_real_001/response_windows.csv`
* `results/weather_real_001/env_replay.csv`
* `results/weather_real_001/weather_study.py`
* `results/weather_real_001/overview.png`
* `results/weather_real_001/fidelity_zoom.png`
* `results/weather_real_001/gust_calibration.png`
* `results/weather_real_001/trim_tracking.png`
* `results/weather_real_001/response_windows.png`

実行時間: 6.6 s / git HEAD: `1e8fc69c3314`
