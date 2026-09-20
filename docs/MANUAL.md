# KMT ドローン飛行艇 運用マニュアル

**対象機体：** 双発モーター駆動・EPS（発泡スチロール）製・15 m 翼幅の Li-Po ドローン飛行艇
**対象読者：** 機体運用・整備・操縦支援に携わる技術者
**文書範囲：** ローカル Python シミュレータ、MAVLink 風インターフェース、強化学習オートパイロット、Phi3.5 ローカルパイロット
**文書範囲外：** 実機ハードウェア、ワイヤレス MAVLink、無線免許、法的運用資格

---

## 0. この文書の位置付け

本プロジェクトは**飛行力学・海面外乱・制御ソフトウェアの机上検証**を目的とするシミュレータです。
実機飛行、運用限界の実機校正、PX4/ArduPilot との電気的接続は対象外です。
運用者は本マニュアルに記載された**モデル制約と監査前の成果物の扱い**を理解したうえで作業してください。

| 用語 | 意味 |
|---|---|
| 機体 / ドローン飛行艇 | 双発・EPS 製 15 m 翼幅の試験機 |
| 1D 海面 | 時間×距離の縦断面のみの不規則波モデル |
| 方向分散海面 | cos²s 型の指向性を持つ 2D 不規則波（縦断面は y=0 スライス） |
| オートパイロット | 上記シミュレータ上のスクリプト／RL 制御器 |
| MAVLink 風 | プロセス内模擬 API。ワイヤシリアライズ・ACK・実機接続なし |
| NDBC | NOAA 国立データブイセンター。本文書では同梱スナップショットのみ参照 |
| 学生モデル | `teacher_student.py` で蒸留された小型 NumPy ネットワーク |

---

## 1. 機体設計パラメータ

`aircraft.py` の `Aircraft()` が既定値を返します。設計変更は当該データクラスのみを更新してください。

### 1.1 主要諸元

| 項目 | 値 | 単位 | 備考 |
|---|---:|---|---|
| 翼幅（b） | 15.0 | m | 主翼翼端間距離 |
| 平均翼弦（c） | 1.5 | m | 翼面積の算出基準 |
| 翼面積（S） | 22.5 | m² | b × c |
| アスペクト比（AR） | 10.0 | — | b² / S |
| 翼厚比（t/c） | 0.12 | — | 発泡翼プロファイル |
| 水線長（Lwl） | 2.6 | m | 船体水線長 |
| 水線幅（Bwl） | 0.55 | m | 船体水線幅 |
| 水平尾翼面積（S_t） | 2.4 | m² | — |

### 1.2 質量内訳

| 部位 | 質量 [kg] |
|---|---:|
| 主翼構造（EPS＋カーボン＋グラス） | 32.0 |
| 船体構造（EPS＋グラス） | 18.0 |
| 尾翼構造 | 4.0 |
| モーター（双発・大型 outrunner） | 3.5 |
| ESC（双発） | 0.6 |
| プロペラ（22×10 カーボン） | 2.0 |
| 電池（6S 22000 mAh Li-Po ×2） | 6.5 |
| アビオニクス（FC／GPS／RC／サーボ） | 3.0 |
| 配線・雑材 | 1.5 |
| ペイロード（カメラ等） | 5.0 |
| 翼下フロート（左右） | 4.0 |
| マージン | 3.9 |
| **合計** | **84.0** |

重量 W = m·g = 84.0 × 9.80665 = **823.76 N**（設計点）
搭載エネルギー：3.52 MJ（公称 6S 22000 mAh ×2）

### 1.3 空力・推進

| パラメータ | 値 | 単位 | 備考 |
|---|---:|---|---|
| 最大揚力係数 CL_max | 1.4 | — | フラップ装備 |
| 無揚力迎角 CL0 | 0.30 | — | — |
| 揚力勾配 CL_alpha | 5.5 / 4.56 | /rad | 二次元翼 / Helmbold 三次元 |
| 零揚力抗力 CD0 | 0.025 | — | クリーン状態 |
| オスワン効率 e | 0.85 | — | — |
| 双発ピークパワー | 12 000 | W | 電気入力 |
| 巡航パワー（合計） | 3 000 | W | — |
| プロペラ径 | 0.71 | m | 28 inch |
| 公称回転数 | 5 500 | rpm | — |
| 静推力 | 約 549 | N | T_static（推力係数 0.105 基準） |
| **推力／重量比** | **0.67** | — | 静推力時に浮力補助が必須 |
| 失速速度 V_stall | 6.5 | m/s | フロート込み 84 kg |
| 巡航速度 V_cruise | 11.3 | m/s | V_stall × √3 |
| 最大揚抗比 L/D_max | 16.3 | — | CL\* = √(CD0·π·e·AR) で算出 |

### 1.4 既知の制約

- 従来モードは縦運動のみ。spatialモードでは横運動・指令追従のロール／ヨーを追加。**完全な6自由度・独立舵面の空力モーメントは未モデル。**
- 推力係数 0.70 のため、**離水は船体浮力を補助として成立する設計**です。
  `validate.py` の物理検証は "T_static < W" を通知しています。
- 従来モードの外乱源は海面のみです。密度は 1D / spatial とも ISA。spatial の 10 m 風は対数則（突風も同じ係数）。
- 誘導抗力は McCormick 地面効果、推力は密度比例＋パワー上限、推力軸は α_T=2°。船体抵抗係数は未変更です。

---

## 2. 海面モデル

### 2.1 合成 PM スペクトル（`ocean.py`）

Pierson–Moskowitz スペクトルに基づく不規則波の縦断面モデル。既定パラメータ：

| パラメータ | 既定値 | 単位 | 説明 |
|---|---:|---|---|
| 有義波高 Hs | 1.5 | m | 観測値相当 |
| 卓越周期 Tp | 6.0 | s | ピーク周期 |
| シード | 42 | — | 再現性用 |

`Hs / 4 = η_rms` を満たすよう校正されています（`validate.py` で再検証）。

### 2.2 方向分散海面（`ocean_directional.py`）

2D 不規則波。エネルギー指向分布は cos²s（s = spreading 指数）。
本シミュレータは y=0 の縦断面スライスを返します。

| 引数 | 既定値 | 効果 |
|---|---:|---|
| directional | False | True で 2D 分散海面を有効化 |
| theta_mean_deg | 0.0 | 平均波向（機体進行方向を 0° とする） |
| spread_s | 10 | cos²s の s（大きいほど鋭い指向性） |

`theta_mean_deg = 35`〜`60` で波浪エネルギーが機体軸に対して斜めに入射する条件を模擬します。
本シミュレータは縦断面スライスのみを用いるため、**斜め波の横方向運動は再現できません**。

### 2.3 実海面 NDBC スナップショット（`ocean_real.py`）

`data/ndbc_46026_spectral.txt`（Cape San Martin, CA）から海況を読み込みます。
同梱のスナップショットはダウンロード時刻時点のものです。**データの最新性は本シミュレータでは確認しません。**

実測例（46026, スウェル＋風浪の2成分）：

| 成分 | Hs [m] | Tp [s] |
|---|---:|---:|
| スウェル | 1.50 | 7.1 |
| 風浪 | 0.80 | 5.0 |
| 合成（√(Hs²_s + Hs²_w)） | 1.70 | — |

### 2.4 海況の評価手順

1. `Ocean.statistics()` で η_rms、最大値、Hs_observed、Hmax などを取得
2. `validate.py` を実行し、`Hs / 4 = η_rms` の整合性を確認
3. 業務で必要なら deep-water 分散関係 `ω² = g·k` の最大相対誤差（≤ 1e-12）を確認

---

## 3. 動力学モデル

### 3.1 モデル化範囲

`dynamics.py` の `simulate_takeoff` / `simulate_landing` は**縦方向（x）のみ**を扱います。
状態：`[x, z, Vx, Vz, α, …]`、積分刻み既定 `dt = 0.001 s`。

| フェーズ（離水） | 条件 |
|---|---|
| 水上滑走（taxi） | 浮力 > 重量、推進力大 |
| ハンプ抵抗（hump） | 船体が抵抗ピーク近傍 |
| 揚力オフ（lift-off） | N_water → 0 |

| フェーズ（着水） | 条件 |
|---|---|
| 進入（approach） | 高度 > 10 m、滑空角 8° |
| フレア（flare） | 高度低下、ピッチ上げ |
| 接水（touchdown） | N_water > 0 |
| 滑走停止 | Vx < 0.2 m/s |

### 3.2 既知の制約

- 0.05 s 刻みの RL／MAVLink 風経路は**接水判定・波浪相対速度に刻み依存性**があります。
  物理検証は `dt = 0.001 s` の再計算で行います。
- 縦運動のみ。**旋回・斜め波・回転運動はモデル外。**
- 電池消費・熱制約は未モデル。

### 3.3 設計と現実の整合性

`validate.py` 実行結果（直近の監査時）：

| シナリオ | 結果 |
|---|---|
| 離水 | t = 3.89 s で浮上、x = 18.0 m、Vx = 7.51 m/s |
| 離水後の最高速度 | 18.49 m/s |
| 離水時の推力／抗力 | 493 N > 35 N（離水成立） |
| 着水（接水） | t = 24.78 s、x = 351.5 m、Vx = 14.46 m/s、\|Vz\|=1.18 m/s |
| 着水停止 | t = 40.0 s、Vx ≈ 0.08 m/s |
| 着水時ピーク N_water | 2 207 N（≈ 2.8 W） |

---

## 4. 損傷モデル（`damage.py`）

### 4.1 海水飛沫と推力低下

- プロペラ上端が波面に近づくと効率係数 K_eta が 0.55 〜 1.0 で変動
- 推力に K_eta を乗じる
- プロペラ上端が水面以下（**スリング**）に達すると重大事象
  - スリング 2 回で**致命的故障**
  - スリングは連続水中進入を 1 事象として計数（毎ステップの水没をカウントしない）
- 既定のハブ高さは CG 上 0.90 m。静水平衡では上端クリアランス約 1.50 m、下端約 0.79 m となり、静水では K_eta=1.0
- 旧値 0.30 m では下端約 0.19 m・K_eta≈0.82 で離水がハンプ停滞した（`results/takeoff_spray_001/`）
- テレメトリの `prop_clearance_m` / `prop_bottom_clearance_m` で確認できる。採用後の再評価は `results/takeoff_mount_001/`

### 4.2 機体浸水

- 船底が水没している時間に比例した浸水
- 船首 Veta > 0 の打ち込みで追加浸水（**空中では発生しない**）
- 警告閾値 0.20 kg、**臨界閾値 0.50 kg でアビオニクス故障**
- 浸水は質量増加として動力学に反映

### 4.3 故障状態の影響

- 臨界浸水またはスリング 2 回に達すると `damage.failed = True`
- 故障時は**推力停止、自動再武装拒否**（`dynamics.py` / `mavlink_if.py`）
- `fly_ollama.py` の評価は `damage_failure` で終了

### 4.4 設計上の閾値

| 状態 | 水量 [kg] | 対応 |
|---|---:|---|
| 正常 | < 0.20 | 通常運用 |
| 警告 | ≥ 0.20 | 注意。早期着水を検討 |
| 臨界 | ≥ 0.50 | アビオニクス故障。**直ちに着水・点検** |
| スリング 2 回 | — | 致命的故障。**再運用不可** |

---

## 5. 環境・強化学習・オートパイロット

### 5.1 `env.py` — Gym 風環境

状態は 11 次元（前方遭遇波面プレビュー付き）：

| # | 観測 | 単位 |
|---|---|---|
| 0 | z（高度） | m |
| 1 | Vx（対気速度水平成分） | m/s |
| 2 | Vz（垂直速度） | m/s |
| 3 | η（直下の波面） | m |
| 4 | dη/dt（波面時間変化） | m/s |
| 5 | 水没量（船底の水線下量） | m |
| 6 | 速度比 Vx / V_stall | — |
| 7 | 前回推力 | — |
| 8 | 前方 5 m 遭遇時刻の η | m |
| 9 | 前方 15 m 遭遇時刻の η | m |
| 10 | 前方 30 m 遭遇時刻の η | m |

行動は連続値で、`action_low = [0, -1]`、`action_high = [1, 1]`：
- 0 番目 → 推力（0 = 停止、1 = 全開）
- 1 番目 → 機体ピッチ。前段で `−3°〜12°` にクリップされて機体に渡る
  （`damage.py` 等の機体経路で許容値が前後するため注意）

エピソードの成功判定：
- **離水（`evaluate.py` / `fly_ollama.py` 既定）**：12 s 以内に `z ≥ 8 m` かつ `Vx ≥ 8.5 m/s`
- **着水（`evaluate.py` 既定）**：30 s 以内に着水し `|Vz| < 1.5 m/s` かつ最大 N_water < 3 W
- **訓練エピソード長（`train.py`）**：離水 300 ステップ・着水 400 ステップ（`dt = 0.05 s` で 15 s／20 s）

### 5.2 ベクトル環境（`vectorized.py`）

- 単一プロセスの逐次処理です。**並列実行ではありません。**
- 完了済み環境は軌跡に追加されず固定されます（追加報酬・成功集計を行わない）
- `sea_state_sweep(n_seeds=..., n_envs=...)` で海況スイープ。`n_envs` は同時処理数

### 5.3 Actor–Critic（`policy.py`）

- NumPy 実装の A2C + clipped surrogate + 価値 MSE + エントロピー
- GAE(λ)（λ = 0.95、γ = 0.99）で Advantage を推定
- 行動・価値・確率を**全てスカラー／shape (2,) で返す**こと
  （監査前の不具合：バッチ配列を返していた）
- PPO クリッピング・スロー値・全勾配ノルム制限を実装

### 5.4 訓練ループ（`train.py`）

着水は既定で波プレビュー付きフレア教師から 24 エピソードの行動クローンを行ったあと RL する（`--bc-episodes`）。

既定の訓練設定：

| パラメータ | 既定値 |
|---|---:|
| エピソード数 | 400 |
| ローリング内更新回数 | 4 |
| 学習率（本体） | 3e-4 |
| 学習率（ヘッド） | 5e-4 |
| 割引率 γ | 0.99 |
| λ（GAE） | 0.95 |
| シード（離水／着水） | 0 / 1 |
| ログ間隔 | 10 |

タグ命名規則（既定）：
- シナリオ名のみ → `takeoff` / `landing`
- `--directional` 指定時 → `_dir<θ>`
- `--seed-per-episode` 指定時 → `_rand` を付与

### 5.5 訓練済みモデルの互換性

- `ActorCritic` のチェックポイント（`.npz`）は現行の学習済み RL 方策
- **`teacher_student.StudentPilot` の NumPy 学生ネットワークとは形式が別**
  - 入力特徴・ピッチ範囲・キー名が専用
  - 共通ロード関数は存在しません

---

## 6. MAVLink 風インターフェース（`mavlink_if.py`）

### 6.1 対応コマンド

| コマンド ID | 名前 | パラメータ |
|---:|---|---|
| 22 | `MAV_CMD_NAV_TAKEOFF` | `alt`（目標高度 [m]）、`speed`（目標対気速度 [m/s]） |
| 21 | `MAV_CMD_NAV_LAND` | `alt`（目標高度 0）、`glide`（滑空角 [deg]） |
| 16 | `MAV_CMD_NAV_WAYPOINT` | `x`, `z`, `speed`（プロセス内座標系） |
| 178 | `MAV_CMD_DO_CHANGE_SPEED` | `speed` |
| 183 | `MAV_CMD_DO_SET_SERVO` | `servo` (1=throttle, 2=elevator), `pwm`（1000〜2000） |

未対応コマンドは拒否し、`MAV_RESULT_UNSUPPORTED` を返します。

### 6.2 テレメトリ出力

| メッセージ ID | 名前 | 内容 |
|---:|---|---|
| 90 | `HIL_STATE` | 機体状態 |
| 33 | `GLOBAL_POSITION_INT` | 緯度経度高度（内部座標） |
| 30 | `ATTITUDE` | 姿勢（ロール／ピッチ／ヨー） |

### 6.3 ロウレベル制御器

`LowLevelController` が PX4 のカスケード制御を模倣：
- `MAV_CMD_NAV_TAKEOFF` → 推力・ピッチのセットポイントへ変換
- 比例・微分制御で状態を目標に追従

### 6.4 状態機械

`vehicle` の `mode` は次のいずれか：
- `DISARMED` → 武装待ち
- `ARMED` → 通常運用
- `TAKEOFF` → 離水中
- `LAND` → 進入中
- `FAILSAFE` → 故障発生

### 6.5 既知の制約

- プロセス内 API。**ワイヤ形式（バイト列）・ACK・実機接続は未実装**
- PX4/ArduPilot との電気的・論理的な相互運用性は**未検証**
- 機体直接操作（`step(action=...)`）での RL 行動は**次のステップで自動的に制御器に戻される**点に注意。
  監査後の `step(action=...)` は RL 行動を**当該ステップで適用**し、訓練環境と同じ観測を返します
- `MAV_CMD_DO_SET_SERVO` の servo 1 PWM 1000 は推力停止、2000 は全開に対応
- pitch（servo 2）の PWM は機体ハードウェア可動域 ±15° にクリップされます
- `MAV_CMD_NAV_WAYPOINT` の `x, z` はプロセス内座標（実機 GPS ではない）

### 6.6 評価経路

`evaluate.py` は次のフローで実行：

1. 機体生成、`arm()` で武装
2. `send_command(MAV_CMD_NAV_TAKEOFF, {"alt":10,"speed":13})` 送信
3. `veh.step()` を `dt = 0.05 s` で 240 ステップ（12 s）
4. 軌跡を 3D レンダリング・CSV 出力

実機の運用限界や安全性を判定する目的には使用できません。

---

## 7. 視覚化（`visual3d.py` / `evaluate.py`）

### 7.1 3D シーン構成

- 主翼＋船体モデル
- 不規則波面（青半透明メッシュ）
- 飛行経路（オレンジ）／現在軌跡（赤）

### 7.2 同時表示パネル（4〜6 面）

| パネル | 系列 |
|---|---|
| 推力スロットル | servo 1（PWM 1000〜2000） |
| ピッチ α | servo 2（elevator） |
| 高度 | z / setpoint |
| 速度 | Vx / target |
| 浸水・推力係数 | 損傷状態 |

### 7.3 出力例（既定で `results/` に生成）

- `takeoff.png` / `landing.png`：6 面プロット
- `wave_field.png`：海面スライス
- `profile.png`：機体経路の側面図
- `scene_takeoff_early/mid/final.png`：3D シーン
- `scene_landing_early/mid/final.png`：3D シーン
- `mavlink_takeoff_profile.png` / `mavlink_landing_profile.png`：MAVLink 経路プロファイル
- `mavlink_takeoff_telemetry.csv` / `mavlink_landing_telemetry.csv`：テレメトリ CSV

---

## 8. ローカル Ollama / Phi3.5 パイロット

### 8.1 前提

- 別途 `ollama serve` を起動する
- `phi3.5` モデルを `ollama pull phi3.5` で取得する
- 既定接続先：`http://127.0.0.1:11434`
- `phi3.5:latest` のモデル ID 先頭：`61819fb370a3`

接続・モデルの確認は `fly_ollama.py --check`。

### 8.2 実行例

```bash
# 接続のみ検証
python3 fly_ollama.py --check

# 離水シミュレーション（既定：12 秒）
python3 fly_ollama.py --scenario takeoff --duration 12

# 着水シミュレーション
python3 fly_ollama.py --scenario landing --duration 30

# 同梱 NDBC 海況
python3 fly_ollama.py --scenario takeoff --ndbc --duration 12

# 接続・推論エラー時に即時停止（フォールバックなし）
python3 fly_ollama.py --duration 3 --interval 1 --strict
```

### 8.3 入出力

| | 内容 |
|---|---|
| 入力（観測） | 高度・速度・昇降速度・波面・船底クリアランス・浸水量・目標・前回指令 |
| 出力 | JSON：`throttle`（0〜1）、`pitch_deg`（-8 〜 +15 度） |
| 検証項目 | 型・有限値・範囲・必須キー |

### 8.4 動作モード

| モード | 効果 |
|---|---|
| 標準 | 推論失敗時にスクリプト制御器へフォールバック。次の判断まで既存制御器が物理ステップを刻む |
| `--strict` | フォールバックを使わず終了コード 2 で停止 |
| `--student <path>` | Ollama 不在でも `StudentPilot` を使用（後述） |

### 8.5 タイミング

- 既定：シミュレーション時刻 **1 秒ごと**に判断
- 物理ステップは **`--dt`（既定 0.05 s）刻み**で適用
- **推論中はシミュレーションを一時停止**。実時間制御ではありません
- ソケットタイムアウトはすべての実行時間の上限ではありません

### 8.6 終了状態（`summary.json` の `status`）

| status | 意味 |
|---|---|
| `success` | 目標達成 |
| `time_limit` | 経過時間で終了（飛行目標達成を意味しない） |
| `damage_failure` | 損傷モデルによる故障 |
| `impact_failure` | 着水反力 > 8 W |
| `hard_landing` | |Vz| ≥ 1.5 または N_water ≥ 3 W |
| `pilot_error` | 推論失敗（`--strict` 下の例外終了を含む） |
| `interrupted` | キーボード割り込み |

### 8.7 出力ディレクトリ

`results/ollama_<UTC日時>/` に下記を保存（既存ディレクトリの上書きは拒否）：
- `config.json` — 設定
- `decisions.jsonl` — 観測＋Phi判断ログ
- `trajectory.jsonl` — 各物理ステップの状態
- `summary.json` — 結果サマリ

### 8.8 既知の制約

- LLM の判断は観測時点のローカル状態のみ。前後整合の保証はありません
- 推論待ち時間のぶんだけ実時間が伸びます（実測：離水 ≈ 28 s／3 s シミュレーション）
- `fallback_decisions` の数が大きいほど Phi の応答品質が低下しています

---

## 9. 教師 → 学生蒸留（`teacher_student.py`）

### 9.1 目的

Phi3.5 の判断は遅く、また可用性に依存します。
判断時の応答だけを教師ログから**模倣学習**し、Ollama 不在でも軽量 NumPy ネットワークで操縦可能にします。

### 9.2 教材データ

- ソース：`fly_ollama.py --output <dir>` の `decisions.jsonl`
- `source == "ollama"`（`fallback`／`aborted`／`student` は除外）
- JSON 形式が正しくても**判断が正しいとは限りません**。教材の品質は別途評価が必要

### 9.3 学習プロセス

```bash
# 1. 教師ログの収集
python3 fly_ollama.py --scenario takeoff --duration 12 --seed 10 \
  --output results/teacher_takeoff_10
python3 fly_ollama.py --scenario landing --duration 30 --seed 20 \
  --output results/teacher_landing_20

# 2. 教材から学習（Ollama 不要）
python3 teacher_student.py \
  results/teacher_takeoff_10/decisions.jsonl \
  results/teacher_landing_20/decisions.jsonl \
  --output results/student_new

# 3. 学生で操縦（Ollama 不要・各物理ステップで判断）
python3 fly_ollama.py --student results/student_new/student.npz \
  --scenario takeoff --duration 12 --interval .05
```

### 9.4 同梱モデル

- `results/phi_student_v1/student.npz`
- 教師応答 5 件での学習結果。**一般化性能や飛行成功率を意味しません**
- 比較評価は別海況・別フェーズで実施する必要があります

### 9.5 既知の制約

- 成功軌跡の選別や報酬による改善は**未実装**。Phi の判断を模倣するだけです
- `StudentPilot` の入力・ピッチ範囲は専用。`ActorCritic` の RL チェックポイントとは非互換
- 評価で `--student` を使う場合、`--interval` は `--dt` 以上に揃える必要があります

---

## 10. 育成サイクル（`educate.py`）

### 10.1 概要

Phi に 32 件の操縦指令を依頼し、24 件を学習・8 件（別シード）で検証します。
4 つの海況／シード条件 × 8 段階（離水：滑走・抵抗域・引き起こし・上昇・高度到達／着水：進入・降下・フレア）。

### 10.2 実行

```bash
python3 educate.py --output results/education_002
```

完了済みモデル（既存ディレクトリ）は上書きしません。新しい出力先で実行してください。

### 10.3 出力

| ファイル | 内容 |
|---|---|
| `curriculum.json` | 教材リスト（実行条件・段階・適用後ピッチ） |
| `teacher.jsonl` | 教師判断の収集ログ |
| `train.jsonl` | 学習エピソード記録 |
| `validation.jsonl` | 検証エピソード記録 |
| `student.npz` | 最良検証誤差モデル |
| `report.json` | 成否・終了理由・最終状態・学習曲線・ハッシュ |

### 10.4 評価

- 学習／モデル選択に使用しないシード 101・102、波高 0／0.3／0.8 m で離水・着水を比較
- 旧生徒・新生徒・既存制御器を同条件で 12 試行ずつ評価
- 初期化 3 種・最大 4000 更新を試し、検証誤差が最小のモデルを保存

### 10.5 既知の制約

- 学習・評価のみを再開する場合は `--evaluate-only`
- 教師収集中に停止した場合、同じコマンドで再実行すれば既存ログを再利用します

---

## 11. フィードバック再指導（`feedback_education.py`）

### 11.1 目的

第 1 サイクルで収集した各状態を復元し、スロットル・ピッチ候補を**それぞれ 2 秒間**試します。
最も良かった候補を Phi に提示して採用案を決定させ、新しい教材とします。

### 11.2 実行

```bash
python3 feedback_education.py \
  --source results/education_001/teacher.jsonl \
  --output results/education_002
```

### 11.3 評価指標（短期スコア）

| シナリオ | 重み |
|---|---|
| 離水中の加速 | 大 |
| 飛行中の高度追従 | 大 |
| 着水時の降下速度 | 大 |
| 着水時の接水反力 | 大 |

これらは**局所的な代理指標**です。飛行完遂や実機安全性を保証しません。

### 11.4 既知の制約

- Phi の返答は候補番号として検証します。候補に存在しない指令は採用しません
- 物理・損傷係数は変更しません（再評価時に同一条件）
- チェックポイントは**評価結果ではなく新教材での学習結果**から選びます
- 評価シード 201/202 は再利用します。**これらは既知のベンチマーク**として扱ってください

---

## 12. 監査前の成果物の扱い

`results/` 配下の既存モデル・図・CSV は**監査前の出力**です。
学習・成功判定・海況設定の不具合を修正済みのため、旧成功率をそのまま再利用しないでください。
再評価は本マニュアルの監査（第 13 章）に従ってください。

---

## 13. 監査と検証

### 13.1 監査スクリプト

```bash
python3 audit.py           # results/audit.json に保存
python3 audit.py --output audit_other.json
```

| 項目 | 動作 |
|---|---|
| `regressions` | `tests/` の単体テスト 52 件を実行 |
| `physics` | `validate.py` を呼び出し、機体・海面・離水・着水を再計算 |
| `source_sha256` | 各 `.py` の SHA-256 を保存 |

監査出力には以下が含まれます：
- タイムスタンプ（UTC）
- Python バージョン
- 各チェックの `passed` フラグと出力テキスト
- ソースファイルの SHA-256 一覧

### 13.2 単体テスト

```bash
python3 -m unittest discover -s tests -v
```

| ファイル | 件数（直近） | 主な確認項目 |
|---|---:|---|
| `test_regressions.py` | 15 | 勾配数値微分、PPO、終了環境固定、海況反映、故障時推力 |
| `test_new_modules.py` | 18 | 方向分散海面・CLI 既定値・シード毎ランダム化 |
| `test_teacher_student.py` | 3 | 学習・保存・読込・Ollama 不在 |
| `test_education.py` | 2 | カリキュラム網羅・成否判定 |
| `test_feedback_education.py` | 3 | 候補提示・状態復元・候補検証 |
| `test_ollama_pilot.py` | 11 | スキーマ・HTTP ペイロード・タイムアウト・strict |

### 13.3 物理サニティ（`validate.py`）

- 機体：W = m·g、L/D_max、V_cruise = V_stall·√3
- 海面：Hs/4 = η_rms、分散関係 ω² = g·k
- 離水：離水後に Vx > V_stall、T > 抗力、トップ速度を評価
- 着水：Vx > V_stall、|Vz| < 5 m/s、最終 Vx < 0.5 m/s、ピーク N_water < 10 W

### 13.4 制約事項

- 監査は**環境とコードの整合性**を確認するものです
- 実機の運用限界・安全性は本監査では判定できません
- 浮力・抗力・損傷係数は簡易モデルです（特に 0.05 s 刻みの経路で刻み依存性）
- 学習の時間上限は**有限長エピソードの終端**として扱われます。time-limit bootstrap は未実装
- NOAA データの最新性・地点ラベルは手動で再確認してください
- pymavlink の Python 3.14 互換ビルドが失敗するため、MAVLink 風インターフェースは自前実装です

---

## 14. 標準作業フロー

### 14.1 初回セットアップ

```bash
python3 -m pip install -r requirements.txt
```

依存パッケージは `requirements.txt` に列挙。`pymavlink` の Python 3.14 ビルドが失敗するため
MAVLink 風インターフェースはプロセス内実装のみ対応。

### 14.2 毎日の作業手順

```bash
# 1. 監査（回帰テスト＋物理検証）
python3 audit.py

# 2. 機体・海面・離水・着水の基本出力
python3 main.py

# 3. MAVLink 経路での再評価
python3 evaluate.py

# 4. RL 訓練（必要時）
python3 train.py --scenario takeoff --hs 1.5 --tp 6.0
# 方向分散・エピソード毎ランダム化
python3 train.py --scenario takeoff --directional \
  --theta-mean-deg 35 --seed-per-episode --episodes 400
```

### 14.3 LLM パイロット評価

```bash
# Ollama 接続確認
python3 fly_ollama.py --check

# Phi で離水
python3 fly_ollama.py --scenario takeoff --duration 12

# 接続検証のみ
python3 fly_ollama.py --duration 3 --interval 1 --strict
```

### 14.4 蒸留・再指導

```bash
# 教師データの追加収集
python3 fly_ollama.py --scenario takeoff --duration 12 --seed 10 \
  --output results/teacher_takeoff_10
python3 fly_ollama.py --scenario landing --duration 30 --seed 20 \
  --output results/teacher_landing_20

# 学生学習 → 評価
python3 teacher_student.py \
  results/teacher_takeoff_10/decisions.jsonl \
  results/teacher_landing_20/decisions.jsonl \
  --output results/student_new

python3 fly_ollama.py --student results/student_new/student.npz \
  --scenario takeoff --duration 12 --interval .05
```

### 14.5 育成サイクル

```bash
python3 educate.py --output results/education_<NNN>
python3 feedback_education.py \
  --source results/education_<NNN-1>/teacher.jsonl \
  --output results/education_<NNN>
```

---

## 15. トラブル対応の初動

| 症状 | 推定原因 | 初動 |
|---|---|---|
| `audit.json` のチェック失敗 | 環境／コード改変／乱数シード | 直前のコミット差分を確認、`git status`、`python3 -m unittest discover -s tests -v` を単体実行 |
| `phi3.5` 接続不可 | Ollama 停止／ポート競合 | `ollama serve`、`curl http://127.0.0.1:11434/api/tags`、`fly_ollama.py --check` |
| 推論失敗が連続する | プロンプト不整合／モデル差替 | `--strict` で停止 → 応答内容を確認 → `--interval` を延ばす／`--timeout` を増やす |
| 離水が常に失敗（特定シード） | 波面位相のトラップ | `--seed-per-episode` を有効化／`Hs` を下げて段階的に訓練 |
| 着水で `hard_landing` 連続 | フレア時のピッチ不足 | 進入高度を 25 m に設定／`glide` を 6° に下げる |
| `damage.failed == True` | 海水飛沫・浸水 | 進入速度を抑える、`Hs` を下げる、Flap 装備の可否を確認 |
| 学んだモデルの互換性不明 | 保存形式の違い | `ActorCritic` と `StudentPilot` は別形式。RL は `policy.npz`、学生は `student.npz` を流用しない |

---

## 16. 安全設計上の境界

| 境界 | 値 | 出典 |
|---|---|---|
| 縦方向のみ | — | 本シミュレータは横・回転を扱わない |
| `MAV_CMD_NAV_WAYPOINT` の座標 | プロセス内（実機 GPS ではない） | `mavlink_if.py` |
| 推力／重量比 | 0.70（静推力） | 浮力補助が必須 |
| 浸水警告／臨界 | 0.20 / 0.50 kg | `damage.py` |
| スリング致命的故障 | 連続水中進入 2 回 | `damage.py` |
| 着水最大降下速度（設計） | 5 m/s | `validate.py` |
| ピーク N_water 検証上限 | 10 W | `validate.py` |
| hard landing 判定 | N_water ≥ 3 W または \|Vz\| ≥ 1.5 m/s | `fly_ollama.py` |
| タイマー上限（離水） | 12 s | 既定シナリオ |
| タイマー上限（着水） | 30 s | 既定シナリオ |

---

## 17. 参照ファイル一覧

| ファイル | 用途 |
|---|---|
| `aircraft.py` | 機体設計パラメータ |
| `ocean.py` | 1D 不規則波（PM スペクトル） |
| `ocean_real.py` | NOAA NDBC 入力の 2 成分海面 |
| `ocean_directional.py` | 方向分散 2D 海面（y=0 縦断面） |
| `dynamics.py` | 縦運動シミュレータ（離水・着水） |
| `damage.py` | 海水飛沫・浸水 |
| `mavlink_if.py` | MAVLink 風 API・PX4 風カスケード |
| `env.py` | RL Gym 風環境ラッパー |
| `policy.py` | Actor-Critic（NumPy） |
| `vectorized.py` | ベクトル環境（逐次） |
| `train.py` | 訓練スクリプト |
| `evaluate.py` | MAVLink+実海洋+3D 統合評価 |
| `visual3d.py` | 3D シーン＋操縦ログ |
| `validate.py` | 物理サニティ |
| `audit.py` | 回帰テスト＋監査 JSON |
| `advisor.py` | Phi 助言＋短期試行 |
| `ollama_pilot.py` | Ollama クライアント／観察・判断 |
| `fly_ollama.py` | Phi パイロットランナー |
| `teacher_student.py` | 蒸留ネットワーク |
| `educate.py` | 育成サイクル |
| `feedback_education.py` | フィードバック再指導 |
| `tests/` | 単体テスト |
| `data/` | NOAA NDBC データ |
| `results/` | 出力グラフ・CSV・モデル |
| `AUDIT.md` | 監査報告（直近） |
| `audit.json` | 監査出力 |


## 18. 横方向・大気の学習モード

`train.py --spatial` で横位置・横速度・バンク・方位と大気を有効にします。
行動は4次元、既定観測は19次元で、既存モデルの再利用はできません。
風設定・成功判定・モデル制約・実行例は [README](../README.md#横方向の運動と大気spatial-モード) を参照してください。
本拡張は `FlyingBoatVehicle(spatial=True)` とOllama／RL操縦・3D表示にも接続されています。
実行例とテレメトリ仕様はREADMEの「空間運動の機体・操縦・表示への接続」を参照してください。
