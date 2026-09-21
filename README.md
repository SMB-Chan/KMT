# KMT - 15 m 翼幅 スチロール製 ドローン飛行艇 実験プロジェクト


EPS（発泡スチロール）製・双発モーター駆動のLi-Poドローン飛行艇を設計し、
実海洋データに基づく不規則波浪モデル上で離水・着水の動力学シミュ
レーションを行う実験プロジェクト。

MAVLink風のローカル・コマンドインターフェースを実装し、強化学習で
オートパイロットを訓練、3D可視化で挙動を確認できる。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `aircraft.py` | 機体設計パラメータ（質量・空力・推進系） |
| `ocean.py` | PM海面スペクトル（合成、不規則波） |
| `ocean_real.py` | NOAA NDBCブイデータの実海面モデル |
| `dynamics.py` | 離水・着水の縦運動シミュレータ |
| `damage.py` | 海水飛沫・浸水のダメージモデル |
| `mavlink_if.py` | PX4/ArduPilot互換のMAVLinkインターフェース |
| `env.py` | 強化学習用 Gym 風環境ラッパー |
| `policy.py` | numpy製 Actor-Critic ポリシーネットワーク |
| `train.py` | 強化学習訓練スクリプト（離水・着水） |
| `evaluate.py` | MAVLink + 実海洋 + 3D 統合評価 |
| `visual3d.py` | mplot3d による 3D シーン＋操縦ログ可視化 |
| `validate.py` | 再計算による物理サニティチェック |
| `audit.py` | 回帰テスト・物理検証・JSON監査レポート |
| `system_one.py` | 型付き Choice/Score/Noul と信頼度（ローカル System One 原語） |
| `flight_jev.py` | 飛行状態に対する Jev 風決定。内側は内蔵セットポイント |
| `advisor.py` | Phi助言＋短horizon試行によるカリキュラム生成（Ollama不在時は規則スタブ） |
| `ocean_directional.py` | 方向分散付き2D不規則波（cos^2s、NDBCのMWD対応） |
| `atmosphere.py` | ISA大気・対数風・シード固定突風 |
| `weather.py` | 決定論的気象層（湿り空気密度・降雨/雲/霧・着氷・下降気流・天候プリセット） |
| `weather_real.py` | NDBCブイ実測毎時気象のリプレイ（`HistoricalWeather`、`Weather` 互換の時間変動気象） |
| `weather_study.py` | 実測リプレイの定量解析（忠実度・突風校正・エンベロープ追従・動的反応・Env端到端） |
| `weather_forecast.py` | 天気予報モデル（`VarForecastModel`: 実測初期値から将来を予報、`forecast_weather` で予報下飛行） |
| `weather_forecast_study.py` | 予報モデルの検証（次数選択・スキル・減衰・嵐ケース・予報下飛行） |
| `design_optimize.py` | 機体設計の決定論的パラメトリック最適化（コンパス探索） |
| `wind_tunnel.py` | 大気モデル駆動の仮想風洞（ポーラ・速度スイープ・突風荷重計測と空力設計最適化） |
| `spatial_dynamics.py` | 横運動・バンク・フロート復原の空間動力学 |
| `accelerate.py` | ベクトル化訓練・海況スイープ |
| `vectorized.py` | 並列環境ロールアウト |
| `flight_diagnostics.py` | 力収支・ plateau 診断 |
| `ollama_pilot.py` | ローカルOllama操縦・フォールバック制御 |
| `fly_ollama.py` | 操縦実行・Jev／方策・教材記録ランナー |
| `policy_pilot.py` | 学習済み方策の操縦ラッパー |
| `teacher_student.py` | 教師模倣の生徒学習 |
| `educate.py` | Phi教材 cycles（カリキュラム・検証） |
| `feedback_education.py` | フィードバック付き教材 cycles |
| `operator_training/` | ブラウザ操縦コックピット・MAVLink UDP SITL・フライトレポート |
| `web/operator/` | QGC 代替のローカル操縦 UI |
| `docs/` | 運用マニュアル・MAVLink参照・チェックリスト |
| `tests/` | 学習・評価・故障処理の回帰テスト |
| `data/` | 取得した NOAA NDBC データ（リアルタイム・スペクトル） |
| `results/` | 出力グラフ・CSV・テキスト |

## 機体設計（aircraft.py）

| 項目 | 値 |
|---|---|
| 翼幅 / 翼弦 / 翼面積 | 15.0 m / 1.5 m / 22.5 m² |
| アスペクト比 | 10.0 |
| 総質量 | 84.0 kg（翼下フロート 4 kg を含む） |
| 双発モーター（28"プロペラ） | ピーク12 kW |
| 静止推力 / 重量比 | 0.67 |
| 失速速度 / 巡航速度 | 6.5 / 11.3 m/s |
| 最大揚抗比 L/D | 16.3 |
| Li-Po 6S 22000 mAh ×2 | 3.52 MJ |

## 海面モデル

| モデル | ソース | パラメータ例 |
|---|---|---|
| `ocean.py` | 合成PMスペクトル | Hs=1.5 m, Tp=6 s |
| `ocean_real.py` | NOAA NDBC リアルタイムデータ | スウェル+風浪の2成分 |

実データ例（NDBC 46026, Cape San Martin, CA）：
- Hs_swell=1.50 m, Tp_swell=7.1 s
- Hs_wind=0.80 m, Tp_wind=5.0 s
- Hs_total=√(Hs_s²+Hs_w²)=1.70 m

## ダメージモデル（damage.py）

**海水飛沫（プロペラ負荷）**
- プロペラハブは CG 上 0.90 m（径 0.71 m）。静水の上端クリアランスは約 1.50 m
- プロペラ上端と波面との距離から効率係数を計算（0.55〜1.0）
- 推力に効率係数を乗じる
- プロペラ上端が水面以下になるとスリング事象 → 2回で致命的故障

**機体浸水**
- 船底が水没している時間で比例的に浸水
- 船首への波の打ち込み（Veta>0）で追加浸水
- 警告0.20 kg、臨界0.50 kgで avionics 故障
- 浸水は質量増加 → 重量増加分も考慮

## MAVLink / PX4 互換インターフェース（mavlink_if.py）

対応 MAV_CMD：
- `MAV_CMD_NAV_TAKEOFF` (22) - 目標高度・速度を指定
- `MAV_CMD_NAV_LAND` (21) - 目標高度・グライド角を指定
- `MAV_CMD_NAV_WAYPOINT` (16) - ウェイポイント
- `MAV_CMD_DO_SET_SERVO` (183) - サーボ直接制御

テレメトリメッセージ：
- `HIL_STATE` (90) - 機体状態
- `GLOBAL_POSITION_INT` (33) - 緯度経度高度
- `ATTITUDE` (30) - 姿勢

内部の `LowLevelController` がPX4のカスケード制御を模倣し、
`MAV_CMD_NAV_TAKEOFF` を推力＋ピッチのセットポイントに変換する。

## 強化学習（env.py / policy.py / train.py）

- **環境**：11次元状態（z, Vx, Vz, η, dη/dt, 水没, 速度比, 前回推力, 前方5/15/30 mの遭遇波面）
- **行動**：推力 [0,1]、ピッチ（[-8°, 12°]にマップ）
- **アルゴリズム**：numpy製 Actor-Critic、GAE(λ)  Advantage Estimation
- **報酬**：高度・速度に対する密報酬＋成功ボーナス
- **旧実績の扱い**：既存の学習済みモデル・学習曲線は監査前の出力です。
  学習・成功判定の不具合を修正したため、性能評価には再学習・再評価が必要です。

## 3D 可視化（visual3d.py / evaluate.py）

- 機体モデル（主翼＋船体）+ 不規則波面（青半透明メッシュ）
- 飛行経路（オレンジ、現在軌跡を赤）
- **右側4パネルで操縦ログ同時表示**：
  - 推力スロットル（servo 1, PWM 1000-2000）
  - ピッチ α（servo 2, elevator）
  - 高度 vs 目標高度（setpoint）
  - 速度 vs 目標速度
  - 浸水量とスラスト係数（ダメージ状態）

## 実行

```bash
python3 -m pip install -r requirements.txt
python3 audit.py       # 回帰テスト＋再計算した物理検証、results/audit.jsonへ記録
python3 main.py        # 機体・海面・離水・着水（基本、可視化）
python3 train.py       # RL による離水・着水オートパイロット訓練
# 例：方向性海面＋エピソード毎ランダム化で離水を学習（既定の上書き防止に自動タグ）
python3 train.py --scenario takeoff --directional --theta-mean-deg 35 --seed-per-episode
python3 evaluate.py    # スクリプト制御器 + 保存済み海洋データ + 3D評価
python3 validate.py    # 物理サニティチェック
```

## 実験結果

旧表（実海洋 NDBC 46026、Hs=1.7 m、監査前の制御器・物理）は履歴として残す：離水は
Vx=2.5 m/s でハンプ停滞（失敗）、着水は成功。下記が修正後の現状である。

現状（合成海面 Hs=1.5 m、Tp=6 s、スクリプト／RL再学習後）：

| 経路 | 離水 | 着水 |
|---|---|---|
| スクリプト制御器（ハブ0.30 m当時） | 失敗（ハンプ停滞、実推力444N） | 成功 |
| RL方策（ダメージなし環境、保持評価） | takeoff_preview_best 7/10 | landing_preview_best 10/10（50波 47/50） |
| 空間＋横風10条件・ハブ0.30 m | どちらも 0/10 | 内蔵 8/10、既存RL＋横 7/10 |
| 空間＋横風10条件・ハブ0.90 m（採用） | 内蔵 10/10（Hs=0.3 と 1.5）、既存縦RL＋横 7/10 | 内蔵 8/10、既存RL＋横 5/10 |
| 同上・ISA/地面効果後の内蔵制御 | 10/10（Hs=0.3 と 1.5） | 6/10（低空で沈下不足） |
| 同上・着水沈下則（アイドル＋揚力捨て） | 10/10 | 10/10 |
| 空間離水RL 400ep（検証で選択） | 4軸 0/10（横ずれ）、推力ピッチ＋横補助 10/10 | — |
| 新機体84 kg 空間RL 400ep | 4軸 1/10、横補助 8/10 | 4軸 0/10、横補助 9/10 |
| 新機体 縦RL＋横補助 | 7/10（旧方策 2/10） | 10/10（旧方策 6/10） |
| 離水 BC＋縦RL | 7/10（内蔵 10/10のまま） | — |

実機の運用限界は検証していない。詳細は `results/hump_analysis_001/`、
`results/reward_envelope_001/`、`results/warmstart_001/`、
`results/takeoff_control_001/`、`results/landing_spatial_001/`、
`results/takeoff_spray_001/`、`results/takeoff_mount_001/`、
`results/spatial_takeoff_001/`、`results/model_fidelity_001/`、
`results/model_fidelity_002/`、`results/landing_settle_001/`、
`results/wing_floats_001/`、`results/new_airframe_001/`、
`results/new_airframe_long_001/`、`results/takeoff_bc_001/`、`results/flight_jev_001/`、`results/envelope_001/`、`results/envelope_002/`、`results/lateral_001/`、`results/land_phase_001/`、`results/audit_fixes_001/`、`results/remaining_001/`、`results/size_001/` を参照。

設計最適化系の結果は `results/design_opt_001/`（ミッション目的関数 J を 39.8% 改善、
1.068→0.643）と `results/wind_tunnel_001/`（`atmosphere.py` の ISA・対数シアー・突風を
使う仮想風洞で空力5変数を探索し、風洞目的関数 J を 26.2% 改善、1.000→0.738。
巡航電力 1082→658 W、(L/D)max 16.3→21.2、突風荷重 n_std 1.031→1.007）を参照。
再実行は `python3 design_optimize.py` / `python3 wind_tunnel.py`（どちらも決定論的）。

## 注意点・限界

- 従来モードは縦方向（x）のみ。`spatial=True` では横運動・旋回と現在y位置の方向性海面を使います。翼下フロートが接水するとロール復元が入り、波面の左右差も荷重に反映します。空中のバンクは指令追従のままです。
- ピッチ指令の範囲は経路で異なる。RL環境・機体直接操作はオートパイロット包絡線 -8〜12°、サーボ直接操作はハードウェア可動域 ±15°（Ollamaは-8〜15°で応答）。`advisor.short_simulate` とカリキュラムは環境で実際に適用された値（クリップ後）を記録する
- 完全な6自由度剛体運動・独立した舵面の空力モーメントは未モデル
- 従来モードの外乱は海面のみ。`--spatial` では横運動と合成大気モデルを使用（後述）
- pymavlink のPython 3.14互換ビルドが失敗したため、MAVLinkメッセージ
  識別子を使う自前のプロセス内実装（シリアライズ・通信・ACK未実装、実機接続やプロトコル互換性は未検証）
- 訓練は既定で固定シード。`--seed-per-episode` でエピソード毎に海面シードを変更可能
- 着水成功は船底が波面に最初に触れた瞬間に確定する（波頂では平均水面より最大約1 m上方でも確定）。進入点の波位が成功率に影響しうる
- `fly_ollama.py` の合成海面既定は `--hs 0.3` で、env/train の既定 Hs=1.5 と意図的に異なる（軽海況の操縦確認用）

## 監査と再検証

[監査報告](AUDIT.md) に修正内容と残る制約を記載しています。
`python3 audit.py` は既存のモデル・図を上書きせず、検証結果とソースのSHA-256を保存します。
`python3 -m unittest discover -s tests -v` で回帰テストのみ実行できます。

`EnvConfig(Hs=..., Tp=...)` で学習環境の海況を指定できます。
`directional=True`（＋`theta_mean_deg`、`spread_s`）で方向分散付き2D海面の縦断面を使います。
`sea_state_sweep(..., n_seeds=..., n_envs=...)` は各海況で `n_seeds` 回の試行を行い、
`n_envs` は一度に処理する環境数です。ベクトル環境は逐次実行であり、並列プロセスではありません。
`sea_state_sweep` と `mavlink_sea_sweep` は同じ `directional` 引数を受け付けます。
MAVLink風評価の `step(action=...)` は、そのステップにRL行動を直接適用します。

## ローカルOllama / phi3.5パイロット

`ollama_pilot.py` と `fly_ollama.py` で、ローカルの `phi3.5:latest` が観測から
スロットル・機体ピッチを決定し、既存の `FlyingBoatVehicle` を操縦します。
標準ライブラリのHTTPクライアントを使用するため追加Pythonパッケージは不要です。

```bash
# Ollamaサービスが未起動の場合、別ターミナルで実行
ollama serve
# phi3.5 が未導入の場合のみ
ollama pull phi3.5

python3 fly_ollama.py --check
python3 fly_ollama.py --scenario takeoff --duration 12
python3 fly_ollama.py --scenario landing --duration 30
# 同梱のNDBC海況で評価
python3 fly_ollama.py --scenario takeoff --ndbc --duration 12
# 応答に問題があれば即座に打ち切る接続検証
python3 fly_ollama.py --duration 3 --interval 1 --strict
# ローカル System One（Jev 風）。内側は内蔵制御。Ollama不要
python3 fly_ollama.py --jev --spatial --lateral-assist --scenario takeoff --duration 12
python3 fly_ollama.py --jev --jev-calibrator results/flight_jev_002/calibrator.npz --spatial --lateral-assist --scenario takeoff --duration 12
# 性能包絡（環境変数）
HAMA_HS=0.3,1.5,3.5 HAMA_WIND_Y=-5,5 python3 results/envelope_001/sweep.py
```

既定の接続先は `http://127.0.0.1:11434`。`--host` でローカルの別ポートを指定できます。
モデルは `--model`、推論のソケットタイムアウトは `--timeout`（既定60秒）、
海況は `--hs` / `--tp`、乱数シードは `--seed` で指定します。
モデル確認に失敗した場合は開始せず、モデルのダウンロードも自動実行しません。

- 入力：高度・速度・昇降速度・波面・船底クリアランス・浸水量・目標・前回指令。
- 出力：JSONの `throttle`（0〜1）と `pitch_deg`（−8〜15度）。型・有限値・範囲・必須キーを検証。
- 制御：既定でシミュレーション1秒ごとに判断し、0.05秒刻みで指令を適用。
  **推論中はシミュレーションを一時停止します。実時間制御・実機接続はありません。**
- エラー：推論失敗や不正な出力は記録し、次の判断まで既存の制御器を各物理ステップで使用。
  `--strict` ではフォールバックせず終了コード2で停止します。
  故障・非武装時の推力停止は既存の機体処理が維持します。
- 記録：`results/ollama_<UTC日時>/` に `config.json`、`decisions.jsonl`、
  `trajectory.jsonl`、`summary.json` を保存。出力先は `--output` で変更でき、既存ディレクトリの上書きは拒否します。
  判断ログには生のモデル応答・推論時間・フォールバック理由を残します。

`summary.json` の `status` は `success`、`time_limit`、`damage_failure`、
`impact_failure`、`hard_landing`、`pilot_error`、`interrupted`。
`time_limit` は飛行目標達成を意味しません。通常の飛行評価結果は終了コード0、
接続・推論エラーや中断は2です。自動評価では `status` と `fallback_decisions` を確認してください。

実装時にこの環境の `phi3.5:latest`（モデルID先頭 `61819fb370a3`）で実推論を確認しました。
[離水シナリオの接続試験](results/ollama_phi35_smoke/summary.json) は3判断・60物理ステップ、
フォールバック0回、シミュレーション3秒／実時間約28秒。離水成功の試験ではありません。
単体テストはサービス不要で `python3 audit.py` に組み込まれています。

API仕様の参照：[Ollama Chat API](https://docs.ollama.com/api/chat)、
[Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)。

## Phiを教師として使う（推奨構成）

Phiの推論は教材収集時に使い、操縦時は小さなNumPyネットワークに任せます。
`teacher_student.py` は `decisions.jsonl` の観測・目標から教師のスロットルと
ピッチを模倣学習します。`source=ollama` の有効な指令だけを採用し、
フォールバック、生徒自身の出力、中断記録は教材から除外します。
教材ファイルのSHA-256と学習誤差を保存します。

```bash
# 1. 教師データの追加収集（Ollamaが必要）
python3 fly_ollama.py --scenario takeoff --duration 12 --seed 10 --output results/teacher_takeoff_10
python3 fly_ollama.py --scenario landing --duration 30 --seed 20 --output results/teacher_landing_20

# 2. 保存済み教師ログから生徒を学習（Ollama不要）
python3 teacher_student.py results/teacher_takeoff_10/decisions.jsonl results/teacher_landing_20/decisions.jsonl --output results/student_new

# 3. 生徒で操縦（Ollama不要、各物理ステップで判断）
python3 fly_ollama.py --student results/student_new/student.npz --scenario takeoff --duration 12 --interval .05
```

接続確認用の生徒モデル `results/phi_student_v1/student.npz` を同梱しています。
既存の教師応答5件で学習したモデルで、学習誤差は汎化性能や飛行成功率を意味しません。
未経験の海況・進入高度・飛行段階での評価と、教材の増量が必要です。
教師のJSON形式が正しくても操縦判断が正しいとは限りません。
現行学習は教師の判断を模倣するもので、成功軌跡の選別や報酬による改善は未実装です。
生徒の入力・ピッチ範囲は専用で、既存RLの `ActorCritic` チェックポイントとは互換ではありません。

## 育成サイクルの実行

```bash
python3 educate.py --output results/education_002
```

`educate.py` はPhiに32件の操縦指令を依頼します。離水の滑走・抵抗域・引き起こし・
上昇・高度到達、着水の進入・降下・フレアという8段階を、4つの海況／シード条件で提示します。
24件を学習、別シードの8件を検証に使います。初期化3種類・最大4000更新を試し、
検証誤差が最小のモデルを保存します。旧5件モデルは比較用として保持します。

保存後、学習にもモデル選択にも使わないシード101・102、波高0・0.3・0.8 mで
離水・着水を評価します（各制御器12試行）。旧生徒・新生徒・既存制御器を同じ条件で比較し、
`report.json` に成否・終了理由・最終状態・学習曲線・データハッシュを記録します。
教師の判断を収集した初期状態は設計したカリキュラムであり、成功軌跡の保証はありません。

出力：`curriculum.json`、`teacher.jsonl`、`train.jsonl`、`validation.jsonl`、
`student.npz`、`report.json`。教師収集中に停止した場合は同じコマンドで保存済みの指令を
再利用します。完了済みモデルを上書きせず、新しい出力先で次のサイクルを実行します。
学習・評価だけを再開する場合は `--evaluate-only` を指定できます。

## シミュレーション結果を返して再指導する

```bash
python3 feedback_education.py --source results/education_001/teacher.jsonl --output results/education_002
```

第1回の各状態を復元し、スロットルとピッチの候補をそれぞれ2秒間試します。
元の指令とスコア上位候補の実測結果をPhiへ提示し、選択した候補を新しい教材にします。
短期スコアは離水時の加速、飛行中の高度追従、着水時の降下速度・接水反力を評価します。
このスコアは局所的な代理指標で、飛行完遂や実機安全性を保証しません。

Phiの返答は候補番号として検証し、候補に存在しない指令は採用しません。
各教材に全候補の結果・元の指令・Phiの選択・生の応答を残します。
物理・損傷係数を変更せずに再学習し、第1回モデルと比較します。
第2回では新しい評価シード201/202を使い、評価結果でチェックポイントを選びません。
後続の反復でもシード201/202を再利用する場合は、既知のベンチマークとして扱ってください。


## 横方向の運動と大気（spatial モード）

`EnvConfig(spatial=True)` または `train.py --spatial` で横方向を含む学習を実行できます。
座標は x＝前方基準、y＝横方向、z＝上向き。位置 x/y/z、速度 Vx/Vy/Vz、
バンク角、方位角を積分します。バンクは時定数0.5秒の指令追従、方位は空中の
協調旋回近似とラダーによるヨーレート指令で更新します。

```bash
# 横風3 m/s、各成分の突風RMS 0.5 m/s、方向性海面で学習
python3 train.py --scenario takeoff --spatial --directional \
  --wind 0 3 0 --gust-rms 0.5 --seed-per-episode --episodes 400
```

- `--wind WX WY WZ`：高度 10 m での空気速度（m/s、世界座標）。気象の「吹いてくる方位」とは異なります。水平平均風と突風は z0=0.001 m の対数則で高度換算します。
- 大気密度は ICAO ISA 対流圏（海面 1.225 kg/m³）。1D 経路も同じ密度を使います。突風はシード固定の滑らかな合成正弦波で、同時刻の再読込は同じ値になります。
- 空力は対気速度（対地速度 − 風速）から算出。揚力勾配は有限翼、誘導抗力は地面効果付き。水抵抗は対地水平速度に作用します（海流なし）。
- 行動4次元：スロットル[0,1]、ピッチ[-1,1]、バンク[-1,1]（±45°）、ラダー[-1,1]（ヨーレート±20°/s）。
- 観測は既定19次元。従来の11次元に y/10、Vy/15、バンク/45°、sin方位、cos方位、風速3成分/15を追加します。前方波面は実際のy位置と水平進行方向から予測します。
- 成功には従来条件に加え |y|<10 m、|Vy|<1.5 m/s、|バンク|<10° が必要です。|y|>50 mで失敗終了します。横ずれ・横速度にも報酬ペナルティを適用します。
- 学習出力名には既定で `_spatial` が付きます。旧11入力・2出力モデルとは互換性がありません。着水の従来BC教師は使用せず、新規学習します。

`atmosphere.py`・`weather.py`・`spatial_dynamics.py` は簡易モデルです。完全な6自由度剛体運動、
空力モーメント・失速後の横安定性、波面傾斜によるロール、左右フロート接触、
気象予報・標準大気の厳密実装ではありません（気象層は解析的・定常で、時間変動する
前線やメソスケール現象は扱いません）。学習環境・MAVLink風機体は共通の
空間運動計算を使用します。Ollama／RL操縦と3D表示への接続は以下を参照してください。


## 気象（weather.py）

`Weather` は `Atmosphere` と同じインターフェース（temperature / pressure /
density / shear_factor / wind）を持つ決定論的な気象層で、spatial モードの大気を
そのまま置き換えます。加えて `effects()`（機体に働く全気象項のスナップショット）と
`step()`（着氷質量の積分）を提供します。未設定時の挙動は従来とビット単位で同一です。

```bash
# 雷雨プリセットで飛行（--wind/--gust-rms 未指定時はプリセット推奨値を採用）
python3 fly_ollama.py --spatial --weather storm --policy runs/…/best.pt --duration 12
# 着氷を伴う降雪下でRL学習（出力タグは takeoff_spatial_snow）
python3 train.py --scenario takeoff --spatial --weather snow --seed-per-episode
```

### 天候プリセット（`--weather`）

| プリセット | 気象 | 海面密度 kg/m³ | 推奨風・突風RMS m/s |
|---|---|---|---|
| `clear` | 晴天（温圧偏差なし・湿度50%） | 1.221（ISA比 −0.3%） | なし |
| `heat_wave` | 猛暑 +15 K・乾燥 | 1.160（−5%） | 0 / 0.3 |
| `low_pressure` | 低気圧 −25 hPa・薄雲 | 1.181（−4%） | (−4,−2) / 0.8 |
| `overcast` | 曇天・雲底800 m | 1.223 | (−5, 2) / 1.0 |
| `fog` | 濃霧・視程150 m | 1.222 | (−1, 0) / 0.2 |
| `rain` | 降雨 8 mm/h・視程4 km | 1.217 | (−7, 3) / 1.5 |
| `storm` | 雷雨 35 mm/h・下降気流 −3 m/s・雷リスク0.8 | 1.206（−2%） | (−12, 5) / 3.5（Dryden） |
| `snow` | 降雪 −10 °C・着氷条件 | 1.334（+9%） | (−4,−2) / 1.0 |

### 物理法則とドローンへの影響

- **湿り空気密度**：水蒸気は乾燥空気より軽い（R_V=461.5 対 R=287.05 J/(kg·K)）ため、
  同圧・同温なら湿度が高いほど空気は軽く、揚力・プロペラ推力（ともに ∝ρ）が落ちます。
  飽和蒸気圧は Magnus 式（0 °C 以上は水面、未満は氷面）。
  ρ = (p−e)/(R·T) + e/(R_V·T)。気温・気圧の視覚的偏差（offset）も同じ経路で密度高度を変えます。
- **降雨・降雪**：降水強度 R (mm/h) から含水量 LWC = R/(3.6×10⁶·v_t)、
  落下終端速度 v_t = min(9, 2.5·R^0.24) m/s（雪は 1 m/s）。雨粒の運動量交換で
  ΔCD = LWC·(V+v_t)/(½ρV) の追加抗力、プロペラ濡れで推力最大 −10%、視程低下。
- **雲・霧**：雲は cloud_base–cloud_top の層に cover×0.3 g/m³（プリセット既定）。
  霧は視程 <1 km かつ湿度 ≥0.95 のとき LWC = 0.15·√(200/vis) g/m³ で
  高度300 m まで線形減衰。降雨による視程低下（>1 km）は霧として扱いません。
- **着氷**：Messinger 風付着 dm/dt = 0.45·LWC·V·S·f(T)（雪は効率 ×0.4）。
  f(T) は 0 °C で 0、−8 °C で最大 1、−28 °C で再び 0（雲が氷晶化）。
  氷 1 kg ごとに CD0 +0.004、CL_max −3.5%、プロペラ効率 −2%（上限 12 kg、
  CL_max は最低 50%、プロペラは最低 60%）。付着した氷は `extra_mass` として
  機体質量にも加算されます。
- **下降気流**：storm の −3 m/s は風速 z 成分に常時加算され、上昇性能を直接奪います。
- **雷**：リスク指標（0–1）をテレメトリに載せるのみで、落雷・故障はシミュレートしません。

### 接続点

- `EnvConfig(weather=WeatherConfig(…))`（spatial=True 必須）：info に
  `ice_mass_kg` と `weather` サマリー、`episode_conditions["weather"]` に設定を記録。
- `FlyingBoatVehicle(…, weather=WeatherConfig(…))`：テレメトリ snapshot に
  `ice_mass_kg` / `weather` を追加。`reset()` で着氷もリセットされます。
- `integrate(…, weather=w)`：気象項は呼び出しごとに初期状態で1回評価し、
  サブステップ中は一定（定常気象）とします。`weather=None` では従来とビット同一。
- `fly_ollama.py --weather PRESET` / `train.py --weather PRESET`：
  どちらも `--wind`/`--gust-rms` を明示した場合はそちらが優先されます。

すべて解析的・シード固定で、同一シードなら風・突風・着氷は完全に再現します。
プリセットは `weather.PRESETS`、任意条件は `WeatherConfig` の13フィールド
（気温/気圧 offset、湿度、降水強度、雪、視程、雲底/雲頂/雲量/雲LWC、
下降気流、雷リスク）で構成します。


### 過去気象リプレイ（weather_real.py）

`weather_real.py` は NOAA NDBC ブイの実測毎時データ（realtime2 `.txt` の気象列
WDIR/WSPD/GST/PRES/ATMP/DEWP）を読み、`HistoricalWeather`（`Weather` のサブクラスで
`Atmosphere` 互換）としてシミュレーション時間に沿ってリプレイします。プリセットではなく
実データで観測された気象経過をたどれます（同梱 `data/ndbc_46012_realtime.txt` は
Humboldt Bay, CA の1096時間＝45.7日分）。

| 観測列 | モデル量 | 変換 |
|---|---|---|
| PRES [hPa] | `pressure_offset_Pa` | (PRES−1013.25)×100 を ISA 高度分布に加算 |
| ATMP [°C] | `temp_offset_K` | ATMP−15.0 を ISA 減率に沿って高度展開 |
| DEWP [°C] | `humidity` | RH = es(Td)/es(T)（Magnus）を [0,1] クリップ |
| WDIR+WSPD | `wind` (m/s) | −WSPD·(cosθ, sinθ)、+x=北 +y=東、z_ref≈10 m |
| GST−WSPD | `gust_rms` | ×0.363（校正値、下記） |

- 風向はスカラーでなく (u,v) 成分で補間するため、359°→1° の折返しで逆回りの偽回転が起きません。
- GST は「1時間内のピーク風速」です。sum4 突風過程の1時間ピークは実測で ≈2.755×gust_rms
  だったため、`gust_rms = GUST_SIGMA_FACTOR×(GST−WSPD)`（係数0.363）でピーク風速を再現します
  （`weather_study.py` で校正：ピークRMSE 3.18→0.28 m/s、約91%改善）。
- 記録に無い降水・雲・視程はベース `WeatherConfig`（既定: 乾燥リプレイ）から重ねます。
- 記録の前後は端値を保持。`time_scale=0` で観測状態を凍結（乱流のみ時間変動）。すべて解析的・決定論的です。

```bash
# 実測リプレイで飛行（rate=360 なら実1時間をシミュレーション10秒で消化）
python3 fly_ollama.py --spatial --weather-real data/ndbc_46012_realtime.txt \
    --weather-real-t0 964 --weather-real-rate 360 --duration 10

# 解析スタディ（忠実度・突風校正・エンベロープ追従・動的反応・Env端到端）
python3 weather_study.py            # -> results/weather_real_001/
```

`EnvConfig(weather_real=…, weather_real_t0=…, weather_real_rate=…)`（spatial=True 必須）と
`FlyingBoatVehicle(…, weather_real=…)` の両方から利用でき、`info["weather"]["record_utc"]` に
リプレイ中の観測時刻、`episode_conditions["weather_real"]` に設定が記録されます。
検証結果は `results/weather_real_001/REPORT.md`：全1096ノードで気温・気圧・密度・風ベクトルの
最大偏差 0（変換チェーン厳密）、突風ピークRMSE 0.28 m/s、荒天/静穏窓の動的反応比が実測
ガスト強度比と向き・オーダーで整合、Env端到端リプレイのテレメトリ偏差 0.000 °C / 0.000 hPa。


### 天気予報モデル（weather_forecast.py）

リプレイが「観測された過去をなぞる」のに対し、`weather_forecast.py` は同じ実測記録から
**将来を予報する統計モデル**へ気象層を成長させます。`VarForecastModel` は日周期気候値
（UTC時刻別の平均、学習窓のみでフィット）からの異常ベクトル
y = (PRES, ATMP, DEWP, u, v, GST−WSPD) 上に VAR(p)（線形逆モデル）をリッジ最小二乗で
フィットし、発行時刻の実測状態を初期条件として再帰予測します。コンパニオン行列の
スペクトル半径をチェックし、全モードが減衰するまで係数を縮小するため（半径 < 1 保証）、
予報はリードとともに必ず気候値へ緩和する有界な誘導曲線になります。lead 0 は解析値
（実測）そのものです。

- **学習の誠実性**: `train_stop` の既定は発行時刻＝モデルは発行より前の記録だけで学習します。
  `MIN_TRAIN_HOURS = 336`（2週間）未満の履歴しかない発行時刻は `ValueError`。
- `forecast_series()` は予報軌道を `MetSeries` に戻し変換（u,v→風速/風向、ガスト=風速+超過、
  RHクリップ）、`forecast_weather()` はそれを `HistoricalWeather` に渡すため、
  env / mavlink_if / fly_ollama の既存消費者は無変更で「予報の下」を飛べます。
  シミュレーション時刻 0 が発行時刻、`time_scale` はリプレイ同様リード時間へ対応。
- 基準予報 `persistence_forecast`（解析値保持）/ `climatology_forecast`（日周期のみ）と
  `rmse_vs_lead` / `skill_score` を同梱。すべて解析的・決定論的です。

```bash
# 予報下飛行（発行700h＝2026-09-03 04:00 UTC の実測から48h先まで予報、rate=360）
python3 fly_ollama.py --spatial --weather-forecast data/ndbc_46012_realtime.txt \
    --weather-forecast-issue 700 --weather-forecast-rate 360 --duration 10

# 検証スタディ（次数選択・スキル・減衰・嵐ケース・予報下飛行）
python3 weather_forecast_study.py     # -> results/weather_forecast_001/
```

`EnvConfig(weather_forecast=…, weather_forecast_issue=…, weather_forecast_rate=…)` と
`FlyingBoatVehicle(…, weather_forecast=…)`（いずれも spatial=True 必須、`weather_real` と
排他）から利用でき、`episode_conditions["weather_forecast"]` に設定が記録されます。

検証結果は `results/weather_forecast_001/REPORT.md`：全予報が発行前の記録のみで学習する
expanding window プロトコル（順序選択窓14件・検証窓24件・H=48h）で選択された VAR(6) は、
検証窓平均で lead 48 h に全6変数で永続予報を上回ります（スキル +0.14〜+0.42）。
短リード（1〜6h）は実況慣性が強く永続予報が優位、という実務どおりの使い分けが現れ、
正規化異常ノルムは lead 0→48 h で比 0.40 へ単調減衰。嵐ケース（発行 2026-09-05 23:00 UTC）
でも風速・気圧・気温を実測追跡し、`FlyingBoatEnv` 端到端エピソードのテレメトリは
予報系列と偏差 0.000 °C / 0.000 hPa で一致します。


## 空間運動の機体・操縦・表示への接続

`FlyingBoatVehicle(..., spatial=True, atmosphere=AtmosphereConfig(...))` で
横運動・大気をMAVLink風機体にも適用します。`spatial_dynamics.integrate` は学習環境と共通で、
機体側は飛沫による推力低下・浸水による質量増加・非武装／故障時の推力停止を追加します。
積分刻みは最大0.01秒です。従来モードは引き続き使用できます。

```bash
# ローカルPhiの4軸操縦。3秒の接続試験＋図の保存
python3 fly_ollama.py --scenario landing --spatial --directional \
  --wind 0 3 0 --gust-rms 0.5 --duration 3 --interval 1 --strict --render

# 保存済み空間RLモデルを機体側で動かす（Ollama不要）
python3 fly_ollama.py --policy results/spatial_001/landing_policy.npz \
  --scenario landing --spatial --directional --wind 0 3 0 --gust-rms 0.5 \
  --duration 3 --interval .05 --render
```

上記RLファイルは2エピソードだけの動作確認モデルです。操縦性能を実証したものではありません。
`--policy` は既定の11入力・2出力または19入力・4出力のRLモデルに対応します。
従来のPhi生徒モデル（`--student`）は空間モードでは使用できません。

- Ollamaは `throttle`、`pitch_deg`、`bank_deg`、`rudder` の厳密なJSONスキーマで応答します。値の範囲と有限値を検証します。
- サーボ1/2は推力／ピッチ、3/4はバンク±45°／ラダー±1。`step(action=...)` は空間モードで4要素を受け取ります。
- TAKEOFF/LANDはy=0の進路維持、WAYPOINTはローカル `x`/`y` と `alt` を使用します。`heading` は北を0°・東を90°とする絶対方位で、指定時は位置からの方位算出より優先します。CONDITION_YAWは `heading` を更新します。
- 推論失敗時は既存の縦制御と横位置／横速度を使う制御器で4軸フォールバックします。`--strict` は適用前に停止します。
- `read_telemetry()` の既存3要素を保ち、snapshotにy・Vy・バンク・方位・角速度・風・対気速度・密度を追加。`read_attitude()` でATTITUDE相当のメッセージを取得できます。
- 位置は内部で北／東／上。GLOBAL_POSITION_INT相当の速度は北／東／下のcm/s、緯度はx、経度はyから換算します。HIL相当は従来の模擬フィールド（速度m/s、加速度m/s²）で、ワイヤ形式との完全互換はありません。
- `trajectory.jsonl` に観測・4軸指令・適用状態・テレメトリを保存。`config.json` に実行時ソースのハッシュ、`--render` で `scene.png` を保存します。3D表示は横移動・バンク・方位と2D海面に対応します。

実接続記録は `results/spatial_connected_phi_001/`（Phi、3判断・60ステップ・フォールバック0回）です。
3秒の試験は飛行完遂を意味しません。Phiはこの試験で横舵を変更せず、約1.43 mの横流れが残りました。
本接続はローカルOllamaとプロセス内シミュレータであり、実機・UDP／シリアル・PX4 SITLへの接続ではありません。


### Phi／RLの横方向補助

`fly_ollama.py --spatial --lateral-assist` は推力・ピッチを操縦モデルに任せ、
バンク・ラダーを物理ステップごとのy=0進路維持制御で置き換えます。
推論間隔中も補正を更新します。既定では無効で、Phi自体の学習改善とは区別します。

```bash
python3 fly_ollama.py --scenario landing --spatial --lateral-assist --directional \
  --wind 0 3 0 --gust-rms 0.5 --duration 3 --interval 1 --strict --render
```

`decisions.jsonl` は操縦モデルの生の指令、`trajectory.jsonl` は元の `control` と
実際に適用した `applied_control`、`lateral_assist` を保存します。
補助なしの場合も `applied_control` に適用指令を記録します。
本補助はy=0の進路維持用です。操縦モデルによる横方向の旋回指示も置き換えるため、
任意の横方向ミッションでは補助を無効にしてください。
比較試験・離水の固定ピッチ試験は `results/control_improvement_001/` に保存しています。


### 離水停滞の診断

空間モードの環境info・機体snapshotには `force_budget` を記録します。
`thrust_x_N`、`aerodynamic_x_N`、`water_x_N` は前後方向の力（N）の物理ステップ平均。
その和が `net_x_N`、`mean_ax_m_s2` は同区間の前進加速度、`total_mass_kg` は浸水を含む質量です。

`fly_ollama.py --scenario takeoff` の `summary.json` には `takeoff_diagnostics` を保存します。
最後3秒間があり、全期間で推力指令≥0.95、速度0〜4 m/s未満、速度幅≤0.5 m/sなら
`low_speed_plateau=true`。診断のみで指令・成功判定・終了時刻を変更しません。
スナップショットの `prop_clearance_m` / `prop_bottom_clearance_m` は飛沫判定に使った幾何です。
離水制御12案は `results/takeoff_control_001/`、取付高さのスイープは
`results/takeoff_spray_001/`、採用後の再評価は `results/takeoff_mount_001/` を参照してください。
既定の `prop_z_offset` は 0.90 m です。

## ブラウザでの手動操縦

```bash
python3 -m operator_training serve --port 8766
```

`http://127.0.0.1:8766/cockpit/` を開き、`Start Manual` で水上から手動操縦を開始します。
`Start Takeoff` は自動離陸です。飛行中の `Take Control` は自動操縦の入力値へ合わせ、
0.5秒の一致を経て手動へ切り替えます。`Pause` で停止し、`Resume` で再開できます。

- W/S: ピッチ、E/D または上下矢印: スロットル
- A/F または左右矢印: バンク、Q/C: ラダー
- ゲームパッド接続時は左スティックでピッチ・バンク、右スティック横でラダー、RTでスロットル

通信断からの自動再接続では同じセッションを使用します。ページ再読み込みは新規セッションを作成します。
Pythonコードを変更した場合はサーバーを再起動してください。

操縦モードの回帰テスト:

```bash
python3 -m unittest discover -s tests -p 'test_operator_*.py'
node tests/test_operator_cockpit.cjs
```

記録済みセッションのフライトレポート（離水後の成否・機体性能・操縦特性・損傷・ASCII 高度プロファイル）:

```bash
python3 -m operator_training report <session_dir> --md report.md --csv series.csv
python3 -m operator_training report --compare <dir1> <dir2> [dir3 ...]
```

カメラの「船上の操作者」は、発進位置付近の船上（眼高3.2 m）から機体を見続ける固定位置の視点です。
追従モードは旋回時も水平を維持し、位置と注視点を滑らかに追従します。視野角スライダーで拡大率を調整できます。
「FPV（機上カメラ+OSD）」は機首（キャノピー）に固定した一人称視点で、機体姿勢に 1:1 で追従します。
キャンバス OSD は人工地平線・ピッチラダー・バンク指針、対気速度テープ（左）・高度テープ（右）・方位ストリップ（上）・
スロットルバー・キールクリアランス、および FAILURE / FLOODING / LOW CLEARANCE などの警報を overlay 表示します。
FPV 選択中は外部視点の機体メッシュを非表示にし、視野角スライダー（20〜100）で拡大できます。
Xboxは手前に倒すと機首上げが既定です。「ピッチ反転」で逆向きに変更できます。
入力が効かない場合、HUMAN表示、コントローラー名、生入力（LY/LX/RX）、適用値を順に確認してください。
AUTO中のスティック入力は飛行操作に適用されません。実機コントローラーでの動作確認は別途必要です。

## OSS 地上局（QGroundControl / MAVProxy / Mission Planner）

プロセス内の模擬 API に加え、MAVLink v2 を UDP で話します（pymavlink 不要）。

```bash
# シミュレータ単体。QGC は既定で UDP 14550 を待ち受ける
python3 -m operator_training mavlink --port 14551 --gcs 127.0.0.1:14550

# コックピットと同時。セッション開始後に同じ機体のテレメトリが出る
python3 -m operator_training serve --port 8765 --mavlink-port 14551
```

QGroundControl: 自動接続（UDP 14550）または Comm Link `udp:127.0.0.1:14551`。
MAVProxy: `mavproxy.py --master=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14550`

GCS から Arm → Takeoff。ジョイスティックは `MANUAL_CONTROL`（推力・ピッチ・バンク・ラダー）。
対応コマンド: HEARTBEAT / ATTITUDE / GLOBAL_POSITION_INT / GPS_RAW_INT / VFR_HUD、
`COMMAND_LONG` の ARM/DISARM (400)、TAKEOFF (22)、LAND (21)、DO_SET_SERVO (183)。
PX4/ArduPilot 実機・SITL ファームウェアそのものではなく、KMT 動力学への MAVLink 橋です。

## 変動する海況・風況での4軸学習

```bash
MPLCONFIGDIR=/tmp/kmt-mpl python3 train_robust.py \
  --scenario takeoff --episodes 40 --bc-episodes 12 --eval-episodes 10 \
  --output results/robust_spatial_v1
```

方向分散波と高度による風の変化を持つ空間環境で、教師模倣後に強化学習します。
教師はピッチ・スロットルに加えて、横ずれと横速度を補正するバンク、方位を補正するラダーを出力します。
エピソードごとに、基準波高の0.5〜1.5倍、周期の0.8〜1.2倍、波向±45度、
水平風の各成分±3 m/s、突風RMSの追加0〜0.8 m/sを乱数シードから決定します。
これは訓練用の条件分布であり、実測した気象分布ではありません。

`report.json` に環境設定、訓練履歴、未学習シードの評価条件・成功率と教師制御の比較を保存します。
`takeoff_policy.npz`（着水時は `landing_policy.npz`）と学習曲線も同じ出力先に保存します。
従来のCLIでも `--spatial --directional --randomize-conditions --bc-episodes 12` を利用できます。
乱数条件の有効化で観測・行動の次元は変わりません。新モデルはコックピットのAUTOへ自動採用しません。
力学は既存の簡略モデルのままで、実機性能の保証や実測値による校正を行ったものではありません。

### 学習収束とモデル選択（v2）

```bash
MPLCONFIGDIR=/tmp/kmt-mpl OPENBLAS_NUM_THREADS=1 python3 train_robust.py \
  --episodes 40 --bc-episodes 12 --bc-epochs 80 \
  --eval-seed 10040 --validation-seed 20000 --eval-episodes 10 \
  --output results/robust_spatial_v2
```

教師模倣は128サンプル単位でシャッフルし、正規化した操作量の誤差で学習します。
`bc_report.normalized_action_mse` は教師データ上の各エポック終了時の誤差です。
教師模倣直後（`*_bc_policy.npz`）と強化学習後（`*_policy.npz`）を別保存し、
検証用シードで成功率、同率なら平均報酬を比較して `*_selected_policy.npz` を選びます。
比較評価用シードは選択に使いません。訓練・検証・比較評価シードの重複はエラーになります。
前回と同じ条件で比較する場合は `--eval-seed` と `--eval-episodes` を固定してください。
強化学習を追加しても必ず性能が向上するとは限らないため、選択済みモデルと評価報告を利用してください。
