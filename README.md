# KUMOTA - 15 m 翼幅 スチロール製 ドローン飛行艇 実験プロジェクト

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
| `advisor.py` | Phi助言＋短horizon試行によるカリキュラム生成（Ollama不在時は規則スタブ） |
| `ocean_directional.py` | 方向分散付き2D不規則波（cos^2s、NDBCのMWD対応） |
| `tests/` | 学習・評価・故障処理の回帰テスト |
| `data/` | 取得した NOAA NDBC データ（リアルタイム・スペクトル） |
| `results/` | 出力グラフ・CSV・テキスト |

## 機体設計（aircraft.py）

| 項目 | 値 |
|---|---|
| 翼幅 / 翼弦 / 翼面積 | 15.0 m / 1.5 m / 22.5 m² |
| アスペクト比 | 10.0 |
| 総質量 | 80.0 kg |
| 双発モーター（28"プロペラ） | ピーク12 kW |
| 静止推力 / 重量比 | 0.70 |
| 失速速度 / 巡航速度 | 6.4 / 11.0 m/s |
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

- **環境**：8次元状態（z, Vx, Vz, η, dη/dt, 水没, 速度比, 前回推力）
- **行動**：推力 [0,1]、ピッチ（[-3°, 12°]にマップ）
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
python3 evaluate.py    # スクリプト制御器 + 保存済み海洋データ + 3D評価
python3 validate.py    # 物理サニティチェック
```

## 実験結果（実海洋データ NDBC 46026 における離水）

実海洋（Hs=1.7 m）の条件下では：

| シナリオ | 結果 | 原因 |
|---|---|---|
| 離水 | **失敗**（Vx=2.5 m/s でハンプ停滞） | スラストが海水飛沫で20%低下、船体浸水0.22 kg |
| 着水 | **成功**（\|Vz\|<1 m/s, Vx=0.8 m/s で停止） | スラスト不要、波の影響は小さい |

この表は監査前の旧結果です。修正後の性能を示すものではなく、実機の運用限界も検証していません。

## 注意点・限界

- 動力学は縦方向（x）のみ。`EnvConfig(directional=True)` と `FlyingBoatVehicle` は方向分散付き2D海面の縦断面（y=0）を使える。横方向の運動・旋回・斜め波の影響は未モデル
- ピッチ指令の範囲は経路で異なる。RL環境・機体直接操作はオートパイロット包絡線 -3〜12°、サーボ直接操作はハードウェア可動域 ±15°（Ollamaは-8〜15°で応答）。`advisor.short_simulate` とカリキュラムは環境で実際に適用された値（クリップ後）を記録する
- 機体縦運動のみ（横方向・回転・制御面の独立度は未モデル）
- 風の擾乱は含まず、海面のみが外乱源
- pymavlink のPython 3.14互換ビルドが失敗したため、MAVLinkメッセージ
  識別子を使う自前のプロセス内実装（シリアライズ・通信・ACK未実装、実機接続やプロトコル互換性は未検証）
- 訓練エピソードのランダム化はせず、固定シードで再現性を確保

## 監査と再検証

[監査報告](AUDIT.md) に修正内容と残る制約を記載しています。
`python3 audit.py` は既存のモデル・図を上書きせず、検証結果とソースのSHA-256を保存します。
`python3 -m unittest discover -s tests -v` で回帰テストのみ実行できます。

`EnvConfig(Hs=..., Tp=...)` で学習環境の海況を指定できます。
`sea_state_sweep(..., n_seeds=..., n_envs=...)` は各海況で `n_seeds` 回の試行を行い、
`n_envs` は一度に処理する環境数です。ベクトル環境は逐次実行であり、並列プロセスではありません。
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
