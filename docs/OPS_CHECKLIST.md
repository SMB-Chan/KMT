# KMT 運用チェックリスト／障害対応

**対象読者：** 機体運用者・整備担当・オートパイロット評価担当
**文書範囲：** ローカルシミュレータ・MAVLink 風インターフェース・RL／LLM パイロット

---

## 1. 毎日の運用前チェック

### 1.1 環境

- [ ] Python 3.14.x で起動（コードベースは `3.14.4` 動作確認済）
- [ ] 依存パッケージ：`python3 -m pip install -r requirements.txt` 完了
- [ ] `git status` で意図しない差分がないこと
- [ ] `audit.py` 直近結果が `OK` であることを確認

### 1.2 監査

```bash
python3 audit.py
cat results/audit.json | head -50
```

期待値：
- `checks[0].name == "regressions"` → `passed: true`
- `checks[1].name == "physics"` → `passed: true`
- `source_sha256` に**意図しない差分**がないこと

### 1.3 既知の差分

次のファイルを編集した場合、監査の差分（SHA-256 変化）が出ます。**正常な変更か、差分を記録しているか**確認してください。

| ファイル | 役割 |
|---|---|
| `train.py` | 訓練エントリポイント |
| `tests/test_new_modules.py` | 新規モジュール単体テスト |
| `README.md` | 利用者向け説明 |
| `results/audit.json` | 直近の監査出力（自動再生成） |

操縦モード（コックピット／QGC SITL）は次の回帰テストで守られています：

```bash
python3 -m unittest discover -s tests -p 'test_operator_*.py'
node tests/test_operator_cockpit.cjs
```

---

## 2. 標準作業フロー

### 2.1 基本出力の生成

```bash
python3 main.py
```

期待出力（`results/`）：
- `takeoff.png` / `landing.png`
- `wave_field.npy` / `wave_field.png`
- `profile.png`
- `summary.txt`
- `timeseries_takeoff.csv` / `timeseries_landing.csv`

`summary.txt` のチェックポイント：

| 項目 | 期待値（Hs=1.5, Tp=6.0, seed=42 既定） |
|---|---|
| Lift-off time | ≈ 3.9 s |
| Lift-off Vx | ≈ 7.5 m/s |
| Touchdown \|Vz\| | ≈ 1.2 m/s |
| Peak N_water | ≈ 2.8 W |

外れた場合はシード差ではなく**コード改変／環境異常**を疑ってください。

### 2.2 MAVLink 経路での評価

```bash
python3 evaluate.py
```

期待出力：
- `mavlink_takeoff_profile.png` / `mavlink_landing_profile.png`
- `mavlink_*_telemetry.csv`
- `scene_takeoff_*.png` / `scene_landing_*.png`

**12 秒／30 秒のシナリオでは目標未達で終了します。** これは既設計上で、MAVLink 経路の性能限界を表します。

### 2.3 OSS 地上局（QGroundControl / MAVProxy）からの SITL 操縦

```bash
# SITL 単体（既定で QGC のオートコネクト UDP 14550 へ送出）
python3 -m operator_training mavlink --port 14551 --gcs 127.0.0.1:14550

# ブラウザコックピットと同じセッションを MAVLink で配信
python3 -m operator_training serve --port 8765 --mavlink-port 14551
```

- QGroundControl は `Application Settings → Comm Links` で `UDP Listen on 14550` を追加するか、起動時に 14550 でオートコネクト
- MAVProxy: `mavproxy.py --master=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14550`
- GCS から Arm → Takeoff / Land。ジョイスティックは `MANUAL_CONTROL`
- 1 GCS 接続でも機体側スレッドは 50 ms 周期でテレメトリを送出

### 2.4 ブラウザコックピット

```bash
python3 -m operator_training serve --port 8766
# → http://127.0.0.1:8766/cockpit/
```

- `Start Manual` / `Start Takeoff` / `Take Control` / `Pause` / `Resume` ボタン
- W/S: ピッチ、E/D・↑/↓: スロットル、A/F・←/→: バンク、Q/C: ラダー
- ゲームパッド: 左スティック＝ピッチ・バンク、右スティック横＝ラダー、RT＝スロットル
- 通信断からの自動再接続は同じセッションを再利用します

### 2.5 物理サニティ

```bash
python3 validate.py
```

`OK` 行が出ていれば通過。`AssertionError` が出たら次の項目を確認：

- 環境変数 `MPLCONFIGDIR` を確認
- `audit.py` と `validate.py` を別々に再実行して切り分け
- 直前のコミットで `aircraft.py` / `ocean.py` / `dynamics.py` の係数を変更していないか確認

---

## 3. RL 訓練時のチェック

### 3.1 既定での着手

```bash
python3 train.py                          # 離水・着水の 400 エピソード訓練
python3 train.py --scenario takeoff        # 離水のみ
python3 train.py --scenario landing        # 着水のみ
```

### 3.2 ロバスト訓練

シード毎の波面ランダム化：

```bash
python3 train.py --scenario takeoff \
  --directional --theta-mean-deg 35 \
  --seed-per-episode --episodes 400
```

タグは自動で `takeoff_dir35_rand` のように付与されます。手動上書きは `--tag takeoff_<任意の文字列>` で可能です。
**監査前のタグは固定シード前提であり、ランダム化後の性能と直接比較しないでください。**

### 3.3 評価時の注意

- 1 試行の `n_seeds=5` はパイロット規模です。統計的有意性はありません
- 訓練済みモデル（`results/*_policy.npz`）は**修正後の成功判定で再評価されていません**
- 必ず `--seed-per-episode` を付けて学習し、複数海況で評価してください

---

## 4. Ollama / Phi3.5 パイロットのチェック

### 4.1 接続

```bash
ollama serve                           # 別ターミナル
python3 fly_ollama.py --check
```

`model.name` が `phi3.5:latest`、またはモデル ID 先頭が `61819fb370a3` であれば接続成功です。

### 4.2 接続失敗時の初動

| エラー | 推定原因 | 対処 |
|---|---|---|
| `Connection refused` | Ollama 未起動 | `ollama serve` を別ターミナルで起動 |
| `model not found` | モデル未取得 | `ollama pull phi3.5` |
| `PilotError: incomplete` | モデル出力の JSON 不整合 | `--timeout` を増やす／`--strict` で即停止 |
| 推論がタイムアウト | モデルが重いかプロンプトが大きすぎる | `--model` を小さいものに変更／`--interval` を大きく |

### 4.3 動作確認

```bash
python3 fly_ollama.py --duration 3 --interval 1 --strict
```

3 秒・1 秒間隔・strict モードで、推論・記録系の最小動作を確認します。
`status: success` でも飛行成功ではありません。`status: time_limit` が正常終了です。

### 4.4 NDBC 海況での評価

```bash
python3 fly_ollama.py --scenario takeoff --ndbc --duration 12
```

同梱の NDBC スナップショット（Hs_total ≈ 1.7 m）は訓練教材より**厳しい条件**です。
最初の段階では `--hs 0.3` で評価してください。

---

## 5. 学生モデル（蒸留ネットワーク）のチェック

### 5.1 既存モデルの確認

```bash
ls -l results/phi_student_v1/student.npz
python3 -c "from teacher_student import StudentPilot; s = StudentPilot.load('results/phi_student_v1/student.npz'); print('ok')"
```

### 5.2 推論実行

```bash
python3 fly_ollama.py --student results/phi_student_v1/student.npz \
  --scenario takeoff --duration 12 --interval .05
```

`--interval` は `--dt`（既定 0.05 s）以上に設定してください。

### 5.3 訓練時に注意

- 教師ログの SHA-256 を保存します。同じ教材で再学習しないこと
- 学習データは `source: "ollama"` のレコードのみ。`fallback` / `aborted` / `student` は除外
- 学習回数は最大 4000 更新（既定）から始め、必要に応じて調整

---

## 6. 育成サイクル・フィードバック再指導のチェック

### 6.1 育成サイクル

```bash
python3 educate.py --output results/education_<NNN>
ls results/education_<NNN>/
```

期待出力（5 種＋モデル）：

| ファイル | 必須 |
|---|---|
| `curriculum.json` | ✓ |
| `teacher.jsonl` | ✓ |
| `train.jsonl` | ✓ |
| `validation.jsonl` | ✓ |
| `student.npz` | ✓ |
| `report.json` | ✓ |

### 6.2 中断からの再開

`--evaluate-only` で学習・評価部分のみを再実行できます。
教師収集中に停止した場合、同じ出力先で再実行すれば既存ログを再利用します。

### 6.3 フィードバック再指導

```bash
python3 feedback_education.py \
  --source results/education_<N>/teacher.jsonl \
  --output results/education_<N+1>
```

- 候補番号として Phi の返答を検証
- 候補に存在しない指令は採用されません
- 物理・損傷係数は変更しません

### 6.4 既知の落とし穴

- シード 201/202 は再利用されるベンチマーク。**後続サイクルでも同一条件で評価**
- 検証誤差が最小のモデルを保存（評価結果では選びません）
- 旧生徒・新生徒・既存制御器の比較は同条件・同試行数で実施

---

## 7. 障害対応マトリクス

### 7.1 監査失敗

| 症状 | 推定原因 | 対処 |
|---|---|---|
| 回帰テスト失敗 | ロジック改変／環境差 | `python3 -m unittest discover -s tests -v` で個別実行 → 失敗箇所を読む |
| 物理サニティ失敗 | `aircraft.py` / `ocean.py` / `dynamics.py` の係数改変 | `git diff` で差分確認、戻す |
| ソース SHA-256 差分 | コード変更（想定内）か不正改変か | 変更意図と比較 |

### 7.2 訓練が収束しない

| 症状 | 推定原因 | 対処 |
|---|---|---|
| 成功率が 0% で停滞 | シード毎の波面トラップ | `--seed-per-episode` |
| loss が NaN | 学習率過剰／データ破損 | `lr_body 3e-4 → 1e-4`、教材ハッシュ確認 |
| 学習が遅い | 海況が厳しすぎる | `--hs` を下げて訓練（0.3 → 1.0 → 1.5） |

### 7.3 Phi パイロットの不具合

| 症状 | 推定原因 | 対処 |
|---|---|---|
| 推論失敗が頻発 | プロンプト／モデル精度 | `--interval` を大きく、`--timeout` を増やす |
| JSON 形式の不整合 | モデル差替 | 接続テストで確認 `--check` |
| `interrupted` 終了 | キーボード割り込み | CI 環境では非対話運用に切り替え |
| 連続 `hard_landing` | フレア不足 | 進入高度 25 m、`glide 6°` から調整 |

### 7.4 損傷・アビオニクス故障

| 症状 | 推定原因 | 対処 |
|---|---|---|
| スリング 2 回故障 | プロペラ没水 | `Hs` を下げる／進入速度を抑える |
| 浸水 ≥ 0.50 kg | 接水時間が長い | `MAV_CMD_NAV_LAND` のフレアを確認、`speed` を下げる |
| `damage_failure` | 上記の累積 | 機体再校正後に `--seed-per-episode` で再学習 |

---

## 8. 過去事例スタディ（要約）

`results/directional_study_*.json` の所見：

| スタディ | 主な所見 | 推奨 |
|---|---|---|
| `directional_study_001` | 1D と dir0 で 0/5、dir35 と dir60 で 2/5 | 斜め波が滑走を妨げない条件に着目 |
| `directional_study_002` | n=20 でも dir60 が 10/20 (50%)。Hs2 では dir35 が 1D の 5 倍成功率 | 方向分散込みの訓練が必要 |
| `directional_study_003` | 訓練シード依存性：seed=1 で 100%、seed=0 で 0% | **シード毎ランダム化が必須** |
| `directional_study_004` | ランダム化訓練で 1D 60% / dir35 30%。seed=0 のトラップを回避 | n_seeds=10 規模の検証を推奨 |

活用上の注意：

- `n=5`〜`n=10` はパイロット規模であり**統計的有意性はありません**
- 学習済みモデルは**修正後の成功判定で再評価されていません**
- 結果の解釈は AUDIT.md の制約事項を踏まえてください

---

## 9. 資材の完全性

### 9.1 必要なファイルが揃っているか

```bash
ls -1 *.py | sort
ls -1 tests/*.py
ls -1 data/
ls -1 results/ | head
```

期待される構成は `MANUAL.md` 第 17 章参照。

### 9.2 監査前の成果物かどうか

`results/` 配下のファイルは**監査前の出力**である可能性があります。
学習済みモデルを実機や重要な意思決定に用いないでください。
再評価は本チェックリストの手順で実施してください。

---

## 10. 連絡先・エスカレーション

シミュレータ・コード上の不具合：

1. `audit.py` を実行して再現条件を記録
2. 直前の `git diff` で差分を保存
3. 失敗時の `audit.json` と該当 `tests/test_*.py` の単体実行結果を併せて報告

報告時に含める情報：

| 項目 | 取得方法 |
|---|---|
| Python バージョン | `python3 --version` |
| OS | `uname -a` |
| 監査結果 | `results/audit.json` |
| シード・引数 | 実行コマンドをそのまま |
| 学んだモデル | `results/*.npz` の `ls -l` |
| NDBC のコピー | `data/` の更新日 |

---

## 11. 用語集

| 用語 | 意味 |
|---|---|
| audit before | 監査前の成果物。再利用不可 |
| HIL_STATE | Hardware-In-the-Loop 機体状態（シミュレータ用 MAVLink メッセージ） |
| FAILSAFE | 損傷故障後の状態。推力停止・再武装拒否 |
| tail success | 学習曲線末尾 30 エピソードの成功率 |
| train succ | 訓練中の成功率 |
| V_stall | 失速速度 6.4 m/s |
| V_cruise | 巡航速度 11.0 m/s |
| Hs / Tp | 有義波高・卓越周期 |
| MWD | Mean Wave Direction（NDBC）。本シミュレータは `theta_mean_deg` に対応 |
| cos²s 分布 | 方向分散関数。s が大きいほど指向性が鋭い |
| PW | Pitch Width（エレベータ／ピッチ可動域） |
| A2C / GAE | Advantage Actor-Critic / Generalised Advantage Estimation |
| clipping | クリップ後の値。`advisor.short_simulate` のログには適用後値が記録される |

---

## 12. チェックリスト：新規着手者向け

- [ ] 本文書と `MANUAL.md` を通読
- [ ] `AUDIT.md` で監査内容と残る制約を確認
- [ ] `audit.py` を実行して通過を確認（regressions + physics）
- [ ] `python3 main.py` で基本出力を生成
- [ ] `python3 evaluate.py` で MAVLink 経路を実行
- [ ] `python3 -m operator_training mavlink` を起動し、QGroundControl / MAVProxy から接続確認
- [ ] `python3 -m operator_training serve --port 8766` でブラウザコックピットを開く
- [ ] Ollama 環境があれば `fly_ollama.py --check` で接続
- [ ] 同梱 `results/phi_student_v1/student.npz` で学生モデルを試走
