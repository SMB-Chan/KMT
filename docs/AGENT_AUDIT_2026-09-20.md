# 監査レポート（エージェント実施、2026-09-20）

対象：KMT リポジトリ全体（Python 33 ファイル＋tests 9 ファイル＋docs・data・README/AUDIT）。
実施内容：全ソース精読、回帰テスト 86 件実行（全 PASS）、`python3 audit.py`
（regressions PASS / physics PASS）、および疑わしい挙動の数値再現検証。
Python 3.11.2 / numpy 2.4.6 / matplotlib（venv）で実行。

判定区分：
- **F（修正すべき）**：挙動が仕様・文書・物理と矛盾する、または特定経路で機能しない欠陥
- **I（改善すべき）**：正しさは担保されているが、維持性・整合性・文書でリスクがある事項

---

## F. 修正すべき点

### F1. 船体抵抗曲線に不連続ステップ（`dynamics.py` `HullDrag.resistance`, 88–141 行）

- λ = V/√(g·Bwl) が 1.4 を越える瞬間（V ≈ 3.25 m/s）に、planing 係数が
  `hump_Cv=0.18` から `Cv_planing/√λ ≈ 0.0101` へ**即座に切り替わる**。
  再現数値：V=3.25 m/s で R≈478 N → V=3.30 m/s で R≈195 N（283 N ≈ 重量 36%
  の後方力が一瞬に発生）。
- docstring は "The hump and planing curves meet smoothly without
  singularities" と書いているが、実際には硬い段差がある。
- 3–4 m/s はまさに離水の「ハンプ停滞」帯（README の診断基準 0–4 m/s と重なる）。
  RL 学習・スクリプト制御器の離水挙動に非線形の「崖」を仕込んでいる。
- 修正案：λ のある区間（例 1.2–1.8）で Cv を滑らか補間（smoothstep 等）する。
  修正時は `results/` の離水関連結論の再確認が必要（既往の成功率データは
  この曲線上で得たもの）。

### F2. NDBC MWD → 波進行方向の座標変換が誤り（`ocean_directional.py` `from_ndbc`, 234–300 行）

- 本プロジェクトの座標は x=北、y=東、z=上（`spatial_dynamics.py` docstring、
  README「空間運動」節とも一致）。この系では波の進行方位（コンパス方角）が
  θ（+x からの角度）と数値上一致する。MWD は「波が来る方位」なので
  正しい変換は `theta_deg = (mwd_deg + 180.0) % 360.0`。
- 現行 `(270.0 - mwd_deg) % 360.0` は、コメントに書かれた別座標系
  （"0 deg = from North (+y in our convention), 90 = from East (-x)"、
  つまり北=+y・東=-x）で導いた式を、実座標（北=+x・東=+y）にそのまま
  適用したための誤り。
- 再現：MWD=305°（NW〜N 由来）のとき、正しい進行方向は方位 125°
  (N-0.57, E+0.82) だが、コードは 325° (N+0.82, E-0.57) を出す。
- **テストが誤った式を固定化している**：`tests/test_new_modules.py:34-35`
  が `theta_mean ≈ radians((270.0 - 305.0) % 360.0)` をアサートしており、
  修正時はコード＋テストの同時更新が必要。
- 影響範囲：現行パイプライン（train/evaluate/educate）は自前で
  `theta_mean_deg` を渡すため被害は限定的。`ocean_directional.py __main__`
  のデモ表示が誤った指向性を出している状態。

### F3. `VectorizedEnv.step` が行動次元を (n, 2) にハードコード（`vectorized.py:39`）

- 空間モード（action_dim=4）では `ValueError: actions must have shape (n, 2)`
  で再現確認。空間環境のベクトル化ロールアウトは現状実行不能。
- 修正案：環境の `action_dim` から期待形状を導く（`train_vectorised` の
  `action_low/high` も同様に次元を固定化しており、併せて整理すべき）。

### F4. 海況スイープが着水に 15 秒のホライズンを適用（`accelerate.py:167`）

- `sea_state_sweep`（および `mavlink_sea_sweep`）は両シナリオで
  `max_steps=300`（dt=0.05 で 15 s）固定。一方 `train.py::_config_for`
  は着水を 600 ステップ（30 s）にしている。
- 再現：25 m / 8° グライドのスクリプト着水は初接水まで **約 27 s** であり、
  15 s ホライズンの着水評価はほぼ必ず中途打ち切り（truncated）になり、
  成功率が系統的低く出る。README の着水関連スイープ表をこの経路で再生成
  する場合は誤った結論になる。
- 修正案：シナリオ別の既定ホライズン、または `max_steps` パラメータ追加。

### F5. `Aircraft.V_cruise` の注釈が物理と矛盾（`aircraft.py:125-129`）

- 注釈は "Speed for minimum power (L/D max × √3)" だが、実装
  `V_stall·√3 = 11.0 m/s` は「CL = CL_max/3 での速度」にすぎず、
  最小パワー速度（本機では失速速度 6.38 m/s とほぼ一致）でも、
  V_maxLD×√3 = 14.5 m/s でもない。
- `validate.py:36` も同一恒等式をアサートしており、誤りを検出できない
  循環的チェックになっている。
- 機能影響は表示・README の「巡航速度 11.0 m/s」のみだが、設計文書として
  誤った根拠提示になる。数値は設計意図（巡航点）として残すなら、注釈と
  validate のアサート内容を書き換えるべき。

### F6. `advisor.parse_action` の既定ピッチ範囲が -3..12°（`advisor.py:93-95`）

- プロジェクト全体の包絡線は -8..12°（`EnvConfig`、
  `PhiAdvisor.PITCH_LO_DEG`）。既定引数は `PhiAdvisor` 経由では上書き
  されるが、単体呼び出しでは -8..-3° の命令が黙って -3° にクリップされる
  地雷。既定を -8 に揃えるか、必須引数化を検討。

---

## I. 改善すべき点

### I1. RL ピッチ包絡線 [-8,12]° が 3 箇所にハードコード（整合性リスク）

- `mavlink_if.py:419` `step(action=...)`：`math.radians(2.0 + 10.0*u)`。
  コメントは "shared with EnvConfig" とするが実際はハードコード。
- `policy_pilot.py:40`：`2 + 10 * action[1]`。
- `train.py::_init_bias_for` だけが `EnvConfig` から導出（正しい做法）。
- `EnvConfig.pitch_lo/hi` を調整すると訓練環境と評価経路が黙ってずれる。
  単一の所望（例：`EnvConfig` 導出のヘルパー）に集約すべき。
  （F3 と同種の「既定値の復写」問題が複数箇所に存在）

### I2. 積分器の刻みが経路で非統一（`mavlink_if.py` 縦方向 `step`）

- RL 環境は安定性注記付きで 5 回の 0.01 s サブステップ、空間系は
  0.01 s 以下のサブステップ（`spatial_dynamics.integrate`）だが、
  MAVLink 風機体の縦方向 `step` は 0.05 s の単一 Euler。
  激しい高速・急降下状態での発散耐性が環境より低い（今回の試行では
  25 m/s/-15 m/s の急降下でも発散しなかった）。
- README「積分刻みは最大0.01秒」の主張と縦方向機体経路が食い違う。
- 統合した `step`（サブステップ付き）を機体縦方向にも使うと良い。

### I3. `Propulsion.power_required` にプロペラ効率がなく静止電力が過小（`aircraft.py:101-108`）

- `P = T·V/(η_motor·η_esc)` はプロペラ効率 100% を仮定。V=0 付近は
  クランプで約 314 W になるが、静止推進の理想軸電力は
  T·√(T/2ρA) ≈ 13 kW 級（T_static=549 N、A=0.396 m²）。
- 現状 `aircraft.__main__` の航続時間デモのみに使用（航続を過大評価し得る）。
- シミュレーション本体には電池状態が未実装（AUDIT.md でも既知事項）。
  航続・エネルギー設計に使うならプロペラ効率項を追加すべき。

### I4. `Ocean.elevation_at` の恒零項（`ocean.py:103`）

- `self.eta(np.array([x0]), float(0)) * np.zeros_like(t) + ...` は初項が
  常に 0 の死に表現（かつ wasteful）。残りの式単体で等価。

### I5. `evaluate._save_telemetry_csv` が不正な CSV を出力（`evaluate.py:83-107`）

- データ行の後に空行・`# summary` 行・列数が 6 列のサマリ行（ヘッダ 12 列）を
  書くため、pandas 等では壊れる。サマリは別ファイル（JSON/TXT）に分離、
  またはヘッダと一致した列数で出力すべき。

### I6. `audit.py` のタイムアウトが未処理例外（`audit.py:25`）

- 120 s 超過で `subprocess.TimeoutExpired` が飛ぶとレポート出力なしで
  トレースバック終了。catch して `passed=False, output="timeout"` と
  記録すべき（CI での失敗モードとして不親切）。

### I7. 死にコード・未使用変数

- `mavlink_if.py:574-578` `a_z_safe` / `V_safe`：呼び出し元なし。
- `visual3d.py:28` `_wing_polygon` の `c, s = ...`（次行で `ca, sa` を再定義）。
- `accelerate.py:66` `action_dim` 読み取りは未使用（2 次元ハードコードに埋没）。
- `env.py:118` は `0.5 * 1.225 * V**2` と RHO をハードコード（`aircraft.RHO`
  との整合性を担保できない。他モジュールは RHO import）。

### I8. `main.write_summary` の停止時刻判定（`main.py:205`）

- `stop_idx = int(np.argmax(res_ld.Vx < 0.2))` は Vx が 0.2 未満に
  ならない場合に argmax=0 を返し、停止時刻・距離を 0 として**黙って**
  報告する。検出不能なら明示的に NaN にすべき。

### I9. CWD 依存の既定パス（AUDIT.md の修正趣旨に残る漏れ）

- `educate.py:91` `previous_path="results/phi_student_v1/student.npz"`、
  `feedback_education.py:144` `previous_path='results/education_001/student.npz'`。
- `ocean_real` の NDBC 読込はモジュール相対化済みだが、これらは相対パス。
  別 CWD から実行すると誤った失敗（ファイル不存在）になる。

### I10. 文書の同期ずれ

- README「ファイル構成」表に `accelerate.py`、`vectorized.py`、
  `atmosphere.py`、`spatial_dynamics.py`、`flight_diagnostics.py`、
  `ollama_pilot.py`、`fly_ollama.py`、`teacher_student.py`、
  `educate.py`、`feedback_education.py`、`policy_pilot.py`、`docs/` がない
  （後半節では説明されている）。
- `fly_ollama.py` の合成海面既定 `--hs 0.3` は env/train の既定 Hs=1.5 と
  違い、海況の意図的な違いとして README に明記されていない。
- AUDIT.md は「回帰テスト 15 件／合計 26 件」と记载しているが、現在は
  86 件（README/AUDIT のいずれにも 86 件の記載なし）。

### I11. `results/` が 50 MB・231 ファイル git 管理下

- プロブェナンス（旧成果物の保持）を意図的にコミットしているように見える
  が、リポジトリ肥大の要因（pack 34 MB）。`audit.py` の実行は
  `results/audit.json`（追跡ファイル）を更新するため、監査の実行が
  ワーキングツリーの差分を生む。
- 継続するなら git-lfs、または .gitignore＋SHA-256 マニフェスト方式への
  移行を検討（現行規約との取捨選択を明示すること）。

### I12. 接水判定の許容幅の不統一（軽微）

- `dynamics.py` の phase フラグは +0.05 m のヒステリシス
  （`z - h_keel > eta + 0.05`）を使うが、`env.py::_build_state` の
  `hull_in_water` は厳密比較（`eta + h_keel > z`）。数値寄与は小さいが
  定義を揃えるとログ比較が楽になる。

### I13. 着水成功判定は「最初の波面接触」で確定（挙動の明記）

- 環境・機体評価とも、船底が**波頂**に最初に触れた瞬間（平均水面より
  最大 ~1 m 上方でも）に着水成功／失敗が確定する（`env.py:322-338`、
  `fly_ollama.py`、`accelerate.mavlink_sea_sweep` は同基準）。
  Hs=1.5 では進入点の波位に成功率が大きく依存し得る。設計意図と
  思われるが、README「注意点・限界」に明記すると良い。

---

## 検証して問題なしと確認した事項

- **PPO 更新**（`policy.py`）：クリップ条件の勾配マスク
  （A≥0: r≤1+ε / A<0: r≥1−ε）は標準 PPO と一致、数値勾配テストも PASS。
  エントロピー勾配（次元毎 -0.01）も正。`MLP.backward` に渡される
  「z」は活性化側だが ReLU の等価マスクで無害。
- **GAE**（`train.py` / `accelerate.py`）：終端ブートストラップなしの
  標準実装。訓練ループは行動前状態と対応行動を記録（監査修正済みの
  状態が維持されている）。
- **チェックポイント**：`ActorCritic.load` は shape・有限値・dtype を
  検証してから一括反映（アトミック）。欠落/破損時に旧重みを保持。
- **シード再現性**：環境・海面・大気（解析的ガスト、再読込で乱数消費なし）・
  車両 `reset(seed=...)` 再現を確認。
- **Ollama クライアント**（`ollama_pilot.py`）：localhost のみ許可、
  リダイレクト拒否、1 MiB 上限、JSON スキーマ＋範囲・有限値検証、
  `done/length` の不完了応答拒否。
- **損傷モデル**：スリングは「水中へ進入」イベントで計数（ステップで
  二重計数しない）、空中浸水なし、故障時推力停止・再武装拒否、
  非武装時推力 0（テストで固定）。
- **NDBC 読込**：実ファイル（spectral 15 列・realtime 16 列）の列位置、
  MM 欠測、実測 Hs を保つ成分分割（√(0.954²+0.3²)=1）を再現確認。
  `load_buoy_default` の現データ（Hs=1.7、swell 1.5/7.1 s、wind 0.8/5.0 s）
  と README の記載が一致。
- **空間運動**（`spatial_dynamics.py`）：コオディネート旋回近似、
  バンクによる揚力傾き（正バンク→+y へ）、風速による対気速度、
  力診断 `force_budget` の合計一致性を確認。
- 回帰テスト 86 件・`audit.py`（regressions+physics）は全 PASS。

## 推奨対応順序

1. F1（抵抗曲線の段差）→ 離水結論の再評価を伴うため最優先検討
2. F4（着水ホライズン）→ スイープの着水評価が誤るため
3. F2（MWD 変換、テスト含む）
4. F3（ベクトル環境の空間対応）
5. F5 / F6（文書・既定値の是正）
6. I1 / I2（包絡線・積分器の統合）
7. 其余の I 群（死にコード除去、CSV 整形、タイムアウト処理、README 同期）
