# 監査指摘（2026-09-20 エージェント監査）の対応

対象の指摘 F1–F6・I1–I13 を現行ツリーに照合し、stale（対応済み）の
3件を除いて対処した。新規テスト4件、全109件通過、
`audit.py`（regressions+physics）PASS。

## F（修正すべき）：全6件を修正

- **F1 船体抵抗の段差**：`HullDrag.resistance` の λ=1.4 硬切り替えを
  λ∈[1.2,1.8] の smoothstep ブレンドに置換（係数は dataclass フィールド化）。
  0.05 m/s 刻みの最大ジャンプ 283 N → 18 N。
  離水再検証：包絡離水 96/96（`takeoff_envelope.json`）、
  相似則 60/60（`size_001` 再実行）。RL再学習はしていない。
- **F2 MWD変換**：`(270-mwd)` を `(mwd+180)` に修正（2箇所）。
  軸は x=北・y=東のためコンパス方位がそのまま CCW 数学角になる。
  誤式を固定していたテスト2件も同時更新し、進行方向の象限まで assertions。
- **F3 ベクトル環境**：`VectorizedEnv.step` の (n,2) ハードコードを
  `envs[0].action_dim` 参照に。`train_vectorised` の action Low/High も
  次元追従（dim0=[0,1]、残り=[-1,1]）。
- **F4 ホライズン**：`sea_state_sweep`／`train_vectorised` に
  `max_steps` 引数、`mavlink_sea_sweep` の `duration` 既定を
  シナリオ別（離水15秒／着水30秒、train と一致）に。
- **F5 V_cruise**：値（設計巡航点）は不変。注釈・表示・validate の
  assert を「定義」に書き換え（最小パワー速度の誤主張を除去）。
- **F6 parse_action**：既定下限 -3° → -8°（包絡線に一致）。

## I（改善）：対応／stale／見送りの内訳

- 対応：I4（`elevation_at` の恒零項を除去、値は同一）、I5（CSV サマリを
  `#` コメント行化。`comment="#"` で読める）、I6（audit タイムアウトを
  PASS/FAIL 記録に）、I7 のうち死にコード除去（`a_z_safe`／`V_safe`、
  `visual3d` の `c,s`）、I8（停止未達は NaN）、I9（`previous_path` を
  リポジトリ相対に解決）、I10（README 構成表に12行追加＋hs 既定差の明記）、
  I12（接水ヒステリシス 0.05 m を `HULL_HYSTERESIS_M` に集約し dynamics／env
  で共有。着水 Hs=1.5 で 10/10 を再確認）、I13（初接触確定を README に明記）。
- stale（既に対応済みで変更なし）：I7 の env RHO ハードコードと
  accelerate `action_dim` 未使用（F3 対応で使用側に回った）、AUDIT.md の
  件数（日付付き歴史記録のため不変）。
- 見送り：I1（包絡線集約）、I2（積分器統一）はリファクタ回帰リスクのため
  現状維持。I3（プロペラ効率）は電池未実装のため効果なし。
  I11（results の git 管理）は運用規約の判断が必要なため現状維持。

## 残る既知の角

着水 Hs=3.5・横風±5 の拡張シード（9503/9504）で波頭位相による
ハード着水が各1件（上昇中接水・N 超過、横ずれなし）。
確定シード範囲（9500–9502）の 24/24 は維持。I13 の注意書きどおり
進入波位の依存であり、包絡主張の変更はない。
