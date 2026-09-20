# 残課題 I1・I2・I3・I11 の対応

## I1 ピッチ包絡線の集約

`env.pitch_from_normalized(u, lo, hi)`＋`pitch_from_config(u, cfg)` を
単一の所望にし、3箇所（env／機体 action 経路／policy_pilot）を置換。
`train._init_bias_for` は従来どおり EnvConfig 導出の正例。
機体・パイロット側は `EnvConfig.pitch_lo/hi` の既定値を参照する。
既定包絡では旧式とビット等価に近い（policy_pilot のみ rad 往復で
約4e-15° の差が出るため、該当テストの比較を almostEqual 化）。

## I2 縦方向機体 step のサブステップ化

`FlyingBoatVehicle` の縦積分を h≤0.01 s の陽 Euler サブステップにし、
RL 環境・空間積分器と統一。波面はサブステップ毎に再評価する。
損傷更新はステップ単位のまま（スリング計数の意味を保つ）。
無風 Faired 試験：dt=0.05×1 と dt=0.01×5 がビット一致。
縦方向の離水・着水スポット（Hs=1.5、各3シード）は 6/6 成功。

## I3 power_required にプロペラ効率

`Propulsion.eta_prop = 0.70`（巡航域）を追加し、電力式の分母に入れた。
巡航電力の表示は約3816 W → 約5452 W、航続デモは約12.3分 → 約8.6分に
下方修正（過大評価の方向に是正）。V→0 の T·V モデル限界は docstring に
明記し、静止の大電力は主張しない。本関数はデモ専用のまま。

## ついでに修正した既存バグ

`Aircraft.summary` が `scaled_aircraft` 挿入時にクラス外の到達不能な
内部関数へ脱落し、`python3 aircraft.py` が AttributeError になっていた
（HEAD でも再現する既存不具合）。`L_D` の後ろに復帰させた。

## I11 results 管理方針の判断

現状：results/ 63 MB（gif・CSV・npy・trajectory jsonl が大半）、
.git/ 56 MB（未pack）。各スタディは SHA-256 付き JSON で来歴を保持する
設計であり、追跡が再現性の根拠になっている。
判断：現状維持。履歴書換えを伴う git-lfs 移行や .gitignore＋マニフェスト化は
push 権・共同運用の合意が必要なため実施しない。再訪の目安は 200 MB。

## 検証

- 新規テスト4件（包絡線単一化・サブステップ一致・プロペラ効率・summary）
  を含む全113件通過、`audit.py` は regressions・physics とも PASS
- I2 の影響範囲（縦機体経路）はスポット 6/6 で確認。
  dynamics.py の simulate 系は触っていない
