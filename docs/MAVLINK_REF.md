# KMT MAVLink 風インターフェース技術リファレンス

**適用範囲：** `mavlink_if.py`（プロセス内 API）
**対象読者：** 機体制御ソフトウェア担当・オートパイロット実装担当

> **注意：** `mavlink_if.py` はプロセス内 API です。UDP ワイヤは
> `operator_training/mavlink_wire.py` + `mavlink_udp.py`（MAVLink v2、pymavlink なし）。
> QGroundControl / MAVProxy が接続できます。PX4/ArduPilot ファームウェアそのものではありません。

---

## 1. 概要

`FlyingBoatVehicle` は `dynamics.py` の縦運動シミュレータを PX4 風にラップします。
外部オートパイロットは次の 2 操作だけで機体にアクセスします。

| 方向 | API |
|---|---|
| ダウンリンク（機体 → 外部） | `read_telemetry()` → `HIL_STATE` / `GLOBAL_POSITION_INT` / `ATTITUDE` |
| アップリンク（外部 → 機体） | `send_command(MAV_CMD, params)` |

物理ステップは `step(dt=...)` で進め、`dt` 既定は 0.05 s です。
離水・着水シナリオの既定時間：

| シナリオ | 既定時間 |
|---|---:|
| 離水 | 12 s |
| 着水 | 30 s |

---

## 2. コマンド ID 一覧

`common.xml` MAVLink 2.0 のサブセットのみサポートします。

| ID | 定数名 | パラメータ | 用途 |
|---:|---|---|---|
| 16 | `MAV_CMD_NAV_WAYPOINT` | `x`, `z`, `speed` | ホールドウェイポイント（プロセス内座標） |
| 21 | `MAV_CMD_NAV_LAND` | `alt`, `glide` | 目標高度 0、滑空角 [deg] |
| 22 | `MAV_CMD_NAV_TAKEOFF` | `alt`, `speed` | 目標高度 [m]、目標対気速度 [m/s] |
| 115 | `MAV_CMD_CONDITION_YAW` | — | 未対応（拒否） |
| 178 | `MAV_CMD_DO_CHANGE_SPEED` | `speed` | 目標速度の変更 |
| 183 | `MAV_CMD_DO_SET_SERVO` | `servo`, `pwm` | サーボ直接指令（1=throttle, 2=elevator） |

未対応コマンドは `MAV_RESULT_UNSUPPORTED` を返します。

---

## 3. コマンドごとの仕様

### 3.1 `MAV_CMD_NAV_TAKEOFF`（22）

| パラメータ | 型 | 単位 | 説明 |
|---|---|---|---|
| `alt` | float | m | 目標高度 |
| `speed` | float | m/s | 目標対気速度 |

**受領後の動作：**
- 機体が DISARMED の場合は ARMED に遷移（`arm()` が別途必要）
- `LowLevelController.takeoff_setpoint` で推力・ピッチを計算

```python
veh.arm()
veh.send_command(MAV_CMD_NAV_TAKEOFF, {"alt": 10.0, "speed": 13.0})
```

**制御則：**
- `e_alt = alt - z`
- `climb_rate = clip(Kp_alt·e_alt − Kd_alt·Vz, −5, 8)`
- `target_pitch = atan2(climb_rate, Vx) + pitch_trim`（`pitch_trim = 4°`）
- `target_pitch ∈ [−8°, 15°]` にクリップ
- スロットル `= 0.85 + 0.4·tanh(0.3·e_alt) + Kp_speed·(speed − Vx)` を [0, 1] でクリップ

### 3.2 `MAV_CMD_NAV_LAND`（21）

| パラメータ | 型 | 単位 | 説明 |
|---|---|---|---|
| `alt` | float | m | 目標高度（通常 0） |
| `glide` | float | deg | 滑空角（既定的には 8°） |

**制御則：**
- `e_alt = z − max(alt, 0)`
- 高高度（`e_alt > 1 m`）：`pitch = −5°`、`throttle = 0.10`
- 低高度（フレア）：`flare_factor = exp(−e_alt/1.5)`、`pitch = −5°·(1−flare) + 2°·flare`、`throttle = 0.05`
- `pitch ∈ [−8°, 15°]` にクリップ

### 3.3 `MAV_CMD_NAV_WAYPOINT`（16）

| パラメータ | 型 | 単位 | 説明 |
|---|---|---|---|
| `x` | float | m | 目標 x（プロセス内） |
| `z` | float | m | 目標 z（プロセス内） |
| `speed` | float | m/s | 目標速度 |

`takeoff_setpoint(z=alt, speed=...)` と同じホールド則を流用します。
座標は実機 GPS ではなくシミュレータ内部座標です。

### 3.4 `MAV_CMD_DO_CHANGE_SPEED`（178）

| パラメータ | 型 | 単位 | 説明 |
|---|---|---|---|
| `speed` | float | m/s | 目標対気速度 |

現在の高度目標をそのまま保持し、目標速度のみ更新します。

### 3.5 `MAV_CMD_DO_SET_SERVO`（183）

| パラメータ | 型 | 単位 | 説明 |
|---|---|---|---|
| `servo` | int | — | 1 = throttle, 2 = elevator |
| `pwm` | int | μs | 1000 〜 2000 |

| servo | 意味 | PWM 1000 | PWM 2000 |
|---:|---|---|---|
| 1 | throttle | 推力停止 | 全開 |
| 2 | elevator（ピッチ） | ハードウェア可動域 −15° | +15° |

`pitch` はハードウェア可動域 ±15° にクリップされます。
**RL オートパイロット経路の `−3°〜12°` とは範囲が異なる点に注意**してください。

---

## 4. テレメトリメッセージ

### 4.1 `HIL_STATE`（90）

`VehicleState` データクラスのフィールドを反映。

| フィールド | 型 | 単位 | 意味 |
|---|---|---|---|
| `timestamp_us` | int | μs | シミュレータ時刻 |
| `roll`, `pitch`, `yaw` | float | rad | 姿勢（ロール／ピッチ／ヨー） |
| `rollspeed`, `pitchspeed`, `yawspeed` | float | rad/s | 角速度 |
| `lat`, `lon` | int | degE7 | 緯度／経度（内部原点からの相対、参考値） |
| `alt` | int | mm | 高度（z × 1000） |
| `vx`, `vy`, `vz` | float | m/s | 機体軸速度 |
| `ind_airspeed`, `true_airspeed` | float | m/s | 対気速度 |
| `xacc`, `yacc`, `zacc` | float | m/s² | 加速度 |

### 4.2 `GLOBAL_POSITION_INT`（33）

| フィールド | 型 | 単位 | 意味 |
|---|---|---|---|
| `lat`, `lon` | int | degE7 | `origin_lat`/`origin_lon` からの相対位置 |
| `alt` | int | mm | AMSL 高度 |
| `relative_alt` | int | mm | 相対高度 |

### 4.3 `ATTITUDE`（30）

`roll`, `pitch`, `yaw` のみ（rad）。
ロール／ヨーは本シミュレータでは 0 固定です。

---

## 5. 状態機械（vehicle モード）

```
DISARMED  ──arm()──▶  ARMED  ──TAKEOFFコマンド──▶  TAKEOFF
                                              │
                                              └──LANDコマンド──▶ LAND
                                                                │
                                          damage.failed──▶ FAILSAFE
```

| モード | 内容 | 主な API |
|---|---|---|
| `DISARMED` | 武装待ち。推力停止 | `arm()` |
| `ARMED` | 待機。R/C 入力待ち | — |
| `TAKEOFF` | 離水コマンド実行中 | `send_command(MAV_CMD_NAV_TAKEOFF)` |
| `LAND` | 着水コマンド実行中 | `send_command(MAV_CMD_NAV_LAND)` |
| `FAILSAFE` | 損傷故障。推力停止・再武装拒否 | — |

故障後は `vehicle.damage.failed` が `True` となり、`arm()` を再呼びしても拒否されます。

---

## 6. 機体状態と物理量

テレメトリ以外の主要な内部状態：

| 属性 | 型 | 単位 | 説明 |
|---|---|---|---|
| `veh.x`, `veh.z` | float | m | 機体位置 |
| `veh.Vx`, `veh.Vz` | float | m/s | 速度 |
| `veh.alpha` | float | rad | 機体ピッチ |
| `veh.throttle` | float | — | 直近の推力指令 |
| `veh.t` | float | s | シミュレータ時刻 |
| `veh._msg_log` | list | — | テレメトリ送信履歴 |
| `veh.damage.failed` | bool | — | 損傷故障フラグ |
| `veh._active_cmd` | int | — | 受理済みコマンド ID |

---

## 7. 安全・運用上の制約

### 7.1 同時に存在できるコマンド

- `MAV_CMD_NAV_TAKEOFF` と `MAV_CMD_NAV_LAND` のいずれも送信されていない状態が正常な開始条件
- 同一コマンドの再送は受理されます（パラメータ上書き）
- `MAV_CMD_NAV_*` と `MAV_CMD_DO_SET_SERVO` の併用は可能ですが、`MAV_CMD_DO_SET_SERVO` の効果は次ステップから反映されます

### 7.2 物理安全

- 故障状態の機体は再武装拒否（`arm()`）
- 浸水 ≥ 0.50 kg でアビオニクス故障
- スリング（水没プロペラ）連続水中進入 2 回で致命的故障

### 7.3 RL／MAVLink 経路の相互作用

- `env.step(action)` と `veh.step()` は同じ `dynamics` を共有しません
- `evaluate.py` のように MAVLink 経路を使う場合、**RL 行動は経路の制御則を上書きしません**
- 訓練で RL 行動を MAVLink 経路に注入したい場合は `veh.step(action=...)` を使用します
  - 監査後の実装では当該ステップで適用され、訓練環境と同じ観測を返します

### 7.4 物理ステップ

| メソッド | 動作 |
|---|---|
| `veh.step()` | 既定 `dt = 0.05 s` で 1 ステップ |
| `veh.step(dt=...)` | 指定 `dt` で進める（duration 超過時は縮められます） |
| `veh.reset()` | 機体・海面・損傷モデルをリセット（**シードを渡すと海面オブジェクトも再構築**） |

### 7.5 NDBC 海面との組み合わせ

```python
from ocean_real import load_buoy_default
from mavlink_if import FlyingBoatVehicle
from aircraft import Aircraft

sea = load_buoy_default()           # data/ndbc_46026_spectral.txt
veh = FlyingBoatVehicle(Aircraft(), sea)
veh.arm()
veh.send_command(MAV_CMD_NAV_TAKEOFF, {"alt":10, "speed":13})
for _ in range(240):
    veh.step()                     # 12 s
```

---

## 8. 動作例（最短）

### 8.1 離水 → 着水の連続実行

```python
import math
from aircraft import Aircraft
from mavlink_if import (FlyingBoatVehicle,
                        MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_LAND)
from ocean_real import load_buoy_default

sea = load_buoy_default()
veh = FlyingBoatVehicle(Aircraft(), sea)

# 1. 離水
veh.arm()
veh.send_command(MAV_CMD_NAV_TAKEOFF, {"alt": 10.0, "speed": 13.0})
for _ in range(int(12 / veh.dt)):
    veh.step()

# 2. 着水のため高度をセット
veh.z = 25.0
gs = math.radians(8.0)
veh.Vx = 13.0 * math.cos(gs)
veh.Vz = -13.0 * math.sin(gs)

veh.send_command(MAV_CMD_NAV_LAND, {"alt": 0.0, "glide": 8.0})
for _ in range(int(30 / veh.dt)):
    veh.step()
    if veh.damage.failed:
        break
```

### 8.2 テレメトリ CSV 出力

```python
import csv

with open("telemetry.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","msg","x","y","z","Vx","Vz","roll","pitch","yaw","alt_mm","mode"])
    for entry in veh._msg_log:
        if len(entry) == 4:
            t, hil, gpi, snap = entry
        else:
            t, hil, gpi = entry
            snap = dict(x=veh.x, z=veh.z, Vx=veh.Vx, Vz=veh.Vz,
                        alpha=veh.alpha, throttle=veh.throttle, mode=veh._mode)
        w.writerow([t, "HIL_STATE", snap["x"], 0.0, snap["z"],
                    snap["Vx"], snap["Vz"], 0.0,
                    math.degrees(snap["alpha"]), 0.0,
                    int(snap["z"]*1000), snap["mode"]])
```

### 8.3 直接サーボ操作（サーボ PWM を指定）

```python
from mavlink_if import MAV_CMD_DO_SET_SERVO

# スロットル PWM 1500（中立）
veh.send_command(MAV_CMD_DO_SET_SERVO, {"servo": 1, "pwm": 1500})
# エレベーター PWM 1700（軽いエレベータ・アップ）
veh.send_command(MAV_CMD_DO_SET_SERVO, {"servo": 2, "pwm": 1700})
```

**ピッチ範囲に関する注意：**
- 直接サーボ操作は**ハードウェア可動域 ±15°**
- RL/MAVLink 経路は **−3°〜12°** または **−8°〜15°**（`fly_ollama.py`）
- 機体に渡る値は経路ごとにクリップされます。`advisor.short_simulate` のログには**適用後の値**が記録されます

---

## 9. 既知の不具合／監査後の挙動

監査（`AUDIT.md` 第 20 章）で修正された挙動の一覧：

| 修正前 | 修正後 |
|---|---|
| `MAV_CMD_DO_SET_SERVO` が実行されなかった | 物理ステップ内で PWM → 機体指令に変換 |
| RL 行動が次のステップで制御器に上書きされた | `step(action=...)` が当該ステップで適用 |
| 故障・非武装でも推力が発生 | 推力停止、故障機の再武装拒否 |
| 静止時にも船体抵抗 | 速度依存の抵抗、零速度で消失 |
| 速度ゼロ時に水中抵抗方向が反転 | 速度に対し逆向きに適用 |
| 連続水没の各ステップを別の衝突として計数 | 水中進入を 1 事象として計数 |
| 空中でも船首飛沫による浸水 | 船体接水を条件に追加 |
| 速度変更コマンドが実行されなかった | `MAV_CMD_DO_CHANGE_SPEED` を処理 |
| 未対応コマンドが黙って受理された | `MAV_RESULT_UNSUPPORTED` で拒否 |

---

## 10. 簡易チェックリスト

- [ ] `arm()` 呼び出し済みか
- [ ] `send_command` 後に `step()` を呼んでいるか
- [ ] テレメトリは `read_telemetry()` または `_msg_log` から取得
- [ ] 故障判定 `veh.damage.failed` を監視しているか
- [ ] 浸水・スリング閾値（0.20 kg / 0.50 kg / 2 回）を踏まえた運用か
- [ ] 物理ステップ `dt` を RL／物理サニティと整合させているか
- [ ] OSS 地上局は `python3 -m operator_training mavlink`（UDP 14551 → GCS 14550）

---

## 11. 参照

- `mavlink_if.py` — プロセス内インターフェース
- `operator_training/mavlink_udp.py` — QGC/MAVProxy 向け UDP SITL
- `dynamics.py` — 縦運動シミュレータ・HullContact/HullDrag
- `damage.py` — 損傷モデル
- `evaluate.py` — 統合評価スクリプト
- `fly_ollama.py` — Phi パイロットランナー（MAVLink 経由）
- マニュアル本体 — `MANUAL.md` 第 6 章
- 監査報告 — `AUDIT.md`


## 空間モードの追加API

`FlyingBoatVehicle(..., spatial=True, atmosphere=AtmosphereConfig(...), seed=42)` で有効化。
サーボ3＝バンク（1000–2000 µs → -45〜45°）、4＝ラダー（-1〜1）を追加。
直接RL行動は4次元で、従来2次元との混用を拒否します。
WAYPOINTの `x`/`y` はローカル北／東位置（m）、`heading` は北0°・東90°です。
`MAV_CMD_CONDITION_YAW` は `heading` で絶対方位を指定します。

`read_telemetry()` は従来通り `(hil, global_position, snapshot)` を返します。
`read_attitude()` はATTITUDE相当を返し、reset直後はNoneです。
内部zは上向き、HIL/GLOBAL_POSITION相当のvzは下向き。経度はyから算出します。
HILの速度・加速度は模擬APIのSI値で、MAVLinkのシリアライズ／単位変換を実装したものではありません。
学習環境と共通の空間積分器を使い、機体側の損傷・推力停止を適用します。
