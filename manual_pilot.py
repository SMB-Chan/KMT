"""Interactive human/manual pilot for the KMT seaplane drone.

Controls:
    W / S       : Pitch Up / Down (+1 deg / -1 deg)
    E / D       : Throttle Up / Down (+10% / -10%)
    A / F       : Bank left / right (spatial mode)
    Space       : Maintain current control
    Q           : Quit / Abort
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import select
import termios
import tty
import time
from pathlib import Path

from aircraft import Aircraft
from dynamics import hull_force
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean
from ocean_real import load_buoy_default
from ocean_directional import DirectionalOcean
from atmosphere import AtmosphereConfig
from ollama_pilot import Control, SpatialControl, observe


def get_key(timeout=0.08):
    """Read a single key without waiting for enter (non-blocking with timeout)."""
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            # Handle arrow keys
            if ch == '\x1b':
                rlist2, _, _ = select.select([fd], [], [], 0.05)
                if rlist2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A': return 'UP'
                        if ch3 == 'B': return 'DOWN'
                        if ch3 == 'C': return 'RIGHT'
                        if ch3 == 'D': return 'LEFT'
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def print_hud(t, z, Vx, Vz, eta, thr, pitch, bank, water, spray, scenario, target_alt, status="FLYING"):
    # Clear screen
    sys.stdout.write("\033[2J\033[H")
    thr_bar = "█" * int(thr * 20) + "░" * (20 - int(thr * 20))
    alt_ratio = z / max(1.0, target_alt) if target_alt > 0 else (25.0 - z) / 25.0
    alt_bar = "█" * min(20, max(0, int(alt_ratio * 20)))
    
    print("=" * 65)
    print(f"      ✈ KMT ドローン飛行艇 人間操縦モード (Manual Flight) ✈")
    print("=" * 65)
    print(f"  シナリオ: {scenario.upper():<10} | 時間: {t:6.2f} s | 状態: {status}")
    print("-" * 65)
    print(f"  高度 (z)      : {z:6.2f} m   [目標: {target_alt:4.1f} m]  [{alt_bar:<20}]")
    print(f"  対気速度 (Vx)  : {Vx:6.2f} m/s  (失速速度: {Aircraft().V_stall:4.1f} m/s)")
    print(f"  昇降速度 (Vz)  : {Vz:+6.2f} m/s")
    print(f"  海面波高 (η)  : {eta:+6.2f} m   (船底クリアランス: {z - eta:+.2f} m)")
    print("-" * 65)
    print(f"  スロットル    : [{thr_bar}] {int(thr * 100):3d} %  (E: 増 / D: 減)")
    print(f"  ピッチ迎角    : {pitch:+5.1f} °                      (W: 上 / S: 下)")
    if bank != 0.0:
        print(f"  バンク角      : {bank:+5.1f} °                      (A: 左 / F: 右)")
    print(f"  機体浸水量    : {water:6.3f} kg (警戒: 0.20 kg / 限界: 0.50 kg)")
    print(f"  飛沫負荷      : {spray:6.2f}")
    print("=" * 65)
    print("  [キー操作]  W/S: ピッチ変更  |  E/D or ↑/↓: スロットル  |  Q: 終了")
    print("=" * 65)
    sys.stdout.flush()


def run_manual(scenario="takeoff", duration=25.0, dt=0.05, ndbc=False, spatial=False, render_on_exit=True):
    sea = load_buoy_default() if ndbc else Ocean(Hs=0.8, Tp=6.0, seed=42)
    ac = Aircraft()
    veh = FlyingBoatVehicle(ac, sea, spatial=spatial)
    veh.dt = dt

    if scenario == "landing":
        veh.z = 25.0
        veh.Vx = 13.0 * math.cos(math.radians(8.0))
        veh.Vz = -13.0 * math.sin(math.radians(8.0))
        throttle = 0.05
        pitch_deg = -5.0
        target_alt = 0.0
    else:
        veh.z = 0.14
        veh.Vx = 0.5
        veh.Vz = 0.0
        throttle = 1.0
        pitch_deg = 4.0
        target_alt = 10.0

    bank_deg = 0.0
    rudder = 0.0
    veh.arm()

    traj = []
    status = "FLYING"
    print("操縦開始の準備をしています... (1秒後に開始)")
    time.sleep(1)

    while veh.t < duration:
        obs = observe(veh)
        eta = obs["wave_elevation_m"]
        water = obs.get("water_mass_kg", 0.0)
        spray = getattr(veh.damage, "cumulative_spray", 0.0)
        snap = veh.read_telemetry()[2]
        N_water = snap["N_water"] if snap else 0.0

        # 判定
        if veh.damage.failed:
            status = f"FAILED: {veh.damage.failure_reason}"
            print_hud(veh.t, veh.z, veh.Vx, veh.Vz, eta, throttle, pitch_deg, bank_deg, water, spray, scenario, target_alt, status)
            break
        if N_water > 8 * ac.W:
            status = "FAILED: 水面衝撃超過 (Impact)"
            print_hud(veh.t, veh.z, veh.Vx, veh.Vz, eta, throttle, pitch_deg, bank_deg, water, spray, scenario, target_alt, status)
            break
        if scenario == "takeoff" and veh.z >= target_alt and veh.Vx >= 10.0:
            status = "SUCCESS: 離水・目標高度達成！"
            print_hud(veh.t, veh.z, veh.Vx, veh.Vz, eta, throttle, pitch_deg, bank_deg, water, spray, scenario, target_alt, status)
            break
        elif scenario == "landing" and veh.z - eta <= 0.05 and veh.t > 3.0:
            if abs(veh.Vz) < 1.5 and N_water < 3 * ac.W:
                status = "SUCCESS: 安全着水完了！"
            else:
                status = "HARD LANDING: 着水衝撃大"
            print_hud(veh.t, veh.z, veh.Vx, veh.Vz, eta, throttle, pitch_deg, bank_deg, water, spray, scenario, target_alt, status)
            break

        print_hud(veh.t, veh.z, veh.Vx, veh.Vz, eta, throttle, pitch_deg, bank_deg, water, spray, scenario, target_alt, status)

        # 非ブロッキングでキー入力を取得
        key = get_key(timeout=0.08)
        if key:
            k = key.lower() if len(key) == 1 else key
            if k == 'q':
                status = "ABORTED: 手動中断"
                break
            elif k == 'w':
                pitch_deg = min(15.0, pitch_deg + 1.0)
            elif k == 's':
                pitch_deg = max(-8.0, pitch_deg - 1.0)
            elif k in ('e', 'UP'):
                throttle = min(1.0, throttle + 0.10)
            elif k in ('d', 'DOWN'):
                throttle = max(0.0, throttle - 0.10)
            elif k in ('a', 'LEFT'):
                bank_deg = max(-30.0, bank_deg - 5.0)
            elif k in ('f', 'RIGHT'):
                bank_deg = min(30.0, bank_deg + 5.0)

        # 制御適用
        if spatial:
            ctrl = SpatialControl(throttle=throttle, pitch_deg=pitch_deg, bank_deg=bank_deg, rudder=rudder)
        else:
            ctrl = Control(throttle=throttle, pitch_deg=pitch_deg)
        ctrl.apply(veh)

        veh.step(dt=dt)
        traj.append({
            "t": veh.t, "x": veh.x, "y": getattr(veh, "y", 0.0), "z": veh.z,
            "Vx": veh.Vx, "Vy": getattr(veh, "Vy", 0.0), "Vz": veh.Vz,
            "alpha": math.radians(pitch_deg), "throttle": throttle,
            "bank": math.radians(bank_deg), "heading": getattr(veh, "heading", 0.0),
            "eta": eta, "mode": "MANUAL"
        })

    print(f"\nフライト終了: {status}")
    print(f"最終結果: 時刻={veh.t:.2f}s, 飛行距離={veh.x:.1f}m, 高度={veh.z:.2f}m, 速度={veh.Vx:.2f}m/s")

    if render_on_exit and traj:
        try:
            from visual3d import render_scene
            out_img = f"results/manual_{scenario}_final.png"
            render_scene(traj, sea, time_index=-1, output=out_img)
            print(f"フライト結果の3Dシーン画像を保存しました: {out_img}")
        except Exception as e:
            print(f"画像保存スキップ: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KMT ドローン飛行艇 人間操縦モード")
    p.add_argument("--scenario", choices=["takeoff", "landing"], default="takeoff")
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--ndbc", action="store_true", help="実測NOAA波浪を使用")
    p.add_argument("--spatial", action="store_true", help="3次元空間飛行モード")
    args = p.parse_args()

    run_manual(scenario=args.scenario, duration=args.duration, ndbc=args.ndbc, spatial=args.spatial)
