# 環境変数による性能包絡

```bash
HAMA_PILOT=native,jev HAMA_HS=0.3,1.5,2.5,3.5 HAMA_WIND_Y=-5,-3,3,5 \
HAMA_SEEDS=9500,9501,9502 HAMA_JEV_CALIBRATOR=results/flight_jev_002/calibrator.npz \
python3 results/envelope_001/sweep.py
```

変数: `HAMA_PILOT` `HAMA_SCENARIO` `HAMA_HS` `HAMA_WIND_Y` `HAMA_SEEDS` `HAMA_JEV_CALIBRATOR` `HAMA_OUTPUT`。

## 主グリッド（192試行）

Hs≤3.5 m、|横風|≤5 m/s、シード3。native と jev は離水・着水とも **96/96**。差なし（内側が同じ内蔵制御）。

## 限界側（native）

| 条件 | 離水 | 着水 |
|---|---|---|
| Hs=4.5、横風 ±8 | 4/4 | 0/4（lateral_limit） |
| Hs=6.0、横風 ±8 | 4/4 | 0/4（lateral_limit） |

離水はこの範囲では折れない。着水の壁は高波＋強い横風で |y|>50 m。
その後のクラブ制御は `results/lateral_001/`。
