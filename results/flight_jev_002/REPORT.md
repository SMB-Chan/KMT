# FlightJev 校正

内蔵ロールアウト（学習シード9500–9503、保持9504）でロジスティックを学習。
ラベルは未来2秒以内の rotate（Vx≥7）と未来1秒以内の低空浮上（z<1.5 かつ Vz>−0.2）。

| Noul | ヒューリスティック Brier | 学習後 Brier |
|---|---:|---:|
| rotate_soon | 0.198 | 0.082 |
| balloon_soon | 0.057 | 0.024 |

`calibrator.npz` を `FlightJev.load` または `--jev --jev-calibrator results/flight_jev_002/calibrator.npz` で使う。
内側の制御は内蔵のまま。
