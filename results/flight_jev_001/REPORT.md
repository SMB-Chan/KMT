# FlightJev v1（ローカル System One）

Jev の型付き決定を既存の観測と内蔵セットポイントの上に載せた。新学習はしていない。
Choice（位相）／Score（進捗）／Noul（地面効果・浮上）と信頼度。内側の推力・ピッチは内蔵制御。
着水で ballooning が高く信頼できるときだけ −4°・スロットル0。

`python3 fly_ollama.py --jev --spatial --lateral-assist ...`

10条件（シード9500–9504×横風±3）：離水 Hs=0.3 と着水 Hs=1.5 はともに 10/10。
