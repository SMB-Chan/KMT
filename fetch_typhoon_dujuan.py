"""Fetcher for the JMA bulletins of Typhoon No. 25 (Dujuan, TC2630).

Downloads, into ``data/jma_typhoon_2625/``, the machine-readable Japan
Meteorological Agency bulletins that the typhoon forecast module and the
operational study consume:

* ``targetTc.json`` / ``pastTracks.json`` -- active tropical cyclone list and
  recent tracks (bosai typhoon API).
* ``TC2630_*.json`` -- the typhoon bulletins proper: analysis + forecast
  positions, warning areas, probability circles (``forecast.json``),
  intensity / size / wind radii (``specifications.json``) and the previous
  issue of both.
* ``information_typhoon.json`` + ``denbun/*.json`` -- every typhoon-related
  meteorological bulletin (general / regional / prefectural weather
  explanation information) listed by the bosai information service, with the
  full bulletin text (positions, intensity, forecast positions and rainfall
  outlooks are inside the text).
* ``amedas_stations.csv`` + ``amedas_table_subset.json`` -- hourly AMeDAS
  observations (rain, wind, temperature) of the stations along the expected
  track, extracted from the hourly nationwide maps.
* ``feed_prefecture_R1/*.xml`` -- the latest prefectural weather forecast
  (R1) XML of the affected prefectures, i.e. the official rain-probability
  forecast, extracted from the JMA developer XML feed.
* ``manifest.json`` -- fetch time, source URLs and record counts.

The study and the tests run off the committed snapshot, so results stay
reproducible; re-running this script refreshes the snapshot while the
typhoon is still active.  Everything is plain HTTP GET, no API key.
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "jma_typhoon_2625")
BOSAI = "https://www.jma.go.jp/bosai"
FEED_URL = "https://www.data.jma.go.jp/developer/xml/feed/regular_l.xml"
TC_ID = "TC2630"          # tropical cyclone id of Typhoon 2625 (Dujuan)
TY_NUMBER = "2625"

# AMeDAS stations along / near the forecast track (kjName of the station
# table); the first exact or substring match wins.
STATION_NAMES = [
    "八丈島", "三宅島", "大島", "銚子", "東京", "横浜", "千葉", "甲府",
    "静岡", "水戸", "仙台", "宮古", "高知", "大阪",
]
# prefecture codes whose latest R1 (prefectural weather forecast) XML is kept
PREFECTURE_CODES = {
    "130000": "tokyo", "140000": "kanagawa", "120000": "chiba",
    "220000": "shizuoka", "080000": "ibaraki", "040000": "miyagi",
    "030000": "iwate", "390000": "kochi", "270000": "osaka",
}


def _get(url, timeout=40, retries=3, sleep=0.1):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HAMA-sim/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness
            last = exc
            time.sleep(1.0)
    raise RuntimeError(f"GET {url} failed: {last}")


def _get_json(url, **kw):
    return json.loads(_get(url, **kw).decode("utf-8"))


def fetch_typhoon_jsons(out, log):
    urls = {
        "targetTc.json": f"{BOSAI}/typhoon/data/targetTc.json",
        "pastTracks.json": f"{BOSAI}/typhoon/data/pastTracks.json",
        f"{TC_ID}_forecast.json": f"{BOSAI}/typhoon/data/{TC_ID}/forecast.json",
        f"{TC_ID}_specifications.json":
            f"{BOSAI}/typhoon/data/{TC_ID}/specifications.json",
        f"{TC_ID}_forecastPreviousIssue.json":
            f"{BOSAI}/typhoon/data/{TC_ID}/forecastPreviousIssue.json",
        f"{TC_ID}_probabilityThrough.json":
            f"{BOSAI}/typhoon/data/{TC_ID}/probabilityThrough.json",
        f"{TC_ID}_probabilityTimeseries.json":
            f"{BOSAI}/typhoon/data/{TC_ID}/probabilityTimeseries.json",
    }
    for name, url in urls.items():
        blob = _get_json(url)
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=1)
        log[name] = url
    return urls


def fetch_bulletin_list(out, log):
    items = _get_json(f"{BOSAI}/information/data/r8/information.json")
    ty = [x for x in items
          if "台風" in json.dumps(x, ensure_ascii=False)
          or TY_NUMBER in json.dumps(x, ensure_ascii=False)]
    ty.sort(key=lambda x: x["datetime"])
    with open(os.path.join(out, "information_typhoon.json"), "w",
              encoding="utf-8") as fh:
        json.dump(ty, fh, ensure_ascii=False, indent=1)
    log["information_typhoon.json"] = f"{BOSAI}/information/data/r8/information.json"
    return ty


def fetch_denbun(out, items, sleep, log):
    ddir = os.path.join(out, "denbun")
    os.makedirs(ddir, exist_ok=True)
    n = 0
    for it in items:
        name = it["jsonName"]
        path = os.path.join(ddir, name + ".json")
        if os.path.exists(path):
            n += 1
            continue
        blob = _get_json(f"{BOSAI}/information/data/r8/denbun/{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=1)
        n += 1
        time.sleep(sleep)
    log["denbun/"] = f"{BOSAI}/information/data/r8/denbun/{{jsonName}}.json"
    return n


def fetch_amedas(out, hours, sleep, log):
    table = _get_json(f"{BOSAI}/amedas/const/amedastable.json")
    with open(os.path.join(out, "amedas_table.json"), "w",
              encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False)
    picked = []
    for want in STATION_NAMES:
        hit = None
        for sid, st in table.items():
            if st.get("kjName") == want:
                hit = (sid, st)
                break
        if hit is None:
            for sid, st in table.items():
                if want in (st.get("kjName") or ""):
                    hit = (sid, st)
                    break
        if hit is not None:
            picked.append(hit)
    subset = {sid: st for sid, st in picked}
    with open(os.path.join(out, "amedas_table_subset.json"), "w",
              encoding="utf-8") as fh:
        json.dump(subset, fh, ensure_ascii=False, indent=1)

    latest = _get(f"{BOSAI}/amedas/data/latest_time.txt").decode("utf-8").strip()
    t0 = datetime.fromisoformat(latest)
    t0 = t0.replace(minute=(t0.minute // 10) * 10, second=0, microsecond=0)
    rows = []
    stamps = []
    for i in range(hours):
        stamp = t0 - timedelta(hours=i)
        key = stamp.strftime("%Y%m%d%H%M%S")
        stamps.append(stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        m = _get_json(f"{BOSAI}/amedas/data/map/{key}.json")
        for sid, st in picked:
            rec = m.get(sid)
            if rec is None:
                continue
            rows.append([
                stamps[-1], sid, st.get("kjName", ""),
                _num(rec, "precipitation1h"), _num(rec, "precipitation3h"),
                _num(rec, "precipitation24h"), _num(rec, "wind"),
                _num(rec, "windDirection"), _num(rec, "temp"),
                _num(rec, "humidity"),
            ])
        time.sleep(sleep)
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(os.path.join(out, "amedas_stations.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["utc", "station_id", "station_name", "precip1h_mm",
                    "precip3h_mm", "precip24h_mm", "wind_m_s", "wind_dir_deg",
                    "temp_C", "humidity_pct"])
        w.writerows(rows)
    log["amedas_stations.csv"] = f"{BOSAI}/amedas/data/map/{{YYYYMMDDHHMMSS}}.json"
    return len(rows), len(picked), stamps[0], stamps[-1]


def _num(rec, key):
    val = rec.get(key)
    if isinstance(val, list):
        val = val[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return ""


def fetch_feed_prefectures(out, log):
    xml = _get(FEED_URL).decode("utf-8", errors="replace")
    entries = re.findall(r"<entry>.*?</entry>", xml, flags=re.S)
    fdir = os.path.join(out, "feed_prefecture_R1")
    os.makedirs(fdir, exist_ok=True)
    best = {}
    for ent in entries:
        if "府県天気予報（Ｒ１）" not in ent:
            continue
        mid = re.search(r"<id>(https://[^<]+_VPFD51_(\d{6})\.xml)</id>", ent)
        if mid is None:
            continue
        code = mid.group(2)
        if code not in PREFECTURE_CODES:
            continue
        upd = re.search(r"<updated>([^<]+)</updated>", ent).group(1)
        if code not in best or upd > best[code][0]:
            best[code] = (upd, mid.group(1))
    for code, (_, url) in sorted(best.items()):
        blob = _get(url)
        with open(os.path.join(fdir, PREFECTURE_CODES[code] + ".xml"), "wb") as fh:
            fh.write(blob)
        time.sleep(0.1)
    log["feed_prefecture_R1/"] = FEED_URL
    return len(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=72,
                    help="AMeDAS history window in hours (default 72)")
    ap.add_argument("--sleep", type=float, default=0.1,
                    help="politeness delay between requests, s")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "denbun"), exist_ok=True)
    log = {}
    fetch_typhoon_jsons(DATA_DIR, log)
    items = fetch_bulletin_list(DATA_DIR, log)
    n_den = fetch_denbun(DATA_DIR, items, args.sleep, log)
    n_rows, n_st, s0, s1 = fetch_amedas(DATA_DIR, args.hours, args.sleep, log)
    n_r1 = fetch_feed_prefectures(DATA_DIR, log)
    manifest = {
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tc_id": TC_ID, "typhoon_number": TY_NUMBER,
        "bulletins": len(items), "denbun_files": n_den,
        "amedas_rows": n_rows, "amedas_stations": n_st,
        "amedas_window_utc": [s1, s0],
        "prefecture_R1_xml": n_r1,
        "sources": log,
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
