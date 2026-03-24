#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按日期范围提取 10km×10km 矩形内的工人轨迹点（只要在框内的点；出了框不记录）
仅需修改 START_DAY / END_DAY / RAW_DIR / OUT_DIR 即可。
输出: workers_YYYY-MM-DD.csv
列:  day, wid, timestamp, time_str, lon_gcj, lat_gcj, lon_wgs, lat_wgs
"""

import os, io, tarfile, glob, re
import pandas as pd
import numpy as np
from datetime import datetime, date, timezone, timedelta

# ============ 路径与日期范围 ============#
RAW_DIR = r"D:\2016年10月成都市二环局部区域轨迹数据"
OUT_DIR_1 = r"D:\biyelunwen\data\worker"
OUT_DIR_2 = r"D:\biyelunwen\data\worker_id"
os.makedirs(OUT_DIR_1, exist_ok=True)
os.makedirs(OUT_DIR_2, exist_ok=True)


START_DAY = "2016-10-28"
END_DAY   = "2016-10-31"


# ============ 空间范围（10 km × 10 km） ============#
CENTER_LON, CENTER_LAT = 104.06, 30.67     # 中心点
HALF_SIZE_KM = 5.0                          # 半边长 5km -> 10×10 km

def km_to_deg_lon(lat, km): return km / (111.32 * np.cos(np.deg2rad(lat)))
def km_to_deg_lat(km):      return km / 110.574

DLON = km_to_deg_lon(CENTER_LAT, HALF_SIZE_KM)
DLAT = km_to_deg_lat(HALF_SIZE_KM)
LON_MIN, LON_MAX = CENTER_LON - DLON, CENTER_LON + DLON
LAT_MIN, LAT_MAX = CENTER_LAT - DLAT, CENTER_LAT + DLAT
print(f"[BOX] lon {LON_MIN:.5f} ~ {LON_MAX:.5f}, lat {LAT_MIN:.5f} ~ {LAT_MAX:.5f}  (center=({CENTER_LON},{CENTER_LAT}), size=10×10 km)")

# ============ 时区/时间格式 ============#
TZ = timezone(timedelta(hours=8))
def ts2str(ts): return datetime.fromtimestamp(ts, tz=TZ).strftime("%Y-%m-%d %H:%M:%S")

# ============ GCJ-02 -> WGS84 ============#
def _out_of_china(lat, lon):
    return not (0.8293 <= lat <= 55.8271 and 72.004 <= lon <= 137.8347)
def _tlat(x, y):
    ret = -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*abs(x)**0.5
    ret += (20.0*np.sin(6.0*x*np.pi) + 20.0*np.sin(2.0*x*np.pi))*2.0/3.0
    ret += (20.0*np.sin(y*np.pi) + 40.0*np.sin(y/3.0*np.pi))*2.0/3.0
    ret += (160.0*np.sin(y/12.0*np.pi) + 320*np.sin(y*np.pi/30.0))*2.0/3.0
    return ret
def _tlon(x, y):
    ret = 300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*abs(x)**0.5
    ret += (20.0*np.sin(6.0*x*np.pi) + 20.0*np.sin(2.0*x*np.pi))*2.0/3.0
    ret += (20.0*np.sin(x*np.pi) + 40.0*np.sin(x/3.0*np.pi))*2.0/3.0
    ret += (150.0*np.sin(x/12.0*np.pi) + 300.0*np.sin(x/30.0*np.pi))*2.0/3.0
    return ret
def gcj02_to_wgs84(lat, lon):
    if _out_of_china(lat, lon): return float(lat), float(lon)
    a = 6378245.0; ee = 0.00669342162296594323
    dLat = _tlat(lon - 105.0, lat - 35.0)
    dLon = _tlon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * np.pi
    magic = np.sin(radLat); magic = 1 - ee * magic * magic
    sqrtMagic = np.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * np.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * np.cos(radLat) * np.pi)
    mgLat = lat + dLat; mgLon = lon + dLon
    return lat * 2 - mgLat, lon * 2 - mgLon

# ============ 日期解析/过滤 ============#
def parse_day_from_tar(filename: str) -> date | None:
    """
    支持两种命名：
      - '10-19.tar.gz'        -> 2016-10-19
      - '2016-10-19.tar.gz'   -> 2016-10-19
    其他格式返回 None
    """
    base = os.path.basename(filename).lower()
    base = base[:-7] if base.endswith(".tar.gz") else base
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", base)
    if m:
        y, mth, d = map(int, m.groups())
        try: return date(y, mth, d)
        except: return None
    m = re.match(r"^10-(\d{1,2})$", base)
    if m:
        d = int(m.group(1))
        if 1 <= d <= 31:
            return date(2016, 10, d)
    return None

def in_target_range(day_dt: date, start_s: str, end_s: str) -> bool:
    sd = date.fromisoformat(start_s)
    ed = date.fromisoformat(end_s)
    return sd <= day_dt <= ed

# ============ 主处理（只保留框内点） ============#
def process_one_day_tar(tar_path: str):
    day_dt = parse_day_from_tar(tar_path)
    if day_dt is None or not in_target_range(day_dt, START_DAY, END_DAY):
        return
    day_str = day_dt.isoformat()

    out_path = os.path.join(OUT_DIR_1, f"workers_{day_str}.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    rows = []          # (day, wid, ts, time_str, lon_gcj, lat_gcj, lon_wgs, lat_wgs)
    wid_set = set()

    print(f"[+] 处理 {os.path.basename(tar_path)} -> {day_str}")
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in (mm for mm in tar.getmembers() if mm.isfile()):
            f = tar.extractfile(m)
            if f is None:
                continue
            buf = io.TextIOWrapper(f, encoding="ascii", errors="replace", newline="")
            for chunk in pd.read_csv(buf, header=None, chunksize=1_000_000, low_memory=False):
                if chunk.shape[1] < 5:
                    continue
                chunk = chunk.iloc[:, :5]
                chunk.columns = ["wid", "order_id", "timestamp", "lon", "lat"]

                # 统一类型
                # wid 保留为字符串（避免 1/01/1.0 之类混淆）
                chunk["wid"] = chunk["wid"].astype(str)

                # ts 处理：毫秒->秒
                chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
                if chunk["timestamp"].head(1000).mean() > 1e12:
                    chunk["timestamp"] = chunk["timestamp"] / 1000.0

                chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
                chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
                chunk = chunk.dropna(subset=["wid","timestamp","lon","lat"])

                # 空间过滤：只保留 10×10 km 框内点
                mask = (chunk["lon"].between(LON_MIN, LON_MAX)) & (chunk["lat"].between(LAT_MIN, LAT_MAX))
                if not mask.any():
                    continue
                sub = chunk.loc[mask, ["wid","timestamp","lon","lat"]].copy()

                # 生成输出列
                # 注意：过滤判断用 GCJ，经纬度转换用于输出（WGS）
                la_wgs, lo_wgs = gcj02_to_wgs84, gcj02_to_wgs84  # 只是别名，下面逐行调用

                # 逐行追加（更稳；如果追求极致速度可向量化再拼接）
                for wid, ts, lo, la in zip(sub["wid"], sub["timestamp"], sub["lon"], sub["lat"]):
                    try:
                        ts_i = int(ts)
                    except Exception:
                        continue
                    lat_wgs, lon_wgs = gcj02_to_wgs84(float(la), float(lo))
                    rows.append([day_str, wid, ts_i, ts2str(ts_i), float(lo), float(la), float(lon_wgs), float(lat_wgs)])
                    wid_set.add(wid)

    if rows:
        df = pd.DataFrame(rows, columns=["day","wid","timestamp","time_str","lon_gcj","lat_gcj","lon_wgs","lat_wgs"])
        # 为便于后续处理，按 wid, timestamp 排序
        df.sort_values(by=["wid","timestamp"], inplace=True, kind="mergesort")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"    -> {out_path}  ({len(rows)} 条点位，{len(wid_set)} 位工人)")
        # 可选：导出当天出现的工人清单
        wid_path = os.path.join(OUT_DIR_2, f"workers_{day_str}_wids.csv")
        pd.DataFrame(sorted(wid_set), columns=["wid"]).to_csv(wid_path, index=False, encoding="utf-8")
        print(f"    -> {wid_path}  (工人清单)")
    else:
        print("    [warn] 该日框内没有工人点")

def main():
    tars = sorted(glob.glob(os.path.join(RAW_DIR, "*.tar.gz")))
    if not tars:
        raise SystemExit(f"未在 {RAW_DIR} 找到 .tar.gz")
    for t in tars:
        process_one_day_tar(t)

if __name__ == "__main__":
    main()
