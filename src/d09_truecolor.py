"""
D9 第一步：给 QGIS 目视解译做一张"真彩色底图"
B04 红 / B03 绿 / B02 蓝 → 2%~98% 分位拉伸 → uint8 三波段 tif
注意：分位拉伸只用于"给人看"的显示,不能再用它算指数(跨期NDVI被破坏过,就是这个问题)
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

with open("aoi.json", encoding="utf-8") as f:
    aoi = json.load(f)

scene = Path("data/raw/S2_20250906.SAFE")


def read_10m(band):
    # 和 d08 一样的取窗方式：从 SAFE 里找 R10m 的波段，按 AOI 裁窗读
    p = next(scene.glob(f"GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2"))
    with rasterio.open(p) as src:
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
        win = win.round_offsets().round_lengths()
        return src.read(1, window=win)


def stretch(a):
    # 2%~98% 分位拉伸：把"最亮 2%"和"最暗 2%"剪掉，中段映射到 0~255
    p2, p98 = np.nanpercentile(a, [2, 98])  #新换的AOI有NAN像元，不这样会污染全图，全变成NAN
    out = (np.clip((a - p2) / (p98 - p2), 0, 1) * 255)
    return np.nan_to_num(out, nan=0).astype("uint8")
    #这里只是为了肉眼看，存unit8存储量小，其他软件也读这个


b02 = read_10m("B02")
b03 = read_10m("B03")
# B04 用昨天清洗后的版本，保证和特征栈是同一块地
with rasterio.open("data/processed/S2_20250906_B04_clean.tif") as src:
    b04 = src.read(1).astype("float32")
    profile = src.profile.copy()

#stack把数组摞起来，新增一维，一次就能把三个波段全写进去
rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)])  # 顺序：R,G,B

profile.update({"count": 3, "dtype": "uint8", "nodata": None, "compress": "lzw"})
out = "data/processed/S2_20250906_RGB.tif"
with rasterio.open(out, "w", **profile) as dst:
    dst.write(rgb)
    for i, name in enumerate(["B04_red", "B03_green", "B02_blue"], start=1):
        dst.set_band_description(i, name)

# 回读体检
with rasterio.open(out) as src:
    logging.info("RGB: count=%d, descriptions=%s, shape=%s, dtype=%s",
                 src.count, src.descriptions, src.shape, src.dtypes[0])
    for i, name in enumerate(src.descriptions, start=1):
        arr = src.read(i)
        logging.info("  %s: min=%d, max=%d, mean=%.1f", name, arr.min(), arr.max(), arr.mean())
