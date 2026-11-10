"""
读B08波段
"""

import json
import logging
import numpy as np
import rasterio
from pathlib import Path
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

scene = Path("data/raw/S2_20250414.SAFE")
b08_path = next(scene.glob("GRANULE/*/IMG_DATA/R10m/*_B08*.jp2"))

with open("aoi.json", encoding="utf-8") as f:
    aoi = json.load(f)

#1.B04：昨天的掩膜成果，float32，云区已是 NaN
with rasterio.open("data/processed/S2_20250414_B04_clean.tif") as src:
    b04 = src.read(1)  #就是float格式吗
    profile = src.profile.copy() #如果不copy，关掉了文件以后也不会影响原来的吧，除非要update
    logging.info("B04读取完成:%s, NaN=%d", b04.shape, np.isnan(b04).sum())

#2.B08：从 SAFE 读窗口，和批量脚本里是同一套写法
with rasterio.open(b08_path) as src:
    win = from_bounds(aoi["left"],aoi["bottom"],aoi["right"],aoi["top"],src.transform)
    win = win.round_offsets().round_lengths()
    b08 = src.read(1,window=win).astype("float32")

#window_transform函数只传入一个win参数怎么运算的，也不知道其他分辨率什么信息
#b08没去云:NDVI 里 nan - b08 还是 nan、nan / x 还是 nan，B04 的 NaN 会顺着算术传播，云区 NDVI 自动是 NaN。这就是"掩膜继承"

#3.NDVI = (NIR - Red) / (NIR + Red)
ndvi = (b08-b04)/(b08+b04)
ndvi = np.clip(ndvi, -1, 1).astype("float32")

#4.存tif(profile沿用B04_clean的，本来就是对的)
profile.update({"dtype":"float32", "nodata":np.nan ,"compress":"lzw"})
out = "data/processed/S2_20250414_NDVI.tif"
with rasterio.open(out, "w", **profile) as dst:
    dst.write(ndvi,1)

#验证结果
logging.info("NDVI: NaN=%d, min=%.3f, max=%.3f, mean=%.3f",
             np.isnan(ndvi).sum(),
             np.nanmin(ndvi), np.nanmax(ndvi), np.nanmean(ndvi))