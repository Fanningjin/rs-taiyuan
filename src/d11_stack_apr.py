"""1:得到四月的特征栈并保存"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

with open("aoi.json",encoding="utf-8") as f:
    aoi = json.load(f)

scene = Path("data/raw/S2_20250414.SAFE")

def read_window(p):
    """从jp2读AOI窗口,返回(数组,窗口transform,crs)"""
    with rasterio.open(p) as src:
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(1, window=win).astype("float32")
        wt = src.window_transform(win)
        crs = src.crs
    return arr, wt, crs

# 1) 10m 原始波段: B02/B03/B08 从 SAFE 现读
b02 = read_window(next(scene.glob("GRANULE/*/IMG_DATA/R10m/*_B02_10m.jp2")))[0]  #读出来是三个东西，取第一个数组结果
b03 = read_window(next(scene.glob("GRANULE/*/IMG_DATA/R10m/*_B03_10m.jp2")))[0]
b08 = read_window(next(scene.glob("GRANULE/*/IMG_DATA/R10m/*_B08_10m.jp2")))[0]

# 2) B04 用去云成品, 掩膜的 NaN 从这里来
with rasterio.open("data/processed/S2_20250414_B04_clean.tif") as src:
    b04 = src.read(1)
    profile = src.profile.copy()

# 3) B11: 20m 窗口读出来, 双线性重采样到 10m 网格
b11_20, b11_tf, b11_crs = read_window(next(scene.glob("GRANULE/*/IMG_DATA/R20m/*_B11_20m.jp2")))
b11 = np.zeros(b04.shape, dtype="float32")
reproject(
    source=b11_20, destination=b11,
    src_transform=b11_tf, src_crs=b11_crs,
    dst_transform=profile["transform"], dst_crs=profile["crs"],
    resampling=Resampling.bilinear,
)

# 4) 从 d08_features.py 抄五个指数公式(照抄, 别改字母)
#B02:蓝, B03:绿, B04:红, B08:近红外, B11:短波红外
ndvi  = (b08 - b04) / (b08 + b04)    # 植被
ndwi  = (b03 - b08) / (b03 + b08)    # 水体（McFeeters 1996）
mndwi = (b03 - b11) / (b03 + b11)    # 水体增强版（Xu 2006）
ndbi  = (b11 - b08) / (b11 + b08)    # 建成区
bsi   = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02))  # 裸土

# 5) 堆 9 波段: 4 原始波段在前, 5 指数在后(顺序与 9 月一致)
raw = np.stack([b02, b03, b04, b08])  #为啥不用b11？
idx = np.stack([np.clip(ndvi,-1,1),np.clip(ndwi,-1,1),np.clip(mndwi,-1,1),
               np.clip(ndbi,-1,1),np.clip(bsi,-1,1)])   #clip将范围框在里面
stacked = np.concatenate([raw,idx], axis=0)  #再看看stack和con的区别？
log.info("堆叠完成: %s", stacked.shape)

nan_mask = np.isnan(stacked).any(axis=0)
log.info("含NaN像元: %d个", nan_mask.sum())

names = ["B02_blue", "B03_green", "B04_red", "B08_nir",
         "NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]
profile.update({"count":len(names), "dtype":"float32",
                 "nodata":np.nan, "compress":"lzw"})
out = "data/processed/features9_20250414.tif"
with rasterio.open(out, "w", **profile) as dst:
    dst.write(stacked)
    for i,name in enumerate(names, start=1):
        dst.set_band_description(i,name)


# 6) 回读体检
with rasterio.open(out) as src:
    log.info("回读: count=%d, dtype=%s, shape=%s", src.count,src.dtypes[0],src.shape)
    #这些信息存在profile里，读的时候是直接读不用调用profile读吗。是的，存的时候要用，读时不用
    for i,d in enumerate(src.descriptions, start=1):
        log.info("  波段%d: %s", i, d)

# 7) 交叉验证: 与 d06 产出的 4-14 NDVI 逐像元对比(加分项)
with rasterio.open("data/processed/S2_20250414_NDVI.tif") as src:
    ndvi_old = src.read(1)
diff = np.abs(stacked[4] - ndvi_old)
log.info("与d06 NDVI对比:最大差= %.2e", np.nanmax(diff))
