"""
读入5个波段:B02:蓝, B03:绿, B04:红, B08:近红外, B11:短波红外
特征工程：读 B02/B03/B04/B08/B11, 计算 NDVI/NDWI/MNDWI/NDBI/BSI,
输出多波段特征栈 features_20250906.tif(5 波段，带波段描述)
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import json
import logging
from pathlib import Path
from rasterio.warp import reproject, Resampling

logging.basicConfig(level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S")

with open("aoi.json", encoding="utf-8") as f:
    aoi = json.load(f)

scene = Path("data/raw/S2_20250906.SAFE")

def read_10m(band):
    p = next(scene.glob(f"GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2"))
    with rasterio.open(p) as src:
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
        win = win.round_offsets().round_lengths()
        b = src.read(1, window=win).astype("float32")
        return b

with rasterio.open("data/processed/S2_20250906_B04_clean.tif") as src:
     b04 = src.read(1)
     profile = src.profile.copy()

b02 = read_10m("B02")  #传参是这个形式吗？
b03 = read_10m("B03")
b08 = read_10m("B08")

p11 = next(scene.glob("GRANULE/*/IMG_DATA/R20m/*_B11_20m.jp2"))
with rasterio.open(p11) as src:
    win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
    win = win.round_offsets().round_lengths()
    b11_20 = src.read(1,window=win).astype("float32")
    b11_transform = src.window_transform(win)
    b11_crs = src.crs

b11 = np.zeros(b04.shape, dtype="float32")
reproject(
     source = b11_20, destination=b11,
     src_transform=b11_transform,src_crs=b11_crs,
     dst_transform=profile["transform"], dst_crs=profile["crs"],
     resampling=Resampling.bilinear,
)

#对好几个同类型的东西做同样的事时，把 (名字, 数据) 攒成列表再循环
for name, arr in [("B02", b02), ("B03", b03), ("B04", b04),
                  ("B08", b08), ("B11", b11)]:
    logging.info("%s: shape=%s, mean=%.1f", name, arr.shape, np.nanmean(arr))


#算五个指数，逐个做数值体检
#B02:蓝, B03:绿, B04:红, B08:近红外, B11:短波红外
ndvi  = (b08 - b04) / (b08 + b04)    # 植被
ndwi  = (b03 - b08) / (b03 + b08)    # 水体（McFeeters 1996）
mndwi = (b03 - b11) / (b03 + b11)    # 水体增强版（Xu 2006）
ndbi  = (b11 - b08) / (b11 + b08)    # 建成区
bsi   = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02))  # 裸土

features = {"NDVI":ndvi, "NDWI":ndwi, "MNDWI":mndwi, "NDBI":ndbi, "BSI":bsi}
for name, arr in features.items(): #为什么不是直接in features
    arr = np.clip(arr, -1, 1) #防止个别像元越界
    features[name] = arr
    logging.info("%s:min=%.3f, max=%.3f, mean=%.3f", name,
                 np.nanmin(arr),np.nanmax(arr),np.nanmean(arr))

#写features.tif，5个波段组合成一幅图
names = ["NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]

profile.update({"count":len(names), "dtype":"float32", "nodata":np.nan, "compress":"lzw"})

out = "data/processed/features_20250906.tif"
with rasterio.open(out, "w", **profile) as dst:
    for i, name in enumerate(names, start=1):  #波段号从1开始; enumerate等价于手动维护一个计数器
        dst.write(features[name], i)
        dst.set_band_description(i,name) #给波段贴名字标签
        #TIFF文件头里可以存每个波段的文字描述，这行让文件自带"说明书"，以后QGIS里打开能直接看到波段叫NDVI而不是Band 1

#回读体检
with rasterio.open(out) as src:
    logging.info("features: count=%d, names=%s, shape=%s",
                 src.count, src.descriptions, src.shape)