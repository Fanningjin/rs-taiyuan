"""D10 任务②：堆叠 9 波段特征栈(4 原始波段 + 5 指数)"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
#format为啥不是%s(asctime这种):logging 没有给它喂裸变量，只有这一堆命名字段

with open("aoi.json", encoding="utf-8") as f:
    aoi = json.load(f)

scene = Path("data/raw/S2_20250906.SAFE")

def read_10m(band):
    p = next(scene.glob(f"GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2"))
    with rasterio.open(p) as src:
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
        win = win.round_offsets().round_lengths()
        b = src.read(1,window=win).astype("float32")
    return b

b02 = read_10m("B02")
b03 = read_10m("B03")
b08 = read_10m("B08")

with rasterio.open("data/processed/S2_20250906_B04_clean.tif") as src:
    b04 = src.read(1)
    transform_b04 = src.transform

with rasterio.open("data/processed/features_20250906.tif") as src:  #5个计算出的指数
    indices = src.read().astype("float32")
    transform_f = src.transform
    profile = src.profile.copy()

#4个原始波段先堆在一起
raw = np.stack([b02, b03, b04, b08]) #不能去掉方括号吗:np.stack 只接收一个参数，而且这个参数必须是"装着一堆数组的容器"
logging.info("raw shape=%s, indices shape=%s", raw.shape, indices.shape)

assert transform_b04 == transform_f, "transform不一致!"
assert raw.shape[1:] == indices.shape[1:], "尺寸不一致!" 
#读出来是后面两个吧，但为什么不是从0开始 :shape 是 (波段, 行, 列) 三元组

#组合成9个波段一起
stacked = np.concatenate([raw, indices], axis=0)
logging.info("堆叠完成:%s", stacked.shape)

#nan_mask 是"9 个波段里任一为NaN就算"，它统计的是所有波段NaN的并集
nan_mask = np.isnan(stacked).any(axis=0)
logging.info("含NAN像元:%d个", nan_mask.sum())

names = ["B02_blue", "B03_green", "B04_red", "B08_nir",
         "NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]

profile.update({"count":len(names), "dtype":"float32", "nodata":np.nan, "compress":"lzw"})
out = "data/processed/features9_20250906.tif"

with rasterio.open(out, "w", **profile) as dst:
    dst.write(stacked)
    for i, name in enumerate(names,start=1):
        dst.set_band_description(i, name)

with rasterio.open(out) as src:
    logging.info("回读:count=%d, dtype=%s", src.count, src.dtypes[0])
    for i,d in enumerate(src.descriptions, start=1):
        logging.info(" 波段%d: %s", i,d)
