"""
D9 补点：水体/裸地两类在城区窗口里误判率高，核订后数量不足 100。
用更严格的阈值从特征栈里再抽一批"高置信"候选，进 QGIS 快速补审。

水:MNDWI>0.15 且 NDVI<0.1(汾河主槽级别)
裸:BSI>0.15 且 NDVI<0.1(真正的土面)
林:NDVI>0.58(城市窗口里最像树冠的一小撮)
耕:NDVI 0.35~0.5 + 局部纹理>0.06 + 只取北部农田带(行<150@20m)
    ——西山和城区的绿地光谱和田一模一样，靠"田块纹理碎"+"田只在北带"两个先验把田圈出来
已有的 720 个候选像元会被排除，避免重复。
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

N_WATER = 60    # 水体补多少个候选（目标：86 + 补审后剩下的 ≥100）
N_BARE = 120    # 裸地补多少个候选（目标：33 + 补审后剩下的 ≥100）
N_FOREST = 80   # 林地补多少个候选（目标：80 + 补审后剩下的 ≥100）
N_CROP = 150    # 耕地补多少个候选（目标：7 + 补审后剩下的 ≥100，耕地基数最差所以多补些）
N_GRASS = 100   # 草地补多少个候选（目标：54 + 补审后剩下的 ≥100）
SEED = 7

with rasterio.open("data/processed/features_20250906.tif") as src:
    names = [n.upper() for n in src.descriptions]
    stack = {n: src.read(names.index(n) + 1) for n in ["NDVI", "MNDWI", "BSI"]}
    transform = src.transform

ndvi, mndwi, bsi = stack["NDVI"], stack["MNDWI"], stack["BSI"]

# 已有的 720 个像元，排除掉不重复抽
old = pd.read_csv("data/samples/candidates_20250906.csv")
taken = set(zip(old["row"], old["col"]))


def local_std(a, k=5):
    w = sliding_window_view(a, (k, k))
    s = w.std(axis=(-2, -1))
    out = np.full_like(a, np.nan)
    pad = k // 2
    out[pad:-pad, pad:-pad] = s
    return out


# 20m 分辨率上算耕地先验，再放大回 10m
ndvi20 = ndvi[::2, ::2]
ls20 = local_std(ndvi20, 5)
crop20 = (ndvi20 > 0.35) & (ndvi20 <= 0.5) & (ls20 > 0.06)
env20 = np.zeros_like(crop20, dtype=bool)
env20[:150, 140:] = True                      # 北部农田带
crop_mask = np.repeat(np.repeat(crop20 & env20, 2, axis=0), 2, axis=1)

# 草地同样带空间先验：只从西山（西侧三分之一）的稀疏植被里抽
grass20 = (ndvi20 > 0.2) & (ndvi20 <= 0.35)
env_grass20 = np.zeros_like(grass20, dtype=bool)
env_grass20[:, :150] = True
grass_mask = np.repeat(np.repeat(grass20 & env_grass20, 2, axis=0), 2, axis=1)

water_mask = (mndwi > 0.15) & (ndvi < 0.1)
bare_mask = (bsi > 0.15) & (ndvi < 0.1) & ~water_mask
forest_mask = (ndvi > 0.58) & ~water_mask & ~bare_mask
logging.info("严格候选像元：水 %d，裸 %d，林 %d，耕 %d，草 %d",
             water_mask.sum(), bare_mask.sum(), forest_mask.sum(),
             crop_mask.sum(), grass_mask.sum())

rng = np.random.default_rng(SEED)
records = []
for cls, mask, n_want in [("water", water_mask, N_WATER),
                          ("bare", bare_mask, N_BARE),
                          ("forest", forest_mask, N_FOREST),
                          ("cropland", crop_mask, N_CROP),
                          ("grassland", grass_mask, N_GRASS)]:
    cand = [(r, c) for r, c in np.argwhere(mask) if (r, c) not in taken]
    n_take = min(n_want, len(cand))
    if n_take < n_want:
        logging.warning("%s 严格候选不足：%d < %d", cls, len(cand), n_want)
    for idx in rng.choice(len(cand), size=n_take, replace=False):
        row, col = cand[idx]
        x, y = transform * (col + 0.5, row + 0.5)
        records.append({"class": cls, "x_utm": round(x, 2), "y_utm": round(y, 2),
                        "row": int(row), "col": int(col)})

df = pd.DataFrame(records)
df.insert(0, "sample_id", range(1, len(df) + 1))
out = "data/samples/candidates_extra_20250906.csv"
df.to_csv(out, index=False)
logging.info("写出 %s：\n%s", out, df["class"].value_counts().to_string())
