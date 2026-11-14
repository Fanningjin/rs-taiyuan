"""
D9 采样：从特征栈自动生成 6 类候选样本点（"弱监督"初标），再拿进 QGIS 人工核订。

思路：
1. 用指数阈值给每个像元一个"初判类别"（规则互斥：按优先级，后面的规则要避开已经占掉的像元）
2. 每类随机抽 N_PER_CLASS 个点
3. 像元行列 → UTM 坐标，输出 CSV(x_utm / y_utm)
4. 顺手画一张预览图，先自己看一眼点位对不对

类别(6 类，和计划一致):
    cropland 耕地 / forest 林地 / grassland 草地 / water 水体 / builtup 建设用地 / bare 裸地及其他
"""

import logging

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

# ---------------- 参数区：今天主要就调这里 ----------------
N_PER_CLASS = 120    # 每类抽多少个候选点（计划要求核订后每类 100~150）
SEED = 42            # 随机种子：同一个种子每次抽的点完全一样，方便复现

# 阈值（2026-11-13 换 AOI 后重新校准：新窗口=汾河城区）
TH = {
    "water_mndwi":    0.1,    # MNDWI 大于它，且 NDVI 很低 → 水体（新窗口汾河水面 MNDWI 约 0.1~0.3）
    "water_ndvi_max": 0.2,
    "built_ndbi":     0.0,    # NDBI 大于它，且 NDVI 低 → 建设用地
    "built_ndvi_max": 0.25,
    "bare_bsi":       0.05,   # BSI 大于它，且 NDVI 很低 → 裸土
    "bare_ndvi_max":  0.2,
    "forest_ndvi":    0.5,    # NDVI 很高 → 先当林地（山里成片树冠；9月 NDVI>0.5 只占 8.95%）
    "crop_ndvi":      0.35,   # 中等偏高的 NDVI → 先当耕地（9月庄稼正绿）
    "grass_ndvi":     0.2,    # 低一点的稀疏植被 → 先当草地
}

COLORS = {"cropland": "#2ecc40", "forest": "#1b5e20", "grassland": "#b2c21a",
          "water": "#1e6fff", "builtup": "#e61919", "bare": "#e0a01e"}

# ---------------- 读特征栈 ----------------
with rasterio.open("data/processed/features_20250906.tif") as src:
    # 用文件里存的波段描述来找波段，不再靠"第几个"死记
    names = [n.upper() for n in src.descriptions]  #upper:变大写
    # [表达式 for 元素 in 序列] 读成：对序列里每个元素，算一遍表达式，结果攒成一个列表
    stack = {n: src.read(names.index(n) + 1) for n in ["NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]} #三维波段数组
    transform = src.transform
    bounds = src.bounds

ndvi = stack["NDVI"]
mndwi = stack["MNDWI"]
ndbi = stack["NDBI"]
bsi = stack["BSI"]

# ---------------- 阈值规则（按优先级排序，互斥） ----------------
priority = [
    ("water",    (mndwi > TH["water_mndwi"]) & (ndvi < TH["water_ndvi_max"])), #后面是True/False网格，和数据ndvi这些一样大
    ("builtup",  (ndbi > TH["built_ndbi"]) & (ndvi < TH["built_ndvi_max"])),
    ("bare",     (bsi > TH["bare_bsi"]) & (ndvi < TH["bare_ndvi_max"])),
    ("forest",   (ndvi > TH["forest_ndvi"])),
    ("cropland", (ndvi > TH["crop_ndvi"]) & (ndvi <= TH["forest_ndvi"])),
    ("grassland", (ndvi > TH["grass_ndvi"]) & (ndvi <= TH["crop_ndvi"])),
]

labels = np.zeros(ndvi.shape, dtype="uint8")   # 0 = 未分类
for code, (name, mask) in enumerate(priority, start=1): #code就是enumerate发的号
    mask = mask & (labels == 0)  #已被高优先级规则占掉的像元不再重复计数；是0也就是还没分类的才留下
    labels[mask] = code
    logging.info("%-10s 候选像元 %8d (%.2f%%)", name, mask.sum(), mask.sum() / labels.size * 100)

# ---------------- 每类随机抽点 ----------------
rng = np.random.default_rng(SEED)
records = []
#循环各个类别
for code, (name, _) in enumerate(priority, start=1):
    cand = np.argwhere(labels == code)  #cand是所有候选像元的(行, 列)；argwhere找到非0元素的位置
    n_take = min(N_PER_CLASS, len(cand)) #N_PER_CLASS设定好的120，每类抽120个点
    if n_take < N_PER_CLASS:  #不够120个点抽
        logging.warning("%s 候选不足：只有 %d 个像元，想抽 %d", name, len(cand), N_PER_CLASS)
    picked = cand[rng.choice(len(cand), size=n_take, replace=False)] #choice随机选一个元素返回；每类抽120个；replace不放回抽样
    for row, col in picked:
        x, y = transform * (col + 0.5, row + 0.5)   # 像元行列 → UTM 坐标（取像元中心）
        #transform是Affine类型，乘时是调用方法代式子计算，计算式就在Affine里
        records.append({"class": name, "x_utm": round(x, 2), "y_utm": round(y, 2),  #保留2位的四舍五入
                        "row": int(row), "col": int(col)})

df = pd.DataFrame(records)  #把这串字典“抻”成一张二维表格，所有字典的键变成列名，每个字典变成一行，对不上的自动补空值
df.insert(0, "sample_id", range(1, len(df) + 1)) #最左边插一列编号
out = "data/samples/candidates_20250906.csv"
df.to_csv(out, index=False)
logging.info("写出 %s, 共 %d 个点", out, len(df))
logging.info("各类点数：\n%s", df["class"].value_counts().to_string())

# ---------------- 预览图：NDVI 灰度底 + 六色点 ----------------
fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
ax.imshow(ndvi, cmap="gray", vmin=-0.2, vmax=0.8, origin="upper",
          extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
for name, _ in priority:
    sub = df[df["class"] == name]
    ax.scatter(sub["x_utm"], sub["y_utm"], s=6, c=COLORS[name], label=name, linewidths=0)
ax.set_title("Candidate samples over NDVI (2025-09-06)")
ax.legend(loc="lower right", fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig("outputs/d09_candidates.png")
logging.info("预览图: outputs/d09_candidates.png")
