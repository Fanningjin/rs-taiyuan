# 做云掩膜
# 读 SCL → 标出坏像元 → 把标记铺到 B04 的网格 → 在 B04 上把坏位置挖空 → 存文件

import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

with open("data/processed/aoi.json", encoding="utf-8") as f:
    aoi = json.load(f)

# 公共路径存成变量，用来拼子路径；r 前缀防止反斜杠被当转义符
base = r"D:\rs-taiyuan\data\raw\S2_20250414.SAFE\GRANULE\L2A_T49SFC_A042329_20250414T032306\IMG_DATA"

#1. 读 SCL（20m）并按 AOI 裁剪
#Scene Classification Layer，场景分类图层——是 Sentinel-2 L2A 产品里 ESA 附赠的一个“地物标签图”

#f-string 拼路径：{base} 替换成变量值，\\ 是“要一个反斜杠”（因为字符串里单个 \ 是转义符，写两个才表示字面反斜杠）
scl_path = f"{base}\\R20m\\T49SFC_20250414T031519_SCL_20m.jp2"  #地物分类文件
with rasterio.open(scl_path) as scl_src:
    win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], scl_src.transform)
    win = win.round_offsets().round_lengths()
    scl = scl_src.read(1,window = win)
    scl_transform = scl_src.window_transform(win)
    scl_crs = scl_src.crs   #坐标系

# “左边两个变量接一个函数的两个返回值”叫解包
# 把数组里所有出现过的值挑出来、去重、从小到大排序。加了 return_counts=True 之后，它不只返回值，还同时返回每个值出现了几次
values, counts = np.unique(scl, return_counts=True)
print("SCL类别:", dict(zip(values, counts)))  #先把values,counts拉链式配对，再变成一个字典


#2. 生成20m的坏像元掩膜
#对 scl 的每个像元判断“值在不在这个名单里”，返回同形状的布尔数组：云、云影、饱和、雪的位置是 True；就是昨天ndvi > 0.3那种布尔掩膜
bad = np.isin(scl, [1,3,8,9,10,11])  #根据地类分类，这几个是有云的
print("SCL中坏像元比例:", bad.mean()*100, "%")

#3. 把掩膜重采样到 B04 的 10m 网格
b04_path = "data/processed/S2_20250414_B04_aoi.tif"  #d03_crop.py用aoi裁剪出来的
with rasterio.open(b04_path) as src:
    b04 = src.read(1).astype("float32")  #因为下一步要往数组里塞 NaN，而原来的 uint16 装不下 NaN
    profile = src.profile.copy()

#先造一个和 B04 同尺寸的全零数组，作为目标画布，重采样结果会填进来
mask10 = np.zeros(b04.shape, dtype=np.uint8)
#把 500×500 的掩膜“铺”到 1000×1000 的网格上

print("scl_crs =", repr(scl_crs))
reproject(
    source = bad.astype(np.uint8), #源是地物分类这个组
    destination = mask10,
    #下面两行说明源数据在地球的什么位置、怎么摆放
    src_transform = scl_transform,
    src_crs = scl_crs,
    dst_transform = profile["transform"],
    dst_crs = profile["crs"],
    resampling = Resampling.nearest,  #用最近邻法，连续量（高程、温度）才用双线性或三次
)

print("10m 网格上被掩掉的像元比例:", mask10.mean()*100, "%")

# 4. 掩膜并写出:把原来图像有云的地方写成NAN再输出
b04[mask10==1] = np.nan #一行完成掩膜：mask10 == 1 生成布尔掩膜，把 B04 上对应位置全部赋成 NaN
# nodata: np.nan 告诉文件“NaN 代表无效”，这样 GIS 软件打开时会正确识别
profile.update({"driver": "GTiff", "dtype": "float32", "nodata": np.nan, "compress": "lzw"})

with rasterio.open("data/processed/S2_20250414_B04_clean.tif", "w", **profile) as dst:
    dst.write(b04,1)

print("saved data/processed/S2_20250414_B04_clean.tif")
