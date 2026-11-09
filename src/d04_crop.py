# 用aoi把10m波段裁剪齐

import json
import rasterio
from rasterio.windows import from_bounds

with open("data/processed/aoi.json", encoding="utf-8") as f:   #encoding：告诉Python 用什么字符编码读这个文件
    aoi = json.load(f)  #把文件内容解析成python的字典

base = r"D:\rs-taiyuan\data\raw\S2_20250414.SAFE\GRANULE\L2A_T49SFC_A042329_20250414T032306\IMG_DATA\R10m"
bands = {
    "B02": "T49SFC_20250414T031519_B02_10m.jp2",
    "B03": "T49SFC_20250414T031519_B03_10m.jp2",
    "B04": "T49SFC_20250414T031519_B04_10m.jp2",
    "B08": "T49SFC_20250414T031519_B08_10m.jp2",
}

#循环波段组合里的各个波段
for name, fname in bands.items():  #name是键，fname是值
    with rasterio.open(f"{base}\\{fname}") as src:
        # 你手里是地理范围（left/bottom/right/top，单位是米），rasterio 需要的是窗口（从第几行第几列开始、多少行多少列）。
        # 它用 transform 的逆运算来换算;必须按 left, bottom, right, top（西、南、东、北） 的顺序传
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
        # round_offsets() 把起点四舍五入成整数，round_lengths() 把宽高也取整，两步合起来就把窗口“对齐到像元格”
        win = win.round_offsets().round_lengths()  #后一步在前一步的结果上再调用；前是起点，后是宽高
        #aoi是地理坐标，是m，所以要先变成像素坐标，以此来转换

        #这里就和昨天的d03_crop.py一样了，裁剪小窗口
        data = src.read(1, window=win)  #在这里裁剪
        transform = src.window_transform(win) #window转成transform

        profile = src.profile.copy()
        profile.update({
            "driver":"GTiff",
            "width":data.shape[1],
            "height":data.shape[0],
            "transform":transform,
            "compress":"lzw",
        })

    out = f"data/processed/S2_20250414_{name}_aoi.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
    print(name, data.shape, "->", out)

