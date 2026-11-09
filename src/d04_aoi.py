#确定好项目的研究范围和坐标：10kmx10km

import json
import rasterio

crop_path = "data/processed/d03_b04_crop.tif"  #d03剪出来的范围稍大的那一块
full_path = r"D:\rs-taiyuan\data\raw\S2_20250414.SAFE\GRANULE\L2A_T49SFC_A042329_20250414T032306\IMG_DATA\R10m\T49SFC_20250414T031519_B04_10m.jp2"

with rasterio.open(crop_path) as src:
    b = src.bounds

    print("bounds:", b)  #自己看一下bounds什么形式

    left = b.left + 2000
    bottom = b.bottom + 2000
    aoi = {
        "left":left,
        "bottom":bottom,
        "right": left + 10000,
        "top": bottom + 10000,
    }

with rasterio.open(full_path) as src:
    fb = src.bounds
    inside = (aoi["left"] >= fb.left and aoi["right"] <= fb.right   #为什么一个取值用#，一个用点号
              and aoi["top"] <= fb.top and aoi["bottom"] >= fb.bottom) #为什么打括号？
    print("AOI在整景范围内:", inside)

km2 = (aoi["right"] - aoi["left"]) * (aoi["top"] - aoi["bottom"]) / 1e6
print("AOI面积", km2, "km²")

with open("aoi.json", "w", encoding="utf-8") as f:
    json.dump(aoi, f, indent=2)

print("saved data/processed/aoi.json")
