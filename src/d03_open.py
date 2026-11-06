# 预测：width, height 大概是多少？（提示：整景 Sentinel-2 在 10 米分辨率下约 10980 像素见方，你下的是整景）
# crs 会打印出什么？（提示：UTM 投影，49 度带，北半球）
# band4 的数组形状是 (height, width) 还是 (height, width, 3)？

import rasterio

img_path = r"D:\rs-taiyuan\data\raw\S2_20250414.SAFE\GRANULE\L2A_T49SFC_A042329_20250414T032306\IMG_DATA\R10m\T49SFC_20250414T031519_B04_10m.jp2"

with rasterio.open(img_path) as src:
    print("driver:", src.driver)
    print("width, height:", src.width, src.height)
    print("band count:", src.count)
    print("crs:", src.crs )
    print("bounds:", src.bounds )
    print("resolution:", src.res )

    band4 = src.read(1)
    print("array shape:", band4.shape)
    print("array dtype:", band4.dtype)
    print("min / max:", band4.min(), band4.max())