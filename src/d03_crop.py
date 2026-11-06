# 练习：将全景影像裁一小块出来

import rasterio
from rasterio.windows import Window

img_path = r"D:\rs-taiyuan\data\raw\S2_20250414.SAFE\GRANULE\L2A_T49SFC_A042329_20250414T032306\IMG_DATA\R10m\T49SFC_20250414T031519_B04_10m.jp2"

with rasterio.open(img_path) as src:
    # 读进来是像素格子，没有经纬度地址，要用transform实时计算各个格子的经纬度地址
    col_off = (src.width - 2000) // 2
    row_off = (src.height - 2000) // 2
    window = Window(col_off, row_off, 2000, 2000)  #在正中央构建了一个长款各为2000像素的窗口

    b04 = src.read(1, window = window)  #将文件里对应区域读取出来
    win_transform = src.window_transform(window)  #根据窗口在原图中的位置，算出这一小块的地理变换

    print("window shape:", b04.shape)
    print("window transform:", win_transform)

    profile = src.profile.copy()  #参数卡片
    profile.update({
            "driver": "GTiff",
            "width": b04.shape[1],
            "height": b04.shape[0],
            "transform": win_transform,
            "compress": "lzw", #无损压缩
    })

    print(src.profile)

    # 检查尺寸对不对
    print("profile 宽高:", profile["width"], profile["height"])
    print("b04 形状:", b04.shape)


with rasterio.open("data/processed/d03_b04_crop.tif", "w", **profile) as dst:  #“w”写入模式
    dst.write(b04,1)    #**profile将卡片里的键值展开成关键字参数
