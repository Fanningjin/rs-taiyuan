"""
#D5 批量预处理：遍历文件夹里的 .SAFE, 逐景裁剪 + 云掩膜
用法:python batch_preprocess.py --input data/raw --aoi aoi.json --out data/processed
"""


import argparse
import logging
from pathlib import Path

# 日志系统
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,  #最低输出级别，日志级别从低到高是 DEBUG < INFO < WARNING < ERROR
        format="%(asctime)s | %(levelname)s | %(message)s",  #3个占位符
        datefmt="%H:%M:%S",
    )

#从命令行读取一些输入参数
def parse_args():
    p = argparse.ArgumentParser(description="Sentinel-2 批量预处理") #造一个命令行解析器，description会显示在 --help 开头
    #"--input" 这个带 -- 的名字是"长选项"，用户在命令行写 --input data/raw；argparse 自动把它转成属性名 args.input
    p.add_argument("--input", required=True, help="原始影像文件夹,如data/raw")  #各管一个命令行参数
    p.add_argument("--aoi", default="aoi.json", help="AOI边界文件")
    p.add_argument("--out", default="data/processed", help="输出文件夹")
    return p.parse_args()


def process_one(scene_dir, aoi_path, out_dir):
    #D4 脚本里“读取→裁剪→掩膜→保存”整段
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
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], scl_src.transform) #transform转窗口
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

    logging.info("处理:%s", scene_dir.name)
#可是这个函数都是按照单个具体目录来的，不用修改内部吗？


def main():
    setup_logger() #先运行日志系统
    args = parse_args()  #读取命令行输入的参数
    in_dir = Path(args.input) #将字符串包成Path对象
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    #parents=True 表示中间层缺了也一起建（像 mkdir -p）；exist_ok=True 表示目录已存在时不报错

    scenes = sorted(in_dir.glob("*.SAFE"))  #收成列表并按字母排序（就是按时间顺序排列）
    logging.info("共找到 %d 景", len(scenes))
    #if是0都不会进入循环

    ok = 0
    skipped = 0
    for s in scenes:  #s是列表里的一个Path
        try: #有任何异常会跳过Ok这一行进入except
            process_one(s, args.aoi, out_dir)
            ok += 1
        except Exception as e:  #e是异常对象本身
            skipped += 1
            logging.warning("跳过 %s:%s", s.name, e)

    logging.info("结束: 成功 %d 景, 跳过 %d 景", ok, skipped) #汇总


#只有我被当作脚本跑时才执行 main()，被 import 时只提供函数不自动跑
if __name__ == "__main__":
     main()