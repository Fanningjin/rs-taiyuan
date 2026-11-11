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

    with open(aoi_path, encoding="utf-8") as f:
        aoi = json.load(f)


    #1. 读 SCL（20m）并按 AOI 裁剪
    #scene_dir是到.SAFE这一层文件夹
    scl_path = next(scene_dir.glob("GRANULE/*/IMG_DATA/R20m/*_SCL_20m.jp2"))
    with rasterio.open(scl_path) as scl_src:
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], scl_src.transform) #transform转窗口
        win = win.round_offsets().round_lengths()
        scl = scl_src.read(1,window = win)
        scl_transform = scl_src.window_transform(win)
        scl_crs = scl_src.crs   #坐标系

    values, counts = np.unique(scl, return_counts=True)
    logging.info("SCL类别:%s", dict(zip(values, counts)))

    #2. 生成20m的坏像元掩膜
    bad = np.isin(scl, [1,3,8,9,10,11])  #根据地类分类，这几个是有云的
    logging.info("SCL中坏像元比例:%.4f%%", bad.mean()*100)

    #3. 把掩膜重采样到 B04 的 10m 网格
    b04_path = next(scene_dir.glob("GRANULE/*/IMG_DATA/R10m/*_B04*.jp2"))
    scene_id = scene_dir.name.replace(".SAFE","")
    with rasterio.open(b04_path) as src:
        win = from_bounds(aoi["left"], aoi["bottom"], aoi["right"], aoi["top"], src.transform)
        win = win.round_offsets().round_lengths()
        b04 = src.read(1,window=win).astype("float32")
        profile = src.profile.copy()
        profile.update({
        "width": b04.shape[1],                    # 列数
        "height": b04.shape[0],                   # 行数
        "transform": src.window_transform(win),   # 窗口的仿射变换
        })

    #先造一个和 B04 同尺寸的全零数组，作为目标画布，重采样结果会填进来
    mask10 = np.zeros(b04.shape, dtype=np.uint8)

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

    logging.info("10m 网格上被掩掉的像元比例:%.4f%%", mask10.mean()*100)

    # 4. 掩膜并写出:把原来图像有云的地方写成NAN再输出
    b04[mask10==1] = np.nan #一行完成掩膜：mask10 == 1 生成布尔掩膜，把 B04 上对应位置全部赋成 NaN
    profile.update({"driver": "GTiff", "dtype": "float32", "nodata": np.nan, "compress": "lzw"})

    with rasterio.open(out_dir/f"{scene_id}_B04_clean.tif", "w", **profile) as dst:
        dst.write(b04,1)

    logging.info("保存: %s", out_dir/f"{scene_id}_B04_clean.tif")

    logging.info("处理:%s", scene_dir.name)


def main():
    setup_logger() #先运行日志系统
    args = parse_args()  #读取命令行输入的参数
    in_dir = Path(args.input) #将字符串包成Path对象
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    #parents=True 表示中间层缺了也一起建（像 mkdir -p）；exist_ok=True 表示目录已存在时不报错

    scenes = sorted(in_dir.glob("*.SAFE"))  #收成列表并按字母排序（就是按时间顺序排列）
    logging.info("共找到 %d 景", len(scenes))  #有几个.SAFE文件就会有几景
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