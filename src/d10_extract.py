"""D10 任务③：按坐标给 train/val 样本采 9 波段特征值"""

"""
把 452 个训练样本、194 个验证样本的 (x_utm, y_utm) 坐标，转换成 9 波段特征栈上的行列号，取出每条样本对应的 9 个特征值，
存成 train_feat.csv / val_feat.csv
——这是 RF 的直接口粮。
"""

import logging

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRAIN_CSV = "data/samples/train_20250906.csv"
VAL_CSV = "data/samples/val_20250906.csv"
STACK = "data/processed/features9_20250906.tif"

LABEL_COL = "class"

def extract(csv_path, src, stack, band_names):
    df = pd.read_csv(csv_path)
    log.info("%s: %d 个样本, 类别分布 %s", csv_path, len(df),
             df[LABEL_COL].value_counts().to_dict())
    #为啥有时候用logging.info有时候用这个，这个log.info是干嘛的
    #具名 logger 打出的记录带 logger 名字（配置里加 %(name)s 就能看到），以后多文件项目里能看出是谁打的；而且可以单独给某个模块调级别

    xs = df["x_utm"].to_numpy(dtype="float64") #为啥是64
    ys = df["y_utm"].to_numpy(dtype="float64")
    #float64 精细到小数点后十几位,用 float64 零成本地消掉一切精度疑虑。另外 pandas 从 CSV 读数字默认就是 float64

    #正变换x=a*col+c, y=e*row+f 的逆（从utm坐标计算像素坐标）
    cols = np.floor((xs - src.transform.c) / src.transform.a).astype(int)
    rows = np.floor((ys - src.transform.f) / src.transform.e).astype(int)

    #in_bounds 是长度 452 的布尔数组（每个样本点一个 True/False）
    in_bounds = ((cols>=0) & (cols<src.width) & (rows>=0) & (rows<src.height))
    #一幅 1000 列的图,合法列号是 0–999,所以是严格小于
    
    # ~ 是按位取反，~in_bounds 就是"出界的那些点"，sum就是加起来这些越界的1
    log.info("越界样本：%d 个", (~in_bounds).sum())
    #.all() 问"是不是全部为 True"，not一翻，逻辑就是"只要有一个出界，就抛错停住"
    if not in_bounds.all():
        raise RuntimeError("有样本点落在影像范围外,先停下来处理")
    
    #stack[:, rows, cols] 的机制是"竖着捅吸管"：9 个波段各吸一根，结果是 (9, n)——先按波段分组。
    # .T 把它翻过来变成 (n, 9)——先按样本分组，每行一条样本、每列一个特征。RF 要的就是这个朝向
    #对每一对 (rows[i], cols[i])，把所有波段的那个像元一起取出来。结果不再是三维的，而是降成二维 (9, n)：
    # 9 行波段 × n 列样本——就像竖着捅了 n 根吸管，每根吸出一段长 9 的向量，9 根波段各自排一行
    #这里只是对从stack中花式索引取出来的二维数组进行转置，stack没动
    values = stack[:, rows, cols].T   #(n,9)  .T是转置
    log.info("采样矩阵形状: %s", values.shape)

    bad = np.isnan(values).any(axis=1)  #为啥是1：在每个样本内部横着扫 9个特征，只要有一个 NaN，这一行就判 True
    log.info("特征含NaN的样本: %d 个", bad.sum())
    if bad.any():
        log.info("其中类别分布: %s", df.loc[bad, LABEL_COL].value_counts().to_dict())
        #df.loc[bad, LABEL_COL] 是"行用布尔数组 bad 筛，列只取 class"，得到一个只含 NaN 样本类别的 Series；
        # .value_counts() 数每类各几个；.to_dict() 转成 {"water": 3, ...} 好打印。
        # 如果真出现这行输出，说明某些地类的样本正好压在掩膜像元上，要重点盯
        values = values[~bad]
        df = df.loc[~bad]

    X = pd.DataFrame(values, columns=band_names)
    X[LABEL_COL] = df[LABEL_COL].values  #LABEL_COL是"class"
    return X


def main():
    with rasterio.open(STACK) as src:
        stack = src.read()  #(9,1000,1000)
        band_names = list(src.descriptions)
        # 调用函数
        train = extract(TRAIN_CSV, src, stack, band_names) #训练集
        val = extract(VAL_CSV, src, stack, band_names) #验证集

    train.to_csv("data/samples/train_feat.csv", index=False)
    val.to_csv("data/samples/val_feat.csv", index=False)
    log.info("已写出 train_feat.csv(%d 行) / val_feat.csv(%d 行)", len(train), len(val))


if __name__ == "__main__":
    main()