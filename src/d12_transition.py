"""D12 任务①: 两期分类叠加 -> 6x6 转移矩阵
行 = 4月(前期) 列 = 9月(后期), 值 = 像元数
"""
import logging

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

NAMES = ["bare", "builtup", "cropland", "forest", "grassland", "water"]  # 编码 1~6

def load_cls(path):
    with rasterio.open(path) as src:
        return src.read(1)  #uint=8, 0=无效


def main():
    #导入分类后的文件：像素格子是地物编码0-6
    apr = load_cls("data/processed/classified_20250414.tif")   # 前期
    sep = load_cls("data/processed/classified_20250906.tif")   # 后期

    valid = (apr>0) & (sep>0)  # 两期都有效才进矩阵
    log.info("总像元 %d, 两期都有效 %d, 剔除 %d", apr.size, valid.sum(), (~valid).sum())

    # 布尔掩膜选取 → 一维 ~998480 个 → 再转整型
    a = apr[valid].astype(np.int32)  #1-6
    s = sep[valid].astype(np.int32)
    #uint8只能装 0~255，换成宽整数避免溢出

    #把 (行, 列)里代表的编码值 压成 0~35 的单整数:  code = (a - 1) * 6 + (s - 1)
    code = (a - 1) * 6 + (s - 1)
    #"把二维组合压成一维类别"——就像把"第3行第5列"压成"六进制数 35"，然后用最基础的 bincount 画一维直方图
    #本来有6x6种组合，但是是两个数字，比如(3,2)，无法用bincount计数，所以把它转成一个数字，就像0-9是10进制，这里0-5就是6进制
    #3x6 + 2就计算出了对应数字
    #减1是因为bincount编号从0开始

    # 一次数完全部 36 种组合, 再摊成 6x6
    counts = np.bincount(code, minlength=36).reshape(6,6)
    #bincount统计非负整数数组中每个值的出现次数

    """
    自己的思路:对6x6的结果矩阵遍历,然后计算每一个格子代表组合的数目
    counts = np.zeros((6,6), dtype=np.int64)
    for i in range(6):
        for j in range (6):
            counts[i,j] = ((a == i+1) & (s == j+1)).sum()  #注意1是0-5,所以这里要+1
    """

    # ---- 屏幕打印: 像元数矩阵 ----
    log.info("行=4月, 列=9月, 单位:像元")
    #f-string 只能一次性拼你手写的若干段；join 能拼运行时才知道有几段的一串
    #前面的""是分隔符。join是字符串的方法，语法是 分隔符.join(字符串列表)  （也可以是","）在相邻两个元素之间插入分隔符，开头结尾都不插
    log.info("4月\\9月     "+"".join(f"{n:>12}" for n in NAMES)+ "     行合计")  #> 右对齐、宽度 12；join连接字符串
    for i in range(6):
        row = "".join(f"{counts[i,j]:>12d}" for j in range(6))
        log.info("%-10s%s%12d", NAMES[i], row, counts[i].sum())

    n = int(valid.sum())  #有效像元
    unchanged = int(counts.diagonal().sum())  #取矩阵的主对角线，也就是没变化的
    log.info("未变化 %d(%.2f%%), 变化 %d(%.2f%%)",
             unchanged, unchanged/n*100, n-unchanged, (n-unchanged)/n*100)
    
    # 行归一化: 每行 = 前期某类的去向占比
    pct = counts / counts.sum(axis=1, keepdims=True)*100
    # counts.sum(axis=1) 的意思是"把第 1 号轴（列方向）压掉"，剩下的就是每行一个和，形状从 (6,6) 变 (6,)
    # keepdims=True 要求压掉后保留这一维、长度记 1 （6,1）

    # ---- 写 CSV ----
    idx = [f"4月_{n}" for n in NAMES]
    col = [f"9月_{n}" for n in NAMES]
    df = pd.DataFrame(counts, index=idx, columns=col)  #index 给每行贴名字
    # DataFrame 本身的数值不变，只是给行列戴上"名牌"。这样 to_csv 写出去第一行第一列就是可读的标签，而不是光秃秃 36 个数
    df.to_csv("outputs/d12_transition_matrix.csv", encoding="utf-8-sig")
    df_pct = pd.DataFrame(pct.round(2), index=idx, columns=col)
    df_pct.to_csv("outputs/d12_transition_matrix_pct.csv", encoding="utf-8-sig")
    log.info("CSV已写出: outputs/d12_transition_matrix.csv / _pct.csv")


if __name__ == "__main__":
    main()