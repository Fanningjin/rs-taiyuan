"""D13 任务②: 双向阈值掩膜 -> 初版变化图
增绿: Δ > +0.20 | 变秃: Δ < -0.10 | 稳定: 其余 | 白: NaN
"""
import logging

import numpy as np
import rasterio
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

T_UP, T_DN = 0.20, -0.10  #刚刚计算和画图后判定出来的

def main():
    with rasterio.open("data/processed/NDVI_diff_20250906_minus_20250414.tif") as src:
        diff = src.read(1)
    valid = np.isfinite(diff)

    #掩膜:结果都是布尔，(1000,1000)
    up = (diff>T_UP) & valid
    dn = (diff<T_DN) & valid

    log.info("增绿 %d(%.2f%%), 变秃 %d(%.2f%%), 合计 %d(%.2f%%)",
             up.sum(),up.sum()/valid.sum()*100,
             dn.sum(),dn.sum()/valid.sum()*100,
             (up|dn).sum(),(up|dn).sum()/valid.sum()*100)  #并集
            #布尔数组的 | 是逐像元 OR，增绿或变秃也就是就是有显著变化的像元

    #img是三维矩阵(1000, 1000, 3)
    img = np.zeros((*diff.shape, 3), dtype=np.float32)
    #diff.shape 是元组 (1000, 1000)；前面的 * 把元组"拆开"成两个参数
    img[:] = 0.92  #浅灰：无显著变化  [:]所有维度的所有元素
    img[up] = [0.13, 0.55, 0.13]
    img[dn] = [0.55, 0.27, 0.07]
    img[~valid] = [1.0, 1.0, 1.0]
    #img[up] 里up是 (1000,1000)的布尔图，它会挑出所有True的像元，形状变成(n, 3)，右侧给一个3元素颜色向量就逐行广播填进去
    #img[up].shape是(445801, 3)，只有增绿的这些像元会被染色，up和dn的互不重合，各自占一部分

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img)
    ax.set_title("ΔNDVI 阈值变化图 v0(增绿>+0.20 / 变秃<-0.10)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/d13_change_v0.png", dpi=150)
    log.info("图已写: outputs/d13_change_v0.png")

if __name__ == "__main__":
    main()