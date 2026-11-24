"""D13 任务①: ΔNDVI 直方图 + 阈值试算
方向: 9月 - 4月, 正=变绿 负=变秃
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

def main():
    with rasterio.open("data/processed/NDVI_diff_20250906_minus_20250414.tif") as src:
        diff = src.read(1)  # float32, NaN=无效

    valid = np.isfinite(diff)  #无穷大或者NaN是False  那无效的既可能是无限大也可能是NAN，为啥输出只写NAN➡️保险写法
    #diff是上面的变量，不是函数
    v = diff[valid]  #一维有效值：把二维图里 valid 为 True 的像元挑出来，拍平成一维
    log.info("总像元 %d, 有效 %d, NaN %d", diff.size, v.size, (~valid).sum())
    log.info("mean %+.3f, std %.3f, mean+2σ=%+.3f, mean-2σ=%+.3f",  #+3.f:正数也把 + 打出来。所以 0.179 打印成 +0.179
             v.mean(), v.std(), v.mean()+2*v.std(), v.mean()-2*v.std())

    #分位数
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for q in qs:
        log.info("  P%02d: %+.3f", q, np.percentile(v,q))  #02d：两位数、不够补零（5 → 05）

    #增绿试算表：测试在这些阈值下算作变绿的像元比例分别有多少；大于这个值的算作变绿
    #算的是这个超过这个阈值的所有像元数；下面直方图画的是一个个小区间内的像元数
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        log.info("增绿 > +%.2f: %.2f%%", t, (v>t).mean()*100)

    #变秃试算表
    for t in [-0.05, -0.08, -0.10, -0.15, -0.20]:
        log.info("变秃 < %.2f: %.2f%%", t, (v<t).mean()*100)

    # ---- 直方图 ----
    fig, ax = plt.subplots(figsize=(9,5))
    #画直方图。bins 是横轴分箱边界，每一步 0.02
    #直方图每一根柱子 = 落在这个0.02宽度窄区间里的像元数
    ax.hist(v, bins=np.arange(-0.4, 0.62, 0.02), color="0.65")
    ax.axvline(0.20, color="darkgreen", ls="--", lw=1.5, label="增绿阈值 +0.20")  #画两条竖直参考线当阈值
    ax.axvline(-0.10, color="saddlebrown", ls="--", lw=1.5, label="变秃阈值 -0.10")
    ax.set_xlabel("ΔNDVI(9月 - 4月)")
    ax.set_ylabel("像元数")
    ax.set_title("ΔNDVI 直方图(有效像元 998,480)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/d13_ndvi_hist.png", dpi=150)
    log.info("直方图已写: outputs/d13_ndvi_hist.png")


if __name__ == "__main__":
    main()