"""D10 任务⑦：分类图可视化（RGB vs 分类 双联图）"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  #rcParams是全局配置字典
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = {
    1: ("bare",      "#D2B48C"),
    2: ("builtup",   "#E03131"),
    3: ("cropland",  "#F2C744"),
    4: ("forest",    "#1F7A1F"),
    5: ("grassland", "#8FD18F"),
    6: ("water",     "#1E6FE8"),
}

CLS = "data/processed/classified_20250906.tif"
RGB = "data/processed/S2_20250906_RGB.tif"
OUT = "outputs/d10_classified_map.png"


def main():
    with rasterio.open(CLS) as src:
        cls = src.read(1)
    cls_show = np.where(cls == 0, np.nan, cls) #对数组里每一个元素问一遍条件，成立就取 A、不成立就取B

    with rasterio.open(RGB) as src:
        rgb = src.read().transpose(1, 2, 0) #换维度顺序，括号里是原来的维度，相当于原来第一维换到了最后
        #让数组从"波段优先"变成 matplotlib 要的"通道最后"

    colors = [CLASS_COLORS[i][1] for i in range(1, 7)]  #CLASS_COLORS[i][1]是两层连取，一层层取
    cmap = ListedColormap(colors)
    cmap.set_bad("white")  #设置nan的颜色
    #给"分类图"配一个离散的分档器，每个整数就正好落在某个区间正中央，7 个边界围出 6 个区间，正好一类一格
    norm = BoundaryNorm(np.arange(0.5, 7.5, 1.0), cmap.N)

    # 左是影像RGB图，右是分类结果图
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB 底图 (2025-09-06)")
    axes[1].imshow(cls_show, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("随机森林分类图")

    #先造 6 个色块（Patch），再一次性交给图例函数
    patches = [mpatches.Patch(color=c, label=f"{i} {n}")
               for i, (n, c) in CLASS_COLORS.items()]
    axes[1].legend(handles=patches, loc="lower right",
                   fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    log.info("图已保存：%s", OUT)


if __name__ == "__main__":
    main()