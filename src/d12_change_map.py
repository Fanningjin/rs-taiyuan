"""D12 任务③: 变化类型图(二值 + 按转入类着色)"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = {
    1: ("裸地", "#D2B48C"), 2: ("建设", "#E03131"), 3: ("耕地", "#F2C744"),
    4: ("林地", "#1F7A1F"), 5: ("草地", "#8FD18F"), 6: ("水体", "#1E6FE8"),
}
STABLE_GRAY = "#DDDDDD"

def load_cls(path):
    with rasterio.open(path) as src:
        return src.read(1)

def main():
    apr = load_cls("data/processed/classified_20250414.tif")
    sep = load_cls("data/processed/classified_20250906.tif")

    valid = (apr>0) & (sep>0)
    changed = valid & (apr != sep)
    log.info("有效 %d, 变化 %d (%.2f%%), 稳定 %d",
             valid.sum(), changed.sum(), changed.sum()/valid.sum()*100, (valid & ~changed).sum())

    # 面板1: 0=稳定, 1=变化, 无效区屏蔽
    binary = np.zeros(apr.shape, dtype="float32")
    binary[changed] = 1
    binary = np.ma.masked_where(~valid, binary)  #(条件，数组)
    #返回一个"带遮罩的数组"（masked array）：数据本身不变，但额外挂了一个布尔mask，满足条件的地方是True
    #imshow 遇到 mask 为 True 的格子直接不画，露出图的白底。否则无效的也会被画成不变的灰色，和0值区一样

    # 面板2: 稳定=0, 变化像元填转入类(9月)编码
    disp = np.zeros(apr.shape, dtype="float32")
    disp[changed] = sep[changed]
    disp = np.ma.masked_where(~valid, disp)
    #只保留变化的地方，不变的是灰色，变的地方标示出来是变成了9月份的哪些地物

    cmap1 = ListedColormap([STABLE_GRAY, "#D62728"])
    cmap2 = ListedColormap([STABLE_GRAY] + [c for _, c in CLASS_COLORS.values()])
    norm2 = BoundaryNorm(np.arange(-0.5, 6.6, 1.0), cmap2.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    axes[0].imshow(binary, cmap=cmap1, norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
                   interpolation="nearest")
    axes[0].set_title("变化/未变化(红=变化 %.2f%%)" % (changed.sum() / valid.sum() * 100))
    axes[1].imshow(disp, cmap=cmap2, norm=norm2, interpolation="nearest")
    axes[1].set_title("变化像元按转入类(9月)着色")

    patches = [mpatches.Patch(color=STABLE_GRAY, label="稳定")]
    patches += [mpatches.Patch(color=c, label=n) for n, c in
                [(v[0], v[1]) for v in CLASS_COLORS.values()]]
    axes[1].legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("outputs/d12_change_map.png", dpi=200, bbox_inches="tight")
    log.info("图已保存: outputs/d12_change_map.png")


if __name__ == "__main__":
    main()