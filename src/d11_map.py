"""D11 3:两期分类双联图(4月/9月 RGB+分类 2x2)"""
import logging
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

with open("aoi.json", encoding="utf-8") as f:
    aoi = json.load(f)

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


def stretch(a):
    """2%~98% 分位拉伸，只用于显示，不参与计算"""
    p2, p98 = np.nanpercentile(a, [2,98])
    out = np.clip((a-p2) / (p98-p2) , 0, 1) * 255
    return np.nan_to_num(out, nan=0).astype("uint8")  #该函数将数组中的NAN替换成0


def read_safe_10m(band):
    """从 4 月 SAFE 按 AOI 现读 10m 波段（与 d08/d11_stack 同一套取窗）"""
    scene = Path("data/raw/S2_20250414.SAFE")
    p = next(scene.glob(f"GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2"))
    with rasterio.open(p) as src:
        win = from_bounds(aoi["left"], aoi["bottom"],
                          aoi["right"], aoi["top"], src.transform)
        win = win.round_offsets().round_lengths()
        return src.read(1, window=win).astype("float32")

#合成4月的RGB图
def render_rgb_apr():
    b02 = read_safe_10m("B02")
    b03 = read_safe_10m("B03")
    with rasterio.open("data/processed/S2_20250414_B04_clean.tif") as s:
        b04 = s.read(1)
        profile = s.profile.copy() # 注意：这份 profile 是 float32/NaN 的

    rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)])

    # 存成和 9 月一致的显示产品: uint8 三波段
    #顺便保存一下4月RGB图
    #update改掉profile里上面说过的这两项
    profile.update({"count": 3, "dtype": "uint8", "nodata": None, "compress": "lzw"})
    out = "data/processed/S2_20250414_RGB.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(rgb)
        for i, name in enumerate(["B04_red", "B03_green", "B02_blue"], start=1):
            dst.set_band_description(i, name)
    log.info("RGB 已保存: %s", out)

    return rgb.transpose(1, 2, 0) #imshow 认的是"通道在最后"

# 读取分类结果图
def load_cls(path):
    with rasterio.open(path) as src:
        c = src.read(1)
    return np.where(c==0, np.nan, c)  #0=无效，显示成白  np.where(条件, 真值, 假值)


def main():
    # 合成4月的RGB图，读取9月的RGB图
    rgb_apr = render_rgb_apr()
    with rasterio.open("data/processed/S2_20250906_RGB.tif") as src:
        rgb_sep = src.read().transpose(1, 2, 0)

    # 读取分类结果图
    cls_apr = load_cls("data/processed/classified_20250414.tif")
    cls_sep = load_cls("data/processed/classified_20250906.tif")

    colors = [CLASS_COLORS[i][1] for i in range(1, 7)]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(0.5, 7.5, 1.0), cmap.N)

    # 左是影像RGB图，右是分类结果图
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0,0].imshow(rgb_apr)
    axes[0,0].set_title("4-14 RGB 底图")
    axes[0,1].imshow(cls_apr, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0,1].set_title("4-14 分类(9月模型迁移)")
    axes[1,0].imshow(rgb_sep)
    axes[1,0].set_title("9-06 RGB 底图")
    axes[1,1].imshow(cls_sep, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1,1].set_title("9-06 分类(本季训练)")

    #先造 6 个色块（Patch），再一次性交给图例函数
    patches = [mpatches.Patch(color=c, label=f"{i} {n}")
               for i, (n, c) in CLASS_COLORS.items()]
    axes[0,1].legend(handles=patches, loc="lower right",
                   fontsize=8, framealpha=0.9)
    axes[1,1].legend(handles=patches, loc="lower right",
                       fontsize=8, framealpha=0.9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig("outputs/d11_map_two_seasons.png", dpi=200, bbox_inches="tight")
    log.info("图已保存:outputs/d11_map_two_seasons.png")


if __name__ == "__main__":
    main()