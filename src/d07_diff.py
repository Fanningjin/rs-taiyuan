"""
计算NDVI差值并出图
"""

import logging
import numpy as np
import rasterio
import numpy as np
import rasterio
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,  #最低输出级别，日志级别从低到高是 DEBUG < INFO < WARNING < ERROR
    format="%(asctime)s | %(levelname)s | %(message)s",  #3个占位符
     datefmt="%H:%M:%S")


with rasterio.open("data/processed/S2_20250414_NDVI.tif") as src:
    ndvi_apr = src.read(1)
    profile = src.profile.copy()
    ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
with rasterio.open("data/processed/S2_20250906_NDVI.tif") as src:
    ndvi_sep = src.read(1)

diff = ndvi_sep - ndvi_apr #正=变绿，负=变秃/被占

logging.info("差值: min=%.3f, max=%.3f, mean=%.3f, 增绿占比=%.1f%%",
    np.nanmin(diff), np.nanmax(diff), np.nanmean(diff),
    np.nanmean(diff>0.1)*100)

with rasterio.open("data/processed/NDVI_diff_20250906_minus_20250414.tif", "w", **profile) as dst:
    dst.write(diff,1)

#画对比图

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# 左：0414 ndvi_apr
axes[0].set_title("NDVI 2025-04-14")
im1 = axes[0].imshow(np.ma.masked_invalid(ndvi_apr), cmap="RdYlGn",
                    vmin=-0.3, vmax=0.6, extent=ext, origin="upper")

#中：0906 ndvi_sep
axes[1].set_title("NDVI 2025-09-06")
im2 = axes[1].imshow(np.ma.masked_invalid(ndvi_sep), cmap="RdYlGn",
                    vmin=-0.3, vmax=0.6, extent=ext, origin="upper")
#第一个参数：把 NaN 标记成"屏蔽值"（masked）。imshow 遇到被屏蔽的像元就不画、留透明——终端里看不到 NaN，就变成你想要的"云区留白"

plt.colorbar(im2, ax=axes[1], shrink=0.8, label="NDVI")

#右：差值diff
axes[2].set_title("ΔNDVI (Sep - Apr)")
im3 = axes[2].imshow(np.ma.masked_invalid(diff), cmap="RdYlGn",
                    vmin=-0.5, vmax=0.5, extent=ext, origin="upper")

plt.colorbar(im3, ax=axes[2], shrink=0.8, label="ΔNDVI")

plt.tight_layout()
plt.savefig("outputs/d07_ndvi_2periods.png", dpi=150)
