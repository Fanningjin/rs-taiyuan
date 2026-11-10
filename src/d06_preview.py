#出图

import numpy as np
import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/processed/S2_20250414_B04_clean.tif") as src:
    b04 = src.read(1).astype("float32")
    ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)  # imshow要的地图范围

with rasterio.open("data/processed/S2_20250414_NDVI.tif") as src:
    ndvi = src.read(1)

# 2%-98%分位拉伸，和D3一样；NAN用nanpercentile才不会污染分位
p2, p98 = np.nanpercentile(b04, [2,98])
b04_st = np.clip((b04-p2)/(p98-p2), 0, 1)
#clip 是"钳制"：比 0 小的全变成 0，比 1 大的全变成 1，中间的保持原样
#p2 以下的值会算出负数、p98 以上的会超过 1。而 imshow 显示浮点数据时默认用"数组自己的最小值到最大值"做映射—
# 如果不钳制，那极少数 0 值或饱和像元会重新撑大显示范围，2%–98% 拉伸就白做了，对比度又被打回原样。
# 钳到 [0, 1] 等于强行告诉 imshow："显示范围就定在 0–1，超出的部分按两端处理。"

fig, axes = plt.subplots(1, 2, figsize=(12,5))
#axes 是一个装着两个子图对象的数组（1 行 2 列）。axes[0] 是左边那块画板，axes[1] 是右边那块
axes[0].imshow(b04_st, cmap="gray", extent=ext, origin="upper")
axes[0].set_title("B04(Red)2025-04-14")

#imshow返回图像对象
im = axes[1].imshow(np.ma.masked_invalid(ndvi), cmap="RdYlGn",
                    vmin=-0.3, vmax=0.6, extent=ext, origin="upper")
#第一个参数：把 NaN 标记成"屏蔽值"（masked）。imshow 遇到被屏蔽的像元就不画、留透明——终端里看不到 NaN，就变成你想要的"云区留白"
axes[1].set_title("NDVI 2025-04-14")
plt.colorbar(im, ax=axes[1], shrink=0.8, label="NDVI")

plt.tight_layout()
plt.savefig("outputs/d06_ndvi.png", dpi=150)
