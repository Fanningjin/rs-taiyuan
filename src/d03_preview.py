#将d03_crop.py裁剪出来的成图看一下

import numpy as np
import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/processed/d03_b04_crop.tif") as src:
    b04 = src.read(1).astype(float)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)  #什么？

lo, hi = np.percentile(b04, [2,98])
b04_stretch = np.clip((b04 - lo)/ (hi - lo), 0, 1)

fig, ax = plt.subplots(figsize = (6,6))
ax.imshow(b04_stretch, cmap="gray", extent=extent, origin="upper")
ax.set_xlabel("Easting(m)")
ax.set_ylabel("Morthing(m)")
ax.set_title("S2 B04 2025-04-14 | 2-98% stretch")
plt.tight_layout()
plt.savefig("outputs/d03_b04_preview.png", dpi=150)
print("saved outputs/d03_b04_preview.png")