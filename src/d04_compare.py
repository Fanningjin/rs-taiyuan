import numpy as np
import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/processed/S2_20250414_B04_aoi.tif") as src:
    raw = src.read(1).astype(float)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

with rasterio.open("data/processed/S2_20250414_B04_clean.tif") as src:
    clean = src.read(1)

lo, hi = np.percentile(raw, [2, 98])
stretch = lambda a: np.clip((a - lo) / (hi - lo), 0, 1)

cmap = plt.get_cmap("gray").copy()
cmap.set_bad("red")   # 被掩掉的像元显示为红色

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(stretch(raw), cmap="gray", extent=extent, origin="upper")
axes[0].set_title("Original B04")
axes[1].imshow(stretch(clean), cmap=cmap, extent=extent, origin="upper")
axes[1].set_title("Cloud-masked (red = invalid)")

for ax in axes:
    ax.set_xlabel("Easting (m)")
axes[0].set_ylabel("Northing (m)")

plt.tight_layout()
plt.savefig("outputs/d04_mask_compare.png", dpi=150)
print("saved outputs/d04_mask_compare.png")