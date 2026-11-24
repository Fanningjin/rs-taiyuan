"""D13 任务④: NDVI 变化掩膜 vs 分类叠加变化 交叉 + NDBI 补盲
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

T_UP, T_DN = 0.20, -0.10
MIN_AREA = 20


def load1(path):
    with rasterio.open(path) as src:
        return src.read(1)

def build_clean_mask(diff, valid):
    """复现任务③的清理流程, 得到交付版 NDVI 变化掩膜"""
    #因为③没输出tif，只是成了图
    chg = valid & ((diff > T_UP) | (diff < T_DN))
    st = np.ones((3,3), dtype=bool)
    closed = ndimage.binary_closing(ndimage.binary_opening(chg,structure=st), structure=st)  #bool，变化的都是1
    lab, _ = ndimage.label(closed, structure=st)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (sizes >= MIN_AREA)[lab] & closed  #不足20的斑块抹掉，不算变化

def main():
    diff = load1("data/processed/NDVI_diff_20250906_minus_20250414.tif")
    valid = np.isfinite(diff)
    ndvi_chg = build_clean_mask(diff, valid)  #和d13_clean.py一样最终生成的图，但是是bool，1是变化了的

    apr = load1("data/processed/classified_20250414.tif")   # uint8 1~6, 0=无效
    sep = load1("data/processed/classified_20250906.tif")
    cv = (apr > 0) & (sep > 0) & valid  # 两期都有效
    same = cv & (apr == sep)  # 分类未变
    clsg_chg = cv & ~same     # 分类变化

    # ---- 2x2 象限表 ----各类数量
    q = {
        "分类变 & NDVI变":  (clsg_chg & ndvi_chg).sum(),
        "分类变 & NDVI静默": (clsg_chg & ~ndvi_chg).sum(),
        "分类未变 & NDVI变": (same & ndvi_chg).sum(),
        "都静默":            (same & ~ndvi_chg).sum(),
    }
    for k, v in q.items():
        log.info("%-12s %8d (%.1f%%)", k, v, v / valid.sum() * 100)
    log.info("NDVI掩膜中分类也变 %.1f%% | 分类变化中NDVI也变 %.1f%%",
             q["分类变 & NDVI变"] / ndvi_chg.sum() * 100, 
             q["分类变 & NDVI变"] / clsg_chg.sum() * 100)

    df = pd.DataFrame({"组合": list(q.keys()),
                       "像元数": list(q.values())})
    df.to_csv("outputs/d13_cross_table.csv", index=False, encoding="utf-8-sig")

    # ---- NDBI 补盲 ----
    # 读NDBI：测“建设度”
    with rasterio.open("data/processed/features9_20250414.tif") as s:
        ndbi_apr = s.read(8)     # 第8波段 = NDBI
    with rasterio.open("data/processed/features9_20250906.tif") as s:
        ndbi_sep = s.read(8)
    ndbi_d = ndbi_sep - ndbi_apr
    log.info("NDBI差 全域 mean %+.3f", np.nanmean(ndbi_d))

    # 9月转建设 = 两期都有效 & 9月是建设(编码2) & 4月不是建设
    to_built = cv & (sep==2) & (apr!=2)
    tb = to_built & valid
    log.info("9月转建设 %d 像元 | ΔNDVI mean %+.3f | ΔNDBI mean %+.3f | 在NDVI掩膜内 %.0f%%",
             to_built.sum(), np.nanmean(diff[tb]), np.nanmean(ndbi_d[tb]),
             (to_built & ndvi_chg).sum() / to_built.sum() * 100)

    # NDBI 阈值掩膜 ndbi_up = 有效 & (ndbi_d > 0.05); 算圈住转建设的比例
    ndbi_up = (ndbi_d>0.05) & valid  # &的优先级更高，所以要加个括号
    log.info("NDBI>+0.05: 掩膜 %d 像元, 圈住转建设 %d / %d (%.0f%%)",
             ndbi_up.sum(), (ndbi_up & to_built).sum(), to_built.sum(),
             (ndbi_up & to_built).sum() / to_built.sum() * 100)


if __name__ == "__main__":
    main()