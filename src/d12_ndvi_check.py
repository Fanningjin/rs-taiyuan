"""D12 任务④: NDVI 差值交叉验证主要转移组"""
import logging

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CN = {1: "裸地", 2: "建设", 3: "耕地", 4: "林地", 5: "草地", 6: "水体"}

# 组: (4月类, 9月类)
GROUPS = [
    (2, 2), (1, 1), (6, 6),          # 基准组(稳定, 地表同质)
    (2, 5), (2, 3), (1, 5), (5, 3),  # 季节组(矩阵里的最大流)
    (1, 2),                          # 谱相似组(两种无植被表面)
]

def load(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32")


def main():
    apr = load("data/processed/classified_20250414.tif")
    sep = load("data/processed/classified_20250906.tif")
    diff = load("data/processed/NDVI_diff_20250906_minus_20250414.tif")

    valid = (apr>0) & (sep>0) & np.isfinite(diff)  #?
    log.info("有效 %d, 全域ΔNDVI均值 %+.3f", valid.sum(), diff[valid].mean())  #+?

    rows = []
    for f, t in GROUPS:
        m = valid & (apr == f) & (sep == t)
        n = int(m.sum()) #计数
        if n == 0:
            continue
        d = diff[m]  #提取对应位置的NDVI差值
        rows.append({
            "转移":f"{CN[f]}->{CN[t]}",
            "n": n,
            "mean": round(float(d.mean()), 3),
            "median":round(float(np.median(d)), 3),  #？
            "p10":round(float(np.percentile(d,10)), 3),
            "p90":round(float(np.percentile(d,90)), 3),
        })

    df = pd.DataFrame(rows)
    log.info("\n%s", df.to_string(index=False))  #\n%s什么意思？
    df.to_csv("outputs/d12_transition_ndvi.csv", index=False, encoding="utf-8-sig")
    log.info("CSV已写出:outputs/d12_transition_ndvi.csv")


if __name__ == "__main__":
    main()