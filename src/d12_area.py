"""D12 任务②: 两期面积统计 + 条形图"""
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PX_KM2 = 1e-4   # 10m x 10m = 100 m2 = 1e-4 km2
CN = ["裸地", "建设", "耕地", "林地", "草地", "水体"]

def main():
    # 读回任务①的转移矩阵, 第一列当行索引
    df = pd.read_csv("outputs/d12_transition_matrix.csv", index_col=0)

    apr = df.sum(axis=1)  # 行合计 = 4月各类像元数
    sep = df.sum(axis=0)  # 列合计 = 9月各类像元数

    # 两边都是 bare->water 的顺序, 按位置对齐; 行序若不放心可先 print 出来看一眼
    log.info("行序: %s", list(df.index))
    log.info("列序: %s", list(df.columns))

    out = pd.DataFrame({
        "前期4月_km2": np.round(apr.values*PX_KM2, 2), #计算面积，单位km2
        "后期9月_km2": np.round(sep.values*PX_KM2, 2),  #也可写成(apr.values*PX_KM2).round(2)
    }, index=CN)
    out["净变化_km2"] = (out["后期9月_km2"] - out["前期4月_km2"]).round(2)

    log.info("\n%s", out.to_string())
    out.to_csv("outputs/d12_area_summary.csv", encoding="utf-8-sig")
    log.info("CSV已写出: outputs/d12_area_summary.csv")

    # ---- 条形图 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(6)
    h = 0.35
    ax.barh(y - h/2, out["前期4月_km2"], height=h, label="4月(前期)", color="#F2C744")  #水平条形图
    ax.barh(y + h/2, out["后期9月_km2"], height=h, label="9月(后期)", color="#5B8C5A")
    ax.set_yticks(y)
    ax.set_yticklabels(CN)
    ax.invert_yaxis()   # 让裸地排最上面
    ax.set_xlabel("面积 (km²)")
    ax.set_title("两期各类面积对比(交集基准, 合计 %.2f km²)" % (apr.sum() * PX_KM2))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/d12_area_by_class.png", dpi=150)

if __name__ == "__main__":
    main()