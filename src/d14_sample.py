"""D14 任务①: 变化精度抽检——分层随机抽样 + 判读图册
抽检对象: 分类叠加变化图(D12) —— 两期分类逐像元对比, apr != sep 即"图上说变化"
分层: 变化层抽 60 点 / 未变层抽 40 点 (固定种子, 可复现)
产出: data/samples/chg_check_2025.csv + outputs/d14_sheets/sheet_01~10.png
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

N_CHG, N_SAME = 60, 40   # 分层配额: 变化层多分(主指标 UA 靠它), 未变层保底查漏检
SEED = 20261125          # 固定种子 → 任何机器重跑抽到的都是同一批点
CROP = 60                # 判读窗 60×60 像元 = 600×600 m
Z = 6                    # 最近邻放大 6 倍 → 每个小格 360 px
NAMES = {1: "裸地", 2: "建设", 3: "耕地", 4: "林地", 5: "草地", 6: "水体"}

def draw(mask, n, rng):
    """在 mask 为 True 的像元里无放回随机抽 n 个, 返回 [(row, col), ...]"""
    low = np.flatnonzero(mask.ravel())  #mask里所有 True 的一维下标  flatnonzero获得非0元素索引
    mid = rng.choice(low, size=n, replace=False)  #无放回抽 n 个
    rows_, cols_= np.unravel_index(mid, mask.shape)  #一维下标 → 行、列
    return list(zip(rows_, cols_))  # [(r1,c1), (r2,c2), ...]


def main():
    with rasterio.open("data/processed/classified_20250414.tif") as src:
        apr = src.read(1)
        transform = src.transform
    with rasterio.open("data/processed/classified_20250906.tif") as src:
        sep = src.read(1)

        #读进来4、9月分类结果图再判断是否变化
        valid = (apr>0) & (sep>0)  #两期都有效
        chg = valid & (apr != sep)  #变化图：分类图说“变了”
        same = valid & (apr == sep)  #未变层
        log.info("有效 %d | 变化层 %d (%.2f%%) | 未变层 %d (%.2f%%)",
                 valid.sum(),chg.sum(),chg.sum()/valid.sum()*100,
                 same.sum(),same.sum()/valid.sum()*100)

        rng = np.random.default_rng(SEED)
        rows = []
        #变化
        #draw返回无放回抽取点的（行，列）;N_CHG是抽取数量
        for k, (r,c) in enumerate(draw(chg, N_CHG, rng), 1):
            x, y = transform * (c + 0.5, r+ 0.5)   #像元中心 → UTM
            rows.append({"id": k, "stratum": "chg",  #stratum？
                         "row": int(r), "col" :int(c),
                         "x_utm": round(x, 2), "y_utm": round(y, 2),
                         "apr": int(apr[r,c]), "sep": int(sep[r,c]),
                         "trans": f"{NAMES[int(apr[r,c])]}->{NAMES[int(sep[r,c])]}",
                         "q1_ref":np.nan, "q2_ref":np.nan  #设置的参考真值，等着后面自己看的时候填
            })
        #未变化
        for k, (r, c) in enumerate(draw(same, N_SAME, rng), N_CHG + 1):
            x, y = transform * (c + 0.5, r + 0.5)
            rows.append({"id": k, "stratum": "same",
                         "row": int(r), "col": int(c),
                         "x_utm": round(x, 2), "y_utm": round(y, 2),
                         "apr": int(apr[r, c]), "sep": int(sep[r, c]),
                         "trans": f"{NAMES[int(apr[r,c])]}->{NAMES[int(sep[r,c])]}",
                         "q1_ref": np.nan, "q2_ref": np.nan})

        df = pd.DataFrame(rows)
        df.to_csv("data/samples/chg_check_2025.csv", index=False, encoding="utf-8-sig")
        log.info("CSV 已写出: %d 点 (变化 60 + 未变 40) → data/samples/chg_check_2025.csv", len(df))

        # ---- 判读图册: 每张 10 点, 左 4 月 / 右 9 月, 红十字 = 该点 ----
        # 图上只有编号和行列号, 不写预测类型 → 盲判, 防止"看着答案判卷"
        out = Path("outputs/d14_sheets")
        out.mkdir(parents=True, exist_ok=True)
        with rasterio.open("data/processed/S2_20250414_RGB.tif") as s4, \
             rasterio.open("data/processed/S2_20250906_RGB.tif") as s9:
            #每10个点循环一次,g=0,10,...,90
            for g in range(0, len(df), 10):
                #按位置切出第 g~g+9 行，得到 10 行的小表。iloc = integer location（按行列号切）
                sub = df.iloc[g:g + 10]  #每 10 个点一页，共 10 页
                #10x2的大画布：每张图里都有10行，一行2个小图
                fig, axes = plt.subplots(len(sub), 2, figsize=(7, 2.3*len(sub)), dpi=110)
                #表格的第 1 行配第 1 对画板，对一行行进行循环
                for (_, row), ax_pair in zip(sub.iterrows(), axes): #iterrows：把 DataFrame 一行一行吐出来，每行是一个 Series
                    r,c = int(row["row"]), int(row["col"])
                    #每张小图都是以抽检点为中心周围的一小块区域，所以要裁剪一下
                    #min是保证不越界，代价是点不在中心了；套max夹逼，点靠近图像边缘时窗口往里收，确保60x60整个落在图内
                    #min管上限，max管下限
                    r0 = max(0, min(apr.shape[0]- CROP, r - CROP // 2))
                    c0 = max(0, min(apr.shape[1]- CROP, c - CROP // 2))
                    win = Window(c0, r0, CROP, CROP)  #Window 顺序是（列起点, 行起点, 宽, 高） CROP=60
                    # 小窗口左一个右一个，分别画
                    for ax, src, season in ((ax_pair[0], s4, "4月"), (ax_pair[1], s9, "9月")):
                        img = np.transpose(src.read(window=win), (1, 2, 0))   # (3,H,W)→(H,W,3)
                        #这里是因为裁出来窗口只有60x60，画画的画板却更大
                        big = np.repeat(np.repeat(img, Z, axis=0), Z, axis=1) # 最近邻放大
                        ax.imshow(big)
                        px, py = (c - c0) * Z + Z / 2, (r - r0) * Z + Z / 2  # 判读像元放大后的中心
                        ax.add_patch(Rectangle((px - Z / 2 - 3, py - Z / 2 - 3), Z + 6, Z + 6,
                                               fill=False, edgecolor="red", lw=1.5))
                        ax.set_title(season, fontsize=8)
                        ax.axis("off")
                    ax_pair[0].text(0.02, 0.02, f"#{int(row['id']):02d} ({r},{c})",
                                transform=ax_pair[0].transAxes, fontsize=8,
                                va="bottom", ha="left",
                                bbox=dict(facecolor="white", alpha=0.75, pad=1.5))
                fig.tight_layout()
                fig.savefig(out / f"sheet_{(g // 10) + 1:02d}.png", bbox_inches="tight")
                plt.close(fig)
        log.info("判读图册已写出: %d 张 → outputs/d14_sheets/", (len(df) + 9) // 10)


if __name__ == "__main__":
    main()