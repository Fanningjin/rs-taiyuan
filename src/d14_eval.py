"""D14 任务③: 混淆矩阵 + 分层精度估计
读 data/samples/chg_check_2025.csv (已判读), 输出:
  4 格混淆表 + 变化类 UA/PA + 加权总体精度 OA (含 95% 置信区间) + 变化成因分解
  → outputs/d14_confusion_sample.csv / d14_metrics.csv / d14_confusion.png
判读口径:
  q1_ref: 1=两期影像地表明显不同(图上说的"变化"成立) 0=无明显差异 9=不确定(保守起见按未命中计)
  q2_ref: 1=物候/季节变化  2=身份变化(新增/拆除建筑、水域工程)  3=其他  (只在 q1=1 时填)
"""
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

# 两层总像元数与层权 (d14_sample.py 打印过; 合计 = 有效像元 998,480)
N_CHG, N_SAME = 723258, 275222
W_CHG, W_SAME = N_CHG / (N_CHG + N_SAME), N_SAME / (N_CHG + N_SAME)


def load_csv(path):
    """Excel/WPS 另存 CSV 可能是 GBK, 先试 utf-8 再试 gbk"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit("CSV 编码读不出来: 在 Excel 里另存为 'CSV UTF-8' 再跑")


def main():
    df = load_csv("data/samples/chg_check_2025.csv")

    # ---- 判读完整性检查 ----
    if df["q1_ref"].isna().any():
        raise SystemExit(f"还有 {int(df.q1_ref.isna().sum())} 个点没判 — q1_ref 必须全填 1/0/9")
    lack = df[(df.q1_ref == 1) & df.q2_ref.isna()]
    if len(lack):
        raise SystemExit(f"q1=1 的点里 q2_ref 没填: id = {list(lack['id'])}")

    # 这里的chg和same是根据4、9月分类图判断出来的，hit_c/s是两类里我自己看图判别的
    chg = df[df.stratum == "chg"]
    same = df[df.stratum == "same"]
    hit_c = int((chg.q1_ref == 1).sum())     # 图上说变化 → 判读:确实变了
    hit_s = int((same.q1_ref == 0).sum())    # 图上说未变 → 判读:确实没变
    uns_c = int((chg.q1_ref == 9).sum())
    uns_s = int((same.q1_ref == 9).sum())
    log.info("变化层 %d 点: 命中 %d / 未命中 %d (其中不确定 %d)", len(chg), hit_c, len(chg) - hit_c, uns_c)
    log.info("未变层 %d 点: 命中 %d / 未命中 %d (其中不确定 %d)", len(same), hit_s, len(same) - hit_s, uns_s)

    #各自精度
    p_c = hit_c / len(chg)    # 变化层内部命中比例
    p_s = hit_s / len(same)   # 未变层内部命中比例

    # 变化类 UA (用户精度) 的估计
    ua = hit_c / len(chg)

    # 分层加权总体精度：各自精度乘层权再相加(正确的概率)
    oa = W_CHG * p_c + W_SAME  * p_s

    # 变化类 PA (生产者精度): 真变化里被图上圈进变化层的比例 — 注意未变层漏判也进分母
    pa = (W_CHG * p_c) / (W_CHG * p_c + W_SAME * (1 - p_s))
    # PA = 图说变了且确实变了的 ÷ 全部真的变了的（包括一部分没有识别到变化的） = 88.4%

    # 分层抽样的 95% 置信半径 (两层方差按层权平方加权)  ？
    var = (W_CHG**2 * p_c * (1 - p_c) / (len(chg) - 1)
           + W_SAME**2 * p_s * (1 - p_s) / (len(same) - 1))
    ci = 1.96 * np.sqrt(var)
    log.info("变化类 UA = %.3f | 变化类 PA = %.3f | 加权 OA = %.3f (±%.3f)",
             ua, pa, oa, ci)

    ua2 = hit_c / (len(chg) - uns_c)
    log.info("敏感版(把 9 从分母剔除): UA = %.3f — 只作参考, 报告写保守版", ua2)

    # ---- 变化成因分解 ----
    hits = chg[chg.q1_ref == 1]
    pheno = float((hits.q2_ref == 1).mean())  #季节、物候变化
    ident = float((hits.q2_ref == 2).mean())  #地物类型变化
    other = float((hits.q2_ref == 3).mean())  # 原因不明
    ident_c = int((hits.q2_ref == 2).sum())  #地物类型变化像元数
    ident_abs = N_CHG * ident_c / len(chg)   # 外推: 图上变化像元中判为"身份变化"的占比  #？
    log.info("命中点成因构成: 物候 %.0f%% | 身份变化 %.0f%% | 其他 %.0f%%", pheno * 100, ident * 100, other * 100)
    log.info("外推: 图上 %d 变化像元中, 身份变化约占 %.0f%% (≈%d 像元 ≈ %.1f km2)",
             N_CHG, ident_c / len(chg) * 100, ident_abs, ident_abs * 1e-4)

    # ---- 按转移类型拆分命中率 ----
    t = chg.groupby("trans").agg(n=("id", "size"), hits=("q1_ref", lambda s: int((s == 1).sum())))
    t["hit_rate"] = (t.hits / t.n).round(3)
    t = t.sort_values("n", ascending=False)
    log.info("按转移类型拆分 (n<5 的只当线索看):\n%s", t.to_string())

    # ---- 落盘: 样本混淆表 + 指标表 + 图 ----
    mat = pd.DataFrame({
        "判读:变化": [hit_c, int((same.q1_ref == 1).sum())],
        "判读:未变": [int((chg.q1_ref == 0).sum()), hit_s],
        "判读:不确定": [uns_c, uns_s],
    }, index=["图上:变化", "图上:未变"])
    mat.to_csv("outputs/d14_confusion_sample.csv", encoding="utf-8-sig")

    m = pd.DataFrame({"指标": ["变化类UA", "变化类PA", "加权总体精度", "OA95%半径",
                               "身份变化占命中", "身份变化占图上变化", "不确定点数"],
                      "值": [round(ua, 3), round(pa, 3), round(oa, 3), round(ci, 3),
                             round(ident, 3), round(ident_c / len(chg), 3), uns_c + uns_s]})
    m.to_csv("outputs/d14_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    grid = np.array([[hit_c, int((chg.q1_ref != 1).sum())],
                     [int((same.q1_ref != 0).sum()), hit_s]])
    ax.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if grid[i, j] > grid.max() / 2 else "black")
    ax.set_xticks([0, 1], ["判读:变化", "判读:未变"])
    ax.set_yticks([0, 1], ["图上:变化", "图上:未变"])
    ax.set_title(f"样本混淆表 (n=100)\nUA={ua:.1%}  PA={pa:.1%}  加权OA={oa:.1%}±{ci:.1%}")
    fig.tight_layout()
    fig.savefig("outputs/d14_confusion.png")
    log.info("已落盘: outputs/d14_confusion_sample.csv / d14_metrics.csv / d14_confusion.png")


if __name__ == "__main__":
    main()
