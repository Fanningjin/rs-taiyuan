"""
D9 第三步:QGIS 核订并导出 samples_cleaned_20250906.csv 后，
按类别做 70/30 分层抽样 → 训练集 / 验证集 CSV

分层抽样 = 每一类内部各自打乱、各自按 70/30 切。
这样验证集里六类的比例和总样本一致，精度评价才不会被大类（耕地）主导。
"""

import glob
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

# QGIS 导出的核订文件（主样本 + 补点样本，全部读进来拼在一起）
paths = sorted(glob.glob("data/samples/samples_cleaned*.csv"))
if not paths:
    raise SystemExit("data/samples 下没找到 samples_cleaned*.csv, 先完成 QGIS 核订并导出")
logging.info("读入核订文件：%s", paths)
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df = df.drop(columns=[c for c in ["sample_id"] if c in df.columns], errors="ignore")

need = {"class", "x_utm", "y_utm"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"缺少列：{missing}。检查 QGIS 导出时 x_utm / y_utm 字段还在不在，"
                     f"现在读到的列是：{list(df.columns)}")

logging.info("读入 %d 个已核订样本", len(df))
logging.info("各类数量：\n%s", df["class"].value_counts().to_string())

train_parts, val_parts = [], []
for cls, grp in df.groupby("class"):
    n = len(grp)
    if n < 100:
        logging.warning("%s 只有 %d 个点，低于计划的 100 个下限", cls, n)
    grp = grp.sample(frac=1.0, random_state=42)   # 先打乱这一类内部的顺序
    n_train = int(round(n * 0.7))                 # TODO① 训练集该拿多少个点？（70%，四舍五入）
    train_parts.append(grp.iloc[:n_train])
    val_parts.append(grp.iloc[n_train:])

train = pd.concat(train_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
val = pd.concat(val_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

train.to_csv("data/samples/train_20250906.csv", index=False)
val.to_csv("data/samples/val_20250906.csv", index=False)

# ---------------- 体检：各类在训练/验证中的数量和比例 ----------------
check = pd.DataFrame({
    "train": train["class"].value_counts(),
    "val": val["class"].value_counts(),
}).fillna(0).astype(int)
check["total"] = check["train"] + check["val"]
check["val_ratio"] = (check["val"] / check["total"]).round(3)
logging.info("划分结果：\n%s", check.to_string())
logging.info("总计 train=%d, val=%d (验证集占比 %.3f)",
             len(train), len(val), len(val) / len(df))
