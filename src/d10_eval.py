"""D10 任务⑤：验证集评估"""
import logging

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, classification_report)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "models/rf_20250906.joblib"
VAL_CSV = "data/samples/val_feat.csv"

def main():
    rf = joblib.load(MODEL)
    df = pd.read_csv(VAL_CSV)
    band_names = [c for c in df.columns if c!="class"]
    X = df[band_names].to_numpy(dtype="float32")
    y = df["class"].to_numpy()

    pred = rf.predict(X)  #喂特征、吐预测类别（字符串数组，长度 194） 不再需要标签
    log.info("验证集总精度: %.3f", accuracy_score(y,pred))  #参数顺序有点反直觉：先真值、后预测（y, pred）
    log.info("Kappa: %.3f", cohen_kappa_score(y,pred))

    #classes_ 是模型训练时记下的类别清单（自动按字母序排好：bare, builtup...）。list() 把它转成普通列表
    #confusion_matrix 默认按它认为的类别顺序排，而传 labels=就强制行、列、报告全按同一个顺序走，防止矩阵的行列错位——错位一秒，诊断全歪
    labels = list(rf.classes_)
    cm = confusion_matrix(y, pred, labels=labels)  #得到 6×6 的整数矩阵
    #套进 DataFrame，把 index=labels（行名）和 columns=labels（列名）一挂，打印出来行列都有地类名
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    log.info("混淆矩阵(行=真实, 列=预测): \n%s", cm_df.to_string())

    # 成品报告
    log.info("分类报告: \n%s", classification_report(y, pred, labels=labels,digits=3))

if __name__ == "__main__":
    main()