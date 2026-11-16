"""D10 任务④：训练随机森林"""
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEAT_CSV = "data/samples/train_feat.csv"
MODEL_OUT = "models/rf_20250906.joblib"

def main():
    #读进来一个452 行 × 10 列的 DataFrame：列名是 9 个波段名 + "class"，class列里是 forest、water这类字符串
    df = pd.read_csv(FEAT_CSV)
    #把列名一个个过一遍，只要不是 class 就留下——得到 9 个特征名的列表
    band_names = [c for c in df.columns if c != "class"]
    #分别取出纯值数据和class列
    X = df[band_names].to_numpy(dtype="float32")
    y = df["class"].to_numpy()
    log.info("X=%s, y=%s, 特征:%s", X.shape, y.shape, band_names)  #为啥X大写，y小写：一个是矩阵，一个是向量

    t0 = time.time() #返回当前秒数
    #创建了一个还没训练的空模型
    rf = RandomForestClassifier(n_estimators=300,  #造 300 棵树
                                class_weight="balanced",  #小类样本加权
                                random_state=42,  #固定随机种子，保证可复现
                                n_jobs=-1,  #用满所有 CPU 核并行建树
                                oob_score=True)  #顺手算袋外精度
    rf.fit(X,y)  #真训练；sklearn 全家统一 fit(X, y) 接口

    log.info("训练完成，耗时 %.1f s", time.time() - t0)

    log.info("训练集精度: %.3f", rf.score(X,y)) #拿训练集自己考自己，返回准确率（判对比例）
    log.info("OOB精度: %.3f", rf.oob_score_) #属性名以 _ 结尾是 sklearn 的约定：表示"拟合之后才存在的属性"（拟合前访问会报错）

    order = rf.feature_importances_.argsort()[::-1]  #重要性从高到低的波段编号序列
    #rf.feature_importances_ 是长度 9 的数组，每棵树在每个特征上"用这个特征分叉带来了多少纯度提升"的平均值，9 个数加起来恰为 1
    #argsort() 返回"把数组从小到大排序后的下标序列"
    #[::-1] 是 Python 的步长切片：从头到尾、步长 −1，也就是把列表倒过来，从而从大到小排

    for i in order:
        log.info("重要性 %-8s %.3f", band_names[i], rf.feature_importances_[i])
    #%-8s 是新格式符：%s 基础上加 -8 表示"字符串左对齐、占8格宽"——短名字后面补空格，让后面的数字竖着对齐好看

    Path(MODEL_OUT).parent.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_OUT)  #把整个 rf 对象（300 棵树 + 全部参数 + 训练中记下的一切）序列化写盘
    log.info("模型已保存: %s", MODEL_OUT)

if __name__ == "__main__":
    main()