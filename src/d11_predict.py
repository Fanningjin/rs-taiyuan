"""D11 2:用9月训练出的模型进行4月全图预测:输出分类 GeoTIFF"""
import logging

import joblib
import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "models/rf_20250906.joblib"
STACK = "data/processed/features9_20250414.tif"
TRAIN_CSV = "data/samples/train_feat.csv"
OUT = "data/processed/classified_20250414.tif"

def main():
    rf = joblib.load(MODEL)

    with rasterio.open(STACK) as src:
        stack = src.read().astype("float32")  #(9,1000,1000)
        profile = src.profile.copy()
        band_names = list(src.descriptions)

    # 波段顺序必须与训练时一致，否则特征和模型对不上
    train_cols = pd.read_csv(TRAIN_CSV, nrows=0).columns.tolist()[:-1]  #nrows=0只取表头；砍掉最后class这一列
    assert band_names == train_cols, "波段顺序与训练数据不一致!"

    h, w = stack.shape[1], stack.shape[2]  #保存行数和列数
    # 固定好要变成的二维矩阵的第一维，剩下一个维度用-1让python算出来
    flat = stack.reshape(stack.shape[0],-1).T  # (1e6, 9) 整幅影像一共1000000个点
    valid = ~np.isnan(flat).any(axis=1)  #axis=k 沿着第 k 维扫，扫完这一维消失
    # 先产生一个和flat一样大的布尔数组，再用any；因为用的isnan，所以true是有坏像元，还要翻过来
    log.info("总像元 %d, 有效 %d, NaN(跳过) %d", flat.shape[0], valid.sum(), (~valid).sum())

    pred = rf.predict(flat[valid]) #只预测有效的
    log.info("预测完成, %d 个像元有了类别", len(pred))
    
    #预测只能一个个点形式预测，所以预测完了再把这些弄成矩阵形式；预测结果还是字符串标签形式，还要转成数字编码

    code = {name: i+1 for i,name in enumerate(rf.classes_)}  #推导生产一个字典，前面是 键：值 形式
    pred_full = np.full(h*w, "none", dtype=object)  #造一个全长 100 万、装满占位字符串 “none” 的对象数组（要装字符串）
    pred_full[valid] = pred  #散回（scatter back）：把有效位置一个个填上预测值，无效位置保留 “none”

    out = np.zeros(h*w, dtype="uint8")  # 0 = 无效像元  这里不是(h,w)，所以还不是数组
    for name,c in code.items():
        out[pred_full == name] = c  #遍历 6 个类：pred_full == name 得到布尔数组，凡是预测为这个类的像元，编号改成 c
    out = out.reshape(h, w)

    for name,c in code.items():
        n = int((out == c).sum())
        log.info(" %d = %-9s %6d 像元 (%.2f%%)", c, name, n, n/valid.sum()*100)

    profile.update({"count": 1, "dtype": "uint8", "nodata": 0, "compress": "lzw"})
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "class_code")

    log.info("分类图已写出: %s", OUT)

if __name__ == "__main__":
    main()