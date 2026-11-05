import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 块1：用numpy造30个模拟采样点，存成csv
rng = np.random.default_rng(42)
n = 30

# 从四个地类名字里有放回地随机抽 30 次，每次抽一个名字，得到一个长度 30 的字符串数组
classes = rng.choice(["cropland","builtup","water","other"] , size = n,)
base = {"cropland":0.72, "builtup":0.15, "water":-0.35, "other":0.40}  #一个字典，存各地类的典型NDVI值
ndvi = np.array([base[c] for c in classes]) + rng.normal(0, 0.05, size = n)  #造了一个NDVI值计算函数；加正态分布的噪声
# 每一个点的类型是随机的，计算结果就是根据base和这个函数计算

df = pd.DataFrame({"point_id":np.arange(1,n+1), "lc_class": classes, "ndvi":ndvi})
df.to_csv("data/sample_points.csv", index = False) #不要把我左侧的行号 0、1、2…也写进文件


# 块2：用pandas再把csv读回来
data = pd.read_csv("data/sample_points.csv")
print(data.head())  #打印前五行
print(data.shape)  #30x3 可是有31行
print(data.describe())

# 看每个类型有多少个点
print(data["lc_class"].value_counts())


#块3：按地类分组，算NDVI均值
means = data.groupby("lc_class")["ndvi"].mean()  #计算各个地类的NDVI均值
# 按 lc_class 这一列的值把30行分成几堆（cropland 一堆、water 一堆……），每堆里只关心ndvi这一列，每堆算一个平均数
print(means)

#块4：画柱状图
fig , ax = plt.subplots(figsize=(6,4)) #单位是英寸
# fig是整张图，ax是绘图区，相当于“画纸”和“画纸上的坐标系”
means.plot(kind = "bar", ax = ax) #ax=ax告诉它“画在刚才那张画纸的坐标系里”
ax.set_ylabel("mean NDVI")
ax.set_title("Mean NDVI by land cover class")
# 用英文避开 matplotlib 默认字体不支持中文的问题——显示中文要额外配置字体
plt.tight_layout() #自动调整边距，防止标题或标签被裁掉
plt.savefig("outputs/d02_group_stats.png", dpi=150)
print("saved outputs/d02_group_stats.png")

