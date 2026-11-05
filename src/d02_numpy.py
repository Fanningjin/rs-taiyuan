import numpy as np


#  块 1：建数组
a = np.arange(24).reshape(4, 6)   #生成0-23，再变成4行6列
print(a)
print(a.shape, a.dtype)

# 块 2：切片
print(a[:, 2])
print(a[1:3, :])
print(a[-1])

# 块 3：布尔索引
rng = np.random.default_rng(0)
ndvi = rng.random((5, 5))
mask = ndvi > 0.3  #true或false矩阵  做云掩膜都是这样
print(mask)
print(ndvi[mask])
print(mask.sum())
print(ndvi[mask].mean())

# 块 4：广播
print(a.mean(axis=0))
print(a - a.mean(axis=0))

# 块 5：统计
print(a.mean(), a.std(), np.percentile(a, 90))  #90分位数