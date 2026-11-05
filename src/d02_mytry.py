#自己尝试一遍
import numpy as np

# 块1：建数组
a = np.arange(24).reshape(4,6)
print(a)
print(a.shape,a.dtype)

#2：切片
print(a[1:3,:]) #第2,3行
print(a[:,2])  #第3列
print(a[-1])  #一维


#3：布尔索引
rng = np.random.default_rng(0)
ndvi = rng.random((5,5))
mask = ndvi > 0.5
print(mask)
print(ndvi[mask])
print(mask.sum())
print(ndvi[mask].mean())

#4广播
print(a.mean(axis = 0)) #按列
print(a - a.mean(axis = 0))

print(a.mean(axis = 1)) #按行
print(a - a.mean(axis = 1)[:,None]) #把一串数变成4x1的，这样才对得上
#5统计
print(a.std())
print(np.percentile(a,90))
print(a.mean())