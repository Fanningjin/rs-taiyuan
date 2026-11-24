"""D13 任务③: 后处理 -> 干净变化图
开运算(3x3) -> 闭运算(3x3) -> 8连通 label -> 最小图斑 >=20 像元(0.2 ha)
"""
import logging

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

T_UP, T_DN = 0.20, -0.10
MIN_AREA = 20  # 像元。10m: 1px=100m2, 20px=0.2 ha

def render(ax, mask, diff, valid, title):
    """按方向着色: 增绿深绿 / 变秃棕 / 稳定浅灰 / NaN 白"""
    img = np.zeros((*diff.shape, 3), dtype = np.float32)
    img[:] = 0.92
    #mask依次是 chg、closed、final——它决定"哪些像元有资格被染方向色"，valid只负责"哪些像元不是 NaN"
    #传入的这几个矩阵都已经是去过无效值的了
    img[mask & (diff > T_UP)] = [0.13, 0.55, 0.13]
    img[mask & (diff < T_DN)] = [0.55, 0.27, 0.07]
    img[~valid] = 1.0
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

def main():
    with rasterio.open("data/processed/NDVI_diff_20250906_minus_20250414.tif") as src:
        diff = src.read(1)
    valid = np.isfinite(diff)
    chg = valid & ((diff > T_UP) | (diff < T_DN))  #增绿和变秃的全部，变化的都是1

    st = np.ones((3,3), dtype=bool)  # 3x3 方形结构元(8邻域)

    #开运算：先腐蚀后膨胀，删掉多余小碎斑
    #opened和closed还是 (1000,1000)的布尔数组，只是"形状经过了削边和补洞"
    opened = ndimage.binary_opening(chg, structure=st)
    #闭运算：先膨胀后腐蚀
    closed = ndimage.binary_closing(opened, structure=st)

    log.info("v0 %d -> 开运算 %d (删 %d) -> 闭运算 %d",
             chg.sum(),opened.sum(), chg.sum()-opened.sum(), closed.sum())

    #连通域: 给每块编号, 再数每块多大
    #lab是(1000,1000) 的整数图，每个变化斑块里的像元被编成同一个号（1~775），非变化像元是 0
    # n 就是一个整数 = 775，斑块总数
    lab, n = ndimage.label(closed, structure=st)  # 8连通
    #给每个斑块的像元数计数  ravel函数是用于将多维数组转换为一维数组的工具
    sizes = np.bincount(lab.ravel())  # sizes[37] = 37号斑块有多少像元
    sizes[0] = 0  # 0 是背景, 不算
    log.info("连通域 %d 个, 最大斑块 %d 像元(%.2f km2)",
             n, sizes.max(), sizes.max()*1e-4)  # 1px=100m2=1e-4 km2

    # 最小图斑过滤: 编号->是否保留->铺回图上
    # 9和20的都做一次对比，实际上9的太小了，相当于没过滤，尺度上也没意义
    for area in (9, MIN_AREA):  #MIN_AREA=20
        clean = (sizes >= area)[lab] & closed  #sizes>9取的是计数结果>9的
        #(sizes >= 20)[lab] 拿着每个像元的编号去查这张表，查完贴回原图形状——编号37的所有像元瞬间全部变 False
        #一次算一排斑块，不写循环。这就是 D12 压码思想的重现：让整数编号当索引
        #末尾的 & closed 是保险，把"不在变化掩膜里"的像元剔掉
        #[ ]里的东西是下标，不是被运算的对象。语法 A[idx] 永远是「容器在前面、编号在括号里」。
        # 这里容器恰好是布尔数组，编号恰好是 lab 这张图
        #取出来是和 lab一样大的Bool表

        log.info("min_area>=%2d: 保留 %d (%.2f%%), 删 %d 像元 / %d 个斑块",
                 area, clean.sum(), clean.sum() / valid.sum()*100,
                 closed.sum() - clean.sum(),
                 int((sizes[1:]<area).sum()))
        if area == MIN_AREA:
            final = clean
    

    # ---- 三联对比图 ----
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
    render(axs[0], chg,    diff, valid, "v0: 阈值掩膜")  #第2个参数是mask
    render(axs[1], closed, diff, valid, "开运算+闭运算之后")
    render(axs[2], final,  diff, valid, "再删 <20 像元碎斑（交付版）")
    fig.tight_layout()
    fig.savefig("outputs/d13_change_clean.png", dpi=130)
    log.info("三联图已写: outputs/d13_change_clean.png")


if __name__ == "__main__":
    main()