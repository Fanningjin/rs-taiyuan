# rs-taiyuan ｜ 太原北部 Sentinel-2 多时相遥感分析

用 Sentinel-2 L2A 两期影像，在太原北部 10×10 km 研究区完成一套完整的遥感处理流程：
**批量预处理（AOI 裁剪 + SCL 云掩膜）→ 两期 NDVI 对比 → 五指数特征工程**，
后续在此基础上做土地覆盖分类（随机森林）。

## 数据

| 项 | 说明 |
|---|---|
| 数据源 | Copernicus Data Space（Sentinel-2 L2A，含 SCL 场景分类层） |
| 图幅 | `T49SFC`（UTM 49N，EPSG:32649） |
| 第一期 | `S2A_MSIL2A_20250414`（春季，裸土为主） |
| 第二期 | `S2C_MSIL2A_20250906`（初秋，植被季，AOI 无云） |
| 研究区 | 太原北部 10×10 km（`aoi.json`：646900 / 4237120 / 656900 / 4247120，面积 100 km²） |

## 环境

Miniforge3 + conda 环境 `rs`（Python 3.11），主要依赖：`rasterio`、`numpy`、`pandas`、`matplotlib`。

## 目录结构

```
rs-taiyuan/
├─ aoi.json                 # 研究区边界（米制坐标，EPSG:32649）
├─ batch_preprocess.py      # 批量预处理：遍历 data/raw/*.SAFE → AOI 裁剪 + SCL 云掩膜
├─ src/                     # 逐日流程脚本
│  ├─ d03_*.py              # 影像读写 / 窗口裁剪 / 预览
│  ├─ d04_*.py              # AOI 定义 / 按 AOI 裁剪 / 云掩膜 / 对比图
│  ├─ d06_ndvi.py           # 两期 NDVI 计算
│  ├─ d07_diff.py           # 两期 NDVI 差值图
│  └─ d08_features.py       # 五指数特征栈（features.tif）
├─ data/raw/                # 原始 .SAFE（不入库）
├─ data/processed/          # 中间成果 .tif（不入库）
└─ outputs/                 # 图件
```

## 处理流程（均在项目根目录运行）

```bash
# 1) 批量预处理：裁剪 + 云掩膜（输出 {scene}_B04_clean.tif）
python batch_preprocess.py --input data/raw --aoi aoi.json --out data/processed

# 2) 两期 NDVI（输出 {scene}_NDVI.tif）
python src/d06_ndvi.py

# 3) 两期对比 + 差值图
python src/d07_diff.py

# 4) 五指数特征栈（输出 features_20250906.tif）
python src/d08_features.py
```

## 关键结果

- SCL 坏像元比例：2025-04-14 为 0.0076%（76 px @10m）；2025-09-06 为 0.0000%。
- NDVI 均值：0.099（4 月，裸土为主）→ 0.465（9 月，植被为主）。
- 两期 NDVI 差值：mean 0.366，92.7% 像元显著增绿（ΔNDVI > 0.1）。
- 图件（`outputs/`）：
  - `d03_b04_preview.png` 单景预览
  - `d04_mask_compare.png` 云掩膜前后对比
  - `d06_ndvi.png` 2025-04-14 NDVI
  - `d07_ndvi_2periods.png` 两期 NDVI + 差值三面板

## 特征栈 `features_20250906.tif`

5 波段、float32、10 m 网格、EPSG:32649、nodata = NaN。
波段顺序如下（与 `src/d08_features.py` 中字典顺序一致）：

| 波段 | 指数 | 公式 | 说明 |
|---|---|---|---|
| 1 | NDVI | (B08 − B04) / (B08 + B04) | 植被指数 |
| 2 | NDWI | (B03 − B08) / (B03 + B08) | 水体指数，McFeeters 1996 |
| 3 | MNDWI | (B03 − B11) / (B03 + B11) | 水体增强指数，Xu 2006；B11 由 20 m 双线性重采样到 10 m |
| 4 | NDBI | (B11 − B08) / (B11 + B08) | 建成区指数 |
| 5 | BSI | ((B11 + B04) − (B08 + B02)) / ((B11 + B04) + (B08 + B02)) | 裸土指数 |

## 下一步

- [ ] 训练样本采集（QGIS 目视解译，按类分层抽样 70/30）
- [ ] 随机森林分类 + 精度评价（混淆矩阵、OA、Kappa）
- [ ] 专题图输出

> 注：`data/raw/`、`data/processed/`、`*.tif` 已被 `.gitignore` 排除，仓库只保存代码与配置文件。
