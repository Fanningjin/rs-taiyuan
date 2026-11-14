# rs-taiyuan ｜ 太原汾河沿线 Sentinel-2 多时相遥感与土地覆盖分类

用 Sentinel-2 L2A 两期影像，在太原城区北部·汾河沿线 10×10 km 研究区完成一套完整的遥感处理流程：
**批量预处理（AOI 裁剪 + SCL 云掩膜）→ 两期 NDVI 对比 → 五指数特征工程 → 六类样本采样（70/30 分层）**，
后续在此基础上做土地覆盖分类（随机森林）。

## 数据

| 项 | 说明 |
|---|---|
| 数据源 | Copernicus Data Space（Sentinel-2 L2A，含 SCL 场景分类层） |
| 图幅 | `T49SFC`（UTM 49N，EPSG:32649） |
| 第一期 | `S2A_MSIL2A_20250414`（春季，裸土/裸耕为主） |
| 第二期 | `S2C_MSIL2A_20250906`（初秋，植被季） |
| 研究区 | 太原城区北部·汾河沿线 10×10 km（`aoi.json`：625000 / 4194000 / 635000 / 4204000，面积 100 km²） |

> 2026-11-13 研究区调整：初选的北郊旱作河谷窗口 SCL 水体占比 0.00%（六类中水体几乎无样本），
> 在同一景内改选汾河城区窗口——覆盖汾河、主城区、西山林地与北部农田，六类齐全；
> 两期 SCL 坏像元 0.0192% / 0.1328%，无重新下载数据。

## 环境

Miniforge3 + conda 环境 `rs`（Python 3.11），主要依赖：`rasterio`、`numpy`、`pandas`、`matplotlib`；
样本核订使用 QGIS 3.44（conda-forge 安装于 `qgis` 环境）。

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
│  ├─ d08_features.py       # 五指数特征栈（features.tif）
│  ├─ d09_truecolor.py      # 真彩色底图（QGIS 目视解译用）
│  ├─ d09_sample.py         # 阈值初标 + 每类随机抽样（弱监督采样）
│  ├─ d09_extra.py          # 少数类补点（水体/林地/耕地/草地，带空间先验）
│  └─ d09_split.py          # 70/30 分层划分（train/val）
├─ data/raw/                # 原始 .SAFE（不入库）
├─ data/processed/          # 中间成果 .tif（不入库）
├─ data/samples/            # 样本 CSV（入库）：candidates / cleaned / train / val
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

# 5) 真彩色底图 + 弱监督候选采样点
python src/d09_truecolor.py
python src/d09_sample.py

# 6) 少数类补点（可选）
python src/d09_extra.py

# 7) 70/30 分层划分（读 data/samples/samples_cleaned*.csv → train/val）
python src/d09_split.py
```

## 关键结果

- SCL 坏像元比例：2025-04-14 为 0.0192%（192 px @10m）；2025-09-06 为 0.1328%（1328 px）。
- NDVI 均值：0.134（4 月，春季）→ 0.313（9 月，植被季 + 城区混合地表）。
- 两期 NDVI 差值：mean 0.179，67.5% 像元显著增绿（ΔNDVI > 0.1）。
- 图件（`outputs/`）：
  - `d03_b04_preview.png` 单景预览
  - `d04_mask_compare.png` 云掩膜前后对比
  - `d06_ndvi.png` 2025-04-14 NDVI
  - `d07_ndvi_2periods.png` 两期 NDVI + 差值三面板
  - `d09_candidates.png` 六类候选样本点预览

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

## 采样与训练/验证划分

- 类别体系（6 类）：耕地 / 林地 / 草地 / 水体 / 建设用地 / 裸地及其他。
- 弱监督采样流程：指数阈值初标（NDVI 分层 + MNDWI / NDBI / BSI，互斥规则按优先级）
  → 每类随机抽样 120 个候选 → QGIS 真彩色目视核订删除误判像元。
- 少数类补点：水体/林地/草地用更严阈值补抽；耕地加"田芯匀质性 + 北部农田带"空间先验；
  裸地因屋顶光谱污染改用已知裸地点邻域扩散 + 人工补点。
- 样本量 646：水体 131 / 林地 131 / 耕地 115 / 建设用地 100 / 草地 95 / 裸地及其他 74。
- 70/30 分层抽样（每类内部）：训练 452 / 验证 194（验证比例 0.300，各类 0.297–0.305）。

## 下一步

- [x] 训练样本采集（弱监督候选 + QGIS 目视核订，按类分层抽样 70/30）
- [ ] 随机森林分类 + 精度评价（混淆矩阵、OA、Kappa）
- [ ] 专题图输出

> 注：`data/raw/`、`data/processed/`、`*.tif` 已被 `.gitignore` 排除，仓库只保存代码与样本/配置文件。
