"""D15: CLI 自动化——一条命令串起 预处理→特征→分类→变化检测
用法: python run.py --list                       # 查看步骤表
      python run.py --aoi aoi.json               # 按缓存跳过已完成步骤
      python run.py --force --only ndvi_diff     # 强制重跑指定步骤
"""
import argparse
import logging
import time
from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parent   # run.py 所在目录 = 项目根; 之后所有相对路径都以它为锚

# 步骤表: 一行一个脚本 = 一次调用 + 一组产物 + 一个跳过判据
# 注意: 采样(d09_*)与精度抽检(d14_*)是人工环节, 不进自动链
STEPS = [
    {"name": "preprocess", "script": "batch_preprocess.py",
     "desc": "AOI 裁剪 + 云掩膜(两期 B04_clean)",
     "outputs": ["data/processed/S2_20250414_B04_clean.tif",
                 "data/processed/S2_20250906_B04_clean.tif"]},
    {"name": "ndvi", "script": "src/d06_ndvi.py",
     "desc": "两期 NDVI(d06)",
     "outputs": ["data/processed/S2_20250414_NDVI.tif",
                 "data/processed/S2_20250906_NDVI.tif"]},
    {"name": "ndvi_diff", "script": "src/d07_diff.py",
     "desc": "ΔNDVI(9月−4月) + 三面板图(d07)",
     "outputs": ["data/processed/NDVI_diff_20250906_minus_20250414.tif",
                 "outputs/d07_ndvi_2periods.png"]},
    {"name": "features", "script": "src/d08_features.py",
     "desc": "9 月五指数特征工程(d08)",
     "outputs": ["data/processed/features_20250906.tif"]},
    {"name": "stack_sep", "script": "src/d10_stack.py",
     "desc": "9 月 9 波段特征栈(d10_stack)",
     "outputs": ["data/processed/features9_20250906.tif"]},
    {"name": "extract", "script": "src/d10_extract.py",
     "desc": "训练/验证样本取特征值(d10_extract)",
     "outputs": ["data/samples/train_feat.csv",
                 "data/samples/val_feat.csv"]},
    {"name": "train", "script": "src/d10_train.py",
     "desc": "随机森林训练(有模型跳过, --retrain 重训)",
     "outputs": ["models/rf_20250906.joblib"]},
    {"name": "predict_sep", "script": "src/d10_predict.py",
     "desc": "9 月全图分类预测(d10_predict)",
     "outputs": ["data/processed/classified_20250906.tif"]},
    {"name": "stack_apr", "script": "src/d11_stack_apr.py",
     "desc": "4 月 9 波段特征栈(d11_stack, 含 NDVI 交叉验证)",
     "outputs": ["data/processed/features9_20250414.tif"]},
    {"name": "predict_apr", "script": "src/d11_predict.py",
     "desc": "4 月迁移分类预测(d11_predict)",
     "outputs": ["data/processed/classified_20250414.tif"]},
    {"name": "transition", "script": "src/d12_transition.py",
     "desc": "6×6 转移矩阵 + 百分比表(d12_transition)",
     "outputs": ["outputs/d12_transition_matrix.csv",
                 "outputs/d12_transition_matrix_pct.csv"]},
    {"name": "area", "script": "src/d12_area.py",
     "desc": "面积统计 + 分类面积柱状图(d12_area)",
     "outputs": ["outputs/d12_area_summary.csv",
                 "outputs/d12_area_by_class.png"]},
    {"name": "change_map", "script": "src/d12_change_map.py",
     "desc": "变化类型图(d12_change_map)",
     "outputs": ["outputs/d12_change_map.png"]},
    {"name": "ndvi_check", "script": "src/d12_ndvi_check.py",
     "desc": "转移组 ΔNDVI 交叉表(d12_ndvi_check)",
     "outputs": ["outputs/d12_transition_ndvi.csv"]},
    {"name": "hist", "script": "src/d13_threshold.py",
     "desc": "ΔNDVI 直方图 + 阈值试算(d13_threshold)",
     "outputs": ["outputs/d13_ndvi_hist.png"]},
    {"name": "mask", "script": "src/d13_mask.py",
     "desc": "双向阈值初版变化图(d13_mask)",
     "outputs": ["outputs/d13_change_v0.png"]},
    {"name": "clean", "script": "src/d13_clean.py",
     "desc": "开闭运算 + 最小图斑清理(d13_clean)",
     "outputs": ["outputs/d13_change_clean.png"]},
    {"name": "cross", "script": "src/d13_cross.py",
     "desc": "NDVI 掩膜 × 分类叠加交叉(d13_cross)",
     "outputs": ["outputs/d13_cross_table.csv"]},
]


# 日志系统: 屏幕 + 文件双路, 同一套格式
def setup_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"run_{stamp}.log"

    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()   # 屏幕
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")  # 落盘
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger, log_path


# 从命令行读取输入参数
def parse_args():
    p = argparse.ArgumentParser(
        description="遥感变化检测流水线: 预处理→特征→分类→变化检测")
    # 路径类
    p.add_argument("--aoi", default="aoi.json", help="AOI 边界文件")
    p.add_argument("--raw", default="data/raw", help="原始影像文件夹")
    p.add_argument("--out", default="data/processed", help="中间产物文件夹")
    p.add_argument("--outputs", default="outputs", help="成果文件夹")
    p.add_argument("--model", default="models/rf_20250906.joblib", help="模型文件路径")
    p.add_argument("--log-dir", default="outputs", help="日志文件夹")

    # 行为开关类
    p.add_argument("--skip", default="", help="跳过这些步骤, 逗号分隔, 如 ndvi,hist")
    p.add_argument("--only", default="", help="只运行这些步骤, 逗号分隔")
    p.add_argument("--force", action="store_true",
                   help="忽略缓存, 强制重跑选中的步骤")
    p.add_argument("--retrain", action="store_true",
                   help="强制重新训练模型(默认模型存在即跳过)")
    p.add_argument("--list", action="store_true", dest="list_steps",
                   help="打印步骤表后退出, 不做任何计算")

    return p.parse_args()


# 打印步骤表: 序号 / 名称 / 产物数 / 说明, 列对齐
def show_steps(logger):
    logger.info("步骤表(共 %d 步):", len(STEPS))
    logger.info("%4s  %-12s  %4s  %s", "序号", "步骤名", "产物", "说明")
    for i, s in enumerate(STEPS, start=1):
        logger.info("%4d  %-12s  %4d  %s",
                    i, s["name"], len(s["outputs"]), s["desc"])


def build_cmd(step, args):
    """把步骤(表里的一行)翻译成一条子进程命令"""
    # "script"读出来就是.py文件名
    cmd = [sys.executable, step["script"]]  # 用当前解释器,当前正在跑run.py的python
    if step["name"] == "preprocess":
        # 只有 batch_preprocess.py 支持命令行传路径, 其余老脚本读项目根的默认路径
        cmd += ["--input", args.raw, "--aoi", args.aoi, "--out", args.out]
    return cmd


def run_one(step, args, logger):
    outs = [ROOT / p for p in step["outputs"]]  #表里的相对路径变绝对路径

    # 跳过判据 v1: 产物全部存在 -> SKIP (局限: 不比较新旧, mtime 校验是加分项)
    force = args.force or (step["name"] == "train" and args.retrain)
    if not force and all(p.exists() for p in outs):
        logger.info("SKIP | %-12s | 产物齐全", step["name"])
        return "skip"

    cmd = build_cmd(step, args)
    logger.info("RUN  | %-12s | %s", step["name"], " ".join(cmd[1:]))
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # Windows 管道输出防 GBK 乱码
    proc = subprocess.run(cmd, cwd=ROOT,                # 子进程必须在项目根跑
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    dt = time.time() - t0

    if proc.returncode != 0:
        logger.error("FAIL | %-12s | %.1fs | 退出码 %d", step["name"], dt, proc.returncode)
        for line in (proc.stderr or proc.stdout).strip().splitlines()[-15:]:
            logger.error("   | %s", line)               # 只留最后 15 行, 这是真正的报错现场
        return "fail"

    logger.info("DONE | %-12s | %.1fs", step["name"], dt)
    return "run"


# 循环 STEPS 整张表，用 only/skip 当过滤器；执行顺序永远由 STEPS 决定，和命令行里写的顺序无关
def run_pipeline(args, logger):
    # 逗号串 -> 集合; 空串解析成空集合, 不会出现 "" 这种名字
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    known = {s["name"] for s in STEPS}
    bad = (only | skip) - known
    if bad:
        logger.error("未知步骤名: %s", ", ".join(sorted(bad)))
        logger.error("可用步骤: %s", ", ".join(s["name"] for s in STEPS))
        sys.exit(2)   # 用法错误用 2, 运行失败用 1

    # 退出码 0 成功、1 跑一半死了、2 用法错误
    stats = {"run": 0, "skip": 0, "fail": 0}
    t_all = time.time()
    for step in STEPS:
        name = step["name"]
        if only and name not in only:
            continue    # 没被 --only 选中, 静默跳过不计数
        if name in skip:
            logger.info("SKIP | %-12s | --skip", name)
            stats["skip"] += 1
            continue

        status = run_one(step, args, logger)  #运行
        stats[status] += 1
        if status == "fail":
            logger.error("在 %s 步失败, 中止后续步骤", name)
            break

    logger.info("汇总: 成功 %d | 跳过 %d | 失败 %d | 总耗时 %.1fs",
                stats["run"], stats["skip"], stats["fail"], time.time() - t_all)
    if stats["fail"]:
        logger.error("流水线未完成")
        sys.exit(1)


def main():
    args = parse_args()
    logger, log_path = setup_logger(args.log_dir)
    logger.info("日志文件: %s", log_path)

    if args.list_steps:
        show_steps(logger)
        logger.info("共 %d 步。--list 只读, 未做任何计算。", len(STEPS))
        return

    logger.info("参数: %s", vars(args))
    
    run_pipeline(args, logger)



if __name__ == "__main__":
    main()