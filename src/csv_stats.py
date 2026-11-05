import csv   #专门处理csv这种一行一记录的文件（不是表格形式）
from statistics import mean

with open("data/sample.csv", encoding="utf-8") as f:  #with会在代码块结束后自动关闭文件
    rows = list(csv.DictReader(f))

values = [float(r["value"]) for r in rows]  #csv读出的都是文本，所以要转成数字
print("行数:", len(values))
print("均值:", mean(values))
print("最大值:", max(values))