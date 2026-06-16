import pandas as pd
import glob

files = glob.glob("../data/*.csv")  # 替换路径
all_nodes = set()

for f in files:
    df = pd.read_csv(f)
    all_nodes.update(df['u'].tolist())
    all_nodes.update(df['v'].tolist())

print("总节点数:", len(all_nodes))
