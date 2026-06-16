import csv

# 输入文件
input_file = "edges.txt"   # 改成你的

# 读取全部边
edges = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        u, v = line.split()
        edges.append((u, v))

print(f"共读取 {len(edges)} 条原始边")

# ---------------------------------------------------------
# 处理成有向双边 + 权重 1
# ---------------------------------------------------------
def make_bidirectional(edges):
    new_edges = []
    for u, v in edges:
        new_edges.append((u, v, 1))  # 正向
        new_edges.append((v, u, 1))  # 反向
    return new_edges

# ---------------------------------------------------------
# 写 CSV（带表头 u,v,w）
# ---------------------------------------------------------
def write_csv(filename, edge_list):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "weight"])
        writer.writerows(edge_list)


# 1. 初始 50%
mid = len(edges) // 2
init_edges = edges[:mid]
snap_edges = edges[mid:]

# 双向 + 权
init_full = make_bidirectional(init_edges)
write_csv("edges_init.csv", init_full)
print(f"写入 edges_init.csv，共 {len(init_full)} 条双向边")

# ---------------------------------------------------------
# 2. 每 10 条作为一个 snapshot，且每个 snapshot 包含之前所有边
# ---------------------------------------------------------
snap_size = 10
current_graph_edges = set(init_edges)   # 记录所有已加入的"无向边"

for i in range(0, len(snap_edges), snap_size):
    # 新增加的边（无向）
    new_batch = snap_edges[i : i + snap_size]

    # 加入 set（避免重复）
    for e in new_batch:
        current_graph_edges.add(e)

    # 生成包含全部边的双向图
    full_graph = make_bidirectional(list(current_graph_edges))

    # 文件名
    idx = i // snap_size + 1
    file_name = f"edges_snap_{idx}.csv"

    # 写入
    write_csv(file_name, full_graph)
    print(f"写入 {file_name}: 当前图包含 {len(full_graph)} 条双向边(含权)")


print("全部完成！")
