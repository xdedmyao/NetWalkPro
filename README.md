# NetWalkPro: A Deep Embedding framework for Dynamic Shipping Network Anomaly Detection

本项目面向**航运（船舶 AIS）网络**，提供一套**时序图异常检测**的研究型实现。整体流程为：

> **原始 AIS 航次数据 → 时序快照（snapshots）→ NetWalk 节点嵌入 → 流式 K-Means 边异常打分 → 可视化（路线地图）**。

核心算法基于 NetWalk（KDD'18, Cheng & Yu 等）思想，并使用 PyTorch 进行了重写，配合一个轻量的流式 K-Means 检测器对每条航线（边）进行异常评分，以发现“突增 / 突降 / 突然消失”的异常航线。

> *本 README 反映当前仓库代码状态（2026/6/9 更新）。*

---

## ⚠️ 数据使用须知（重要）
仓库未跟踪任何数据文件，包括但不限于：
- `*.csv`
- `*.txt`
- `*.npy`
- `*.pt`
- `*.html`（地图渲染产物）
- 以及其他生成产物

所有数据集、嵌入、异常结果都应**仅保存在本地** `data/` 与 `output/` 目录下，这两个目录已经通过 `.gitignore` 排除在版本控制之外。

仓库提交内容仅限：源代码、配置文件、文档。

---

## 🏗️ 仓库结构

```
TIDE-shipping-network-anomaly/
├── data/                          # 原始数据 / 时间快照（本地，未跟踪）
│   ├── raw_data.csv               # 原始 AIS 航次记录（需手动准备）
│   ├── snapshots/                 # 全量时间快照
│   ├── snapshots_sampled/         # 采样后的时间快照（默认 1%）
│   └── injected/                  # 注入异常实验的标注文件
├── src/                           # 核心算法库
│   ├── netwalk/                   # NetWalk 嵌入模型（PyTorch 版）
│   │   ├── DenoisingAE.py         # 稀疏去噪自编码器
│   │   ├── Model.py               # 训练封装：clique loss + AE loss
│   │   └── netwalk_update.py      # Reservoir 抽样 + 随机游走 + 快照流水线
│   └── anomaly/
│       └── Kmeans_Anomaly.py      # 流式 K-Means 异常检测器
├── scripts/                       # 可执行实验流水线
│   ├── create_snapshots.py        # 由 raw_data.csv 切分时间快照
│   ├── NetWalk.py                 # 仅做 NetWalk 嵌入
│   ├── NetWalk_Anomaly_Kmeans.py  # 嵌入 + 流式 K-Means 异常检测（主流程）
│   ├── plot_snapshot_route_maps.py# 渲染交互式航线地图（HTML）
│   └── anomaly analysis/
│       ├── inject_anomalies.py    # 注入人工异常用于评测
│       └── analyze_anomaly_scores.py # 评测命中率 / 分数变化
├── output/                        # 实验输出（本地，未跟踪）
│   ├── embeddings/                # 每个快照的节点嵌入
│   ├── anomalies/                 # 每个快照的异常评分 CSV
│   └── maps/                      # 渲染出的 HTML 地图
├── draft/                         # 实验性 / 已弃用 / 历史代码
├── .gitignore
└── README.md
```

---

## 📁 各目录说明

### `data/`
存放原始数据与预处理后的快照（不纳入 Git）。
- `raw_data.csv`：来自微信群 / 内部共享的 AIS 航次记录，需自行放置。
  必需列至少包括：
  - `leg_start_postime`, `arrival_time`：航段起止时间
  - `leg_start_port_code`, `leg_end_port_code`：起止港口代码
  - `dwt`：吨位（用作边权 `w`）
  - 可选坐标列：`start_port_lat`, `start_port_lon`, `end_port_lat`, `end_port_lon`（用于绘图）

### `src/`
全部算法实现。两个子模块：
- `src/netwalk/`：NetWalk 的 **PyTorch 实现**，包含基于水库抽样（Reservoir Sampling）的有向加权随机游走，以及带稀疏正则与 Laplacian 正则（clique loss）的去噪自编码器。
- `src/anomaly/`：流式 K-Means 异常检测器（在线增量更新聚类中心）。

### `scripts/`
端到端可执行管道，每个脚本均可通过 `python scripts/xxx.py --help` 查看完整参数说明。

### `output/`
实验产物，全部保存在本地：
- `embeddings/snapshot_<i>.txt`：第 i 个快照的节点嵌入（首列为节点 ID，其余列为向量分量）；以及 `node_mapping.txt`（节点 ID ↔ 内部索引）
- `anomalies/result_kmeans_<snapshot>.csv`：每条航线的异常分数及变化量
- `maps/snapshot_<period>_routes.html`：可在浏览器交互查看的航线地图

### `draft/`
实验性 / 探索性代码，**不属于主流程**，仅作为参考与对比。包括：
- `Dynamic_iForest_Anomaly.py`、`IsolationForest_Anomaly.py`：基于 Isolation Forest 的对比实验（未完整对接数据流）
- `csvgen.py`、`edges*.csv`、`edges.txt`：早期玩具样例
- `cuda_test.py`：CUDA / TF GPU 可用性测试
- `filter_domestic_routes.py`：把同国（港口代码前两位相同）航线从原始数据中剔除
- `node_count.py`、`path.py`：小工具
- `ts_version_netwalk/`：基于 TensorFlow 1.x / 2.x 的旧版 NetWalk 参考实现

---

## 📝 脚本一览

| Script | 主要用途 |
|---|---|
| `scripts/create_snapshots.py` | 把 `raw_data.csv` 按时间切片为快照 CSV。支持 `--slice monthly`/`biweekly`/`weekly`/`<N>d`，可同时输出全量 (`data/snapshots/`) 与采样 (`data/snapshots_sampled/`) 版本，可选附带港口经纬度 |
| `scripts/NetWalk.py` | 单独执行 NetWalk 嵌入，产物落到 `output/embeddings/` |
| `scripts/NetWalk_Anomaly_Kmeans.py` | **主流程**：逐快照计算 NetWalk 嵌入，再用流式 K-Means 给每条边打异常分。产物写到 `output/embeddings/` 与 `output/anomalies/` |
| `scripts/plot_snapshot_route_maps.py` | 把每个快照渲染成交互式 Plotly Scattergeo HTML 地图，并把 anomaly CSV 中分数最高的 Top-N 航线红色高亮 |
| `scripts/anomaly analysis/inject_anomalies.py` | 在某个指定快照中随机选取一定比例的航线，将其权重缩放（默认 ×5 或 ÷5），生成 `data/injected/<snapshot>_labels.csv` 标注 |
| `scripts/anomaly analysis/analyze_anomaly_scores.py` | 用注入的标注评估检测效果（Precision / Recall / F1），并对比基线 vs. 注入后两次结果分数的变化 |

---

## 🔬 算法概述

### 1. NetWalk 嵌入（`src/netwalk/`）

- **Reservoir（水库抽样）**：对每个节点维护其出邻居的有偏抽样池，权重正比于边权（吨位 `w`），用于支持后续随机游走的快速更新。每次进入新快照时基于当月完整边集重新构建。
- **加权随机游走**：在有向加权图上按边权概率游走 `walk_len` 步，每个节点游走 `walk_per_node` 次，得到训练样本（每行一个游走序列的 one-hot 表达）。
- **去噪自编码器（DenoisingAE）**：编码-解码结构，输入为节点的 one-hot，损失由四部分加权：
  - 重构误差（`gamma`）
  - KL 稀疏正则（`beta`, `rho`）
  - 权重 L2 正则（`lamb`）
  - **Clique embedding loss**：在隐空间中拉近同一条游走里出现的节点（基于 walk-graph 的 Laplacian），保证邻接节点嵌入相似
- **流式更新**：第 0 个快照做"冷启动"训练，之后每个快照都会接着 fit，从而获得平滑变化的节点嵌入序列。
- **设备选择**：自动按 `cuda > mps > cpu` 顺序选取最优设备。

### 2. 流式 K-Means 异常检测（`src/anomaly/Kmeans_Anomaly.py`）

对每条航线 `(u, v)` 构造特征向量：

- **结构特征**：`emb[u] * emb[v]`（Hadamard 积，反映两端节点的关系强度）
- **权重相对变化**：`|log(w_t / w_{t-1})|`（截断到 `±ratio_clip` 抑制极端值）
- **加权下行变化**：`max(0, -log_ratio) * log1p(prev_w / scale)`（突出"高吨位航线突然下跌"）

并把"上一期出现但本期消失"的航线也纳入候选集（`is_disappeared` 标志），这样可以检出**航线消失型异常**。

检测流程：
1. **初始化**：第 0 个快照用标准 K-Means（`k=10`）建立簇中心
2. **流式更新**：后续快照中，先用当前中心给样本打 anomaly score（与最近中心的距离 / 1 - cosine），再按权重 α 衰减地融合新样本到中心：

   $$
   c_{new} = \frac{\alpha \cdot c_{old} \cdot n_{old} + (1-\alpha) \cdot \sum x_{new}}{\alpha \cdot n_{old} + (1-\alpha) \cdot n_{new}}
   $$

3. **输出**：每条边给出 `anomaly_score` 与 `anomaly_score_delta_vs_prev`，按后者降序排序保存到 CSV。

### 3. 异常注入实验（`scripts/anomaly analysis/`）

为了在没有真实标签的情况下定量评估检测器：
1. `inject_anomalies.py`：随机选取某一快照中 5% 的航线，将权重 ×5（"突增异常"）或 ÷5（"突降异常"），并备份基线检测结果
2. 重新跑一遍 `NetWalk_Anomaly_Kmeans.py` 得到注入后的检测结果
3. `analyze_anomaly_scores.py`：对比"基线 vs. 注入后"两次得到的 anomaly score，统计：
   - 注入异常被检出的 Precision / Recall / F1（按 Top-N 排序）
   - 多少注入航线的 anomaly score 上升了，平均上升幅度多少

---

## ⚙️ 依赖环境

建议使用虚拟环境（`venv` 或 `conda`）。

- Python 3.8+（推荐 3.10+）
- 主要依赖：

```bash
pip install numpy pandas scipy scikit-learn torch networkx tqdm plotly
```

| 库 | 用途 |
|---|---|
| `numpy`, `pandas` | 数据 I/O 与处理 |
| `torch` | NetWalk 模型实现 |
| `networkx` | 图结构构建与随机游走 |
| `scipy` | 稀疏矩阵 / Laplacian |
| `scikit-learn` | K-Means、MinMaxScaler |
| `tqdm` | 训练进度条 |
| `plotly` | 渲染交互式航线地图 |

设备说明：
- 有 NVIDIA GPU → 自动使用 CUDA
- Apple Silicon (M 系列芯片) → 自动使用 MPS
- 否则回落到 CPU

---

## ▶️ 运行流程（推荐顺序）

### 步骤 0：准备原始数据
将 `raw_data.csv`（来自微信群 / 内部共享）放到 `data/` 目录下。

### 步骤 1：生成时间快照
```bash
# 默认按月切片，并同时输出 1% 采样版本
python scripts/create_snapshots.py

# 自定义示例：每 10 天切一片，从 2025-01-01 起，不要采样
python scripts/create_snapshots.py --slice 10d --start-date 2025-01-01 --sample-rate 0
```
输出：`data/snapshots/snapshot_<period>.csv`（如 `snapshot_2025-01.csv`）。

### 步骤 2：嵌入 + 异常检测（主流程）
```bash
python scripts/NetWalk_Anomaly_Kmeans.py \
    --input data/snapshots \
    --representation-size 32 \
    --number_walks 20 \
    --walk-length 5 \
    --k 10 --alpha 0.5
```
输出：
- `output/embeddings/snapshot_<i>.txt`
- `output/anomalies/result_kmeans_snapshot_<period>.csv`

> 加 `--fast` 可以将 epoch 减半，便于快速验证。

### 步骤 3：可视化
```bash
python scripts/plot_snapshot_route_maps.py
```
默认会读取 `data/snapshots/` 中所有快照，并叠加 `output/anomalies/` 中的 Top-N 异常航线，输出 HTML 到 `output/maps/`，浏览器打开即可交互查看。

### 步骤 4（可选）：注入实验评估
```bash
# 1) 在某一快照注入异常
python "scripts/anomaly analysis/inject_anomalies.py"

# 2) 重新跑步骤 2 得到注入后的检测结果

# 3) 计算指标
python "scripts/anomaly analysis/analyze_anomaly_scores.py"
```

---

## 🧪 实验性开发规范

在尝试新模型 / 新结构 / 大改动时，请：

- **新建 Git 分支**进行开发，不要直接提交到 `main`。
- 充分验证 + 清理后再合并到 `main`。
- 实验性脚本可以放到 `draft/`，并在文件顶部注明用途与状态。

---

## 📌 已知限制与 TODO

- `scripts/anomaly analysis/inject_anomalies.py` 当前会**直接覆盖原快照**，请先备份。
- `draft/Dynamic_iForest_Anomaly.py` 与 `draft/IsolationForest_Anomaly.py` 仅供对比，尚未对齐当前数据接口。
- 港口经纬度依赖原始 CSV 是否提供 `start_port_lat/lon` 与 `end_port_lon/lat`；若不提供，请使用 `--no-coords` 跳过（同时也无法绘制地图）。
- NetWalk 的 `n × hidden_size` one-hot 输入会随节点数增大而变大，节点数特别多时建议增大采样比例或限定区域。

---

祝开发快乐。 2026.6.9

不会跑或者出bug可召唤@xdedmyao。
