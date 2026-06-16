import numpy as np
import pandas as pd
import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class StreamingKMeansDetector:
    def __init__(self, k=10, alpha=0.5, distance_metric="euclidean", ratio_clip=4.0):
        """
        :param k: 聚类中心数量（论文默认10）
        :param alpha: 衰减因子（0.5代表历史和当前的权重相同）
        :param distance_metric: 距离度量，可选 "cosine" 或 "euclidean"
        :param ratio_clip: log_ratio 截断阈值，默认 4.0
        """
        self.k = k
        self.alpha = alpha
        self.distance_metric = distance_metric
        self.ratio_clip = float(ratio_clip)

        # 核心状态变量
        self.centers = None  # 聚类中心
        self.cluster_counts = None  # 每个簇的历史样本数 (k,)

        self.scaler = MinMaxScaler()
        self.is_initialized = False
        self.prev_weights = {}  # 记录上一期每条航线的权重
        self.prev_anomaly_scores = {}  # 记录上一期每条航线的异常分数

    @staticmethod
    def _l2_normalize(X, eps=1e-12):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + eps)

    def _extract_features(self, embedding_path, edge_file_path):
        """加载 embedding 与边列表，并按唯一航线聚合特征。

        特征采用结构嵌入 + 权重相对变化（对称），避免对绝对权重偏置。
        样本使用“当期边 + 上期边(并集)”，以捕捉航线消失异常。
        """
        try:
            # embeddings: first column is node id (could be string), rest floats
            emb_raw = np.loadtxt(embedding_path, dtype=str)
            emb_raw = np.atleast_2d(emb_raw)
            emb_ids = emb_raw[:, 0]
            emb_vecs = emb_raw[:, 1:].astype(float)
            emb_dict = {emb_ids[i]: emb_vecs[i] for i in range(len(emb_ids))}

            # edges: allow header or no header; keep u,v as string, w as float
            df = pd.read_csv(edge_file_path)
            if df.shape[1] >= 3 and set(['u', 'v', 'w']).issubset(df.columns):
                df = df[['u', 'v', 'w']]
            else:
                df = pd.read_csv(edge_file_path, header=None, names=['u', 'v', 'w'])

            df['u'] = df['u'].astype(str).str.strip()
            df['v'] = df['v'].astype(str).str.strip()
            df['w'] = pd.to_numeric(df['w'], errors='coerce')
            df = df.dropna(subset=['w'])
            df = df.astype({'w': float})
        except Exception as e:
            print(f"[Error] 读取文件失败: {e}")
            return None, None

        route_map = {}
        for _, row in df.iterrows():
            u, v, w = row['u'], row['v'], float(row['w'])
            if u not in emb_dict or v not in emb_dict:
                continue

            struct_feat = emb_dict[u] * emb_dict[v]
            key = (u, v)
            if key not in route_map:
                route_map[key] = {
                    'struct_feat': struct_feat,
                    'weight_sum': 0.0,
                    'u': u,
                    'v': v,
                }
            route_map[key]['weight_sum'] += w

        if not route_map:
            return None, None

        eps = 1e-8
        features = []
        aggregated_rows = []

        prev_weight_values = [float(w) for w in self.prev_weights.values() if w is not None and float(w) > 0]
        weight_scale = float(np.median(prev_weight_values)) if prev_weight_values else 1.0

        # 使用并集：当期出现的边 + 上期有货运量的边（用于识别“本期消失”）
        candidate_keys = set(route_map.keys())
        for key, prev_w in self.prev_weights.items():
            if prev_w > 0:
                candidate_keys.add(key)

        for key in candidate_keys:
            u, v = key
            if u not in emb_dict or v not in emb_dict:
                continue

            in_current = key in route_map
            current_w = route_map[key]['weight_sum'] if in_current else 0.0
            prev_w = self.prev_weights.get(key)
            prev_w_for_feature = current_w if prev_w is None else prev_w

            # 对称的相对变化特征：|log(w_t / w_{t-1})|，并进行截断抑制极端值
            raw_log_ratio = np.log((current_w + eps) / (prev_w_for_feature + eps))
            log_ratio = float(np.clip(raw_log_ratio, -self.ratio_clip, self.ratio_clip))
            abs_log_ratio = abs(log_ratio)
            # 与上一期同航线相比的对数变化率: log(w_t / w_{t-1})
            log_ratio_vs_prev = np.nan if prev_w is None else log_ratio
            downward_log_ratio = np.nan if prev_w is None else max(0.0, -log_ratio)
            is_disappeared = int((not in_current) and (prev_w is not None) and (prev_w > 0))
            prev_score = self.prev_anomaly_scores.get(key, np.nan)

            prev_w_for_weight = 0.0 if prev_w is None else float(prev_w)
            downward_for_feature = 0.0 if prev_w is None else max(0.0, -log_ratio)
            weighted_downward_log_ratio = downward_for_feature * np.log1p(prev_w_for_weight / max(weight_scale, eps))

            struct_feat = route_map[key]['struct_feat'] if in_current else (emb_dict[u] * emb_dict[v])
            combined_feat = np.append(struct_feat, [abs_log_ratio, weighted_downward_log_ratio])
            features.append(combined_feat)
            aggregated_rows.append({
                'u': u,
                'v': v,
                'w': current_w,
                'prev_weight': prev_w,
                'prev_anomaly_score': prev_score,
                'weight_log_ratio_vs_prev': log_ratio_vs_prev,
                'downward_log_ratio': downward_log_ratio,
                'weighted_downward_log_ratio': weighted_downward_log_ratio,
                'is_disappeared': is_disappeared,
            })

        return np.array(features), pd.DataFrame(aggregated_rows)

    def process_month(self, embedding_path, edge_file_path, output_path):
        current_file = os.path.basename(edge_file_path)
        print(f"\n>>> [K-Means] 处理文件: {current_file}")

        X, df_result = self._extract_features(embedding_path, edge_file_path)
        if X is None: return

        # 归一化
        if not self.is_initialized:
            self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        if self.distance_metric == "cosine":
            X_used = self._l2_normalize(X_scaled)
        else:
            X_used = X_scaled

        # 初始
        if not self.is_initialized:
            print("  -> 初始化基准模型 (Standard K-Means)...")
            kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            kmeans.fit(X_used)

            # 保存中心和计数
            self.centers = kmeans.cluster_centers_
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            self.cluster_counts = np.zeros(self.k)
            for u, c in zip(unique, counts):
                self.cluster_counts[u] = c

            self.is_initialized = True

            scores = self._calculate_anomaly_score(X_used)

        else:
            print("  -> 流式检测与更新 (Streaming Update)...")

            # 离现有中心越远越异常
            scores = self._calculate_anomaly_score(X_used)

            # 更新中心
            self._update_centers(X_used)

        df_result['anomaly_score'] = scores
        df_result['anomaly_score_delta_vs_prev'] = df_result['anomaly_score'] - df_result['prev_anomaly_score']
        anomalies = df_result.sort_values(by='anomaly_score_delta_vs_prev', ascending=False, na_position='last').head(10)

        print(f"  -> Top 3 异常候选:")
        print(anomalies[['u', 'v', 'w', 'anomaly_score', 'prev_anomaly_score', 'anomaly_score_delta_vs_prev']].head(3).to_string(index=False))

        output_name = f"result_kmeans_{current_file}"
        save_path = os.path.join(output_path, output_name)
        df_result.sort_values(by='anomaly_score_delta_vs_prev', ascending=False, na_position='last').to_csv(save_path, index=False)
        print(f"  -> 结果已保存: {save_path}")

        # 仅保留本期有货运量的航线，供下一期对比（避免长期 0 权重航线持续累积）
        new_prev_weights = {}
        new_prev_scores = {}
        for _, row in df_result.iterrows():
            w = float(row['w'])
            if w <= 0:
                continue
            key = (row['u'], row['v'])
            new_prev_weights[key] = w
            new_prev_scores[key] = float(row['anomaly_score'])
        self.prev_weights = new_prev_weights
        self.prev_anomaly_scores = new_prev_scores

    def _calculate_anomaly_score(self, X):
        """
        计算每个样本到最近中心的距离
        """
        if self.distance_metric == "cosine":
            similarities = np.dot(X, self.centers.T)
            return 1.0 - np.max(similarities, axis=1)
        dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.min(dists, axis=1)

    def _update_centers(self, X):
        """
        动态更新中心
        """
        if self.distance_metric == "cosine":
            similarities = np.dot(X, self.centers.T)
            labels = np.argmax(similarities, axis=1)
        else:
            dists = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
            labels = np.argmin(dists, axis=1)

        new_counts = np.zeros(self.k)
        new_sums = np.zeros_like(self.centers)

        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_counts[i] = len(cluster_points)
                new_sums[i] = np.sum(cluster_points, axis=0)

        # c_new = (alpha * c_old * n_old + (1-alpha) * sum_new) / (alpha * n_old + (1-alpha) * n_new)

        for i in range(self.k):
            if new_counts[i] > 0:
                n_old = self.cluster_counts[i]
                n_new = new_counts[i]
                c_old = self.centers[i]
                sum_new = new_sums[i]  # 即 centroid_new * n_new

                numerator = self.alpha * c_old * n_old + (1 - self.alpha) * sum_new
                denominator = self.alpha * n_old + (1 - self.alpha) * n_new

                self.centers[i] = numerator / denominator
                self.cluster_counts[i] += n_new
            if self.distance_metric == "cosine":
                self.centers = self._l2_normalize(self.centers)

# 数据需要对齐
if __name__ == "__main__":
    EMBEDDING_DIR = "./tmp"
    DATA_DIR = "./data/monthly_snapshots"

    emb_files = sorted(glob.glob(os.path.join(EMBEDDING_DIR, "embedding_snapshot_*.txt")))
    raw_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    # 论文中 K=10, Alpha=0.5
    detector = StreamingKMeansDetector(k=10, alpha=0.5)

    print("开始 K-Means 动态检测对比实验...")
    for emb, raw in zip(emb_files, raw_files):
        detector.process_month(emb, raw)
