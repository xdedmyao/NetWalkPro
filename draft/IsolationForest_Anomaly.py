import numpy as np
import pandas as pd
import glob
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class ShippingAnomalyDetector:
    def __init__(self, contamination=0.01):
        """
        :param contamination: 预计数据中异常点的比例 (例如 1% 的航线是异常的)
        """
        self.clf = IsolationForest(
            n_estimators=100, # number of iTrees
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler()

    def process_snapshot(self, embedding_path, edge_file_path):
        """
        处理单个时间快照
        """
        # 1. 加载 Embedding (跳过第一列ID)
        try:
            emb_data = np.loadtxt(embedding_path)
            emb_dict = {int(row[0]): row[1:] for row in emb_data}
        except Exception as e:
            print(f"读取向量文件失败: {e}")
            return

        # 原始数据：(u, v, w)
        df = pd.read_csv(edge_file_path)

        # features
        features = []
        valid_indices = []

        for idx, row in df.iterrows():
            u, v, w = int(row['u']), int(row['v']), float(row['w'])

            if u in emb_dict and v in emb_dict:
                # A. 结构特征: Hadamard Product——原航线是否存在
                struct_feat = emb_dict[u] * emb_dict[v]

                # B. 流量特征——原始权重是否突变
                # 结构向量和权重拼接
                combined_feat = np.append(struct_feat, w)

                features.append(combined_feat)
                valid_indices.append(idx)

        if not features:
            print("没有匹配到有效的航线数据。")
            return

        X = np.array(features)

        # 数据归一化
        X = self.scaler.fit_transform(X)

        # fit_predict 返回 1 (正常) 和 -1 (异常)
        self.clf.fit(X)
        preds = self.clf.predict(X)

        # 异常评分：分数越高越异常
        scores = -self.clf.decision_function(X)

        # print(f"\n>>> 处理文件: {os.path.basename(edge_file_path)}")

        result_df = df.iloc[valid_indices].copy()
        result_df['anomaly_score'] = scores
        result_df['is_anomaly'] = preds

        # 筛选出被判定为异常的航线 (-1)
        anomalies = result_df[result_df['is_anomaly'] == -1].sort_values(by='anomaly_score', ascending=False)

        if not anomalies.empty:
            print(f"检测到 {len(anomalies)} 条异常航线 (Top 5):")
            print(anomalies[['u', 'v', 'w', 'anomaly_score']].head(5).to_string(index=False))

            # 保存结果到 CSV
            output_name = f"result_{os.path.basename(edge_file_path)}"
            anomalies.to_csv(os.path.join("./src/tmp", output_name), index=False)
            print(f"异常报告已保存至 ./src/tmp/{output_name}")
        else:
            print("未检测到明显异常。")

# 待修改：与数据文件对齐
if __name__ == "__main__":
    EMBEDDING_DIR = "./src/tmp"
    DATA_DIR = "./data"

    emb_files = sorted(glob.glob(os.path.join(EMBEDDING_DIR, "embedding_snapshot_*.txt")))
    raw_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    detector = ShippingAnomalyDetector(contamination=0.01)  # 假设 1% 异常率

    print(f"找到 {len(emb_files)} 个向量文件和 {len(raw_files)} 个原始数据文件。")

    for emb_file, raw_file in zip(emb_files, raw_files):
        detector.process_snapshot(emb_file, raw_file)