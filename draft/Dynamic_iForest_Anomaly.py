import numpy as np
import pandas as pd
import glob
import os
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class DynamicShippingDetector:
    def __init__(self, contamination=0.01, window_size=3):
        """
        :param contamination: 预计数据中异常点的比例
        :param window_size: 滑动窗口大小，过去月份的数量
        """
        self.window_size = window_size
        self.history_buffer = deque(maxlen=window_size)

        # Isolation Forest 模型
        self.clf = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler()

    def _extract_features(self, embedding_path, edge_file_path):
        """
        辅助函数：仅负责加载文件并提取特征，不进行训练
        """
        try:
            emb_data = np.loadtxt(embedding_path)
            emb_dict = {int(row[0]): row[1:] for row in emb_data}
        except Exception as e:
            print(f"[Error] 读取向量文件失败 {embedding_path}: {e}")
            return None, None

        try:
            df = pd.read_csv(edge_file_path, header=None, names=['u', 'v', 'w'])
        except Exception as e:
            print(f"[Error] 读取CSV失败 {edge_file_path}: {e}")
            return None, None

        features = []
        valid_indices = []

        for idx, row in df.iterrows():
            u, v, w = int(row['u']), int(row['v']), float(row['w'])
            if u in emb_dict and v in emb_dict:
                # 结构特征 (Hadamard Product)
                struct_feat = emb_dict[u] * emb_dict[v]
                # 流量特征
                combined_feat = np.append(struct_feat, w)

                features.append(combined_feat)
                valid_indices.append(idx)

        if not features:
            return None, None

        return np.array(features), df.iloc[valid_indices].copy()

    def process_month(self, embedding_path, edge_file_path):
        """
        执行单个月份的检测（带滑动窗口逻辑）
        """
        current_file_name = os.path.basename(edge_file_path)
        print(f"\n>>> 正在处理: {current_file_name}")

        X_current, df_result = self._extract_features(embedding_path, edge_file_path)
        if X_current is None:
            print("  -> 跳过 (数据无效)")
            return

        # 历史月份窗口 + 本月
        if len(self.history_buffer) > 0:
            X_history = np.vstack(list(self.history_buffer))
            X_train = np.vstack([X_history, X_current])
            print(f"  -> 训练模式: 滑动窗口 (历史样本 {len(X_history)} + 当前样本 {len(X_current)})")
        else:
            X_train = X_current
            print(f"  -> 训练模式: 冷启动 (仅使用当前样本 {len(X_current)})")

        # 数据归一化，使用整个滑动窗口中的数据
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_current_scaled = self.scaler.transform(X_current)

        # Fit包含历史数据和当月数据
        self.clf.fit(X_train_scaled)

        # 只 Predict 本月
        scores = -self.clf.decision_function(X_current_scaled)
        preds = self.clf.predict(X_current_scaled)

        # 更新滑动窗口
        self.history_buffer.append(X_current)

        df_result['anomaly_score'] = scores
        df_result['is_anomaly'] = preds

        anomalies = df_result[df_result['is_anomaly'] == -1].sort_values(by='anomaly_score', ascending=False)

        if not anomalies.empty:
            print(f"  -> [警报] 检测到 {len(anomalies)} 条异常航线 (Top 3):")
            print(anomalies[['u', 'v', 'w', 'anomaly_score']].head(3).to_string(index=False))

            # 保存
            output_name = f"result_window_{current_file_name}"
            save_path = os.path.join("./src/tmp", output_name)
            anomalies.to_csv(save_path, index=False)
            print(f"  -> 报告已保存: {save_path}")
        else:
            print("  -> 本月无明显异常。")

# 是否可以仿照NetWalk中的加权，将历史数据合并？ alpha * (history) + (1 - alpha) * current

# 与data结构待整合
if __name__ == "__main__":
    EMBEDDING_DIR = "./src/tmp"
    DATA_DIR = "./data"
    #
    # emb_files = sorted(glob.glob(os.path.join(EMBEDDING_DIR, "embedding_snapshot_*.txt")))
    # raw_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    #
    # if len(emb_files) != len(raw_files):
    #     print(f"警告: 向量文件数({len(emb_files)}) 与 原始文件数({len(raw_files)}) 不匹配")
    #
    # detector = DynamicShippingDetector(contamination=0.01, window_size=3)
    #
    # print(f"开始时间序列检测 (滑动窗口大小: {detector.window_size})...")
    #
    # for emb_file, raw_file in zip(emb_files, raw_files):
    #     detector.process_month(emb_file, raw_file)