"""
    异常注入脚本：用于生成与记录人工注入异常

    功能：
    1. 从指定快照中随机抽取一定比例的航线
    2. 将这些航线的 weight 增加为原来的 5 倍
    3. 生成注入后的快照和标注文件

    使用示例：
        python inject_anomalies.py
"""
import sys
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODE = "increase"  # "increase" or "decrease"
SNAPSHOT_NAME = "snapshot_2025-05-21.csv"
INPUT_DIR = ROOT / "data" / "snapshots"
BASELINE_RESULT_DIR = ROOT / "output" / "anomalies_baseline_snapshot"
LABELS_DIR = ROOT / "data" / "injected"



def inject_anomalies(
    snapshot_path: Path,
    output_dir: Path,
    ratio: float = 0.05,
    scale_factor: float = 5.0,
    seed: int = 42,
    anomaly_type: str = "increase"
):
    """
    在快照中注入异常值

    Args:
        snapshot_path: 原始快照文件路径
        output_dir: 输出目录
        ratio: 注入异常的航线比例 (0.0-1.0)
        scale_factor: 权重缩放因子 (5 = 增加为 5 倍)
        seed: 随机种子
        anomaly_type: 异常类型 "decrease" 或 "increase"

    Returns:
        injected_df: 注入异常后的 DataFrame
        labels_df: 标注信息 DataFrame
    """
    np.random.seed(seed)

    df = pd.read_csv(snapshot_path)
    print(f"[INFO] 读取快照: {snapshot_path.name}")
    print(f"[INFO] 原始航线数: {len(df)}")

    coord_cols = [c for c in df.columns if c not in ["u", "v", "w"]]
    if coord_cols:
        agg_dict = {"w": "sum"}
        for col in coord_cols:
            agg_dict[col] = "first"
        df_agg = df.groupby(["u", "v"], as_index=False).agg(agg_dict)
    else:
        df_agg = df.groupby(["u", "v"], as_index=False)["w"].sum()

    n_routes = len(df_agg)
    n_inject = int(n_routes * ratio)
    print(f"[INFO] 唯一航线数: {n_routes}")
    print(f"[INFO] 注入异常数: {n_inject} ({ratio*100:.1f}%)")

    inject_indices = np.random.choice(n_routes, size=n_inject, replace=False)

    df_agg["is_injected"] = False
    df_agg["original_w"] = df_agg["w"]

    for idx in inject_indices:
        original_w = df_agg.loc[idx, "w"]
        if anomaly_type == "decrease":
            new_w = original_w / scale_factor
        else:
            new_w = original_w * scale_factor
        df_agg.loc[idx, "w"] = new_w
        df_agg.loc[idx, "is_injected"] = True

    labels_df = df_agg[df_agg["is_injected"]][["u", "v", "original_w", "w"]].copy()
    labels_df.columns = ["u", "v", "original_w", "injected_w"]
    labels_df["anomaly_type"] = anomaly_type
    labels_df["scale_factor"] = scale_factor

    output_cols = ["u", "v", "w"] + coord_cols
    injected_df = df_agg[output_cols].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_name = snapshot_path.stem
    injected_path = snapshot_path
    labels_path = output_dir / f"{snapshot_name}_labels.csv"

    injected_df.to_csv(injected_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    print(f"\n[OUTPUT] 注入后快照: {injected_path}")
    print(f"[OUTPUT] 标注文件: {labels_path}")

    print(f"\n[STATS] 被注入航线示例:")
    print(labels_df.head(10).to_string(index=False))

    return injected_df, labels_df


def main():
    snapshot_path = INPUT_DIR / SNAPSHOT_NAME

    if not snapshot_path.exists():
        print(f"[ERROR] 快照文件不存在: {snapshot_path}")
        sys.exit(1)

    BASELINE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_result = ROOT / "output" / "anomalies" / f"result_kmeans_{SNAPSHOT_NAME}"
    if not baseline_result.exists():
        print(f"[ERROR] 基线结果文件不存在: {baseline_result}")
        sys.exit(1)
    baseline_copy = BASELINE_RESULT_DIR / baseline_result.name
    copy2(baseline_result, baseline_copy)
    print(f"[OUTPUT] 基线结果已备份: {baseline_copy}")

    print("[STEP] 注入异常并替换原始快照")
    inject_anomalies(
        snapshot_path=snapshot_path,
        output_dir=LABELS_DIR,
        ratio=0.05,
        scale_factor=5.0,
        seed=42,
        anomaly_type=MODE,
    )


if __name__ == "__main__":
    main()
