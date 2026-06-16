"""
    异常结果分析脚本：用于评估检测效果与分数上升对比
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SNAPSHOT_NAME = "snapshot_2025-05-21.csv"
LABELS_PATH = ROOT / "data" / "injected" / "snapshot_2025-05-21_labels.csv"
BASELINE_RESULT = ROOT / "output" / "anomalies_baseline_snapshot" / f"result_kmeans_{SNAPSHOT_NAME}"
INJECTED_RESULT = ROOT / "output" / "anomalies" / f"result_kmeans_{SNAPSHOT_NAME}"
OUTPUT_COMPARE = ROOT / "output" / "anomalies_compare" / "injected_score_delta.csv"
OUTPUT_METRICS = ROOT / "output" / "anomalies_compare" / "injected_score_delta_metrics.csv"

import pandas as pd


def evaluate_detection(
    labels_path: Path,
    result_path: Path,
    top_n: int = None,
    metrics_out: Path = None,
    detected_out: Path = None
):
    labels_df = pd.read_csv(labels_path)
    result_df = pd.read_csv(result_path)

    injected_set = set(zip(labels_df['u'].astype(str), labels_df['v'].astype(str)))
    n_injected = len(injected_set)

    if top_n is None:
        top_n = n_injected

    result_df = result_df.sort_values('anomaly_score', ascending=False).head(top_n)
    detected_set = set(zip(result_df['u'].astype(str), result_df['v'].astype(str)))

    true_positives = len(injected_set & detected_set)
    false_positives = len(detected_set - injected_set)
    false_negatives = len(injected_set - detected_set)

    precision = true_positives / len(detected_set) if detected_set else 0
    recall = true_positives / n_injected if n_injected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'n_injected': n_injected,
        'top_n': top_n,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"\n[EVALUATION] 检测效果评估 (Top-{top_n})")
    print(f"  注入异常数: {n_injected}")
    print(f"  正确检测 (TP): {true_positives}")
    print(f"  误报 (FP): {false_positives}")
    print(f"  漏检 (FN): {false_negatives}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")

    if true_positives > 0:
        print(f"\n[DETAILS] 正确检测到的注入异常:")
        result_df['u'] = result_df['u'].astype(str)
        result_df['v'] = result_df['v'].astype(str)
        detected_injected = result_df[
            result_df.apply(lambda r: (r['u'], r['v']) in injected_set, axis=1)
        ]
        cols_to_display = [c for c in ['u', 'v', 'w', 'anomaly_score'] if c in detected_injected.columns]
        print(detected_injected[cols_to_display].head(10).to_string(index=False))
    else:
        detected_injected = result_df.iloc[0:0]

    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
        print(f"\n[OUTPUT] 评估指标已保存: {metrics_out}")

    if detected_out is not None:
        detected_out.parent.mkdir(parents=True, exist_ok=True)
        cols_to_save = [c for c in ['u', 'v', 'w', 'anomaly_score'] if c in detected_injected.columns]
        detected_injected[cols_to_save].to_csv(detected_out, index=False)
        print(f"[OUTPUT] 检测命中结果已保存: {detected_out}")

    return metrics


def compare_anomaly_scores(
    base_result_path: Path,
    injected_result_path: Path,
    output_path: Path,
    min_delta: float = 0.0,
    top_n: int = None,
    labels_path: Path = None,
    metrics_out: Path = None
):
    base_df = pd.read_csv(base_result_path)
    injected_df = pd.read_csv(injected_result_path)

    for df in (base_df, injected_df):
        df['u'] = df['u'].astype(str)
        df['v'] = df['v'].astype(str)

    merged = pd.merge(
        injected_df[['u', 'v', 'w', 'anomaly_score']],
        base_df[['u', 'v', 'anomaly_score']],
        on=['u', 'v'],
        how='inner',
        suffixes=('_injected', '_base')
    )
    merged['score_delta'] = merged['anomaly_score_injected'] - merged['anomaly_score_base']

    increased = merged[merged['score_delta'] > min_delta].sort_values('score_delta', ascending=False)
    if top_n is not None:
        increased = increased.head(top_n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    increased.to_csv(output_path, index=False)

    print(f"\n[COMPARE] 对比完成")
    print(f"  原始结果: {base_result_path}")
    print(f"  注入结果: {injected_result_path}")
    print(f"  分数上升数量: {len(increased)}")
    print(f"  输出文件: {output_path}")

    if labels_path is not None:
        labels_df = pd.read_csv(labels_path)
        injected_set = set(zip(labels_df['u'].astype(str), labels_df['v'].astype(str)))
        n_injected = len(injected_set)

        increased_set = set(zip(
            merged.loc[merged['score_delta'] > min_delta, 'u'].astype(str),
            merged.loc[merged['score_delta'] > min_delta, 'v'].astype(str)
        ))

        increased_injected = len(injected_set & increased_set)
        ratio_increased = increased_injected / n_injected if n_injected else 0

        print(f"  注入异常总数: {n_injected}")
        print(f"  分数上升的注入异常数: {increased_injected}")
        print(f"  分数上升比例: {ratio_increased:.2%}")

        if metrics_out is not None:
            metrics_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                'n_injected': n_injected,
                'increased_injected': increased_injected,
                'ratio_increased': ratio_increased,
                'min_delta': min_delta
            }]).to_csv(metrics_out, index=False)
            print(f"  指标已保存: {metrics_out}")


def compare_injected_routes_same_snapshot(
    labels_path: Path,
    baseline_result_path: Path,
    injected_result_path: Path,
    output_path: Path,
    metrics_out: Path = None,
    min_delta: float = 0.0
):
    labels_df = pd.read_csv(labels_path)
    baseline_df = pd.read_csv(baseline_result_path)
    injected_df = pd.read_csv(injected_result_path)

    for df in (labels_df, baseline_df, injected_df):
        df['u'] = df['u'].astype(str)
        df['v'] = df['v'].astype(str)

    injected_set = set(zip(labels_df['u'], labels_df['v']))

    baseline_injected = baseline_df[baseline_df.apply(lambda r: (r['u'], r['v']) in injected_set, axis=1)]
    injected_injected = injected_df[injected_df.apply(lambda r: (r['u'], r['v']) in injected_set, axis=1)]

    merged = pd.merge(
        injected_injected[['u', 'v', 'anomaly_score']],
        baseline_injected[['u', 'v', 'anomaly_score']],
        on=['u', 'v'],
        how='inner',
        suffixes=('_injected', '_baseline')
    )
    merged['score_delta'] = merged['anomaly_score_injected'] - merged['anomaly_score_baseline']

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    increased_injected = (merged['score_delta'] > min_delta).sum()
    total_injected = len(injected_set)
    ratio_increased = increased_injected / total_injected if total_injected else 0

    print(f"\n[COMPARE-SNAPSHOT] 对比完成")
    print(f"  基线结果: {baseline_result_path}")
    print(f"  注入结果: {injected_result_path}")
    print(f"  注入异常总数: {total_injected}")
    print(f"  分数上升的注入异常数: {int(increased_injected)}")
    print(f"  分数上升比例: {ratio_increased:.2%}")
    print(f"  输出文件: {output_path}")

    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            'n_injected': total_injected,
            'increased_injected': int(increased_injected),
            'ratio_increased': ratio_increased,
            'min_delta': min_delta
        }]).to_csv(metrics_out, index=False)
        print(f"  指标已保存: {metrics_out}")


def main():
    if not LABELS_PATH.exists():
        print(f"[ERROR] 标注文件不存在: {LABELS_PATH}")
        sys.exit(1)
    if not BASELINE_RESULT.exists():
        print(f"[ERROR] 基线结果文件不存在: {BASELINE_RESULT}")
        sys.exit(1)
    if not INJECTED_RESULT.exists():
        print(f"[ERROR] 注入结果文件不存在: {INJECTED_RESULT}")
        sys.exit(1)

    compare_injected_routes_same_snapshot(
        labels_path=LABELS_PATH,
        baseline_result_path=BASELINE_RESULT,
        injected_result_path=INJECTED_RESULT,
        output_path=OUTPUT_COMPARE,
        metrics_out=OUTPUT_METRICS,
        min_delta=0.0
    )


if __name__ == "__main__":
    main()
