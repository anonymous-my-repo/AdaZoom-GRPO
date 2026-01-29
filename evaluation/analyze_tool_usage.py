import os
import json
import argparse
import re
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser(description="分析XLRSBench评估结果中的工具调用情况")
parser.add_argument("--eval_results", type=str, required=True, help="eval阶段的结果JSONL文件路径")
parser.add_argument("--judge_results", type=str, required=True, help="judge阶段的结果JSONL文件路径")
parser.add_argument("--output_dir", type=str, default=None, help="统计结果输出目录，默认为eval结果文件所在目录")
args = parser.parse_args()

# 初始化输出目录
if args.output_dir is None:
    args.output_dir = os.path.dirname(args.eval_results)
os.makedirs(args.output_dir, exist_ok=True)


def count_tool_calls(pred_output):
    """
    计算一个样本中的工具调用次数
    参数:
        pred_output: 模型输出的对话历史
    返回:
        工具调用次数
    """
    tool_calls = 0

    # 遍历对话历史，寻找工具调用模式
    if not pred_output:
        return 0

    for i, message in enumerate(pred_output):
        if message["role"] == "assistant" and "<tool_call>" in message["content"] and "</tool_call>" in message["content"]:
            tool_calls += 1

    return tool_calls


def analyze_results(eval_results_file, judge_results_file):
    """
    分析评估和判断结果文件，统计工具调用情况和正确率
    参数:
        eval_results_file: eval阶段结果JSONL文件路径
        judge_results_file: judge阶段结果JSONL文件路径
    返回:
        统计结果字典
    """
    # 初始化统计数据结构
    stats = {
        "total": {
            "sample_count": 0,  # 样本总数
            "tool_using_samples": 0,  # 使用工具的样本数
            "total_tool_calls": 0,  # 工具调用总次数
            "tool_calls_per_sample": [],  # 每个样本的工具调用次数
            "correct_samples": 0,  # 回答正确的样本数
            "tool_using_correct": 0,  # 使用工具且回答正确的样本数
            "non_tool_using_correct": 0,  # 不使用工具但回答正确的样本数
        }
    }

    # 读取judge结果文件，获取正确性判断
    judge_results = {}
    print(f"加载judge阶段结果: {judge_results_file}")
    with open(judge_results_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                sample_id = result.get("sample_id", "unknown")
                judge_results[sample_id] = result
            except json.JSONDecodeError:
                print(f"警告: 跳过无效JSON行")

    print(f"已加载 {len(judge_results)} 个judge结果")

    # 读取eval结果文件，获取工具调用信息
    samples = []
    print(f"加载eval阶段结果: {eval_results_file}")
    with open(eval_results_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                sample_id = sample.get("sample_id", "unknown")

                # 关联judge结果
                if sample_id in judge_results:
                    sample["correct"] = judge_results[sample_id].get("correct", False)
                    samples.append(sample)
                else:
                    print(f"警告: 样本 {sample_id} 在judge结果中未找到")
            except json.JSONDecodeError:
                print(f"警告: 跳过无效JSON行")

    print(f"成功关联并加载 {len(samples)} 个样本")

    # 分析每个样本
    for sample in tqdm(samples, desc="分析样本"):
        category = sample.get("category", "unknown")

        # 确保该类别在统计中存在
        if category not in stats:
            stats[category] = {
                "sample_count": 0,
                "tool_using_samples": 0,
                "total_tool_calls": 0,
                "tool_calls_per_sample": [],
                "correct_samples": 0,
                "tool_using_correct": 0,
                "non_tool_using_correct": 0,
            }

        # 计算工具调用次数
        tool_calls = 0
        if "pred_output" in sample:
            tool_calls = count_tool_calls(sample["pred_output"])

        # 更新总体统计
        stats["total"]["sample_count"] += 1
        stats["total"]["tool_calls_per_sample"].append(tool_calls)
        stats["total"]["total_tool_calls"] += tool_calls

        if tool_calls > 0:
            stats["total"]["tool_using_samples"] += 1

        # 更新类别统计
        stats[category]["sample_count"] += 1
        stats[category]["tool_calls_per_sample"].append(tool_calls)
        stats[category]["total_tool_calls"] += tool_calls

        if tool_calls > 0:
            stats[category]["tool_using_samples"] += 1

        # 统计正确率与工具使用的关系
        is_correct = sample.get("correct", False)
        if is_correct:
            stats["total"]["correct_samples"] += 1
            stats[category]["correct_samples"] += 1

            if tool_calls > 0:
                stats["total"]["tool_using_correct"] += 1
                stats[category]["tool_using_correct"] += 1
            else:
                stats["total"]["non_tool_using_correct"] += 1
                stats[category]["non_tool_using_correct"] += 1

    # 计算平均值和比例
    for cat in stats:
        cat_stats = stats[cat]
        sample_count = cat_stats["sample_count"]

        if sample_count > 0:
            # 平均每个样本的工具调用次数
            cat_stats["avg_tool_calls"] = cat_stats["total_tool_calls"] / sample_count

            # 使用工具的样本比例
            cat_stats["tool_usage_ratio"] = cat_stats["tool_using_samples"] / sample_count

            # 总体正确率
            cat_stats["accuracy"] = cat_stats["correct_samples"] / sample_count

            # 使用工具样本的正确率
            if cat_stats["tool_using_samples"] > 0:
                cat_stats["tool_using_accuracy"] = cat_stats["tool_using_correct"] / cat_stats["tool_using_samples"]
            else:
                cat_stats["tool_using_accuracy"] = 0

            # 不使用工具样本的正确率
            non_tool_samples = sample_count - cat_stats["tool_using_samples"]
            if non_tool_samples > 0:
                cat_stats["non_tool_using_accuracy"] = cat_stats["non_tool_using_correct"] / non_tool_samples
            else:
                cat_stats["non_tool_using_accuracy"] = 0

    return stats


def print_and_save_report(stats, output_dir):
    """
    打印并保存统计报告
    参数:
        stats: 统计结果字典
        output_dir: 输出目录
    """
    # 保存JSON格式的完整统计数据
    with open(os.path.join(output_dir, "tool_usage_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 准备报告内容
    report_lines = []
    report_lines.append("XLRSBench工具调用统计报告")
    report_lines.append("=" * 50)
    report_lines.append("")

    # 总体统计
    report_lines.append("总体统计")
    report_lines.append("-" * 30)
    total_stats = stats["total"]
    report_lines.append(f"样本总数: {total_stats['sample_count']}")
    report_lines.append(f"调用工具的样本数: {total_stats['tool_using_samples']} ({total_stats['tool_usage_ratio']:.2%})")
    report_lines.append(f"工具调用总次数: {total_stats['total_tool_calls']}")
    report_lines.append(f"平均每个样本调用工具次数: {total_stats['avg_tool_calls']:.2f}")
    report_lines.append(f"总体正确率: {total_stats['accuracy']:.2%}")
    report_lines.append(f"使用工具样本的正确率: {total_stats['tool_using_accuracy']:.2%}")
    report_lines.append(f"不使用工具样本的正确率: {total_stats['non_tool_using_accuracy']:.2%}")
    report_lines.append("")

    # 按类别统计
    report_lines.append("按类别统计")
    report_lines.append("-" * 30)

    # 按平均工具调用次数排序的类别
    categories = [cat for cat in stats.keys() if cat != "total"]
    categories.sort(key=lambda x: stats[x]["avg_tool_calls"], reverse=True)

    for category in categories:
        cat_stats = stats[category]
        report_lines.append(f"\n类别: {category}")
        report_lines.append(f"  样本数: {cat_stats['sample_count']}")
        report_lines.append(f"  调用工具的样本数: {cat_stats['tool_using_samples']} ({cat_stats['tool_usage_ratio']:.2%})")
        report_lines.append(f"  工具调用总次数: {cat_stats['total_tool_calls']}")
        report_lines.append(f"  平均每个样本调用工具次数: {cat_stats['avg_tool_calls']:.2f}")
        report_lines.append(f"  正确率: {cat_stats['accuracy']:.2%}")
        report_lines.append(f"  使用工具样本的正确率: {cat_stats['tool_using_accuracy']:.2%}")
        report_lines.append(f"  不使用工具样本的正确率: {cat_stats['non_tool_using_accuracy']:.2%}")

    # 保存报告到文件
    report_text = "\n".join(report_lines)
    with open(os.path.join(output_dir, "tool_usage_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # 打印到终端
    print(report_text)
    print(f"\n报告已保存至: {os.path.join(output_dir, 'tool_usage_report.txt')}")


if __name__ == "__main__":
    print(f"开始分析XLRSBench评估结果")
    stats = analyze_results(args.eval_results, args.judge_results)
    print_and_save_report(stats, args.output_dir)
    print("分析完成!")
