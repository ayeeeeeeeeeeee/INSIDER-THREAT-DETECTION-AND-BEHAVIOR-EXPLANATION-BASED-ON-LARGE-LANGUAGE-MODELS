
import sys
import os
import json
import logging
import argparse

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = _current_dir
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import setup_logging

setup_logging("e2e_inference_test", __file__)
logger = logging.getLogger(__name__)

from explanation import ReportGenerator
from config import BASE_MODEL_PATH

from evaluate_detection import DetectionEvaluator
from evaluation import EvaluationDataLoader


def test_with_mock(session_id: str, evaluator: DetectionEvaluator):

    print("\n" + "=" * 80)
    print("【Mock模式测试】")
    print("=" * 80)

    print("\n[1/4] 加载评估数据...")
    ground_truth, eval_sequences, all_sessions = EvaluationDataLoader.load_evaluation_data()

    if session_id not in eval_sequences:
        raise ValueError(f"会话 {session_id} 不存在于评估数据中")

    sequence = eval_sequences[session_id]
    user_id = session_id.rsplit('_', 1)[0]
    print(f"  ✓ 加载 {len(sequence)} 条行为事件")

    print("\n[2/4] 构建用户画像模块...")
    evaluator.profile_module, _ = evaluator.build_user_profile_module(
        all_sessions=all_sessions,
        eval_sequences=eval_sequences
    )
    user_profile_text = evaluator.profile_module.get_llm_profile_text(user_id)
    print(f"  ✓ 用户画像构建完成")

    print("\n[3/4] 语义异常检测...")
    semantic_results = evaluator._run_semantic_detection(
        all_sessions=[session_id],
        eval_sequences={session_id: sequence},
        semantic_dir="./evaluation_results/evidences/semantic",
    )
    semantic_dict = semantic_results.get(session_id)
    if semantic_dict:
        print(f"  ✓ 检测到 {semantic_dict['total_anomalies']} 条语义异常")
    else:
        print("  ⚠️ 未检测到语义异常")

    print("\n[4/4] 统计异常检测...")
    all_users = [user_id]  
    stat_result = evaluator.run_statistical_detection(
        all_users=all_users,
        all_sessions=[session_id],
        eval_sequences={session_id: sequence},
        profile_module=evaluator.profile_module,
        statistical_dir="./evaluation_results/evidences/statistical",
    )
    stat_dict = stat_result['results'].get(session_id)
    if stat_dict:
        print(f"  ✓ 检测到 {stat_dict['summary']['total_anomaly_days']} 天统计异常")
    else:
        print("  ⚠️ 未检测到统计异常")

    from evidence_aggregation import MultimodalEvidenceAggregator
    multimodal_agg = MultimodalEvidenceAggregator()
    multimodal_result = multimodal_agg.aggregate(
        user_id=session_id,
        semantic_evidence=semantic_dict,
        statistical_evidence=stat_dict
    )
    multimodal_evidence = multimodal_result.to_dict(max_evidences_per_day=5)

    print(f"\n  证据聚合完成:")
    print(f"    总证据数: {multimodal_evidence['summary']['total_count']}")

    from explanation.llm_reasoning_engine import create_mock_engine

    print("\n[推理] 使用Mock引擎...")
    engine = create_mock_engine()
    result = engine.reason(multimodal_evidence, user_profile_text)

    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    print(f"会话: {result.user_id}")
    print(f"威胁判定: {'是' if result.is_threat else '否'}")
    print(f"威胁类型: {result.threat_type}")
    print(f"威胁分数: {result.threat_score:.2f}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"验证状态: {result.validation_status.value}")

    gt = ground_truth.get(session_id, {})
    if gt:
        true_label = '恶意' if gt.get('is_malicious', False) else '正常'
        pred_label = '恶意' if result.is_threat else '正常'
        print(f"\n真实标签: {true_label}")
        print(f"预测标签: {pred_label}")
        print(f"{'✅ 预测正确' if true_label == pred_label else '❌ 预测错误'}")

    print("\n" + "=" * 60)
    print("生成分析报告")
    print("=" * 60)

    generator = ReportGenerator()
    md_report = generator.generate_markdown(result, multimodal_evidence, user_profile_text)
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, f"report_{session_id}_mock.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"报告已保存: {md_path}")

    print("\n报告摘要:")
    print(md_report[:1000] + "...")

    return result, md_report

def test_with_real_model(session_id: str, evaluator: DetectionEvaluator, model_path: str):

    print("\n" + "=" * 80)
    print("【真实模型测试】")
    print("=" * 80)

    print("\n[1/4] 加载评估数据...")
    ground_truth, eval_sequences, all_sessions = EvaluationDataLoader.load_evaluation_data()

    if session_id not in eval_sequences:
        raise ValueError(f"会话 {session_id} 不存在于评估数据中")

    sequence = eval_sequences[session_id]
    user_id = session_id.rsplit('_', 1)[0]
    print(f"  ✓ 加载 {len(sequence)} 条行为事件")

    print("\n[2/4] 构建用户画像模块...")
    evaluator.profile_module, _ = evaluator.build_user_profile_module(
        all_sessions=all_sessions,
        eval_sequences=eval_sequences
    )
    user_profile_text = evaluator.profile_module.get_llm_profile_text(user_id)
    print(f"  ✓ 用户画像构建完成")

    print("\n[3/4] 语义异常检测（加载语义模型）...")
    semantic_results = evaluator._run_semantic_detection(
        all_sessions=[session_id],
        eval_sequences={session_id: sequence},
        semantic_dir="./evaluation_results/evidences/semantic",
    )
    semantic_dict = semantic_results.get(session_id)
    if semantic_dict:
        print(f"  ✓ 检测到 {semantic_dict['total_anomalies']} 条语义异常")
    else:
        print("  ⚠️ 未检测到语义异常")

    print("  ✓ 语义模型已自动卸载")

    print("\n[4/4] 统计异常检测...")
    all_users = [user_id]  
    stat_result = evaluator.run_statistical_detection(
        all_users=all_users,
        all_sessions=[session_id],
        eval_sequences={session_id: sequence},
        profile_module=evaluator.profile_module,
        statistical_dir="./evaluation_results/evidences/statistical",
    )
    stat_dict = stat_result['results'].get(session_id)
    if stat_dict:
        print(f"  ✓ 检测到 {stat_dict['summary']['total_anomaly_days']} 天统计异常")
    else:
        print("  ⚠️ 未检测到统计异常")

    from evidence_aggregation import MultimodalEvidenceAggregator
    multimodal_agg = MultimodalEvidenceAggregator()
    multimodal_result = multimodal_agg.aggregate(
        user_id=session_id,
        semantic_evidence=semantic_dict,
        statistical_evidence=stat_dict
    )
    multimodal_evidence = multimodal_result.to_dict(max_evidences_per_day=5)

    print(f"\n  证据聚合完成:")
    print(f"    总证据数: {multimodal_evidence['summary']['total_count']}")

    print("\n[推理] 加载COT模型...")
    evaluator.model_path = model_path
    evaluator._load_cot_model()
    engine = evaluator.engine

    try:
        result = engine.reason(multimodal_evidence, user_profile_text)

        print("\n" + "=" * 60)
        print("推理结果")
        print("=" * 60)
        print(f"会话: {result.user_id}")
        print(f"威胁判定: {'是' if result.is_threat else '否'}")
        print(f"威胁类型: {result.threat_type}")
        print(f"威胁分数: {result.threat_score:.2f}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"验证状态: {result.validation_status.value}")
        print(f"重试次数: {result.retry_count}")

        if result.validation_errors:
            print(f"验证错误: {result.validation_errors}")
        if result.validation_warnings:
            print(f"验证警告: {result.validation_warnings}")

        gt = ground_truth.get(session_id, {})
        if gt:
            true_label = '恶意' if gt.get('is_malicious', False) else '正常'
            pred_label = '恶意' if result.is_threat else '正常'
            print(f"\n真实标签: {true_label}")
            print(f"预测标签: {pred_label}")
            print(f"{'✅ 预测正确' if true_label == pred_label else '❌ 预测错误'}")

        print("\n" + "=" * 60)
        print("推理日志（thinking）")
        print("=" * 60)
        print(result.reasoning_log[:1000] + "...")

        print("\n" + "=" * 60)
        print("生成分析报告")
        print("=" * 60)

        generator = ReportGenerator()
        os.makedirs(report_dir, exist_ok=True)

        md_report = generator.generate_markdown(result, multimodal_evidence, user_profile_text)
        md_path = os.path.join(report_dir, f"report_{session_id}_real.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"Markdown报告已保存: {md_path}")

        text_report = generator.generate_text(result, user_profile_text)
        txt_path = os.path.join(report_dir, f"report_{session_id}_real.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        print(f"文本报告已保存: {txt_path}")

        result_path = os.path.join(report_dir, f"result_{session_id}_real.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': session_id,
                'result': result.to_dict(),
                'evidence_summary': multimodal_evidence.get('summary', {}),
                'ground_truth': gt
            }, f, ensure_ascii=False, indent=2)
        print(f"完整结果已保存: {result_path}")

        evaluator._unload_current_model()

        return result, md_report

    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        print(f"\n❌ 推理失败: {e}")
        evaluator._unload_current_model()
        return None, None

def main():

    parser = argparse.ArgumentParser(description="端到端内部威胁检测推理测试（会话级）")
    parser.add_argument("--session", type=str, default="MCF0600_2010-09-20", help="会话ID（格式: {user_id}_{date}，如 BSS0369_2022-01-03）")
    parser.add_argument("--model-path", type=str,
                        default=BASE_MODEL_PATH,
                        help=f"DeepSeek模型路径（默认: {BASE_MODEL_PATH}）")
    parser.add_argument("--mock", action="store_true", help="使用Mock模式（不加载真实模型）")

    args = parser.parse_args()

    print("=" * 80)
    print("内部威胁检测 - 端到端推理测试（会话级）")
    print("=" * 80)
    print(f"会话: {args.session}")
    print(f"模式: {'Mock' if args.mock else '真实模型'}")
    if not args.mock:
        print(f"模型路径: {args.model_path}")
    print("=" * 80)

    evaluator = DetectionEvaluator(
        model_path=args.model_path,
        use_mock=args.mock
    )

    if args.mock:
        result, report = test_with_mock(args.session, evaluator)
    else:
        result, report = test_with_real_model(args.session, evaluator, args.model_path)

    if result:
        print("\n" + "=" * 80)
        print("✅ 测试完成")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 测试失败")
        print("=" * 80)

if __name__ == "__main__":
    main()