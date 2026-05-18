
import logging
from typing import Dict, List, Optional, Generator

from tqdm import tqdm

import config
from multi_modal_features.semantic.analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)

class SemanticDetectorWrapper:

    DETECTABLE_EVENT_TYPES = {'email', 'web', 'file'}

    def __init__(self, analyzer: SemanticAnalyzer, batch_size: int = 256):  

        self.analyzer = analyzer
        self.batch_size = batch_size
        logger.info("SemanticDetectorWrapper初始化完成")

    def _extract_content(self, event: Dict) -> Optional[str]:

        event_type = event.get('type', '').lower()

        if event_type in ('email', 'web', 'file'):
            content = event.get('content', '')
            return content if content else None

        return None

    def _extract_event_fields(self, event: Dict) -> Dict:

        event_type = event.get('type', '').lower()
        fields = {
            'user_id': event.get('user_id'),
            'timestamp': event.get('timestamp'),
        }

        if event_type == 'web':  
            fields['url'] = event.get('url')
        elif event_type == 'email':
            fields['to'] = event.get('to')
            fields['from'] = event.get('from')
            fields['cc'] = event.get('cc')
        elif event_type == 'file':
            fields['filename'] = event.get('filename')

        return {k: v for k, v in fields.items() if v is not None}

    def detect_user(self, user_id: str, behavior_sequence: List[Dict],
                user_profile: Optional[str] = None) -> List[Dict]:

        logger.info(f"开始检测用户 {user_id}，事件数: {len(behavior_sequence)}")

        detectable_events = []
        valid_indices = []  
        for idx, event in enumerate(behavior_sequence):
            event_type = event.get('type', '').lower()

            if event_type not in self.DETECTABLE_EVENT_TYPES:
                continue

            content = self._extract_content(event)
            if not content:
                continue

            event_fields = self._extract_event_fields(event)
            event_fields['user_id'] = user_id

            detectable_events.append({
                'event_type': event_type,
                'content': content,
                'event_fields': event_fields,
                'user_profile': user_profile
            })
            valid_indices.append(idx)  
        if not detectable_events:
            logger.info(f"用户 {user_id} 无可检测事件")
            return []

        total_events = len(detectable_events)
        logger.info(f"批量分析 {total_events} 个事件...")

        all_batch_results = []

        total_batches = (total_events + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, total_events, self.batch_size),
                      desc=f"语义检测 ({user_id})",
                      unit="batch",
                      total=total_batches):
            batch = detectable_events[i: i + self.batch_size]
            batch_results = self.analyzer.batch_analyze(batch)
            all_batch_results.extend(batch_results)

        if len(all_batch_results) != len(valid_indices):
            logger.error(f"批量分析结果数量({len(all_batch_results)})与输入数量({len(valid_indices)})不匹配")
            return []

        anomalies = []
        for idx, result in zip(valid_indices, all_batch_results):
            if result.get('is_anomaly'):
                event = behavior_sequence[idx]  
                anomaly_evidence = self._build_anomaly_evidence(event, user_id, result)
                anomalies.append(anomaly_evidence)

        return anomalies

    def _build_anomaly_evidence(self, event: Dict, user_id: str, result: Dict) -> Dict:

        event_type = event.get('type', '').lower()

        evidence = {

            'timestamp': event.get('timestamp'),
            'event_type': event_type,
            'user_id': user_id,
            'pc': event.get('pc'),
            'content': event.get('content', ''),  
            'event_id': event.get('event_id', ''),  

            'is_anomaly': True,
            'anomaly_score': result.get('anomaly_score', 0.0),
            'category': result.get('category', 'unknown'),
        }

        if 'key_evidence' in result:
            evidence['key_evidence'] = result.get('key_evidence', '')
        if 'explanation' in result:
            evidence['explanation'] = result.get('explanation', '')

        if event_type == 'email':
            evidence['to'] = event.get('to')
            evidence['from'] = event.get('from')
        elif event_type in ('web', 'http'):
            evidence['url'] = event.get('url')
        elif event_type == 'file':
            evidence['filename'] = event.get('filename')

        return evidence

    def detect_batch(self, user_sequences: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:

        results = {}
        total_users = len(user_sequences)

        for idx, (user_id, sequence) in enumerate(user_sequences.items()):
            logger.debug(f"检测用户 {user_id} ({idx + 1}/{total_users})")
            results[user_id] = self.detect_user(user_id, sequence)

        return results

    def iterate_detection(self, user_sequences: Dict[str, List[Dict]]) -> Generator[Dict, None, None]:

        for user_id, sequence in user_sequences.items():
            anomalies = self.detect_user(user_id, sequence)

            yield user_id, anomalies

    def get_statistics(self, user_sequences: Dict[str, List[Dict]]) -> Dict:

        stats = {
            'total_users': len(user_sequences),  
            'total_events': 0,  
            'total_detectable_events': 0,  
            'events_by_type': {'email': 0, 'web': 0, 'file': 0},  
            'users_with_content': 0  
        }

        for user_id, sequence in user_sequences.items():
            has_content = False  

            for event in sequence:
                stats['total_events'] += 1  
                event_type = event.get('type', '').lower()

                if event_type in self.DETECTABLE_EVENT_TYPES:
                    content = self._extract_content(event)  

                    if content:
                        stats['total_detectable_events'] += 1
                        has_content = True  

                        if event_type in stats['events_by_type']:
                            stats['events_by_type'][event_type] += 1

            if has_content:
                stats['users_with_content'] += 1

        return stats

if __name__ == "__main__":

    KNOWN_ANOMALY_USERS = {
        'email': ['BSS0369'],  
        'web': ['ABC0174'],   
        'file': ['JLM0364'],  
        'all': ['BSS0369', 'JLM0364', 'ABC0174']
    }  

    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(description="语义检测包装器 - 调试工具")
    parser.add_argument('--use-real-llm', action='store_true',
                        help='使用真实LLM进行测试（仅运行红队数据检测）')
    parser.add_argument('--sequence-dir', type=str, default=None,
                        help='行为序列目录（默认使用config中的配置）')
    parser.add_argument('--user-limit', type=int, default=10,
                        help='限制测试的用户数量（仅Mock模式）')
    parser.add_argument('--target-users', type=str, nargs='+', default=None,
                        help='指定要检测的用户ID（仅Mock模式）')
    parser.add_argument('--target-type', type=str, default=None,
                        choices=['email', 'web', 'file'],
                        help='按异常类型筛选用户（仅Mock模式）')
    args = parser.parse_args()

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_current_dir))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from utils import setup_logging

    setup_logging("semantic_detector_wrapper_debug", __file__)

    if args.use_real_llm:
        print("=" * 80)
        print("真实LLM模式 - 只运行红队数据检测")
        print("=" * 80)

        from data_preprocessing.redteam_data_loader import RedTeamDataLoader
        from multi_modal_features.semantic import SemanticAnalyzer, OutputMode

        print("\n【步骤1】加载红队数据...")
        redteam_loader = RedTeamDataLoader(include_content=True)
        redteam_data = redteam_loader.load(version_filter=getattr(config, 'DATASET_VERSION', '4.2'))

        if not redteam_data['ground_truth']:
            print("  ⚠️ 未找到红队数据")
            exit(1)

        print("\n【步骤2】初始化真实LLM...")
        real_analyzer = SemanticAnalyzer(llm_model=None, output_mode=OutputMode.DETAILED,
                                         prefer_trained=True, auto_load=True)
        real_detector = SemanticDetectorWrapper(real_analyzer, batch_size=10)

        if args.target_users:
            redteam_users = args.target_users
            print(f"  指定测试用户: {redteam_users}")
        elif args.target_type:
            redteam_users = KNOWN_ANOMALY_USERS.get(args.target_type, [])
            print(f"  按类型 '{args.target_type}' 测试: {redteam_users}")
        else:
            redteam_users = KNOWN_ANOMALY_USERS.get('all', [])
            print(f"  测试所有已知异常用户: {redteam_users}")

        for test_user in redteam_users:
            if test_user not in redteam_data['ground_truth']:
                print(f"\n  ⚠️ 用户 {test_user} 不在红队数据中，跳过")
                continue

            print(f"\n  ========== 用户: {test_user} ==========")

            user_gt = redteam_data['ground_truth'].get(test_user, {})
            scenario = user_gt.get('scenario', 'unknown')
            events_file = user_gt.get('events_file', 'unknown')

            print(f"  异常场景: {scenario}")
            print(f"  红队事件文件: {events_file}")

            redteam_events = []
            if scenario in redteam_data['events'] and test_user in redteam_data['events'][scenario]:
                redteam_events = redteam_data['events'][scenario][test_user]
                print(f"  加载红队事件数: {len(redteam_events)}")

            detectable_events = []
            for event in redteam_events:
                event_type = event.get('type', '').lower()
                content = event.get('content', '')
                if event_type in ['email', 'web', 'file'] and content:
                    detectable_events.append(event)

            print(f"  可检测事件数（含content）: {len(detectable_events)}")

            type_stats = {}
            for event in detectable_events:
                event_type = event.get('type', 'unknown')
                type_stats[event_type] = type_stats.get(event_type, 0) + 1

            if type_stats:
                print(f"  可检测事件类型分布: {type_stats}")

            anomalies = real_detector.detect_user(test_user, redteam_events)
            print(f"  检测到异常数: {len(anomalies)}")

            if anomalies:
                for i, anomaly in enumerate(anomalies[:5]):
                    print(f"\n    异常 {i + 1}:")
                    print(f"      时间: {anomaly.get('timestamp')}")
                    print(f"      类型: {anomaly.get('event_type')}")
                    print(f"      分数: {anomaly.get('anomaly_score')}")
                    if 'key_evidence' in anomaly and anomaly['key_evidence']:
                        print(f"      证据: {anomaly['key_evidence'][:100]}...")
                    if 'explanation' in anomaly and anomaly['explanation']:
                        print(f"      解释: {anomaly['explanation'][:100]}...")
            else:
                print(f"  未检测到异常")

            print(f"\n  ✓ 用户 {test_user} 测试完成")

        print("\n" + "=" * 80)
        print("真实LLM检测完成")
        print("=" * 80)
        exit(0)

    print("=" * 80)
    print("Mock LLM模式 - 完整功能测试")
    print("=" * 80)
    print(f"用户限制: {args.user_limit}")
    print("=" * 80)

    print("\n【步骤1】加载已保存的行为序列...")

    from data_preprocessing.behavior_sequence_builder import UserBehaviorSequenceBuilder
    from config import BEHAVIOR_SEQUENCE_DIR

    if args.sequence_dir:
        sequence_dir = args.sequence_dir
    else:
        sequence_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, "with_content")

    print(f"  序列目录: {sequence_dir}")

    if not os.path.exists(sequence_dir):
        print(f"  ✗ 目录不存在: {sequence_dir}")
        print("  请先运行 behavior_sequence_builder.py --build-full --include-content 构建完整版序列")
        exit(1)

    if args.target_users:
        user_ids = args.target_users
        print(f"  指定加载用户: {user_ids}")
    elif args.target_type:
        user_ids = KNOWN_ANOMALY_USERS.get(args.target_type, KNOWN_ANOMALY_USERS['all'])
        print(f"  按类型 '{args.target_type}' 加载用户: {user_ids}")
    else:
        user_ids = KNOWN_ANOMALY_USERS['all']
        print(f"  默认加载已知异常用户: {user_ids}")

    builder = UserBehaviorSequenceBuilder({})
    sequences = builder.load_sequences(input_dir=sequence_dir, user_ids=user_ids)

    if not sequences:
        print(f"  ✗ 加载失败")
        exit(1)

    print(f"  ✓ 加载成功: {len(sequences)} 个用户")
    print(f"  加载的用户: {list(sequences.keys())}")

    sample_user = list(sequences.keys())[0]
    sample_events = sequences[sample_user]
    has_content = any('content' in e for e in sample_events[:10])
    print(f"  ✓ 序列包含content字段: {'是' if has_content else '否'}")

    print("\n【步骤2】初始化语义分析器...")

    from multi_modal_features.semantic import SemanticAnalyzer, OutputMode

    class MockLLM:
        def generate(self, prompt):
            return '{"is_anomaly": false, "anomaly_score": 0.1, "key_evidence": "", "explanation": "正常内容"}'

    mock_analyzer = SemanticAnalyzer(llm_model=MockLLM(), output_mode=OutputMode.DETAILED)
    print("  ✓ 使用 Mock LLM")

    print("\n【步骤3】创建语义检测包装器...")
    detector = SemanticDetectorWrapper(mock_analyzer, batch_size=10)
    print(f"  ✓ 检测器初始化完成")

    print("\n【测试1】统计功能测试")
    print("-" * 40)
    stats = detector.get_statistics(sequences)
    print(f"  统计结果:")
    print(f"    测试用户数: {stats['total_users']}")
    print(f"    总事件数: {stats['total_events']}")
    print(f"    可检测事件数: {stats['total_detectable_events']}")
    print(f"    有内容用户数: {stats['users_with_content']}")
    print(f"    事件类型分布: {stats['events_by_type']}")

    print("\n【测试2】单用户检测")
    print("-" * 40)
    test_user = list(sequences.keys())[0]
    test_sequence = sequences[test_user]
    print(f"  测试用户: {test_user}")
    print(f"  总事件数: {len(test_sequence)}")
    anomalies = detector.detect_user(test_user, test_sequence)
    print(f"  检测到异常数: {len(anomalies)}")
    if anomalies:
        print(f"\n  异常示例（前3个）:")
        for i, anomaly in enumerate(anomalies[:3]):
            print(f"\n    异常{i + 1}:")
            print(f"      时间: {anomaly.get('timestamp')}")
            print(f"      类型: {anomaly.get('event_type')}")
            print(f"      分数: {anomaly.get('anomaly_score')}")
    else:
        print(f"\n  未检测到异常（Mock LLM返回正常）")

    print("\n【测试3】批量检测")
    print("-" * 40)
    results = detector.detect_batch(sequences)
    print(f"  检测用户数: {len(results)}")
    total_anomalies = sum(len(v) for v in results.values())
    print(f"  总异常数: {total_anomalies}")
    for user_id, anomalies in results.items():
        if anomalies:
            print(f"    用户 {user_id}: {len(anomalies)} 个异常")

    print("\n【测试4】迭代器模式测试")
    print("-" * 40)
    batch_count = 0
    for user_id, anomalies in detector.iterate_detection(sequences):
        batch_count += 1
        if batch_count <= 5:
            print(f"  用户 {batch_count}: {user_id}, 异常数={len(anomalies)}")
    print(f"  迭代完成，共 {batch_count} 个用户")

    print("\n【测试5】简化模式测试")
    print("-" * 40)

    class MockLLMSimple:
        def generate(self, prompt):
            return '{"is_anomaly": true, "anomaly_score": 0.75}'

    analyzer_simple = SemanticAnalyzer(llm_model=MockLLMSimple(), output_mode=OutputMode.SIMPLE)
    detector_simple = SemanticDetectorWrapper(analyzer_simple)
    test_user = list(sequences.keys())[0]
    anomalies_simple = detector_simple.detect_user(test_user, sequences[test_user])
    if anomalies_simple:
        print(f"  简化模式检测到异常: {len(anomalies_simple)} 个")
        print(f"    示例: 分数={anomalies_simple[0].get('anomaly_score')}")
    else:
        print("  简化模式未检测到异常")

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("  ✓ 统计功能")
    print("  ✓ 单用户检测")
    print("  ✓ 批量检测")
    print("  ✓ 迭代器模式")
    print("  ✓ 简化模式")
    print("=" * 80)
    print("调试完成")

