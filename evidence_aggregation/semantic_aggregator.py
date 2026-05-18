
import os
import sys

LIMIT_NUM = 0  

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)  
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import numpy as np

from multi_modal_features.semantic import SemanticAnalyzer, SemanticDetectorWrapper, OutputMode
from multi_modal_features.semantic.categories import SemanticCategory, get_category_by_event_type

logger = logging.getLogger(__name__)

@dataclass
class AggregatedSemanticEvidence:

    user_id: str  
    session_id: str
    total_anomalies: int  
    anomaly_scores: List[float] = field(default_factory=list)  
    anomalies_by_category: Dict[str, List[Dict]] = field(default_factory=dict)  
    anomalies_by_type: Dict[str, List[Dict]] = field(default_factory=dict)  

    first_anomaly_time: Optional[str] = None  
    last_anomaly_time: Optional[str] = None  
    peak_anomaly_time: Optional[str] = None  
    peak_anomaly_score: float = 0.0  

    avg_score: float = 0.0  
    max_score: float = 0.0  
    min_score: float = 1.0  
    std_score: float = 0.0
    high_risk_count: int = 0  
    medium_risk_count: int = 0  
    low_risk_count: int = 0  

    raw_evidences: List[Dict] = field(default_factory=list)  

    def compute_statistics(self):

        if self.anomaly_scores:
            self.avg_score = sum(self.anomaly_scores) / len(self.anomaly_scores)
            self.max_score = max(self.anomaly_scores)
            self.min_score = min(self.anomaly_scores)
            self.std_score = float(np.std(self.anomaly_scores))

            for score in self.anomaly_scores:
                if score >= 0.8:
                    self.high_risk_count += 1
                elif score >= 0.5:
                    self.medium_risk_count += 1
                else:
                    self.low_risk_count += 1
        else:

            self.avg_score = 0.0
            self.max_score = 0.0
            self.min_score = 0.0
            self.std_score = 0.0
            self.high_risk_count = 0
            self.medium_risk_count = 0
            self.low_risk_count = 0

    def get_aggregated_vector(self) -> List[float]:

        return [self.avg_score, self.max_score, self.std_score, self.min_score]

    def _make_json_safe(self, obj):

        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        return obj

    def to_dict(self) -> Dict:

        result = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'total_anomalies': self.total_anomalies,
            'avg_score': round(self.avg_score, 4) if self.total_anomalies > 0 else 0.0,
            'max_score': round(self.max_score, 4) if self.total_anomalies > 0 else 0.0,
            'min_score': round(self.min_score, 4) if self.total_anomalies > 0 else 0.0,
            'std_score': round(self.std_score, 4) if self.total_anomalies > 0 else 0.0,
            'risk_distribution': {
                'high': self.high_risk_count,
                'medium': self.medium_risk_count,
                'low': self.low_risk_count
            },
            'categories': {
                cat: len(evidences)
                for cat, evidences in self.anomalies_by_category.items()
            } if self.anomalies_by_category else {},
            'event_types': {
                evt_type: len(evidences)
                for evt_type, evidences in self.anomalies_by_type.items()
            } if self.anomalies_by_type else {},
            'time_range': {
                'first': self.first_anomaly_time,
                'last': self.last_anomaly_time
            },
            'peak': {
                'time': self.peak_anomaly_time,
                'score': round(self.peak_anomaly_score, 4) if self.peak_anomaly_score > 0 else 0.0
            },
            'raw_evidences': [ev.to_dict() if hasattr(ev, 'to_dict') else ev
                  for ev in self.raw_evidences] if self.raw_evidences else [],
        }
        return self._make_json_safe(result)

class SemanticAnomalyAggregator:

    def __init__(
            self,
            analyzer: Optional[SemanticAnalyzer] = None,
            detector: Optional[SemanticDetectorWrapper] = None,
            output_mode: OutputMode = OutputMode.SIMPLE,
            auto_load_model: bool = True,
            high_risk_threshold: float = 0.8,
            medium_risk_threshold: float = 0.5
    ):

        self.high_risk_threshold = high_risk_threshold

        self.medium_risk_threshold = medium_risk_threshold

        if detector:

            self.detector = detector
            self.analyzer = detector.analyzer  
        elif analyzer:

            self.analyzer = analyzer
            self.detector = SemanticDetectorWrapper(analyzer)  
        else:

            self.analyzer = SemanticAnalyzer(
                llm_model=None,  
                output_mode=output_mode,  
                auto_load=auto_load_model  
            )

            self.detector = SemanticDetectorWrapper(self.analyzer)

        logger.info(f"SemanticAnomalyAggregator初始化完成, "
                    f"high_threshold={high_risk_threshold}, "  
                    f"medium_threshold={medium_risk_threshold}")  

    def _aggregate_user_evidences(
            self,
            session_id: str,
            user_id: str,
            anomalies: List[Dict],
            include_raw: bool = True
    ) -> AggregatedSemanticEvidence:

        evidence = AggregatedSemanticEvidence(
            user_id=user_id,
            session_id=session_id,
            total_anomalies=len(anomalies)
        )

        if not anomalies:

            evidence.anomalies_by_category = {}
            evidence.anomalies_by_type = {}
            evidence.anomaly_scores = []
            evidence.raw_evidences = []

            evidence.first_anomaly_time = None
            evidence.last_anomaly_time = None
            evidence.peak_anomaly_time = None
            evidence.peak_anomaly_score = 0.0

            evidence.compute_statistics()

            return evidence

        for anomaly in anomalies:

            score = anomaly.get('anomaly_score', 0.0)  
            category = anomaly.get('category', 'unknown')  
            event_type = anomaly.get('event_type', 'unknown')  

            evidence.anomaly_scores.append(score)

            if category not in evidence.anomalies_by_category:
                evidence.anomalies_by_category[category] = []
            evidence.anomalies_by_category[category].append(anomaly)

            if event_type not in evidence.anomalies_by_type:
                evidence.anomalies_by_type[event_type] = []
            evidence.anomalies_by_type[event_type].append(anomaly)

            timestamp = anomaly.get('timestamp')
            if timestamp:

                if not evidence.first_anomaly_time or timestamp < evidence.first_anomaly_time:
                    evidence.first_anomaly_time = timestamp

                if not evidence.last_anomaly_time or timestamp > evidence.last_anomaly_time:
                    evidence.last_anomaly_time = timestamp

            if score > evidence.peak_anomaly_score:
                evidence.peak_anomaly_score = score
                evidence.peak_anomaly_time = anomaly.get('timestamp')

        if include_raw:

            per_day = self._calculate_per_day_limit(anomalies)

            by_day = defaultdict(list)
            for a in anomalies:
                ts = a.get('timestamp', '')
                day = str(ts)[:10] if ts else '_no_date'
                by_day[day].append(a)

            trimmed = []
            for day_anomalies in by_day.values():

                day_anomalies.sort(key=lambda x: x.get('anomaly_score', 0), reverse=True)

                trimmed.extend(day_anomalies[:per_day])

            evidence.raw_evidences = trimmed

        evidence.compute_statistics()

        return evidence

    def _calculate_per_day_limit(self, anomalies: List[Dict], global_max: int = 50) -> int:

        return global_max if anomalies else 0

    def _build_timeline(self, anomalies: List[Dict]) -> List[Dict]:

        sorted_anomalies = sorted(
            anomalies,
            key=lambda x: x.get('timestamp', '')
        )

        timeline = []
        for anomaly in sorted_anomalies:
            timeline.append({
                'timestamp': anomaly.get('timestamp'),
                'user_id': anomaly.get('user_id'),
                'event_type': anomaly.get('event_type'),
                'category': anomaly.get('category'),
                'score': round(anomaly.get('anomaly_score', 0.0), 4),
                'key_evidence': anomaly.get('key_evidence', '')[:50] if anomaly.get('key_evidence') else ''
            })

        return timeline

    def _load_user_profile(self, user_id: str) -> Optional[str]:
        try:
            from config import USER_PROFILE_DIR
            import json, os
            profile_path = os.path.join(USER_PROFILE_DIR, f"{user_id}.json")
            if os.path.exists(profile_path):
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                from multi_modal_features.user_profile import UserProfileModule
                pm = UserProfileModule(include_content=False)
                pm.user_profiles[user_id] = profile_data
                return pm.get_llm_profile(user_id)
        except Exception as e:
            logger.warning(f"加载用户画像失败 {user_id}: {e}")
        return None

    def detect(
            self,
            user_id: str,
            session_id: str,
            behavior_sequence: List[Dict],
            user_profile: Optional[str] = None,
            include_raw_evidences: bool = True
    ) -> AggregatedSemanticEvidence:

        anomalies = self.detector.detect_user(user_id, behavior_sequence, user_profile=user_profile)
        sid = session_id

        return self._aggregate_user_evidences(sid, user_id, anomalies, include_raw_evidences)

def create_detector(
        use_real_llm: bool = False,
        model_path: Optional[str] = None,
        prefer_trained: bool = True,
        output_mode: str = 'simple',
        high_risk_threshold: float = 0.8,
        medium_risk_threshold: float = 0.5
) -> SemanticAnomalyAggregator:

    mode = OutputMode.DETAILED if output_mode == 'detailed' else OutputMode.SIMPLE
    if use_real_llm:

        analyzer = SemanticAnalyzer(
            llm_model=None,
            output_mode=mode,
            auto_load=True,
            prefer_trained=prefer_trained,
            model_path=model_path
        )
        return SemanticAnomalyAggregator(
            analyzer=analyzer,
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
        )
    else:

        return SemanticAnomalyAggregator(
            output_mode=mode,
            auto_load_model=False,
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
        )

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import json

    parser = argparse.ArgumentParser(description="语义异常检测 - 单用户测试")
    parser.add_argument('--use-real-llm', action='store_true', help='使用真实LLM')
    DEFAULT_MODEL_PATH = r"E:\university\毕业论文\代码\基于大语言模型的内部威胁检测与行为解释\output\models\semantic_analyzer\lora_model\no_profile\final_lora_model__loss0.2185_20260411_215140"
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='模型路径')
    parser.add_argument('--mode', type=str, default='simple', choices=['detailed', 'simple'],
                        help='输出模式')
    parser.add_argument('--users', type=str, nargs='+', default=None, help='指定用户ID')
    parser.add_argument('--json', action='store_true', help='JSON格式输出（含所有证据）')
    args = parser.parse_args()

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_current_dir))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from utils import setup_logging

    setup_logging("semantic_aggregator_debug", __file__)

    KNOWN_ANOMALY_USERS = ['BSS0369']  
    target_users = args.users if args.users else KNOWN_ANOMALY_USERS

    print("=" * 80)
    print("语义异常检测 - 单用户测试")
    print("=" * 80)
    print(f"目标用户: {target_users}")
    print(f"LLM模式: {'真实' if args.use_real_llm else 'Mock'}")
    print(f"输出模式: {args.mode}")
    print("=" * 80)

    from data_preprocessing.behavior_sequence_builder import UserBehaviorSequenceBuilder
    from config import BEHAVIOR_SEQUENCE_DIR

    builder = UserBehaviorSequenceBuilder({})
    sequences = builder.load_sequences(
        input_dir=os.path.join(BEHAVIOR_SEQUENCE_DIR, "with_content"),
        user_ids=target_users
    )
    print(f"\n✓ 加载成功: {len(sequences)} 个用户")

    mode_enum = OutputMode.DETAILED if args.mode == 'detailed' else OutputMode.SIMPLE

    if args.use_real_llm:
        detector = create_detector(use_real_llm=True, output_mode=args.mode, model_path=args.model_path)
        print(f"✓ 使用真实LLM（{args.mode}模式）")
    else:
        class MockLLM:
            def __init__(self):
                self.count = 0

            def generate(self, prompt):
                self.count += 1
                import hashlib
                score = 0.3 + (int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 70) / 100
                if args.mode == 'simple':
                    return f'异常 {score:.2f}' if score > 0.7 else f'正常 {score:.2f}'
                else:
                    return f'异常 | {score:.2f} | Mock证据 | 测试' if score > 0.7 else f'正常 | {score:.2f}'

        mock_analyzer = SemanticAnalyzer(llm_model=MockLLM(), output_mode=mode_enum, auto_load=False)
        detector = SemanticAnomalyAggregator(analyzer=mock_analyzer)
        print(f"✓ 使用Mock LLM（{args.mode}模式）")

    results = {}
    for user_id, sequence in sequences.items():
        if LIMIT_NUM > 0:
            sequence = sequence[:LIMIT_NUM]
        result = detector.detect(user_id, sequence)
        if result:
            results[user_id] = result

    if args.json:

        output = {}
        for uid, r in results.items():
            d = r.to_dict()

            if 'raw_evidences' in d and d['raw_evidences']:

                max_raw = getattr(args, 'max_raw', 5)  
                d['raw_evidences'] = d['raw_evidences'][:max_raw]
                d['raw_evidences_total'] = len(r.raw_evidences)  
            output[uid] = d
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:

        print("\n" + "=" * 80)
        print(f"{'用户ID':<12} {'异常':<6} {'最高分':<8} {'平均分':<8} {'风险':<6} {'类别'}")
        print("-" * 80)
        for user_id, r in results.items():
            risk = "🔴高" if r.max_score >= 0.8 else ("🟡中" if r.max_score >= 0.5 else "🟢低")
            cats = ','.join(list(r.anomalies_by_category.keys())[:3])
            print(f"{user_id:<12} {r.total_anomalies:<6} {r.max_score:<8.4f} {r.avg_score:<8.4f} {risk:<6} {cats}")

        if results:
            first_user = list(results.keys())[0]
            r = results[first_user]
            print(f"\n{'=' * 80}")
            print(f"详细证据示例 - 用户 {first_user}")
            print('=' * 80)
            print(f"时间范围: {r.first_anomaly_time} ~ {r.last_anomaly_time}")
            print(f"风险分布: 高={r.high_risk_count} 中={r.medium_risk_count} 低={r.low_risk_count}")
            print(f"峰值: {r.peak_anomaly_score:.4f} @ {r.peak_anomaly_time}")

            if args.mode == 'detailed' and r.raw_evidences:
                print(f"\n前3条异常证据:")
                for i, ev in enumerate(r.raw_evidences[:3], 1):
                    print(f"\n{i}. [{ev.get('category')}] {ev.get('anomaly_score', 0):.4f}")
                    print(f"   {ev.get('key_evidence', 'N/A')[:100]}")
                    if ev.get('explanation'):
                        print(f"   解释: {ev['explanation'][:80]}")

        if results:
            first_uid = list(results.keys())[0]
            r = results[first_uid]
            print(f"\n{'=' * 80}")
            print(f"按天截断验证 - {first_uid}")
            print('=' * 80)

            from collections import Counter

            day_counts = Counter(
                str(e['timestamp'])[:10] if hasattr(e['timestamp'], 'strftime') else str(e.get('timestamp', ''))[:10]
                for e in r.raw_evidences)
            print(f"全量异常: {r.total_anomalies} 条")
            print(f"输出证据: {len(r.raw_evidences)} 条")
            print(f"覆盖天数: {len(day_counts)} 天")
            print(
                f"per_day: {detector._calculate_per_day_limit(r.raw_evidences) if hasattr(detector, '_calculate_per_day_limit') else 'N/A'}")
            print(f"\n每天输出条数（前10天）:")
            for day, count in sorted(day_counts.items())[:10]:
                bar = '█' * min(count, 20)
                print(f"  {day}: {count:>4}条 {bar}")
            if len(day_counts) > 10:
                print(f"  ... 还有 {len(day_counts) - 10} 天")

        print("\n" + "=" * 80)
        print("完成")
    print("\n" + "=" * 80)
    print("完成")