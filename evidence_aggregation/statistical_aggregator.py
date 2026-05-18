
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from evidence_aggregation.behavior_abstractor_4W import BehaviorAbstractor
from multi_modal_features.behavior_statistics import BehaviorStatsModule
@dataclass
class StatisticalEvidence:

    evidence_id: str  
    date: str  
    anomaly_types: List[str]  
    raw_signals: List[Dict]  
    max_z_score: float  
    confidence: float  
    severity: str  
    description: str  
    behavior_context: str = ""  
    metrics: Dict[str, float] = field(default_factory=dict)   
@dataclass
class AggregatedStatisticalEvidence:

    user_id: str
    total_anomaly_days: int  
    total_evidence_count: int  
    evidence_list: List[StatisticalEvidence]  

    max_confidence: float  
    avg_confidence: float  
    anomaly_type_distribution: Dict[str, int]  

    user_baseline: Dict = field(default=None, repr=False)

    @classmethod
    def create_empty(cls, user_id: str, user_baseline: Dict = None) -> 'AggregatedStatisticalEvidence':

        return cls(
            user_id=user_id,
            total_anomaly_days=0,
            total_evidence_count=0,
            evidence_list=[],
            max_confidence=0.0,
            avg_confidence=0.0,
            anomaly_type_distribution={},
            user_baseline=user_baseline
        )

    def to_dict(self) -> Dict:

        result = {
            'user_id': self.user_id,
            'source': 'statistical',
            'summary': {
                'total_anomaly_days': self.total_anomaly_days,        
                'total_evidence_count': self.total_evidence_count,    
                'max_confidence': round(self.max_confidence, 4) if self.max_confidence > 0 else 0.0,     
                'avg_confidence': round(self.avg_confidence, 4) if self.avg_confidence > 0 else 0.0,      
                'anomaly_type_distribution': self.anomaly_type_distribution  
            },

            'evidences': [
                {
                    'date': e.date,                              
                    'anomaly_types': e.anomaly_types,            
                    'confidence': round(e.confidence, 4),        
                    'severity': e.severity,                      
                    'description': e.description,                
                    'metrics': e.metrics,                         
                    'behavior_context': e.behavior_context,        
                    'max_z_score': round(e.max_z_score, 2),  
                }
                for e in self.evidence_list  
            ]
        }
        if self.user_baseline:

            login_baseline = self.user_baseline.get('login', {})
            typical_hours = login_baseline.get('typical_login_hours', {})
            mean_hour = typical_hours.get('mean_hour')
            typical_login_str = f"{mean_hour:.0f}:00" if mean_hour is not None else "未知"

            usb_baseline = self.user_baseline.get('usb', {})
            avg_usb = usb_baseline.get('avg_per_day', 0)

            email_baseline = self.user_baseline.get('email', {})
            avg_email = email_baseline.get('avg_per_day', 0)
            avg_internal = email_baseline.get('internal_emails', 0)  
            avg_external = email_baseline.get('external_emails', 0)  

            http_baseline = self.user_baseline.get('http', {})
            avg_http = http_baseline.get('avg_per_day', 0)
            job_visits = http_baseline.get('job_site_visits', 0)
            leak_visits = http_baseline.get('leak_site_visits', 0)

            file_baseline = self.user_baseline.get('file', {})
            avg_file = file_baseline.get('avg_per_day', 0)
            avg_file_types = file_baseline.get('file_type_count', 0)

            result['normal_pattern'] = {
                'typical_login_hours': typical_login_str,
                'avg_usb_per_day': avg_usb,
                'avg_email_per_day': avg_email,
                'internal_emails': avg_internal,
                'external_emails': avg_external,
                'avg_http_per_day': avg_http,
                'job_site_visits': job_visits,
                'leak_site_visits': leak_visits,
                'avg_file_per_day': avg_file,
                'file_type_count': avg_file_types,
            }
        return result
class StatisticalAnomalyAggregator:

    def __init__(
            self,
            z_score_thresholds: Dict[str, float] = None,
            confidence_mapping: Dict[str, float] = None
    ):

        self.abstractor = BehaviorAbstractor()

        self.z_score_thresholds = z_score_thresholds or {
            'high': 3.0,
            'medium': 2.0
        }

        self.confidence_mapping = confidence_mapping or {
            'k': 0.8,  
            'midpoint': 2.5  
        }

    def _z_score_to_confidence(self, z_score: float) -> float:

        k = self.confidence_mapping['k']
        mid = self.confidence_mapping['midpoint']
        return 1 / (1 + np.exp(-k * (z_score - mid)))

    def _get_severity(self, z_score: float) -> str:

        if z_score >= self.z_score_thresholds['high']:
            return 'high'
        elif z_score >= self.z_score_thresholds['medium']:
            return 'medium'
        return 'low'

    def _merge_related_signals(self, signals: List[Dict], full_sequence: List[Dict]) -> List[StatisticalEvidence]:

        by_date: Dict[str, List[Dict]] = {}
        for signal_entry in signals:
            date = signal_entry['date_only']
            if hasattr(date, 'strftime'):
                date = date.strftime('%Y-%m-%d')
            elif hasattr(date, 'isoformat'):
                date = date.isoformat()[:10]
            if date not in by_date:
                by_date[date] = []
            by_date[date].extend(signal_entry['signals'])

        evidences = []
        for date, day_signals in by_date.items():
            if not day_signals:
                continue

            anomaly_types = list(set(s['type'] for s in day_signals))

            max_z = max(s.get('z_score', 0) for s in day_signals)
            confidence = self._z_score_to_confidence(max_z)

            type_summary = '、'.join(self._translate_type(t) for t in anomaly_types[:3])
            description = f"{date}: {type_summary}异常"

            metrics = {}
            for s in day_signals:
                if 'count' in s:
                    metrics[f"{s['type']}_count"] = s['count']
                if 'avg' in s and s['avg'] is not None:
                    metrics[f"{s['type']}_baseline"] = s['avg']

            date_events = [e for e in full_sequence if str(e.get('timestamp', ''))[:10] == date]

            behavior_desc = " ".join(self.abstractor.abstract(date_events)) if date_events else ""
            evidences.append(StatisticalEvidence(
                evidence_id=f"stat_{date}",
                date=date,
                anomaly_types=anomaly_types,
                raw_signals=day_signals,
                max_z_score=max_z,
                confidence=confidence,
                severity=self._get_severity(max_z),
                description=description,
                metrics=metrics,
                behavior_context=behavior_desc
            ))

        return sorted(evidences, key=lambda x: x.date)

    def _translate_type(self, type_str: str) -> str:

        mapping = {

            'abnormal_login_time': '登录时间异常',
            'login_count_anomaly': '登录次数异常',
            'weekend_login': '周末登录',
            'non_work_hour_login': '非工作时段登录',
            'login_other_pc': '陌生设备登录',

            'usb_usage_spike': 'U盘使用激增',

            'non_work_hour_usb': '非工作时段U盘使用',

            'email_count_anomaly': '邮件数量异常',
            'non_work_hour_email': '非工作时段邮件',
            'external_email_ratio_anomaly': '外部邮件占比异常',

            'http_count_anomaly': '网页浏览激增',
            'job_site_visit_spike': '求职网站访问异常',
            'leak_site_visit_spike': '泄露网站访问异常',

            'file_count_anomaly': '文件操作激增',
            'non_work_hour_file': '非工作时段文件操作',

            'post_termination_login': '离职后登录',
            'post_termination_usb': '离职后U盘使用',
            'post_termination_email': '离职后邮件发送',
            'post_termination_file': '离职后文件操作',
        }
        return mapping.get(type_str, type_str)

    def aggregate(
            self, user_id: str, anomaly_signals: List[Dict],
            full_sequence: List[Dict], user_baseline: Dict = None,
    ) -> Optional[AggregatedStatisticalEvidence]:

        if not anomaly_signals:
            return AggregatedStatisticalEvidence.create_empty(user_id, user_baseline)

        evidences = self._merge_related_signals(anomaly_signals, full_sequence)

        if not evidences:
            return None

        type_dist = {}
        for e in evidences:
            for t in e.anomaly_types:
                type_dist[t] = type_dist.get(t, 0) + 1

        MIN_CONFIDENCE = 0.15

        EXEMPT_TYPES = {
            'post_termination_login',
            'post_termination_usb',
            'post_termination_email',
            'post_termination_file',
            'login_other_pc',
        }

        filtered = []
        for e in evidences:

            exempt_in_evidence = [t for t in e.anomaly_types if t in EXEMPT_TYPES]
            scored_types = [t for t in e.anomaly_types if t not in EXEMPT_TYPES]

            if scored_types:

                if e.confidence >= MIN_CONFIDENCE:

                    e.anomaly_types = scored_types + exempt_in_evidence
                    filtered.append(e)

        filtered = [e for e in evidences if e.confidence >= MIN_CONFIDENCE]

        if not filtered:
            return AggregatedStatisticalEvidence.create_empty(user_id, user_baseline)

        confidences = [e.confidence for e in filtered]

        type_dist = {}
        for e in filtered:
            for t in e.anomaly_types:
                type_dist[t] = type_dist.get(t, 0) + 1

        return AggregatedStatisticalEvidence(
            user_id=user_id,
            total_anomaly_days=len(filtered),
            total_evidence_count=len(filtered),
            evidence_list=filtered,
            max_confidence=max(confidences),
            avg_confidence=sum(confidences) / len(confidences),
            anomaly_type_distribution=type_dist,
            user_baseline=user_baseline,
        )
def create_statistical_aggregator(
        high_threshold: float = 3.0,
        medium_threshold: float = 2.0
) -> StatisticalAnomalyAggregator:

    return StatisticalAnomalyAggregator(
        z_score_thresholds={'high': high_threshold, 'medium': medium_threshold}
    )
if __name__ == "__main__":
    import sys
    import os
    import json

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    sys.path.insert(0, _project_root)

    TEST_USER = 'BSS0369'  

    stats = BehaviorStatsModule(user_ids=[TEST_USER])
    stats.build_user_baseline(TEST_USER)

    agg = StatisticalAnomalyAggregator()
    result = agg.aggregate(TEST_USER,
                           stats.detect_all_anomalies(TEST_USER),
                           stats.sequences.get(TEST_USER, []),
                           stats.user_baselines.get(TEST_USER))

    if result:
        print(json.dumps(result.to_dict(),
                         indent=2, ensure_ascii=False))