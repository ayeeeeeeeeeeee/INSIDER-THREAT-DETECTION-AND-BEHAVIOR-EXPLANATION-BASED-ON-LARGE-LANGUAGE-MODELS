from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class UnifiedEvidence:

    evidence_id: str  
    timestamp: str  
    date: str  
    source: str  
    event_type: str  
    categories: List[str]  
    metrics: Dict[str, Any] = field(
        default_factory=dict)  
    raw_data: Dict[str, Any] = field(default_factory=dict)  

    def to_dict(self) -> Dict:
        result = {
            'evidence_id': self.evidence_id,           
            'date': self.date,                         
            'source': self.source,                     
            'event_type': self.event_type,             
            'categories': self.categories,             
            'metrics': self.metrics,                   
            'details': self._extract_details()
        }

        return result

    def _extract_details(self) -> Dict:

        if self.source == 'semantic':
            details = {}

            for field in ['pc', 'filename', 'url', 'to', 'from']:
                if self.raw_data.get(field):
                    details[field] = self.raw_data[field]
            return details
        else:  
            return {
                'anomaly_types': self.raw_data.get('anomaly_types', []),  
                'severity': self.raw_data.get('severity'),  
                'description': self.raw_data.get('description', ''),  
                'behavior_context': self.raw_data.get('behavior_context', ''),  
                'value_baseline_comparison': self.raw_data.get('value_baseline_comparison', {})  
            }

@dataclass
class DailyEvidenceGroup:

    date: str
    evidences: List[UnifiedEvidence] = field(default_factory=list)  

    @property
    def source_distribution(self) -> Dict[str, int]:

        dist = defaultdict(int)
        for ev in self.evidences:
            dist[ev.source] += 1
        return dict(dist)

    @property
    def category_distribution(self) -> Dict[str, int]:

        dist = defaultdict(int)
        for ev in self.evidences:
            for cat in ev.categories:  
                dist[cat] += 1
        return dict(dist)

    def to_dict(self, max_evidences_per_day: Optional[int] = None) -> Dict:
        evidences = self.evidences
        if max_evidences_per_day is not None:
            evidences = evidences[:max_evidences_per_day]
        return {
            'date': self.date,
            'evidence_count': len(self.evidences),
            'source_distribution': self.source_distribution,  
            'category_distribution': self.category_distribution,  
            'evidences': [ev.to_dict() for ev in evidences]  
        }

@dataclass
class MultimodalEvidenceList:

    user_id: str
    time_range: Dict  
    summary: Dict

    daily_evidences: List[DailyEvidenceGroup]  

    normal_pattern: Optional[Dict] = None

    _semantic_raw: Optional[Dict] = field(default=None, repr=False)
    _statistical_raw: Optional[Dict] = field(default=None, repr=False)

    def to_dict(self, max_evidences_per_day: Optional[int] = None) -> Dict:

        result = {
            'user_id': self.user_id,
            'time_range': self.time_range,
            'summary': self.summary,
            'daily_evidences': [
                group.to_dict(max_evidences_per_day)
                for group in self.daily_evidences
            ]
        }

        return result

class MultimodalEvidenceAggregator:

    def __init__(self):
        logger.info("MultimodalEvidenceAggregator初始化（纯证据整合模式）")

    def aggregate(
            self,
            user_id: str,
            semantic_evidence: Optional[Dict] = None,
            statistical_evidence: Optional[Dict] = None
    ) -> MultimodalEvidenceList:

        unified_evidences: List[UnifiedEvidence] = []
        normal_pattern = None

        if semantic_evidence:
            semantic_evidences = self._convert_semantic_evidences(semantic_evidence)
            unified_evidences.extend(semantic_evidences)
            logger.debug(f"转换语义证据: {len(semantic_evidences)}条")

        if statistical_evidence:
            statistical_evidences = self._convert_statistical_evidences(statistical_evidence)
            unified_evidences.extend(statistical_evidences)
            normal_pattern = statistical_evidence.get('normal_pattern')
            logger.debug(f"转换统计证据: {len(statistical_evidences)}条")

        daily_groups = self._group_by_date(unified_evidences)

        time_range = self._extract_time_range(daily_groups)

        category_dist = self._count_categories(unified_evidences)
        event_type_dist = self._count_event_types(unified_evidences)

        semantic_count = len([e for e in unified_evidences if e.source == 'semantic'])
        statistical_count = len([e for e in unified_evidences if e.source == 'statistical'])

        logger.info(f"聚合完成: user={user_id}, total={len(unified_evidences)}, "
                    f"semantic={semantic_count}, statistical={statistical_count}")

        summary = {
            'total_count': len(unified_evidences),  
            'semantic_count': semantic_count,  
            'statistical_count': statistical_count,  
            'monitoring_days': len(daily_groups),  
            'category_distribution': category_dist,  
            'event_type_distribution': event_type_dist,  
        }

        if semantic_evidence:
            summary['semantic_stats'] = {
                'total_anomalies': semantic_evidence.get('total_anomalies'),
                'avg_score': semantic_evidence.get('avg_score'),
                'max_score': semantic_evidence.get('max_score'),
                'min_score': semantic_evidence.get('min_score'),
                'std_score': semantic_evidence.get('std_score'),
                'risk_distribution': semantic_evidence.get('risk_distribution'),
                'peak': semantic_evidence.get('peak')
            }

        if statistical_evidence:
            stat_summary = statistical_evidence.get('summary', {})
            summary['statistical_stats'] = {
                'total_anomaly_days': stat_summary.get('total_anomaly_days'),
                'max_confidence': stat_summary.get('max_confidence'),
                'avg_confidence': stat_summary.get('avg_confidence')
            }

        return MultimodalEvidenceList(
            user_id=user_id,  
            time_range=time_range,  
            summary=summary,  
            daily_evidences=daily_groups,  
            normal_pattern=normal_pattern,  
            _semantic_raw=semantic_evidence,  
            _statistical_raw=statistical_evidence  
        )

    @staticmethod
    def _extract_domain(url: str, max_url_length: int = 50) -> str:

        if not url:
            return ""

        url = url.replace('http://', '').replace('https://', '')
        domain = url.split('/')[0].split('?')[0]
        if len(domain) > max_url_length:
            domain = domain[:max_url_length] + "..."
        return domain

    def _convert_semantic_evidences(self, semantic_data: Dict) -> List[UnifiedEvidence]:

        evidences = []
        raw_evidences = semantic_data.get('raw_evidences', [])

        for idx, raw in enumerate(raw_evidences):
            timestamp = raw.get('timestamp', '')
            date = timestamp[:10] if timestamp else ''
            event_type = raw.get('event_type', 'unknown')

            useful_fields = ['pc', 'filename', 'to', 'from', 'cc']
            raw_data = {k: raw[k] for k in useful_fields if raw.get(k)}

            if raw.get('url'):
                raw_data['domain'] = self._extract_domain(raw['url'])

            category_val = raw.get('category', 'unknown')
            evidence = UnifiedEvidence(
                evidence_id=f"sem_{date}_{event_type}_{idx}",
                timestamp=timestamp,
                date=date,
                source='semantic',
                event_type=event_type,
                categories=[category_val] if category_val else ['unknown'],  
                metrics={
                    'anomaly_score': raw.get('anomaly_score'),

                },
                raw_data=raw_data
            )
            evidences.append(evidence)

        return evidences

    def _convert_statistical_evidences(self, statistical_data: Dict) -> List[UnifiedEvidence]:

        evidences = []
        stat_evidences = statistical_data.get('evidences', [])
        user_id = statistical_data.get('user_id', 'unknown')

        for idx, stat in enumerate(stat_evidences):
            date = stat.get('date', '')

            anomaly_types = stat.get('anomaly_types', [])
            categories = [f"stat_{t}" for t in anomaly_types] if anomaly_types else ['stat_unknown']

            evidence = UnifiedEvidence(
                evidence_id=f"stat_{date}_{idx}",
                timestamp='',  
                date=date,
                source='statistical',
                event_type='composite',  
                categories=categories,  
                metrics={
                    'max_z_score': stat.get('max_z_score'),
                    'confidence': stat.get('confidence'),

                },
                raw_data={
                    'anomaly_types': anomaly_types,
                    'severity': stat.get('severity'),
                    'description': stat.get('description', ''),   
                    'behavior_context': stat.get('behavior_context', ''),  
                    'value_baseline_comparison': stat.get('metrics', {})  
                }
            )
            evidences.append(evidence)

        return evidences

    def _group_by_date(self, evidences: List[UnifiedEvidence]) -> List[DailyEvidenceGroup]:

        date_map: Dict[str, DailyEvidenceGroup] = {}

        for ev in evidences:
            date = ev.date
            if not date:
                continue

            if date not in date_map:
                date_map[date] = DailyEvidenceGroup(date=date)

            date_map[date].evidences.append(ev)

        return [date_map[date] for date in sorted(date_map.keys())]

    def _extract_time_range(self, daily_groups: List[DailyEvidenceGroup]) -> Dict[str, Optional[str]]:

        if not daily_groups:
            return {'first': None, 'last': None}

        dates = [g.date for g in daily_groups]
        return {'first': min(dates), 'last': max(dates)}

    def _count_categories(self, evidences: List[UnifiedEvidence]) -> Dict[str, int]:

        dist = defaultdict(int)
        for ev in evidences:
            for cat in ev.categories:  
                dist[cat] += 1
        return dict(dist)

    def _count_event_types(self, evidences: List[UnifiedEvidence]) -> Dict[str, int]:

        dist = defaultdict(int)
        for ev in evidences:
            dist[ev.event_type] += 1
        return dict(dist)

def create_multimodal_aggregator() -> MultimodalEvidenceAggregator:

    return MultimodalEvidenceAggregator()

if __name__ == "__main__":
    import sys
    import os
    import json

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    sys.path.insert(0, _project_root)

    semantic_data = {
        "user_id": "BSS0369",
        "total_anomalies": 5,
        "avg_score": 0.7164,
        "max_score": 0.7344,
        "min_score": 0.6953,
        "std_score": 0.0151,
        "risk_distribution": {"high": 0, "medium": 5, "low": 0},
        "categories": {"file_anomaly": 5},
        "event_types": {"file": 5},
        "time_range": {"first": "2010-01-03T01:10:24", "last": "2010-01-03T01:10:26"},
        "peak": {"time": "2010-01-03T01:10:24", "score": 0.7344},
        "raw_evidences": [
            {
                "timestamp": "2010-01-03T01:10:24",
                "event_type": "file",
                "user_id": "BSS0369",
                "pc": "PC-8884",
                "content": "25-50-44-46-2D armistice...",
                "event_id": "{B8B8-H8AO48KZ-1180PVIT}",
                "is_anomaly": True,
                "anomaly_score": 0.703125,
                "category": "file_anomaly",
                "filename": "8UXZMQOU.pdf"
            }
        ]
    }

    statistical_data = {
        "user_id": "BSS0369",
        "source": "statistical",
        "summary": {
            "total_anomaly_days": 1,
            "total_evidence_count": 1,
            "max_confidence": 0.6069,
            "avg_confidence": 0.6069,
            "anomaly_type_distribution": {"file_count_anomaly": 1}
        },
        "evidences": [
            {
                "date": "2010-07-13",
                "anomaly_types": ["file_count_anomaly"],
                "confidence": 0.6069,
                "severity": "high",
                "description": "2010-07-13: 文件操作暴增异常",
                "metrics": {
                    "file_count_anomaly_count": 28,
                    "file_count_anomaly_baseline": 10.291479820627803
                },
                "behavior_context": "After working hours at PC-9787 login...",
                "max_z_score": 3.04
            }
        ],
        "normal_pattern": {
            "typical_login_hours": "12:00",
            "avg_usb_per_day": 11.512605042016807,
            "avg_email_per_day": 9.70464135021097,
            "avg_file_per_day": 10.291479820627803
        }
    }

    aggregator = MultimodalEvidenceAggregator()
    result = aggregator.aggregate(
        user_id="BSS0369",
        semantic_evidence=semantic_data,
        statistical_evidence=statistical_data
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print(f"\n总计: {result.summary['total_count']} 条证据 (语义: {result.summary['semantic_count']}, 统计: {result.summary['statistical_count']})")