from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from multi_modal_features.behavior_statistics import BehaviorStatsModule

@dataclass
class TimeSlice:

    time_context: str  
    start_time: str
    end_time: str
    events: List[Dict] = field(default_factory=list)

@dataclass
class DeviceGroup:

    device: str  
    time_context: str  
    events: List[Dict] = field(default_factory=list)  

class BehaviorAbstractor:

    def __init__(
            self,
            work_start: int = 9,
            work_end: int = 18,
            max_events_per_group: int = 10,  
            max_url_length: int = 50,  
            max_email_length: int = 30  
    ):
        self.internal_domains =  ['dtaa.com']
        self.work_start = work_start
        self.work_end = work_end
        self.max_events_per_group = max_events_per_group
        self.max_url_length = max_url_length
        self.max_email_length = max_email_length

        self._is_work_hour = lambda dt: work_start <= dt.hour < work_end
        self._is_weekend = lambda dt: dt.weekday() >= 5

    def _parse_timestamp(self, timestamp) -> Optional[datetime]:

        if not timestamp:
            return None

        if isinstance(timestamp, datetime):
            return timestamp

        if hasattr(timestamp, 'to_pydatetime'):
            return timestamp.to_pydatetime()

        timestamp_str = str(timestamp)

        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            pass

        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            pass

        return None

    def _get_time_context(self, timestamp: str) -> str:

        if not timestamp:
            return "unknown"

        dt = self._parse_timestamp(timestamp)
        if dt is None:
            return "unknown"

        if self._is_weekend(dt):
            return "weekend"
        elif self._is_work_hour(dt):
            return "working_hours"
        else:
            return "after_hours"

    def _extract_domain(self, url: str) -> str:

        if not url:
            return ""

        url = url.replace('http://', '').replace('https://', '')
        domain = url.split('/')[0].split('?')[0]
        if len(domain) > self.max_url_length:
            domain = domain[:self.max_url_length] + "..."
        return domain

    def _extract_email_addr(self, email_str: str) -> str:

        if not email_str:
            return ""
        if len(email_str) > self.max_email_length:
            email_str = email_str[:self.max_email_length] + "..."
        return email_str

    def _extract_filename(self, filepath: str) -> str:

        if not filepath:
            return ""
        return filepath.split('/')[-1].split('\\')[-1]

    def _group_by_time(self, sequence: List[Dict]) -> List[TimeSlice]:

        if not sequence:
            return []

        slices = []
        current_slice = None  

        for event in sequence:
            timestamp = event.get('timestamp', '')
            time_ctx = self._get_time_context(timestamp)  

            if current_slice is None or current_slice.time_context != time_ctx:

                if current_slice:
                    current_slice.end_time = timestamp  
                    slices.append(current_slice)  

                current_slice = TimeSlice(
                    time_context=time_ctx,  
                    start_time=timestamp,  
                    end_time=timestamp,  
                    events=[]  
                )

            current_slice.events.append(event)
            current_slice.end_time = timestamp

        if current_slice:
            slices.append(current_slice)

        return slices

    def _group_by_device(self, time_slice: TimeSlice) -> List[DeviceGroup]:

        if not time_slice.events:
            return []

        by_device = defaultdict(list)
        for event in time_slice.events:

            device = event.get('pc', event.get('device', 'unknown'))
            by_device[device].append(event)

        groups = []
        for device, events in by_device.items():
            groups.append(DeviceGroup(
                device=device,  
                time_context=time_slice.time_context,  
                events=events  
            ))

        return groups

    def _extract_actions(self, events: List[Dict]) -> List[str]:

        actions = []
        action_counts = defaultdict(int)
        action_examples = defaultdict(set)  

        for event in events:
            event_type = event.get('type', event.get('event_type', 'unknown'))
            activity = event.get('activity', '')

            if event_type == 'logon':
                activity_lower = activity.lower() if activity else ''
                if activity_lower == 'logon':
                    action = 'login'
                elif activity_lower == 'logoff':
                    action = 'logoff'
                else:
                    action = 'logon/logoff'
            elif event_type == 'usb':
                activity_lower = activity.lower() if activity else ''
                action = 'connect USB' if activity_lower == 'connect' else 'disconnect USB'
            elif event_type == 'web':
                action = 'browse website'
            elif event_type == 'email':
                action = 'send email'
            elif event_type == 'file':
                action = 'access file'
            else:
                action = event_type

            action_counts[action] += 1

            obj = self._extract_object(event)
            if obj:
                action_examples[action].add(obj)

        for action, count in action_counts.items():

            if count == 1:

                actions.append(action)

            else:

                if action_examples[action]:

                    examples = ', '.join(sorted(action_examples[action]))  
                    actions.append(f"{action} {count} times ({examples})")
                else:

                    actions.append(f"{action} {count} times")

        return actions

    @staticmethod
    def extract_domain(email_str):

        if not email_str:
            return ''

        email_str = email_str.strip().strip('"').strip("'")

        if '@' in email_str:

            return email_str.split('@')[-1].lower()
        return ''

    def extract_email_direction(self, event: Dict) -> str:

        to = event.get('to', '')
        from_addr = event.get('from', '')

        to = to.replace(';', ',')  
        to_domains = []
        for r in to.split(','):
            r = r.strip()
            if r:
                domain = self.extract_domain(r)
                if domain:
                    to_domains.append(domain)

        from_domain = self.extract_domain(from_addr)

        is_internal_from = from_domain in self.internal_domains

        has_internal_to = any(d in self.internal_domains for d in to_domains)
        has_external_to = any(d not in self.internal_domains for d in to_domains)

        if not from_domain:
            if has_internal_to and not has_external_to:
                return "email to internal recipients"
            elif not has_internal_to and has_external_to:
                return "email to external recipients"
            elif has_internal_to and has_external_to:
                return "email to both internal and external recipients"
            else:
                return ""  

        if is_internal_from and not has_external_to:
            return "internal email"

        elif not is_internal_from and not has_internal_to:
            return "external email"

        elif is_internal_from and not has_internal_to:
            return "from insider to outsider"

        elif not is_internal_from and has_internal_to and not has_external_to:
            return "from outsider to insider"

        elif is_internal_from and has_internal_to and has_external_to:
            return "from insider to both internal and external recipients"

        elif not is_internal_from and has_internal_to and has_external_to:
            return "from outsider to both internal and external recipients"
        else:
            return ""  

    def _extract_object(self, event: Dict) -> str:

        event_type = event.get('type', event.get('event_type', ''))

        if event_type == 'web':
            url = event.get('url', '')
            return self._extract_domain(url)

        elif event_type == 'email':

            return self.extract_email_direction(event)

        elif event_type == 'file':
            filename = event.get('filename', '')
            return self._extract_filename(filename)

        elif event_type == 'usb':
            return f"device {event.get('pc', '')}"

        elif event_type == 'logon':
            return f"at {event.get('pc', '')}"

        return ""

    def _render_4w_sentence(
            self,
            when: str,
            where: str,
            actions: List[str],
            objects: List[str]
    ) -> str:

        time_map = {
            'working_hours': 'During working hours',
            'after_hours': 'After working hours',
            'weekend': 'On weekend'
        }

        when_str = time_map.get(when, f"At {when}")
        where_str = f"at {where}" if where and where != 'unknown' else ""

        if len(actions) == 1:
            action_str = actions[0]
        else:
            action_str = ", ".join(actions[:-1]) + f", and {actions[-1]}"

        parts = [p for p in [when_str, where_str, action_str] if p]
        sentence = " ".join(parts) + "."

        return sentence

    def abstract(self, sequence: List[Dict]) -> List[str]:

        if not sequence:
            return []

        time_slices = self._group_by_time(sequence)

        sentences = []

        for time_slice in time_slices:

            device_groups = self._group_by_device(time_slice)

            for device_group in device_groups:
                events = device_group.events

                actions = self._extract_actions(events)

                if len(actions) == 0:
                    continue

                objects = []
                for event in events:
                    obj = self._extract_object(event)
                    if obj:
                        objects.append(obj)

                sentence = self._render_4w_sentence(
                    when=device_group.time_context,
                    where=device_group.device,
                    actions=actions,
                    objects=objects
                )

                sentences.append(sentence)

        return sentences

def create_abstractor(
        work_start: int = 9,
        work_end: int = 18
) -> BehaviorAbstractor:

    return BehaviorAbstractor(work_start=work_start, work_end=work_end)

if __name__ == "__main__":
    import sys
    import os

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from data_preprocessing.behavior_sequence_builder import UserBehaviorSequenceBuilder
    from config import BEHAVIOR_SEQUENCE_DIR

    TEST_USER = "BSS0369"
    MAX_EVENTS = 1000

    print("=" * 80)
    print("4W行为抽象器测试")
    print("=" * 80)

    builder = UserBehaviorSequenceBuilder({})
    seq_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, "light")
    sequences = builder.load_sequences(input_dir=seq_dir, user_ids=[TEST_USER])

    if TEST_USER not in sequences:
        print(f"用户 {TEST_USER} 不存在")
        sys.exit(1)

    sequence = sequences[TEST_USER][:MAX_EVENTS]
    print(f"加载用户 {TEST_USER}: {len(sequence)} 条事件")

    abstractor = BehaviorAbstractor()
    sentences = abstractor.abstract(sequence)

    print(f"\n抽象结果 ({len(sentences)} 个片段):")
    print("-" * 80)
    for i, s in enumerate(sentences, 1):
        print(f"{i}. {s}")

    print("=" * 80)
    print("测试完成")
