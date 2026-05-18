
import os
from typing import List, Dict
import pandas as pd

from config import BEHAVIOR_STATISTICS_DIR, USER_PROFILE_DIR
from data_preprocessing import UserBehaviorSequenceBuilder

class BehaviorStatsModule:

    def __init__(self, work_start: int = 8, work_end: int = 18, user_ids: List[str] = None):

        self.data = {}
        self.user_baselines = {}
        self.abnormal_signals = []   

        self.work_start = work_start
        self.work_end = work_end

        self.thresholds = {
            'login_count_std': 2.0,
            'login_time_std': 1.75,
            'usb_usage_std': 2,  
            'email_count_std': 0.5,
            'file_count_std': 2,  
            'http_count_std': 0.5,
            'weekend_login_std': 1.25,
            'non_work_hour_login_std': 999,
            'non_work_hour_usb_std': 999.0,
            'non_work_hour_email_std': 999.0,
            'job_site_visit_std': 999,  
            'leak_site_visit_std': 999,  
            'external_email_ratio_std': 999,  

            'non_work_hour_login_first_seen': 0.5,  
            'non_work_hour_usb_first_seen': 0.05,
            'non_work_hour_email_first_seen': 999,
            'external_email_ratio_first_seen': 999,
            'job_site_visit_first_seen': 0.5,
            'leak_site_visit_first_seen': 0.5,
        }

        self.disabled_anomaly_types = self._get_disabled_anomaly_types()

        builder = UserBehaviorSequenceBuilder(data={}, include_content=False)
        self.sequences = builder.load_sequences(user_ids=user_ids)  
        self._convert_sequences_to_dataframes()
        self.user_profiles = self._load_user_profiles(user_ids=user_ids)
        self.termination_months = {}  
        for uid in self.sequences.keys():  
            profile = self.user_profiles.get(uid, {})
            term = profile.get('termination_month', None)
            if term:
                self.termination_months[uid] = term

        self.job_site_keywords = [
            'linkedin.com', 'monster.com', 'careerbuilder.com',
            'simplyhired.com', 'job-hunt.org', 'jobhuntersbible.com',
            'indeed.com', 'elance.com', 'freelancer.com', 'taleo.net'
        ]
        self.leak_site_keywords = [
            'mediafire.com', 'wikileaks.org', 'thepiratebay.org',
            'yousendit.com', 'filesonic.com', 'megaupload.com',
            'fileserve.com', 'torrentz.eu', 'demonoid.me',
            'kat.ph', 'btjunkie.org'
        ]
        self.internal_domains = ['dtaa.com']

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

    def _clean_baseline_for_export(self, baseline):

        import copy
        cleaned = copy.deepcopy(baseline)

        if 'non_work_hour_login' in self.disabled_anomaly_types:
            cleaned.get('login', {}).pop('non_work_ratio_avg', None)
            cleaned.get('login', {}).pop('non_work_ratio_std', None)

        if 'weekend_login' in self.disabled_anomaly_types:
            cleaned.get('login', {}).pop('weekend_avg_per_day', None)
            cleaned.get('login', {}).pop('weekend_std_per_day', None)

        if 'non_work_hour_usb' in self.disabled_anomaly_types:
            cleaned.get('usb', {}).pop('non_work_ratio_avg', None)
            cleaned.get('usb', {}).pop('non_work_ratio_std', None)

        if 'non_work_hour_email' in self.disabled_anomaly_types:
            cleaned.get('email', {}).pop('non_work_ratio_avg', None)
            cleaned.get('email', {}).pop('non_work_ratio_std', None)

        if 'external_email_ratio_anomaly' in self.disabled_anomaly_types:
            cleaned.get('email', {}).pop('external_avg_per_day', None)
            cleaned.get('email', {}).pop('external_std_per_day', None)
            cleaned.get('email', {}).pop('external_ratio_avg', None)
            cleaned.get('email', {}).pop('external_ratio_std', None)

        if 'job_site_visit_spike' in self.disabled_anomaly_types:
            cleaned.get('http', {}).pop('job_site_visits', None)
            cleaned.get('http', {}).pop('job_site_avg_per_day', None)
            cleaned.get('http', {}).pop('job_site_std_per_day', None)

        if 'leak_site_visit_spike' in self.disabled_anomaly_types:
            cleaned.get('http', {}).pop('leak_site_visits', None)
            cleaned.get('http', {}).pop('leak_site_avg_per_day', None)
            cleaned.get('http', {}).pop('leak_site_std_per_day', None)

        return cleaned

    def _get_disabled_anomaly_types(self):

        threshold_to_type = {
            'login_time_std': 'abnormal_login_time',
            'weekend_login_std': 'weekend_login',
            'non_work_hour_login_std': 'non_work_hour_login',
            'non_work_hour_usb_std': 'non_work_hour_usb',
            'non_work_hour_email_std': 'non_work_hour_email',
            'job_site_visit_std': 'job_site_visit_spike',
            'leak_site_visit_std': 'leak_site_visit_spike',
            'external_email_ratio_std': 'external_email_ratio_anomaly',
        }
        return [anomaly_type for threshold_key, anomaly_type in threshold_to_type.items()
                if self.thresholds.get(threshold_key, 0) >= 999]

    def filter_disabled_anomalies(self):

        filtered_signals = []
        removed_count = 0
        for signal_entry in self.abnormal_signals:

            filtered_entry_signals = [
                s for s in signal_entry['signals']
                if s['type'] not in self.disabled_anomaly_types
            ]
            if filtered_entry_signals:
                signal_entry['signals'] = filtered_entry_signals
                signal_entry['signal_count'] = len(filtered_entry_signals)
                filtered_signals.append(signal_entry)
            else:
                removed_count += 1
        self.abnormal_signals = filtered_signals
    def _convert_sequences_to_dataframes(self):

        logon_events, device_events, http_events, email_events, file_events = [], [], [], [], []

        for user_id, events in self.sequences.items():
            for event in events:
                event_with_user = event.copy()
                event_with_user['user'] = user_id
                event_with_user['datetime'] = event['timestamp']
                event_with_user['date_only'] = event_with_user['datetime'].date()
                event_with_user['hour'] = event_with_user['datetime'].hour
                event_with_user['is_weekend'] = event_with_user['datetime'].weekday() >= 5
                event_with_user['is_work_hour'] = self.work_start <= event_with_user['hour'] < self.work_end

                if event['type'] == 'logon':
                    logon_events.append(event_with_user)
                elif event['type'] == 'usb':
                    device_events.append(event_with_user)
                elif event['type'] == 'web':
                    http_events.append(event_with_user)
                elif event['type'] == 'email':
                    email_events.append(event_with_user)
                elif event['type'] == 'file':
                    file_events.append(event_with_user)

        self.data['logon'] = pd.DataFrame(logon_events) if logon_events else pd.DataFrame()
        self.data['device'] = pd.DataFrame(device_events) if device_events else pd.DataFrame()
        self.data['http'] = pd.DataFrame(http_events) if http_events else pd.DataFrame()
        self.data['email'] = pd.DataFrame(email_events) if email_events else pd.DataFrame()
        self.data['file'] = pd.DataFrame(file_events) if file_events else pd.DataFrame()

        for key in ['logon', 'device', 'http', 'email', 'file']:
            if not self.data[key].empty:
                self.data[key].set_index(['user', 'date_only'], inplace=True)
                self.data[key].sort_index(inplace=True)

    def _load_user_profiles(self, user_ids: List[str] = None):

        import os
        import json

        profiles = {}
        profile_dir = USER_PROFILE_DIR
        if not os.path.exists(profile_dir):
            return profiles

        if user_ids is not None:
            for user_id in user_ids:
                safe_user_id = user_id.replace('/', '_').replace('\\', '_')
                file_path = os.path.join(profile_dir, f"{safe_user_id}.json")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            profiles[user_id] = json.load(f)
                    except Exception as e:
                        print(f"加载用户画像失败 {file_path}: {e}")
        else:

            for filename in os.listdir(profile_dir):
                if filename.endswith('.json'):
                    user_id = filename.replace('.json', '')
                    file_path = os.path.join(profile_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            profiles[user_id] = json.load(f)
                    except Exception as e:
                        print(f"加载用户画像失败 {file_path}: {e}")

        return profiles

    def build_login_baseline(self, user_id):

        if 'logon' not in self.data or self.data['logon'].empty:
            return {}

        try:
            user_logons = self.data['logon'].xs(user_id, level='user')
            term_month = self.termination_months.get(user_id)
            if term_month and term_month is not None and not user_logons.empty:
                term_date = pd.to_datetime(str(term_month) + '-01')
                user_logons = user_logons[user_logons['datetime'] < term_date]
        except KeyError:
            user_logons = pd.DataFrame()

        if user_logons.empty:
            return {
                'login_count_total': 0,
                'login_days': 0,
                'avg_login_per_day': 0,
                'typical_login_hours': {
                    'mean_hour': None,
                    'std_hour': None,
                    'min_hour': None,
                    'max_hour': None
                },
                'weekend_login_ratio': 0,
                'unique_pcs': [],
                'non_work_hour_ratio': 0,
                'daily_stats': {
                    'mean': 0,
                    'std': 0,
                    'max': 0,
                    'min': 0
                }
            }

        login_hours = user_logons['hour'].dropna()

        weekend_logins = user_logons[user_logons['is_weekend'] == True]

        daily_logins = user_logons.groupby(level='date_only').size()

        daily_weekend = user_logons[user_logons['is_weekend'] == True].groupby(level='date_only').size()

        daily_non_work = user_logons[user_logons['is_work_hour'] == False].groupby(level='date_only').size()
        daily_non_work_ratio = (daily_non_work / daily_logins).fillna(0)

        baseline = {

            'login_count_total': len(user_logons),  
            'login_days': len(user_logons.groupby(level='date_only').size()), 

            'avg_login_per_day': len(user_logons) / max(len(user_logons.index.get_level_values('date_only').unique()), 1),

            'typical_login_hours': {
                'mean_hour': login_hours.mean() if not login_hours.empty else None,  
                'std_hour': login_hours.std() if not login_hours.empty else None,  
                'min_hour': login_hours.min() if not login_hours.empty else None,  
                'max_hour': login_hours.max() if not login_hours.empty else None  
            },

            'weekend_login_ratio': len(weekend_logins) / len(user_logons) if len(user_logons) > 0 else 0,

            'unique_pcs': user_logons['pc'].unique().tolist(),
            'non_work_hour_ratio': len(user_logons[user_logons['is_work_hour'] == False]) / len(user_logons) if len(
                user_logons) > 0 else 0,
            'daily_stats': {
                'mean': daily_logins.mean(),
                'std': daily_logins.std(),
                'max': daily_logins.max(),
                'min': daily_logins.min()
            },
            'weekend_avg_per_day': daily_weekend.mean() if len(daily_weekend) > 0 else 0,
            'weekend_std_per_day': daily_weekend.std() if len(daily_weekend) > 1 else 0,
            'non_work_ratio_avg': daily_non_work_ratio.mean() if len(daily_non_work_ratio) > 0 else 0,
            'non_work_ratio_std': daily_non_work_ratio.std() if len(daily_non_work_ratio) > 1 else 0,
        }

        return baseline

    def build_usb_baseline(self, user_id):

        if 'device' not in self.data or self.data['device'].empty:
            return {}

        try:
            user_devices = self.data['device'].xs(user_id, level='user')
            term_month = self.termination_months.get(user_id)
            if term_month and term_month is not None and not user_devices.empty:
                term_date = pd.to_datetime(str(term_month) + '-01')
                user_devices = user_devices[user_devices['datetime'] < term_date]
        except KeyError:
            user_devices = pd.DataFrame()

        if user_devices.empty:
            return {}

        connects = user_devices[user_devices['activity'].str.lower() == 'connect']

        if connects.empty:
            return {
                'total_connects': 0,
                'unique_days': 0,
                'avg_per_day': 0,
                'std_per_day': 0,
                'max_per_day': 0,
                'unique_devices': 0,
                'non_work_hour_ratio': 0
            }

        daily_connects = connects.groupby(level='date_only').size()
        daily_non_work_usb = connects[connects['is_work_hour'] == False].groupby(level='date_only').size()
        daily_non_work_usb_ratio = (daily_non_work_usb / daily_connects).fillna(0)

        baseline = {
            'total_connects': len(connects),  
            'unique_days': len(connects.index.get_level_values('date_only').unique()),  
            'avg_per_day': daily_connects.mean() if not daily_connects.empty else 0,  
            'std_per_day': daily_connects.std() if len(daily_connects) > 1 else 0,  
            'max_per_day': daily_connects.max() if not daily_connects.empty else 0,  
            'unique_devices': connects['pc'].nunique(),  
            'non_work_hour_ratio': len(connects[connects['is_work_hour'] == False]) / len(connects) if len(connects) > 0 else 0,
            'non_work_ratio_avg': daily_non_work_usb_ratio.mean() if len(daily_non_work_usb_ratio) > 0 else 0,
            'non_work_ratio_std': daily_non_work_usb_ratio.std() if len(daily_non_work_usb_ratio) > 1 else 0,
        }

        return baseline

    def build_email_baseline(self, user_id):

        if 'email' not in self.data or self.data['email'].empty:
            return {}

        try:
            user_emails = self.data['email'].xs(user_id, level='user')
            term_month = self.termination_months.get(user_id)
            if term_month and not user_emails.empty:
                term_date = pd.to_datetime(str(term_month) + '-01')
                user_emails = user_emails[user_emails['datetime'] < term_date]
        except KeyError:
            user_emails = pd.DataFrame()

        if user_emails.empty:
            return {
                'total_emails': 0,
                'unique_days': 0,
                'avg_per_day': 0,
                'std_per_day': 0,
                'max_per_day': 0,
                'avg_recipients': 0,
                'avg_attachment': 0,
                'non_work_hour_ratio': 0,
                'internal_emails': 0,
                'external_emails': 0
            }

        daily_emails = user_emails.groupby(level='date_only').size()

        internal_count = 0
        external_count = 0
        for _, email in user_emails.iterrows():
            direction = self.extract_email_direction(email.to_dict())
            if 'internal email' in direction or 'insider' in direction:
                internal_count += 1
            if 'external email' in direction or 'outsider' in direction:
                external_count += 1

        def _count_external(to_field):
            if not to_field:
                return 0
            return sum(1 for addr in str(to_field).replace(';', ',').split(',')
                       if addr.strip() and '@' in addr and 'dtaa.com' not in addr.lower())

        daily_external = user_emails.groupby(level='date_only').apply(
            lambda x: sum(_count_external(to) for to in x['to'])
        )
        daily_total = user_emails.groupby(level='date_only').size()
        daily_non_work_email = user_emails[user_emails['is_work_hour'] == False].groupby(level='date_only').size()
        daily_non_work_email_ratio = (daily_non_work_email / daily_total).fillna(0)
        daily_external_ratio = (daily_external / daily_total).fillna(0)

        baseline = {
            'total_emails': len(user_emails),  
            'unique_days': len(user_emails.index.get_level_values('date_only').unique()),  
            'avg_per_day': daily_emails.mean() if not daily_emails.empty else 0,  
            'std_per_day': daily_emails.std() if len(daily_emails) > 1 else 0,  
            'max_per_day': daily_emails.max() if not daily_emails.empty else 0,  

            'avg_recipients': user_emails['to'].str.count(',').mean() + 1 if 'to' in user_emails.columns else 0,

            'avg_attachment': user_emails['attachment_count'].mean() if 'attachment_count' in user_emails.columns else 0,

            'non_work_hour_ratio': len(user_emails[user_emails['is_work_hour'] == False]) / len(user_emails) if len(
                user_emails) > 0 else 0,
            'non_work_ratio_avg': daily_non_work_email_ratio.mean() if len(daily_non_work_email_ratio) > 0 else 0,
            'non_work_ratio_std': daily_non_work_email_ratio.std() if len(daily_non_work_email_ratio) > 1 else 0,
            'internal_emails': internal_count,
            'external_emails': external_count,
            'external_avg_per_day': daily_external.mean() if len(daily_external) > 0 else 0,
            'external_std_per_day': daily_external.std() if len(daily_external) > 1 else 0,
            'external_ratio_avg': daily_external_ratio.mean() if len(daily_external_ratio) > 0 else 0,
            'external_ratio_std': daily_external_ratio.std() if len(daily_external_ratio) > 1 else 0,
        }

        return baseline

    def build_file_baseline(self, user_id):

        if 'file' not in self.data or self.data['file'].empty:
            return {}

        try:
            user_files = self.data['file'].xs(user_id, level='user')
            term_month = self.termination_months.get(user_id)
            if term_month and not user_files.empty:
                term_date = pd.to_datetime(str(term_month) + '-01')
                user_files = user_files[user_files['datetime'] < term_date]
        except KeyError:
            user_files = pd.DataFrame()

        if user_files.empty:
            return {
                'total_files': 0,
                'unique_days': 0,
                'avg_per_day': 0,
                'std_per_day': 0,
                'max_per_day': 0,
                'non_work_hour_ratio': 0,
                'file_type_count': 0
            }

        daily_files = user_files.groupby(level='date_only').size()

        def _extract_extension(filename):

            if not filename or '.' not in str(filename):
                return 'no_extension'
            return str(filename).split('.')[-1].lower()

        unique_extensions = set()
        for _, row in user_files.iterrows():
            ext = _extract_extension(row.get('filename', ''))
            unique_extensions.add(ext)

        baseline = {
            'total_files': len(user_files),  
            'unique_days': len(user_files.index.get_level_values('date_only').unique()),  
            'avg_per_day': daily_files.mean() if not daily_files.empty else 0,  
            'std_per_day': daily_files.std() if len(daily_files) > 1 else 0,  
            'max_per_day': daily_files.max() if not daily_files.empty else 0,  
            'non_work_hour_ratio': len(user_files[user_files['is_work_hour'] == False]) / len(user_files) if len(
                user_files) > 0 else 0,
            'file_type_count': len(unique_extensions),
        }

        return baseline

    def build_http_baseline(self, user_id):

        if 'http' not in self.data or self.data['http'].empty:
            return {}

        try:
            user_http = self.data['http'].xs(user_id, level='user')
            term_month = self.termination_months.get(user_id)
            if term_month and not user_http.empty:
                term_date = pd.to_datetime(str(term_month) + '-01')
                user_http = user_http[user_http['datetime'] < term_date]
        except KeyError:
            user_http = pd.DataFrame()

        if user_http.empty:
            return {
                'total_http': 0,
                'unique_days': 0,
                'avg_per_day': 0,
                'std_per_day': 0,
                'max_per_day': 0,
                'non_work_hour_ratio': 0,
                'job_site_visits': 0,
                'job_site_avg_per_day': 0,
                'job_site_std_per_day': 0,
                'leak_site_visits': 0,
                'leak_site_avg_per_day': 0,
                'leak_site_std_per_day': 0,
            }

        daily_http = user_http.groupby(level='date_only').size()

        job_visits = 0
        leak_visits = 0
        for _, row in user_http.iterrows():
            url = str(row.get('url', '')).lower()
            if any(kw in url for kw in self.job_site_keywords):
                job_visits += 1
            if any(kw in url for kw in self.leak_site_keywords):
                leak_visits += 1

        daily_job = user_http.groupby(level='date_only').apply(
            lambda x: sum(any(kw in str(url).lower() for kw in self.job_site_keywords) for url in x.get('url', ''))
        )
        daily_leak = user_http.groupby(level='date_only').apply(
            lambda x: sum(any(kw in str(url).lower() for kw in self.leak_site_keywords) for url in x.get('url', ''))
        )

        baseline = {
            'total_http': len(user_http),
            'unique_days': len(user_http.index.get_level_values('date_only').unique()),
            'avg_per_day': daily_http.mean() if not daily_http.empty else 0,
            'std_per_day': daily_http.std() if len(daily_http) > 1 else 0,
            'max_per_day': daily_http.max() if not daily_http.empty else 0,
            'non_work_hour_ratio': len(user_http[user_http['is_work_hour'] == False]) / len(user_http) if len(
    user_http) > 0 else 0,
            'job_site_visits': job_visits,
            'job_site_avg_per_day': daily_job.mean() if len(daily_job) > 0 else 0,
            'job_site_std_per_day': daily_job.std() if len(daily_job) > 1 else 0,
            'leak_site_visits': leak_visits,
            'leak_site_avg_per_day': daily_leak.mean() if len(daily_leak) > 0 else 0,
            'leak_site_std_per_day': daily_leak.std() if len(daily_leak) > 1 else 0,
        }

        return baseline

    def build_user_baseline(self, user_id):

        baseline = {
            'user_id': user_id,
            'login': self.build_login_baseline(user_id),  
            'usb': self.build_usb_baseline(user_id),  
            'email': self.build_email_baseline(user_id),  
            'file': self.build_file_baseline(user_id),  
            'http': self.build_http_baseline(user_id),  
        }

        self.user_baselines[user_id] = baseline
        return baseline

    def detect_login_anomaly(self, user_id, date):

        baseline = self.user_baselines.get(user_id, {}).get('login', {})
        if not baseline:
            return None

        if 'logon' not in self.data:
            return None

        try:
            day_logons = self.data['logon'].xs((user_id, date), level=['user', 'date_only'])
        except KeyError:
            day_logons = pd.DataFrame()

        signals = []

        term_month = self.termination_months.get(user_id)
        if term_month and not day_logons.empty:
            term_date = pd.to_datetime(str(term_month) + '-01')
            post_start = term_date + pd.DateOffset(months=1)
            if pd.to_datetime(date) >= post_start:
                signals.append({
                    'type': 'post_termination_login',
                    'date_only': date,
                    'count': len(day_logons),

                    'description': f'离职后异常登录: {len(day_logons)}次'
                })

        if not day_logons.empty:
            std_hour = baseline.get('typical_login_hours', {}).get('std_hour')
            if std_hour and std_hour > 0:
                mean_hour = baseline['typical_login_hours']['mean_hour']

            for _, logon in day_logons.iterrows():
                hour = logon.get('hour')
                if hour is not None and std_hour and std_hour > 0:

                    z_score = abs(hour - mean_hour) / std_hour

                    if z_score > self.thresholds['login_time_std']:
                        signals.append({
                            'type': 'abnormal_login_time',
                            'date_only': date,
                            'hour': hour,
                            'z_score': z_score,
                            'description': f'异常登录时间: {hour}:00 (正常范围: {mean_hour - std_hour:.0f}-{mean_hour + std_hour:.0f})'
                        })

        dt = pd.to_datetime(date)
        if dt.weekday() >= 5 and not day_logons.empty:
            weekend_count = len(day_logons)
            weekend_avg = baseline.get('weekend_avg_per_day', 0)
            weekend_std = baseline.get('weekend_std_per_day', 0)
            if weekend_std is not None and weekend_std >= 0:
                if weekend_std == 0:
                    if weekend_count > weekend_avg:
                        weekend_z = weekend_count
                else:
                    weekend_z = (weekend_count - weekend_avg) / weekend_std
                if (weekend_std == 0 and weekend_count > weekend_avg) or (weekend_std > 0 and weekend_z > self.thresholds['weekend_login_std']):
                    signals.append({
                        'type': 'weekend_login',
                        'date_only': date,
                        'count': weekend_count,
                        'z_score': weekend_z,
                        'description': f'周末异常登录: {weekend_count}次'
                    })

        if not day_logons.empty:
            non_work_count = len(day_logons[day_logons['is_work_hour'] == False])
            if non_work_count > 0:
                day_non_work_ratio = non_work_count / len(day_logons)
                ratio_avg = baseline.get('non_work_ratio_avg', 0)
                ratio_std = baseline.get('non_work_ratio_std', 0)
                if ratio_std is not None and ratio_std >= 0:
                    if ratio_std == 0:

                        if day_non_work_ratio > self.thresholds.get('non_work_hour_login_first_seen', 999):
                            signals.append({
                                'type': 'non_work_hour_login',
                                'date_only': date,
                                'count': non_work_count,
                                'z_score': day_non_work_ratio,
                                'description': f'首次非工作时段登录异常: {non_work_count}次 ({day_non_work_ratio:.0%})'
                            })
                    else:
                        ratio_z = (day_non_work_ratio - ratio_avg) / ratio_std
                        if ratio_z > self.thresholds['non_work_hour_login_std']:
                            signals.append({
                                'type': 'non_work_hour_login',
                                'date_only': date,
                                'count': non_work_count,
                                'z_score': ratio_z,
                                'description': f'非工作时段登录异常: {non_work_count}次'
                            })

        if not day_logons.empty:
            user_pcs = set(baseline.get('unique_pcs', []))
            if isinstance(user_pcs, list) and len(user_pcs) > 0:
                for _, logon in day_logons.iterrows():
                    pc = logon.get('pc')

                    if pc and pc not in user_pcs:
                        signals.append({
                            'type': 'login_other_pc',
                            'date_only': date,
                            'pc': pc,
                            'description': f'登录其他用户设备: {pc}'
                        })

        if not day_logons.empty:
            count = len(day_logons)
            avg = baseline.get('daily_stats', {}).get('mean', 0)
            std = baseline.get('daily_stats', {}).get('std', 0)
            if std and std > 0:
                std_val = max(std, 1)
                z_score = (count - avg) / std_val
                if z_score > self.thresholds['login_count_std']:
                    signals.append({
                        'type': 'login_count_anomaly',
                        'date_only': date,
                        'count': count,
                        'avg': avg,
                        'z_score': z_score,
                        'description': f'登录次数异常: {count}次 (平时日均{avg:.1f}次)'
                    })
        return signals

    def detect_usb_anomaly(self, user_id, date):

        baseline = self.user_baselines.get(user_id, {}).get('usb', {})
        if not baseline:
            return None

        if 'device' not in self.data:
            return None

        try:
            day_data = self.data['device'].xs((user_id, date), level=['user', 'date_only'])
            day_connects = day_data[day_data['activity'].str.lower() == 'connect']
        except KeyError:
            day_connects = pd.DataFrame()

        signals = []
        count = len(day_connects)

        term_month = self.termination_months.get(user_id)
        if term_month and count > 0:
            term_date = pd.to_datetime(str(term_month) + '-01')
            post_start = term_date + pd.DateOffset(months=1)
            if pd.to_datetime(date) >= post_start:
                signals.append({
                    'type': 'post_termination_usb',
                    'date_only': date,
                    'count': count,

                    'description': f'离职后异常U盘使用: {count}次'
                })

        if count > 0:

            avg = baseline.get('avg_per_day', 0)
            std = baseline.get('std_per_day', 0)
            if std and std > 0:
                std_val = max(std, 1)
                z_score = (count - avg) / std_val
                if z_score > self.thresholds['usb_usage_std']:
                    signals.append({
                        'type': 'usb_usage_spike',
                        'date_only': date,
                        'count': count,
                        'avg': avg,
                        'z_score': z_score,
                        'description': f'U盘使用量暴增: {count}次 (平时平均{avg:.1f}次)'
                    })

            non_work_connects = day_connects[day_connects['is_work_hour'] == False]
            if len(non_work_connects) > 0 and count > 0:
                day_non_work_ratio = len(non_work_connects) / count
                ratio_avg = baseline.get('non_work_ratio_avg', 0)
                ratio_std = baseline.get('non_work_ratio_std', 0)
                if ratio_std is not None and ratio_std >= 0:
                    if ratio_std == 0:
                        if day_non_work_ratio > self.thresholds.get('non_work_hour_usb_first_seen', 999):
                            signals.append({
                                'type': 'non_work_hour_usb',
                                'date_only': date,
                                'count': len(non_work_connects),
                                'z_score': day_non_work_ratio,
                                'description': f'首次非工作时段异常使用U盘: {len(non_work_connects)}次 ({day_non_work_ratio:.0%})'
                            })
                    else:
                        ratio_z = (day_non_work_ratio - ratio_avg) / ratio_std
                        if ratio_z > self.thresholds['non_work_hour_usb_std']:
                            signals.append({
                                'type': 'non_work_hour_usb',
                                'date_only': date,
                                'count': len(non_work_connects),
                                'z_score': ratio_z,
                                'description': f'非工作时段U盘使用异常: {len(non_work_connects)}次'
                            })

        return signals

    def detect_email_anomaly(self, user_id, date):

        baseline = self.user_baselines.get(user_id, {}).get('email', {})
        if not baseline:
            return None

        if 'email' not in self.data:
            return None

        try:
            day_emails = self.data['email'].xs((user_id, date), level=['user', 'date_only'])
        except KeyError:
            day_emails = pd.DataFrame()

        signals = []
        count = len(day_emails)

        term_month = self.termination_months.get(user_id)
        if term_month and count > 0:
            term_date = pd.to_datetime(str(term_month) + '-01')
            post_start = term_date + pd.DateOffset(months=1)
            if pd.to_datetime(date) >= post_start:
                signals.append({
                    'type': 'post_termination_email',
                    'date_only': date,
                    'count': count,

                    'description': f'离职后异常发送邮件: {count}封'
                })

        day_internal = 0
        day_external = 0
        if count > 0:
            for _, email in day_emails.iterrows():
                direction = self.extract_email_direction(email.to_dict())
                if 'internal email' in direction or 'insider' in direction:
                    day_internal += 1
                if 'external email' in direction or 'outsider' in direction:
                    day_external += 1

        if count > 0:
            avg = baseline.get('avg_per_day', 0)
            std = baseline.get('std_per_day', 0)
            if std and std > 0:
                std_val = max(std, 2)
                z_score = (count - avg) / std_val
                if z_score > self.thresholds['email_count_std']:
                    signals.append({
                        'type': 'email_count_anomaly',
                        'date_only': date,
                        'count': count,
                        'avg': avg,
                        'z_score': z_score,
                        'internal_count': day_internal,
                        'external_count': day_external,
                        'description': f'邮件数量异常: {count}封 (平时平均{avg:.1f}封)'
                    })

            non_work_emails = day_emails[day_emails['is_work_hour'] == False]
            if len(non_work_emails) > 0 and count > 0:
                day_non_work_ratio = len(non_work_emails) / count
                ratio_avg = baseline.get('non_work_ratio_avg', 0)
                ratio_std = baseline.get('non_work_ratio_std', 0)
                if ratio_std is not None and ratio_std >= 0:
                    if ratio_std == 0:
                        if day_non_work_ratio > self.thresholds.get('non_work_hour_email_first_seen', 999):
                            signals.append({
                                'type': 'non_work_hour_email',
                                'date_only': date,
                                'count': len(non_work_emails),
                                'z_score': day_non_work_ratio,
                                'description': f'首次非工作时段邮件: {len(non_work_emails)}封 ({day_non_work_ratio:.0%})'
                            })
                    else:
                        ratio_z = (day_non_work_ratio - ratio_avg) / ratio_std
                        if ratio_z > self.thresholds['non_work_hour_email_std']:
                            signals.append({
                                'type': 'non_work_hour_email',
                                'date_only': date,
                                'count': len(non_work_emails),
                                'z_score': ratio_z,
                                'description': f'非工作时段邮件异常: {len(non_work_emails)}封'
                            })

        if count > 0:
            day_external_ratio = day_external / count
            ratio_avg = baseline.get('external_ratio_avg', 0)
            ratio_std = baseline.get('external_ratio_std', 0)
            if ratio_std is not None and ratio_std >= 0:
                if ratio_std == 0:
                    if day_external_ratio > self.thresholds.get('external_email_ratio_first_seen', 999):
                        signals.append({
                            'type': 'external_email_ratio_anomaly',
                            'date_only': date,
                            'count': day_external,
                            'ratio': round(day_external_ratio, 2),
                            'z_score': day_external_ratio,
                            'description': f'首次外部邮件占比异常: {day_external_ratio:.0%}'
                        })
                else:
                    ratio_z = (day_external_ratio - ratio_avg) / ratio_std
                    if ratio_z > self.thresholds['external_email_ratio_std']:
                        signals.append({
                            'type': 'external_email_ratio_anomaly',
                            'date_only': date,
                            'count': day_external,
                            'ratio': round(day_external_ratio, 2),
                            'z_score': ratio_z,
                            'description': f'外部邮件占比异常: {day_external_ratio:.0%} (平时{ratio_avg:.0%})'
                        })
        return signals

    def detect_all_anomalies(self, user_id):

        if user_id not in self.user_baselines:
            self.build_user_baseline(user_id)

        all_signals = []

        dates = set()
        for data_type in ['logon', 'device', 'email', 'file', 'http']:
            if data_type in self.data and not self.data[data_type].empty:
                try:
                    user_data = self.data[data_type].xs(user_id, level='user')
                    dates.update(user_data.index.get_level_values('date_only').unique())
                except KeyError:
                    pass

        for date in sorted(dates):
            day_signals = []

            login_signals = self.detect_login_anomaly(user_id, date)
            if login_signals:
                day_signals.extend(login_signals)

            usb_signals = self.detect_usb_anomaly(user_id, date)
            if usb_signals:
                day_signals.extend(usb_signals)

            email_signals = self.detect_email_anomaly(user_id, date)
            if email_signals:
                day_signals.extend(email_signals)

            http_signals = self.detect_http_anomaly(user_id, date)
            if http_signals:
                day_signals.extend(http_signals)

            file_signals = self.detect_file_anomaly(user_id, date)
            if file_signals:
                day_signals.extend(file_signals)

            if day_signals:
                signal_entry = {
                    'user_id': user_id,
                    'date_only': date,
                    'signals': day_signals,
                    'signal_count': len(day_signals)
                }
                all_signals.append(signal_entry)
                self.abnormal_signals.append(signal_entry)

        return all_signals

    def detect_http_anomaly(self, user_id, date):

        baseline = self.user_baselines.get(user_id, {}).get('http', {})
        if not baseline:
            return None

        if 'http' not in self.data:
            return None

        try:
            day_http = self.data['http'].xs((user_id, date), level=['user', 'date_only'])
        except KeyError:
            day_http = pd.DataFrame()

        signals = []
        count = len(day_http)  

        day_job_visits = 0
        day_leak_visits = 0
        if count > 0:
            for _, row in day_http.iterrows():
                url = str(row.get('url', '')).lower()
                if any(kw in url for kw in self.job_site_keywords):
                    day_job_visits += 1
                if any(kw in url for kw in self.leak_site_keywords):
                    day_leak_visits += 1

        job_avg = baseline.get('job_site_avg_per_day', 0)
        job_std = baseline.get('job_site_std_per_day', 0)
        if day_job_visits > 0 and job_std is not None and job_std >= 0:
            if job_std == 0:
                if day_job_visits > self.thresholds.get('job_site_visit_first_seen', 999):
                    signals.append({
                        'type': 'job_site_visit_spike',
                        'date_only': date,
                        'count': day_job_visits,
                        'z_score': float(day_job_visits),
                        'description': f'首次求职网站访问: {day_job_visits}次'
                    })
            else:
                job_z = (day_job_visits - job_avg) / job_std
                if job_z > self.thresholds['job_site_visit_std']:
                    signals.append({
                        'type': 'job_site_visit_spike',
                        'date_only': date,
                        'count': day_job_visits,
                        'avg': job_avg,
                        'z_score': job_z,
                        'description': f'求职网站访问异常: {day_job_visits}次 (日均{job_avg:.1f}次)'
                    })

        leak_avg = baseline.get('leak_site_avg_per_day', 0)
        leak_std = baseline.get('leak_site_std_per_day', 0)
        if day_leak_visits > 0 and leak_std is not None and leak_std >= 0:
            if leak_std == 0:
                if day_leak_visits > self.thresholds.get('leak_site_visit_first_seen', 999):
                    signals.append({
                        'type': 'leak_site_visit_spike',
                        'date_only': date,
                        'count': day_leak_visits,
                        'z_score': float(day_leak_visits),
                        'description': f'首次泄露网站访问: {day_leak_visits}次'
                    })
            else:
                leak_z = (day_leak_visits - leak_avg) / leak_std
                if leak_z > self.thresholds['leak_site_visit_std']:
                    signals.append({
                        'type': 'leak_site_visit_spike',
                        'date_only': date,
                        'count': day_leak_visits,
                        'avg': leak_avg,
                        'z_score': leak_z,
                        'description': f'数据泄露网站访问异常: {day_leak_visits}次 (日均{leak_avg:.1f}次)'
                    })

        if count > 0:
            avg = baseline.get('avg_per_day', 0)  
            std = baseline.get('std_per_day', 0)  

            if std and std > 0:
                std = max(std, 2)

                z_score = (count - avg) / std

                if z_score > self.thresholds['http_count_std']:
                    signals.append({
                        'type': 'http_count_anomaly',  
                        'date_only': date,  
                        'count': count,  
                        'avg': avg,  
                        'z_score': z_score,  

                        'job_site_visits': day_job_visits,
                        'leak_site_visits': day_leak_visits,
                        'description': f'网页浏览激增: {count}次 (平时平均{avg:.1f}次)'
                    })

        return signals

    def detect_file_anomaly(self, user_id, date):

        baseline = self.user_baselines.get(user_id, {}).get('file', {})
        if not baseline:
            return None

        if 'file' not in self.data:
            return None

        try:
            day_files = self.data['file'].xs((user_id, date), level=['user', 'date_only'])
        except KeyError:
            day_files = pd.DataFrame()

        signals = []
        count = len(day_files)  

        term_month = self.termination_months.get(user_id)
        if term_month and count > 0:
            term_date = pd.to_datetime(str(term_month) + '-01')
            post_start = term_date + pd.DateOffset(months=1)
            if pd.to_datetime(date) >= post_start:
                signals.append({
                    'type': 'post_termination_file',
                    'date_only': date,
                    'count': count,

                    'description': f'离职后异常文件操作: {count}次'
                })

        day_extensions = set()
        if count > 0:
            for _, row in day_files.iterrows():
                filename = row.get('filename', '')
                if filename and '.' in str(filename):
                    day_extensions.add(str(filename).split('.')[-1].lower())
                else:
                    day_extensions.add('no_extension')

        if count > 0:
            avg = baseline.get('avg_per_day', 0)  
            std = baseline.get('std_per_day', 0)  
            if std and std > 0:
                std_val = max(std, 2)
                z_score = (count - avg) / std_val
                if z_score > self.thresholds['file_count_std']:
                    signals.append({
                        'type': 'file_count_anomaly',  
                        'date_only': date,  
                        'count': count,  
                        'avg': avg,  
                        'z_score': z_score,  
                        'file_type_count': len(day_extensions),
                        'description': f'文件复制暴增: {count}次 (平时平均{avg:.1f}次)'
                    })

        return signals

    def get_behavior_summary(self, user_id):

        if user_id not in self.user_baselines:
            return None

        baseline = self.user_baselines[user_id]
        anomalies = self.detect_all_anomalies(user_id)

        filtered_anomalies = []
        for anomaly_entry in anomalies:
            filtered_signals = [
                s for s in anomaly_entry['signals']
                if s['type'] not in self.disabled_anomaly_types
            ]
            if filtered_signals:
                anomaly_entry['signals'] = filtered_signals
                anomaly_entry['signal_count'] = len(filtered_signals)
                filtered_anomalies.append(anomaly_entry)
        anomalies = filtered_anomalies

        summary = {
            'user_id': user_id,
            '行为基线': {
                '登录行为': {
                    '平均每日登录': baseline.get('login', {}).get('avg_login_per_day', 0),
                    '典型登录时间': f"{baseline.get('login', {}).get('typical_login_hours', {}).get('mean_hour', 0):.0f}:00",
                    '周末登录比例': f"{baseline.get('login', {}).get('weekend_login_ratio', 0) * 100:.1f}%",
                    '非工作时段登录比例': f"{baseline.get('login', {}).get('non_work_hour_ratio', 0) * 100:.1f}%",
                },
                'U盘使用': {
                    '平均每日使用': f"{baseline.get('usb', {}).get('avg_per_day', 0):.2f}次",
                    '最大单日使用': baseline.get('usb', {}).get('max_per_day', 0),
                    '非工作时段使用比例': f"{baseline.get('usb', {}).get('non_work_hour_ratio', 0) * 100:.1f}%",
                },
                '邮件行为': {
                    '平均每日邮件': f"{baseline.get('email', {}).get('avg_per_day', 0):.1f}封",
                    '平均收件人数': f"{baseline.get('email', {}).get('avg_recipients', 0):.1f}人",
                    '非工作时段邮件比例': f"{baseline.get('email', {}).get('non_work_hour_ratio', 0) * 100:.1f}%",
                },
                '文件操作': {
                    '平均每日操作': f"{baseline.get('file', {}).get('avg_per_day', 0):.1f}次",
                    '非工作时段操作比例': f"{baseline.get('file', {}).get('non_work_hour_ratio', 0) * 100:.1f}%",
                }
            },
            '异常信号': {
                '异常天数': len(anomalies),
                '异常详情': anomalies
            }
        }

        return summary

    def export_all_baselines(self, output_dir=None):

        import json

        if output_dir is None:
            output_dir = BEHAVIOR_STATISTICS_DIR

        os.makedirs(output_dir, exist_ok=True)

        for user_id, baseline in self.user_baselines.items():
            safe_user_id = user_id.replace('/', '_').replace('\\', '_')
            file_path = os.path.join(output_dir, f"{safe_user_id}_baseline.json")

            cleaned_baseline = self._clean_baseline_for_export(baseline)
            baseline_copy = json.loads(json.dumps(cleaned_baseline, default=str))
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(baseline_copy, f, ensure_ascii=False, indent=2)

        print(f"已导出 {len(self.user_baselines)} 个用户的行为基线到 {output_dir}")

    def export_anomalies(self, output_dir=None):

        import json
        from collections import defaultdict

        self.filter_disabled_anomalies()
        if output_dir is None:
            output_dir = os.path.join(BEHAVIOR_STATISTICS_DIR, "anomalies")

        os.makedirs(output_dir, exist_ok=True)

        user_anomalies = defaultdict(list)
        for signal in self.abnormal_signals:
            user_anomalies[signal['user_id']].append(signal)

        for user_id, anomalies in user_anomalies.items():
            safe_user_id = user_id.replace('/', '_').replace('\\', '_')
            file_path = os.path.join(output_dir, f"{safe_user_id}_anomalies.json")

            anomalies_copy = json.loads(json.dumps(anomalies, default=str))
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(anomalies_copy, f, ensure_ascii=False, indent=2)

        print(f"已导出 {len(user_anomalies)} 个用户的异常信号到 {output_dir}")

def main():

    import json
    import argparse

    parser = argparse.ArgumentParser(description="行为统计分析模块")
    parser.add_argument('--user-ids', type=str, default=None,
                        help='指定要分析的用户ID，逗号分隔（如: BSS0369,ABC0174,JLM0364）')
    parser.add_argument('--work-start', type=int, default=8,
                        help='工作时段开始时间（小时）')
    parser.add_argument('--work-end', type=int, default=18,
                        help='工作时段结束时间（小时）')
    args = parser.parse_args()

    user_ids = ['MOS0047']
    if args.user_ids:
        user_ids = [uid.strip() for uid in args.user_ids.split(',')]
        print(f"指定分析用户: {user_ids}")

    print("=" * 60)
    print("初始化行为统计分析模块...")
    stats_module = BehaviorStatsModule(
        work_start=args.work_start,
        work_end=args.work_end,
        user_ids=user_ids
    )
    print(f"已加载 {len(stats_module.sequences)} 个用户的行为序列")

    target_users = user_ids if user_ids else list(stats_module.sequences.keys())
    print(f"共 {len(target_users)} 个用户待分析")

    print("\n" + "=" * 60)
    for user_id in target_users:
        print(f"处理用户: {user_id}")
        stats_module.build_user_baseline(user_id)
        anomalies = stats_module.detect_all_anomalies(user_id)
        if anomalies:
            print(f"  发现 {len(anomalies)} 天异常")

    print("\n" + "=" * 60)
    stats_module.export_all_baselines()
    stats_module.export_anomalies()

    if target_users:
        print(f"\n用户 {target_users[0]} 行为分析摘要:")
        summary = stats_module.get_behavior_summary(target_users[0])
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print("\n分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

