
from typing import Dict, List

import pandas as pd

import config
from config import USER_PROFILE_DIR
from data_preprocessing import UserBehaviorSequenceBuilder
from multi_modal_features import utils

class UserProfileModule:

    def __init__(self, include_content: bool = False, sequences: Dict[str, List[Dict]] = None):

        self.psychometric_data = None
        self.ldap_data = None
        self.user_profiles = {}

        self.device_user_stats = None

        self.include_content = include_content
        if sequences is not None:

            self.sequences = sequences
            print(f"使用传入的行为序列，共 {len(self.sequences)} 个用户")
        else:

            builder = UserBehaviorSequenceBuilder({}, include_content=include_content)
            self.sequences = builder.load_sequences()
            print(f"自动加载行为序列，共 {len(self.sequences)} 个用户")

        self.all_user_ids = list(self.sequences.keys())
        self.no_assigned_pc_count = {}  

    def load_data(self):

        print(f"-------------加载用户角色画像模块类所需数据--------------")

        try:
            self.psychometric_data = pd.read_csv(config.PSYCHOMETRIC_FILE)
            print(f"加载心理测量数据: {len(self.psychometric_data)} 条记录")
            print(f"心理测量数据字段: {list(self.psychometric_data.columns)}")
        except Exception as e:
            print(f"加载心理测量数据失败: {e}")
            self.psychometric_data = pd.DataFrame()

        try:
            self.ldap_data = utils.load_ldap_data(config.LDAP_PATH)
            print(f"加载LDAP数据: {len(self.ldap_data)} 条记录")
            if not self.ldap_data.empty:
                print(f"LDAP数据字段: {list(self.ldap_data.columns)}")
        except Exception as e:
            print(f"加载LDAP数据失败: {e}")
            self.ldap_data = pd.DataFrame()

        try:
            logon_data = pd.read_csv(config.LOGON_FILE)
            print(f"加载登录数据: {len(logon_data)} 条记录")

            self.device_user_stats = logon_data.groupby('pc').agg(
                unique_users=('user', 'nunique'),
                total_logons=('user', 'count')
            ).reset_index()
        except Exception as e:
            print(f"加载登录数据失败: {e}")

    def build_psychological_profile(self, user_id):

        if self.psychometric_data.empty:
            return {}

        user_psych = self.psychometric_data[
            self.psychometric_data['user_id'] == user_id
            ]

        if user_psych.empty:
            return {}

        psych_data = user_psych.iloc[0]

        profile = {
            'openness': int(psych_data.get('O', 0)) if psych_data.get('O') is not None else None,  
            'conscientiousness': int(psych_data.get('C', 0)) if psych_data.get('C') is not None else None,
            'extraversion': int(psych_data.get('E', 0)) if psych_data.get('E') is not None else None,
            'agreeableness': int(psych_data.get('A', 0)) if psych_data.get('A') is not None else None,
            'neuroticism': int(psych_data.get('N', 0)) if psych_data.get('N') is not None else None
        }

        return profile

    def reconstruct_status_by_month(self, user_id, target_month):

        profile = self.user_profiles.get(user_id)
        if not profile:
            return None

        if target_month < profile['first_month']:
            return None

        if profile.get('termination_month') and target_month > profile['termination_month']:
            return {'employment_status': 'Terminated', 'termination_month': profile['termination_month']}

        baseline = profile['baseline'].copy()
        current_status = {k: v for k, v in baseline.items() if k != 'month'}

        for event in profile.get('change_events', []):

            if event['month'] > target_month:
                break
            for field, change in event['changes'].items():
                current_status[field] = change['to']

        return current_status

    def build_identity_profile(self, user_id):

        if self.ldap_data.empty:
            return {}

        user_ldap = self.ldap_data[self.ldap_data['user_id'] == user_id]
        if user_ldap.empty:
            return {}

        user_ldap_sorted = user_ldap.sort_values('month')

        latest_info = user_ldap_sorted.iloc[-1]

        profile = {
            'employee_name': latest_info.get('employee_name', ''),
            'email': latest_info.get('email', ''),
            'role': latest_info.get('role', 'Unknown'),
            'business_unit': latest_info.get('business_unit', None),
            'functional_unit': latest_info.get('functional_unit', None),
            'department': latest_info.get('department', None),
            'team': latest_info.get('team', None),
            'supervisor': latest_info.get('supervisor', None)
        }

        first_record = user_ldap_sorted.iloc[0]
        profile['baseline'] = {
            'month': first_record['month'],
            'employee_name': first_record.get('employee_name', ''),
            'email': first_record.get('email', ''),
            'role': first_record.get('role', 'Unknown'),
            'business_unit': first_record.get('business_unit', None),
            'functional_unit': first_record.get('functional_unit', None),
            'department': first_record.get('department', None),
            'team': first_record.get('team', None),
            'supervisor': first_record.get('supervisor', None),
            'employment_status': 'Active'
        }

        tracked_fields = ['employee_name', 'role', 'business_unit', 'functional_unit', 'department', 'team', 'supervisor', 'email']
        profile['change_events'] = []

        if len(user_ldap_sorted) > 1:

            prev_record = user_ldap_sorted.iloc[0]

            for month_idx in range(1, len(user_ldap_sorted)):
                curr_record = user_ldap_sorted.iloc[month_idx]
                month = curr_record['month']

                changes_this_month = {}
                for field in tracked_fields:
                    prev_val = prev_record.get(field)
                    curr_val = curr_record.get(field)

                    if prev_val != curr_val:
                        changes_this_month[field] = {'from': prev_val, 'to': curr_val}

                if changes_this_month:
                    profile['change_events'].append({
                        'month': month,
                        'changes': changes_this_month
                    })

                prev_record = curr_record

        profile['is_it_admin'] = 'ITAdmin' in str(profile['role']) if profile['role'] else False

        email = profile['email']
        is_dtaa_email = '@dtaa.com' in str(email)   

        all_months = sorted(self.ldap_data['month'].unique())
        latest_data_month = all_months[-1] if all_months else None
        first_month = user_ldap_sorted.iloc[0]['month']
        last_month = user_ldap_sorted.iloc[-1]['month']

        profile['first_month'] = first_month  
        profile['last_month'] = last_month    

        if last_month < latest_data_month:

            profile['employment_status'] = 'Terminated'
            profile['termination_month'] = last_month

            profile['change_events'].append({
                'month': last_month,
                'changes': {'employment_status': {'from': 'Active', 'to': 'Terminated'}}
            })
        else:

            profile['employment_status'] = 'Active'
            profile['termination_month'] = None

        if profile['is_it_admin']:
            profile['User_Type'] = 'ITAdmin'
        elif is_dtaa_email:
            profile['User_Type'] = 'Employee'
        else:
            profile['User_Type'] = 'Non_Employee'

        return profile

    def _identify_assigned_pc(self, user_events, user_id=None):

        pc_counts = {}
        for e in user_events:
            if e['type'] == 'logon':
                pc = e['pc']
                pc_counts[pc] = pc_counts.get(pc, 0) + 1

        if not pc_counts:
            if user_id:
                reason = "无登录记录"
                self.no_assigned_pc_count[reason] = self.no_assigned_pc_count.get(reason, 0) + 1  
                print(f"[警告] 用户 {user_id} 无专属设备，原因: {reason}")
            return None

        sorted_pcs = sorted(pc_counts.items(), key=lambda x: x[1], reverse=True)
        candidate_pc = sorted_pcs[0][0]

        if self.device_user_stats is not None:
            device_stat = self.device_user_stats[
                self.device_user_stats['pc'] == candidate_pc
                ]
            if not device_stat.empty:
                unique_users = device_stat['unique_users'].values[0]  
                total_logons = device_stat['total_logons'].values[0]  
                user_logons = pc_counts.get(candidate_pc, 0)  

                user_share = user_logons / total_logons if total_logons > 0 else 0
                if user_share < 0.8:
                    reason = f"登录占比{user_share:.1%}(<80%)"
                    self.no_assigned_pc_count[reason] = self.no_assigned_pc_count.get(reason, 0) + 1
                    print(f"[警告] 用户 {user_id} 无专属设备，原因: {reason}")
                    return None

        return candidate_pc

    def build_permission_profile(self, user_id):

        profile = {
            'global_access': False,  
            'Assigned_PC': None,  

            'accessed_devices': []  
        }
        if user_id not in self.sequences:
            return profile

        user_events = self.sequences[user_id]

        accessed_pcs = list(set([e['pc'] for e in user_events if e['type'] == 'logon']))
        profile['accessed_devices'] = accessed_pcs

        if accessed_pcs:

            profile['Assigned_PC'] = self._identify_assigned_pc(user_events, user_id)

        return profile

    def build_user_profile(self, user_id):

        profile = {
            'user_id': user_id
        }

        profile.update(self.build_psychological_profile(user_id))
        profile.update(self.build_identity_profile(user_id))
        profile.update(self.build_permission_profile(user_id))

        profile['global_access'] = profile.get('is_it_admin', False)

        self.user_profiles[user_id] = profile
        return profile

    def get_single_profile_summary(self, user_id):

        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]

        summary = {
            'user_id': user_id,
            '心理属性': {
                '大五人格': {k: v for k, v in profile.items()
                             if k in ['openness', 'conscientiousness',
                                      'extraversion', 'agreeableness', 'neuroticism']},

            },

            '身份属性': {
                 '员工姓名': profile.get('employee_name', '未知'),
                 '邮箱': profile.get('email', '未知'),
                 '角色': profile.get('role', '未知'),
                 '业务单元': profile.get('business_unit', '未知'),
                 '部门': profile.get('department', '未知'),
                 '职能单元': profile.get('functional_unit', '未知'),
                 '团队': profile.get('team', '未知'),
                 '在职状态': profile.get('employment_status', '未知'),
                 '用户类型': profile.get('User_Type', '未知'),  
                 '是否IT管理员': profile.get('is_it_admin', False),
                 '上级': profile.get('supervisor', '未知'),
                 '入职月份': profile.get('first_month', '未知'),
                 '离职月份': profile.get('termination_month', None)
            },
            '权限属性': {
                '专属设备': profile.get('Assigned_PC', '无'),

                '已访问设备数': len(profile.get('accessed_devices', [])),
                '已访问设备': profile.get('accessed_devices', [])[:5],  
                '全局访问权限': profile.get('global_access', False)
            }
        }

        return summary

    def get_llm_profile(self, user_id: str) -> Dict:

        if user_id not in self.user_profiles:
            self.build_user_profile(user_id)

        profile = self.user_profiles.get(user_id, {})

        llm_profile = {"user_id": user_id}

        role = profile.get('role')
        if role and role != 'Unknown' and role != '未知':
            llm_profile['role'] = role

        dept = profile.get('department')
        if dept and dept != 'Unknown' and dept != '未知' and str(dept) != 'nan':
            llm_profile['department'] = dept

        user_type = profile.get('User_Type')
        if user_type:
            llm_profile['user_type'] = user_type

        emp_status = profile.get('employment_status')
        if emp_status:
            llm_profile['employment_status'] = emp_status

        is_admin = profile.get('is_it_admin')
        if is_admin is not None:
            llm_profile['is_it_admin'] = is_admin

        first_month = profile.get('first_month')
        if first_month:
            llm_profile['first_month'] = first_month

        term_month = profile.get('termination_month')
        if term_month:
            llm_profile['termination_month'] = term_month

        assigned_pc = profile.get('Assigned_PC')
        if assigned_pc:
            llm_profile['assigned_pc'] = assigned_pc

        global_access = profile.get('global_access')
        if global_access is not None:
            llm_profile['global_access'] = global_access

        psych_fields = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        psych_values = {}
        for field in psych_fields:
            val = profile.get(field)
            if val is not None and val != 0:
                psych_values[field] = val

        if psych_values:  
            llm_profile['personality'] = psych_values

        return llm_profile

    def get_llm_profile_text(self, user_id: str) -> str:

        profile = self.get_llm_profile(user_id)

        parts = [f"用户 {profile['user_id']}"]

        identity_parts = []
        if profile.get('role'):
            identity_parts.append(f"岗位为{profile['role']}")
        if profile.get('department'):
            identity_parts.append(f"隶属于{profile['department']}")
        if profile.get('user_type'):
            type_map = {'ITAdmin': 'IT管理员', 'Employee': '正式员工', 'Non_Employee': '非员工'}
            identity_parts.append(f"用户类型为{type_map.get(profile['user_type'], profile['user_type'])}")

        if identity_parts:
            parts.append("，".join(identity_parts))

        status_parts = []
        if profile.get('employment_status'):
            status_map = {'Active': '在职', 'Terminated': '已离职'}
            status_parts.append(status_map.get(profile['employment_status'], profile['employment_status']))
        if profile.get('first_month'):
            status_parts.append(f"入职于{profile['first_month']}")
        if profile.get('termination_month'):
            status_parts.append(f"离职于{profile['termination_month']}")

        if status_parts:
            parts.append("；" + "，".join(status_parts))

        perm_parts = []
        if profile.get('assigned_pc'):
            perm_parts.append(f"专属设备为{profile['assigned_pc']}")
        if profile.get('global_access'):
            perm_parts.append("具有全局访问权限")

        if perm_parts:
            parts.append("；" + "，".join(perm_parts))

        if profile.get('personality'):
            psych = profile['personality']
            psych_map = {
                'openness': '开放性',
                'conscientiousness': '尽责性',
                'extraversion': '外向性',
                'agreeableness': '宜人性',
                'neuroticism': '神经质'
            }
            psych_text = "，".join([f"{psych_map[k]}得分{v}" for k, v in psych.items()])
            parts.append(f"。大五人格：{psych_text}")

        return "".join(parts) + "。"

    def build_all_profiles(self):

        for user_id in self.all_user_ids:
            self.build_user_profile(user_id)
        self.save_all_profiles()

        if self.no_assigned_pc_count:
            print(f"\n无专属设备用户统计（共 {sum(self.no_assigned_pc_count.values())} 人）:")
            for reason, count in sorted(self.no_assigned_pc_count.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count} 人")

    def save_all_profiles(self):

        import os
        import json
        os.makedirs(USER_PROFILE_DIR, exist_ok=True)
        for user_id, profile in self.user_profiles.items():
            safe_user_id = user_id.replace('/', '_').replace('\\', '_')
            file_path = os.path.join(USER_PROFILE_DIR, f"{safe_user_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2, default=str)

def main():

    profile_module = UserProfileModule()

    print("=" * 60)
    print("开始加载数据...")
    print("=" * 60)
    profile_module.load_data()

    print("\n" + "=" * 60)
    print("开始构建用户画像...")
    print("=" * 60)
    profile_module.build_all_profiles()

    print("\n" + "=" * 60)
    print(f"用户画像构建完成！")
    print(f"共处理用户数: {len(profile_module.user_profiles)}")
    print(f"画像保存路径: {USER_PROFILE_DIR}")
    print("=" * 60)

    if profile_module.all_user_ids:
        sample_user = profile_module.all_user_ids[0]
        print(f"\n示例用户 {sample_user} 的画像摘要:")
        summary = profile_module.get_single_profile_summary(sample_user)
        import json
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
