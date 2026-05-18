import os
import re
import sys
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import config
logger = logging.getLogger(__name__)


class RedTeamDataLoader:

    def __init__(self, include_content: bool = False):

        self.include_content = include_content
        self.data: Dict[str, Union[pd.DataFrame, str, Dict, None]] = {
            'insiders': None,  
            'scenarios': None,  
            'events': {},  
            'ground_truth': {}  
        }
        self.user_filter: Optional[List[str]] = None  
        self.scenario_filter: Optional[List[int]] = None

    def load(self, version_filter: str = None, user_filter: Union[str, List[str]] = None,
             scenario_filter: Union[int, List[int]] = None) -> Dict:
        if user_filter is not None:
            if isinstance(user_filter, str):
                self.user_filter = [user_filter]
            else:
                self.user_filter = user_filter
            logger.info(f"用户筛选: {self.user_filter}")  
        if scenario_filter is not None:
            if isinstance(scenario_filter, int):
                self.scenario_filter = [scenario_filter]
            else:
                self.scenario_filter = scenario_filter
            logger.info(f"场景筛选: {self.scenario_filter}")
        if version_filter is None:
            version_filter = getattr(config, 'DATASET_VERSION', None)
            if version_filter is None:
                logger.warning("未指定版本筛选，将加载所有版本")
        logger.info(f"开始加载红队数据，版本筛选: {version_filter if version_filter else '所有版本'}...")
        if hasattr(config, 'INSIDERS_FILE') and os.path.exists(config.INSIDERS_FILE):
            self.data['insiders'] = self._load_insiders_file(version_filter, self.user_filter)
            if self.data['insiders'] is not None:
                logger.info(f"✓ 加载insiders.csv: {len(self.data['insiders'])} 条记录")
                if self.user_filter:
                    logger.info(f"  用户筛选后: {self.data['insiders']['user'].unique().tolist()}")
            else:
                logger.error("✗ 加载insiders.csv失败")
                return self.data
        else:
            logger.warning("insiders.csv文件不存在，跳过红队数据加载")
            return self.data
        if hasattr(config, 'SCENARIOS_FILE') and os.path.exists(config.SCENARIOS_FILE):
            self.data['scenarios'] = self._load_scenarios_file()
            logger.info("✓ 加载scenarios.txt")
        if self.data['insiders'] is not None and not self.data['insiders'].empty:
            self.data['events'], self.data['ground_truth'] = self._load_redteam_event_files(
                self.data['insiders'], version_filter
            )
        return self.data

    def _load_insiders_file(self, version_filter: str = None, user_filter: List[str] = None) -> Optional[pd.DataFrame]:

        try:
            df = pd.read_csv(
                config.INSIDERS_FILE,
                encoding='utf-8',
                dtype={
                    'dataset': 'category',
                    'scenario': 'int8',
                    'details': 'category',
                    'user': 'category',
                    'start': 'str',
                    'end': 'str'
                }
            )
            df['start'] = pd.to_datetime(df['start'], format='mixed', dayfirst=False, errors='coerce')
            df['end'] = pd.to_datetime(df['end'], format='mixed', dayfirst=False, errors='coerce')
            df = df.dropna(subset=['start', 'end'])
            if version_filter:
                df = df[df['dataset'] == version_filter]
            if user_filter:
                df = df[df['user'].isin(user_filter)]
            if self.scenario_filter:
                df = df[df['scenario'].isin(self.scenario_filter)]
            return df
        except Exception as e:
            logger.error(f"加载insiders.csv失败: {e}")
            return None

    def _load_scenarios_file(self) -> str:
        try:
            with open(config.SCENARIOS_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"加载scenarios.txt失败: {e}")
            return ""
    def _load_redteam_event_files(self, insiders_df: pd.DataFrame, version_filter: str = None) -> Tuple[Dict, Dict]:

        events = {}
        ground_truth = {}
        base_path = getattr(config, 'ANSWERS_BASE_PATH', None)
        if base_path is None:
            logger.error("config.py中未配置 ANSWERS_BASE_PATH")
            return events, ground_truth
        if not os.path.exists(base_path):
            logger.error(f"ANSWERS目录不存在: {base_path}")
            return events, ground_truth
        total_users = 0
        total_events = 0
        scenario_set = set()
        total_insiders = len(insiders_df)
        for _, row in insiders_df.iterrows():
            details_file = row['details']  
            user = row['user']
            scenario = row['scenario']
            start = row['start']
            end = row['end']
            dataset = row['dataset']
            match = re.match(r'(r[\d.]+-\d+)-', details_file)
            if match:
                subdir = match.group(1)  
            else:

                subdir = f"r{dataset}-1"  
                logger.warning(f"无法从文件名 {details_file} 提取子目录，使用默认: {subdir}")
            file_path = os.path.join(base_path, subdir, details_file)
            if not os.path.exists(file_path):
                found = False
                for dir_name in os.listdir(base_path):
                    if dir_name.startswith(f"r{dataset}"):
                        test_path = os.path.join(base_path, dir_name, details_file)

                        if os.path.exists(test_path):
                            file_path = test_path
                            found = True
                            break
                if not found:
                    logger.warning(f"红队事件文件不存在: {file_path}")
                    continue
            if scenario not in events:
                events[scenario] = {}

            event_list = self._parse_redteam_event_file(str(file_path))
            event_labels = {}
            for idx, event in enumerate(event_list):
                event_id = event.get('event_id')  
                event_labels[event_id] = {
                    'is_anomaly': True,  
                    'event_index': idx,
                    'event_type': event.get('type')
                }
            events[scenario][user] = event_list
            ground_truth[user] = {
                'scenario': scenario,  
                'start': start,  
                'end': end,  
                'events_file': details_file,  
                'event_count': len(event_list),  
                'dataset': dataset,  
                'subdir': subdir,  
                'event_labels': event_labels  
            }
            total_users += 1
            total_events += len(event_list)
            scenario_set.add(scenario)
            if total_users % 10 == 0:
                logger.info(f"加载中：已处理 {total_users}/{total_insiders} 个红队用户...")
        logger.info(f"✅ 红队数据加载完成 | 总用户={total_users} | 场景={sorted(list(scenario_set))} | 总事件={total_events}")
        return events, ground_truth
    def _parse_redteam_event_file(self, filepath: str) -> List[Dict]:

        events = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if not parts:
                        continue
                    event_type = parts[0].lower()
                    event_id = parts[1] if len(parts) > 1 else f"{os.path.basename(filepath)}_{line_num}"
                    timestamp = pd.to_datetime(parts[2]) if len(parts) > 2 else None
                    pc = parts[4] if len(parts) > 4 else ''
                    base_event = {
                        'event_id': event_id,
                        'timestamp': timestamp,
                        'source': 'redteam',  
                        'pc': pc
                    }
                    if event_type == 'logon' and len(parts) >= 6:
                        activity = parts[5]
                        event = {
                            **base_event,
                            'type': 'logon',
                            'activity': activity,
                            'description': f"{activity} on {pc}"
                        }
                        events.append(event)
                    elif event_type == 'device' and len(parts) >= 6:
                        activity = parts[5]
                        event = {
                            **base_event,
                            'type': 'usb',  
                            'activity': activity,
                            'description': f"USB {activity} on {pc}"
                        }
                        events.append(event)
                    elif event_type == 'http' and len(parts) >= 6:
                        url = parts[5]
                        event = {
                            **base_event,
                            'type': 'web',  
                            'url': url,
                            'description': f"Visited {url}"
                        }

                        if self.include_content and len(parts) > 6:
                            event['content'] = parts[6]
                        events.append(event)

                    elif event_type == 'email' and len(parts) >= 12:
                        event = {
                            **base_event,
                            'type': 'email',
                            'to': parts[5],
                            'cc': parts[6] if len(parts) > 6 else '',
                            'bcc': parts[7] if len(parts) > 7 else '',
                            'from': parts[8],
                            'size': int(parts[9]) if parts[9].isdigit() else 0,
                            'attachment_count': int(parts[10]) if len(parts) > 10 and parts[10].isdigit() else 0,
                            'description': f"Sent email from {parts[8]} to {parts[5]}"
                        }

                        if self.include_content and len(parts) > 11:
                            event['content'] = parts[11]
                        events.append(event)

                    elif event_type == 'file' and len(parts) >= 7:
                        filename = parts[5]
                        event = {
                            **base_event,
                            'type': 'file',
                            'filename': filename,
                            'description': f"Copied file {filename}"
                        }

                        if self.include_content and len(parts) > 6:
                            event['content'] = parts[6]
                        events.append(event)

                    else:
                        logger.debug(f"跳过无法解析的行 {line_num}: 类型={event_type}, 字段数={len(parts)}")
                        event = {
                            **base_event,
                            'type': 'unknown',
                            'raw': line,
                            'description': f"Unknown event: {line[:50]}"
                        }
                        events.append(event)
        except Exception as e:
            logger.error(f"解析红队事件文件失败 {filepath}: {e}")
        return events
    def get_anomaly_users(self) -> List[str]:

        return list(self.data['ground_truth'].keys())
    def get_anomaly_by_scenario(self, scenario_id: int) -> Dict:
        result = {}
        if scenario_id not in self.data['events']:
            logger.warning(f"场景 {scenario_id} 不存在")
            return result

        scenario_users = self.data['events'][scenario_id]
        for user, events in scenario_users.items():
            if user in self.data['ground_truth']:
                result[user] = {
                    'info': self.data['ground_truth'][user],
                    'events': events
                }
        return result

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="红队数据加载器 - 调试工具")
    parser.add_argument('--version', type=str, default=None,
                        help='指定版本筛选（如 4.2, 5.2），默认使用config中的DATASET_VERSION')
    parser.add_argument('--all-versions', action='store_true',
                        help='加载所有版本（忽略version参数）')
    parser.add_argument('--users', type=str, default=None,
                        help='指定用户筛选，多个用户用逗号分隔（如 AAM0658,BCM1234）')
    parser.add_argument('--include-content', action='store_true',  
                        help='是否包含content字段（邮件/网页/文件内容），默认False')
    parser.add_argument('--scenarios', type=str, default=None,
                        help='指定场景筛选，多个场景用逗号分隔（如 1,2,3）')
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"redteam_loader_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.WARNING,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    logger.info(f"📝 详细日志文件：{log_file}")

    print("=" * 60)
    print("红队数据加载器 - 极简调试模式")
    print("=" * 60)

    if args.all_versions:
        version_filter = None
        print(f"🔍 运行模式: 加载所有版本")
    else:
        version_filter = args.version or getattr(config, 'DATASET_VERSION', None)
        print(f"🔍 运行模式: 版本筛选 [{version_filter or '无'}]")

    user_filter = None
    if args.users:
        user_filter = [u.strip() for u in args.users.split(',')]
        print(f"🔍 用户筛选: {user_filter}")
    else:
        print(f"🔍 运行模式: 加载所有用户")

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [int(s.strip()) for s in args.scenarios.split(',')]
        print(f"🔍 场景筛选: {scenario_filter}")
    else:
        print(f"🔍 运行模式: 加载所有场景")

    if not config.init_config(auto_create_dirs=False, auto_validate=True):
        print("❌ 配置验证失败，请检查config.py")
        exit(1)

    required_configs = ['INSIDERS_FILE', 'ANSWERS_BASE_PATH']
    missing_configs = [cfg for cfg in required_configs if not hasattr(config, cfg)]
    if missing_configs:
        print(f"❌ 缺失配置: {', '.join(missing_configs)}")
        exit(1)
    print("✅ 配置检查通过")

    loader = RedTeamDataLoader(include_content=args.include_content)
    print(f"包含content字段: {'是' if args.include_content else '否'}")
    print("\n📊 开始加载红队数据...")
    data = loader.load(version_filter=version_filter, user_filter=user_filter,
                   scenario_filter=scenario_filter)

    if data['insiders'] is not None and not data['insiders'].empty:
        print(f"✅ [Insiders模块] 加载成功 | 记录数: {len(data['insiders'])}")
        print(f"   覆盖版本: {', '.join(data['insiders']['dataset'].unique())}")

        scenarios = sorted(data['insiders']['scenario'].unique())

        scenarios_int = [int(s) for s in scenarios]
        print(f"   覆盖场景: {scenarios_int}")
        loaded_users = data['insiders']['user'].unique().tolist()

        if user_filter is not None:
            missing_users = [u for u in user_filter if u not in loaded_users]
            if missing_users:
                print(f"⚠️  未找到的用户: {missing_users}")
            else:
                print(f"✅ 所有指定用户已成功加载")
        else:
            print(f"✅ 共加载 {len(loaded_users)} 个用户")
    else:
        print("❌ [Insiders模块] 加载失败")
        exit(1)

    print(f"✅ [Scenarios模块] 加载成功 | 内容长度: {len(data['scenarios'])} 字符")

    print(f"✅ [事件解析模块] 加载成功 | 异常用户数: {len(data['ground_truth'])}")
    if data['ground_truth']:

        top_users = list(data['ground_truth'].keys())[:2]
        for user in top_users:
            info = data['ground_truth'][user]
            print(f"   - 用户 {user}: 场景{info['scenario']} | 事件数{info['event_count']}")
        if len(data['ground_truth']) > 2:
            print(f"   - 共{len(data['ground_truth'])}个用户，其余省略...")

    print("\n🔧 API功能测试")
    anomaly_users = loader.get_anomaly_users()
    print(f"✅ [get_anomaly_users] 返回 {len(anomaly_users)} 个异常用户")
    if anomaly_users:
        test_scenario = data['ground_truth'][anomaly_users[0]]['scenario']
        scenario_data = loader.get_anomaly_by_scenario(test_scenario)
        print(f"✅ [get_anomaly_by_scenario] 场景{test_scenario} | 返回 {len(scenario_data)} 个用户")

    all_events = []
    for scenario_id, scenario_data in data['events'].items():

        for user, events_list in scenario_data.items():
            if events_list:  
                all_events.append(events_list[0])  
                break  

        if len(all_events) >= 5:  
            break
    if all_events:
        event_types = [evt['type'] for evt in all_events if 'type' in evt]
        print(f"✅ [事件解析] 抽样验证 | 识别类型: {set(event_types)}")
    else:
        print(f"⚠️ [事件解析] 没有找到事件样本")

    print("\n" + "=" * 60)
    print("🔍 数据结构验证测试")
    print("=" * 60)

    print("\n【1. insiders 模块】")
    if data['insiders'] is not None and not data['insiders'].empty:
        insiders = data['insiders']
        print(f"  ✅ DataFrame形状: {insiders.shape}")
        print(f"  ✅ 列名: {list(insiders.columns)}")
        print(f"  ✅ 数据集版本: {insiders['dataset'].unique().tolist()}")
        print(f"  ✅ 场景ID: {insiders['scenario'].unique().tolist()}")
        print(f"  ✅ 用户数: {len(insiders['user'].unique())}")

        first_row = insiders.iloc[0]
        print(f"  📝 示例: user={first_row['user']}, scenario={first_row['scenario']}, "
              f"dataset={first_row['dataset']}, start={first_row['start']}")
    else:
        print("  ❌ 数据为空或None")

    print("\n【2. scenarios 模块】")
    if data['scenarios']:
        print(f"  ✅ 内容长度: {len(data['scenarios'])} 字符")
        print(f"  📝 前100字符: {data['scenarios'][:100]}...")
    else:
        print("  ❌ 内容为空")

    print("\n【3. events 模块】")
    if data['events']:
        print(f"  ✅ 场景数量: {len(data['events'])}")
        total_events = 0
        for scenario_id, scenario_data in data['events'].items():
            user_count = len(scenario_data)
            events_count = sum(len(events) for events in scenario_data.values())
            total_events += events_count
            print(f"    场景{scenario_id}: {user_count}个用户, {events_count}个事件")
        print(f"  ✅ 总事件数: {total_events}")

        sample_event = None
        sample_user = None
        sample_scenario = None
        for scenario_id, scenario_data in data['events'].items():
            for user, events_list in scenario_data.items():
                if events_list:
                    sample_event = events_list[0]
                    sample_user = user
                    sample_scenario = scenario_id
                    break
            if sample_event:
                break
        if sample_event:
            print(f"\n  📝 事件示例 (场景{sample_scenario}, 用户{sample_user}):")
            print(f"     - type: {sample_event.get('type')}")
            print(f"     - source: {sample_event.get('source')}")
            print(f"     - timestamp: {sample_event.get('timestamp')}")
            print(f"     - pc: {sample_event.get('pc')}")
            print(f"     - description: {sample_event.get('description', '')[:60]}...")
            if 'content' in sample_event:
                print(f"     - content长度: {len(sample_event.get('content', ''))} 字符")
    else:
        print("  ❌ 数据为空")

    print("\n【4. ground_truth 模块】")
    if data['ground_truth']:
        print(f"  ✅ 用户数量: {len(data['ground_truth'])}")

        for i, (user_id, info) in enumerate(list(data['ground_truth'].items())[:2]):
            print(f"\n  📝 用户 {user_id}:")
            print(f"     - scenario: {info['scenario']}")
            print(f"     - start: {info['start']}")
            print(f"     - end: {info['end']}")
            print(f"     - events_file: {info['events_file']}")
            print(f"     - event_count: {info['event_count']}")
            print(f"     - dataset: {info['dataset']}")
            print(f"     - subdir: {info['subdir']}")
        if len(data['ground_truth']) > 2:
            print(f"\n  ... 还有 {len(data['ground_truth']) - 2} 个用户未显示")
    else:
        print("  ❌ 数据为空")

    print("\n【5. 统一格式验证】")
    if data['events'] and sample_event:
        required_fields = ['timestamp', 'type', 'source', 'pc', 'description']
        missing = [f for f in required_fields if f not in sample_event]
        if not missing:
            print(f"  ✅ 事件格式符合规范 (包含所有必需字段)")
            print(f"     - source字段正确标记为: {sample_event.get('source')}")

            event_type = sample_event.get('type')
            if event_type in ['usb', 'web']:
                print(f"     - 类型映射正确: {event_type} (原device/http已映射)")

            if args.include_content:
                if 'content' in sample_event:
                    print(f"     - content字段已包含 (符合--include-content参数)")
                else:
                    print(f"     ⚠️ content字段缺失 (但--include-content已启用)")
            else:
                if 'content' not in sample_event:
                    print(f"     - content字段未包含 (符合默认设置)")
                else:
                    print(f"     ⚠️ content字段存在 (但--include-content未启用)")
        else:
            print(f"  ❌ 事件格式不符合规范，缺少字段: {missing}")

    print("\n" + "=" * 60)
    print("✅ 数据结构验证完成")

    print("\n" + "=" * 60)
    print("🎉 所有模块测试完成 | 详细日志见: " + log_file)
    print("=" * 60)
