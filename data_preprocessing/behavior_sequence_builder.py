import os
import sys
import pandas as pd
import logging
from typing import Dict, List
from tqdm import tqdm

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

logger = logging.getLogger(__name__)


class UserBehaviorSequenceBuilder:

    def __init__(self, data: Dict[str, pd.DataFrame], include_content: bool = False):
        self.data = data
        self.user_sequences = {}
        self.include_content = include_content

    def build_all_sequences(self, user_ids: List[str] = None, save_path: str = None) -> Dict[str, List[Dict]]:
        logger.info("开始构建用户行为序列...")

        if user_ids is None:
            user_ids = self._get_all_users()

        logger.info(f"构建 {len(user_ids)} 个用户的行为序列，include_content={self.include_content}")

        for user_id in tqdm(user_ids, desc="构建用户序列"):
            self.user_sequences[user_id] = self._build_user_sequence(user_id)

        logger.info("用户行为序列构建完成")

        if save_path:
            self.save_sequences(save_path)

        return self.user_sequences

    def _get_all_users(self) -> List[str]:
        users = set()
        if self.data.get('psychometric') is not None:
            users.update(self.data['psychometric']['user_id'].unique())

        for key in ['logon', 'device', 'http', 'email', 'file']:
            if self.data.get(key) is not None and 'user' in self.data[key].columns:
                behavior_users = self.data[key]['user'].dropna().unique()
                users.update(behavior_users)

        return sorted(list(users))

    def _build_user_sequence(self, user_id: str) -> List[Dict]:
        events = []

        if self.data.get('logon') is not None:
            logon_events = self._extract_logon_events(user_id)
            events.extend(logon_events)

        if self.data.get('device') is not None:
            device_events = self._extract_device_events(user_id)
            events.extend(device_events)

        if self.data.get('http') is not None:
            http_events = self._extract_http_events(user_id)
            events.extend(http_events)

        if self.data.get('email') is not None:
            email_events = self._extract_email_events(user_id)
            events.extend(email_events)

        if self.data.get('file') is not None:
            file_events = self._extract_file_events(user_id)
            events.extend(file_events)

        events.sort(key=lambda x: x['timestamp'])

        return events

    def _extract_logon_events(self, user_id: str) -> List[Dict]:
        events = []
        df = self.data['logon']
        user_logs = df[df['user'] == user_id]

        for _, row in user_logs.iterrows():
            event = {
                'event_id': row.get('id'),
                'timestamp': row['date'],
                'type': 'logon',
                'activity': row['activity'],
                'pc': row['pc'],
                'description': f"{row['activity']} on {row['pc']}"
            }
            events.append(event)

        return events

    def _extract_device_events(self, user_id: str) -> List[Dict]:
        events = []
        df = self.data['device']
        user_logs = df[df['user'] == user_id]

        for _, row in user_logs.iterrows():
            event = {
                'event_id': row.get('id'),
                'timestamp': row['date'],
                'type': 'usb',
                'activity': row['activity'],
                'pc': row['pc'],
                'description': f"USB {row['activity']} on {row['pc']}"
            }
            events.append(event)

        return events

    def _extract_http_events(self, user_id: str) -> List[Dict]:
        events = []
        df = self.data['http']
        user_logs = df[df['user'] == user_id]

        for _, row in user_logs.iterrows():
            event = {
                'event_id': row.get('id'),
                'timestamp': row['date'],
                'type': 'web',
                'url': row['url'],
                'pc': row['pc'],
                'description': f"Visited {row['url']}"
            }
            if self.include_content:
                event['content'] = row.get('content', '')
            events.append(event)

        return events

    def _extract_email_events(self, user_id: str) -> List[Dict]:
        events = []
        df = self.data['email']
        user_logs = df[df['user'] == user_id]

        for _, row in user_logs.iterrows():
            event = {
                'event_id': row.get('id'),
                'timestamp': row['date'],
                'type': 'email',
                'to': row['to'],
                'cc': row.get('cc', ''),
                'bcc': row.get('bcc', ''),
                'from': row['from'],
                'size': row['size'],
                'attachment_count': row.get('attachments', 0),
                'pc': row['pc'],
                'description': f"Sent email from {row['from']} to {row['to']}"
            }
            if self.include_content:
                event['content'] = row.get('content', '')
            events.append(event)

        return events

    def _extract_file_events(self, user_id: str) -> List[Dict]:
        events = []
        df = self.data['file']
        user_logs = df[df['user'] == user_id]

        for _, row in user_logs.iterrows():
            event = {
                'event_id': row.get('id'),
                'timestamp': row['date'],
                'type': 'file',
                'filename': row['filename'],
                'pc': row['pc'],
                'description': f"Copied file {row['filename']}"
            }
            if self.include_content:
                event['content'] = row.get('content', '')
            events.append(event)

        return events

    def save_sequences(self, output_dir: str = None, format: str = None) -> Dict[str, str]:
        if not self.user_sequences:
            logger.warning("没有可保存的行为序列，请先调用 build_all_sequences()")
            return {}

        if output_dir is None:
            output_dir = getattr(config, 'BEHAVIOR_SEQUENCE_DIR', './output/behavior_sequences')

        if format is None:
            format = getattr(config, 'BEHAVIOR_SEQUENCE_FORMAT', 'parquet')

        os.makedirs(output_dir, exist_ok=True)

        metadata = {
            'saved_at': pd.Timestamp.now().isoformat(),
            'num_users': len(self.user_sequences),
            'format': format,
            'include_content': self.include_content,
            'user_ids': list(self.user_sequences.keys()),
            'sequence_stats': {}
        }

        saved_files = {}

        logger.info(f"开始保存用户行为序列到 {output_dir}，格式: {format}")

        for user_id, events in tqdm(self.user_sequences.items(), desc="保存用户序列"):
            safe_user_id = user_id.replace('/', '_').replace('\\', '_')
            file_path = os.path.join(output_dir, f"{safe_user_id}.{format}")

            df = pd.DataFrame(events)

            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(str)

            try:
                if format == 'parquet':
                    df.to_parquet(file_path, index=False, engine='pyarrow')
                elif format == 'json':
                    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
                elif format == 'pickle':
                    df.to_pickle(file_path)
                else:
                    logger.error(f"不支持的保存格式: {format}")
                    continue

                saved_files[user_id] = file_path
                metadata['sequence_stats'][user_id] = {
                    'event_count': len(events),
                    'file': file_path
                }

            except Exception as e:
                logger.error(f"保存用户 {user_id} 序列失败: {e}")

        metadata_file = os.path.join(output_dir, "_metadata.json")
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"行为序列保存完成，共 {len(saved_files)} 个用户，元数据: {metadata_file}")

        return saved_files

    def load_sequences(self, input_dir: str = None, format: str = None,
                   user_ids: List[str] = None) -> Dict[str, List[Dict]]:
        is_default_dir = (input_dir is None)
        if input_dir is None:
            base_dir = getattr(config, 'BEHAVIOR_SEQUENCE_DIR', './output/behavior_sequences')
            if self.include_content:
                input_dir = os.path.join(base_dir, 'with_content')
            else:
                input_dir = os.path.join(base_dir, 'light')
            logger.info(f"根据include_content={self.include_content}自动选择目录: {input_dir}")
        if format is None:
            format = getattr(config, 'BEHAVIOR_SEQUENCE_FORMAT', 'parquet')

        if not os.path.exists(input_dir):
            if is_default_dir and self.data:
                logger.warning(f"默认序列目录不存在: {input_dir}")
                logger.info("自动回退到构建模式...")
                return self.build_all_sequences(user_ids=user_ids, save_path=input_dir)
            else:
                logger.error(f"指定的序列目录不存在: {input_dir}")
                return {}

        logger.info(f"开始加载行为序列: {input_dir}")

        sequences = {}

        if user_ids is not None:
            file_paths = []
            for user_id in user_ids:
                safe_user_id = user_id.replace('/', '_').replace('\\', '_')
                file_path = os.path.join(input_dir, f"{safe_user_id}.{format}")
                if os.path.exists(file_path):
                    file_paths.append((user_id, file_path))
                else:
                    # missing_users.append(user_id)
                    logger.warning(f"用户 {user_id} 的序列文件不存在: {file_path}")

                # if missing_users:
                #     if is_default_dir and self.data:
                #         logger.info(f"有 {len(missing_users)} 个用户的序列文件缺失，开始构建...")
                #
                #     else:
                #         logger.warning(f"有 {len(missing_users)} 个用户的序列文件缺失，但未使用默认目录，跳过构建")
        else:
            import glob
            pattern = os.path.join(input_dir, f"*.{format}")
            file_paths = [(os.path.basename(f).replace(f".{format}", ""), f)
                          for f in glob.glob(pattern)]
            if not file_paths:
                if is_default_dir and self.data:
                    logger.warning(f"默认目录 {input_dir} 中没有找到序列文件")
                    logger.info("自动回退到全量构建模式...")
                    return self.build_all_sequences(save_path=input_dir)
                else:
                    logger.error(f"指定目录 {input_dir} 中没有找到序列文件")
                    return {}

        for user_id, file_path in tqdm(file_paths, desc="加载用户序列"):
            try:
                if format == 'parquet':
                    df = pd.read_parquet(file_path, engine='pyarrow')
                elif format == 'json':
                    df = pd.read_json(file_path, lines=True)
                elif format == 'pickle':
                    df = pd.read_pickle(file_path)
                else:
                    logger.error(f"不支持的加载格式: {format}")
                    continue

                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                events = df.to_dict('records')
                sequences[user_id] = events

            except Exception as e:
                logger.error(f"加载用户序列失败 {file_path}: {e}")

        self.user_sequences = sequences
        logger.info(f"行为序列加载完成，共 {len(sequences)} 个用户")

        if sequences:
            sample_user = list(sequences.keys())[0]
            sample_events = sequences[sample_user]
            has_content = any('content' in e for e in sample_events[:10])
            self.include_content = has_content
            logger.info(f"自动检测序列包含content字段: {has_content}")

        return sequences

    def get_sequence_summary(self) -> pd.DataFrame:
        if not self.user_sequences:
            logger.warning("没有行为序列，请先调用 build_all_sequences()")
            return pd.DataFrame()

        summaries = []

        for user_id, events in self.user_sequences.items():
            event_types = {}
            for event in events:
                event_type = event.get('type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1

            timestamps = [e.get('timestamp') for e in events if e.get('timestamp') is not None]
            timestamps = [t for t in timestamps if pd.notna(t)]

            summaries.append({
                'user_id': user_id,
                'total_events': len(events),
                'logon_events': event_types.get('logon', 0),
                'usb_events': event_types.get('usb', 0),
                'web_events': event_types.get('web', 0),
                'email_events': event_types.get('email', 0),
                'file_events': event_types.get('file', 0),
                'first_event_time': min(timestamps) if timestamps else None,
                'last_event_time': max(timestamps) if timestamps else None
            })

        return pd.DataFrame(summaries)

    def get_all_user_ids(self, input_dir: str = None) -> List[str]:
        if input_dir is None:
            base_dir = getattr(config, 'BEHAVIOR_SEQUENCE_DIR', './output/behavior_sequences')
            input_dir = os.path.join(base_dir, 'with_content' if self.include_content else 'light')
        import glob
        pattern = os.path.join(input_dir, "*.parquet")
        user_ids = [os.path.basename(f).replace(".parquet", "") for f in glob.glob(pattern)]
        return user_ids


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from datetime import datetime

    parser = argparse.ArgumentParser(description="用户行为序列构建器")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--test', action='store_true', default=True,
                            help='测试模式（默认）：使用采样数据，输出到独立测试目录')
    mode_group.add_argument('--build-full', action='store_true',
                            help='完整构建模式：构建所有用户的完整行为序列')
    mode_group.add_argument('--load-only', action='store_true',
                            help='仅加载模式：直接加载已保存的序列，不重新构建')

    parser.add_argument('--sample-size', type=int, default=2000,
                        help='测试模式下每个数据源的采样大小')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（测试模式默认使用独立测试目录）')
    parser.add_argument('--force-overwrite', action='store_true',
                        help='强制覆盖已存在的文件（仅全量模式有效）')
    parser.add_argument('--user-limit', type=int, default=10,
                        help='测试模式下限制展示的用户数量')
    parser.add_argument('--include-content', action='store_true',
                        help='是否包含content字段（用于语义检测，默认不包含）')
    parser.add_argument('--user-ids', type=str, default=None,
                        help='指定要加载的用户ID，逗号分隔（如: BSS0369,ABC0174,JLM0364），仅加载模式有效')

    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"behavior_sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    from config import BEHAVIOR_SEQUENCE_DIR
    if args.output_dir:
        output_dir = args.output_dir
    elif args.build_full:
        if args.include_content:
            output_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, 'with_content')
        else:
            output_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, 'light')
    elif args.load_only:
        with_content_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, 'with_content')
        light_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, 'light')

        if os.path.exists(with_content_dir):
            output_dir = with_content_dir
            print(f"  自动选择完整版目录: {output_dir}")
        elif os.path.exists(light_dir):
            output_dir = light_dir
            print(f"  自动选择轻量版目录: {output_dir}")
        else:
            output_dir = BEHAVIOR_SEQUENCE_DIR
            print(f"  使用默认目录: {output_dir}")
        if args.user_ids:
            print(f"  指定加载用户: {[u.strip() for u in args.user_ids.split(',')]}")
    else:
        base_test_dir = getattr(config, 'BEHAVIOR_SEQUENCE_DIR')
        test_suffix = 'with_content' if args.include_content else 'light'
        output_dir = os.path.join(os.path.dirname(base_test_dir), f'behavior_sequences_test_{test_suffix}')
    print(f"这是测试：使用的目录是 {output_dir}")

    print("=" * 80)
    print("用户行为序列构建工具")
    print("=" * 80)

    if args.load_only:
        print("运行模式: 【仅加载模式】- 直接加载已保存的序列")
    elif args.build_full:
        print("运行模式: 【完整构建模式】- 构建所有用户行为序列")
        print(f"包含content: {'是' if args.include_content else '否'}（语义检测需要开启）")
    else:
        print("运行模式: 【测试模式】- 使用采样数据快速验证（不影响全量数据）")
        print(f"采样大小: {args.sample_size} 条/数据源")
        print(f"包含content: {'是' if args.include_content else '否'}")

    print(f"输出目录: {output_dir}")

    if not args.build_full and not args.load_only:
        print("\n⚠️  测试模式说明:")
        print(f"   - 使用独立测试目录: {output_dir}")
        print("   - 不会影响已保存的全量数据")
        print("   - 如需构建全量数据，请使用: --build-full")
        print("   - 如需包含content字段，请使用: --include-content")

    print("=" * 80)

    if not config.init_config(auto_create_dirs=False, auto_validate=True):
        print("配置验证失败，请检查配置文件")
        exit(1)

    if args.load_only:
        print("\n【仅加载模式】从磁盘加载已保存的行为序列...")

        user_ids = None
        if args.user_ids:
            user_ids = [u.strip() for u in args.user_ids.split(',')]
            print(f"  指定加载用户: {user_ids}")

        builder = UserBehaviorSequenceBuilder({})

        sequences = builder.load_sequences(input_dir=output_dir, user_ids=user_ids)

        if sequences:
            print(f"\n✓ 加载成功: {len(sequences)} 个用户")

            summary_df = builder.get_sequence_summary()
            if not summary_df.empty:
                print(f"\n序列摘要:")
                print(f"  总用户数: {len(summary_df)}")
                print(f"  总事件数: {summary_df['total_events'].sum():,}")
                print(f"  平均事件数: {summary_df['total_events'].mean():.2f}")
                print(f"  最大事件数: {summary_df['total_events'].max()}")
                print(f"  最小事件数: {summary_df['total_events'].min()}")

                sample_user = list(sequences.keys())[0]
                sample_events = sequences[sample_user]
                has_content = any('content' in e for e in sample_events[:10])
                print(f"\n  序列包含content字段: {'是' if has_content else '否'}")

            print(f"\n用户详情:")
            for i, (user_id, events) in enumerate(sequences.items()):
                if i >= 5:
                    remaining = len(sequences) - 5
                    print(f"  ... 还有 {remaining} 个用户未显示")
                    break
                event_types = {}
                for e in events:
                    et = e.get('type', 'unknown')
                    event_types[et] = event_types.get(et, 0) + 1
                print(f"  {user_id}: {len(events)} 个事件, 类型分布: {event_types}")
        else:
            print(f"\n✗ 加载失败: 目录 {output_dir} 中没有找到序列文件")

        print("\n" + "=" * 80)
        print("加载完成")
        exit(0)

    metadata_file = os.path.join(output_dir, "_metadata.json")
    should_rebuild = True

    if args.build_full:
        if not args.force_overwrite and os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)

                print(f"\n⚠️  检测到已存在的全量序列文件:")
                print(f"   保存时间: {existing_metadata.get('saved_at', '未知')}")
                print(f"   用户数量: {existing_metadata.get('num_users', 0)}")
                print(f"   保存格式: {existing_metadata.get('format', '未知')}")
                print(f"   包含content: {existing_metadata.get('include_content', '未知')}")

                print("\n是否覆盖已存在的序列文件？")
                print("  [y] 覆盖 (重新构建)")
                print("  [n] 跳过 (使用现有文件)")
                print("  [q] 退出")
                choice = input("请选择 (y/n/q): ").strip().lower()

                if choice == 'q':
                    print("退出程序")
                    exit(0)
                elif choice == 'n':
                    print("使用现有文件，跳过构建")
                    should_rebuild = False
                else:
                    print("将覆盖现有文件，重新构建...")
                    backup_dir = f"{output_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print(f"创建备份: {backup_dir}")
                    if os.path.exists(output_dir):
                        shutil.move(output_dir, backup_dir)
                    should_rebuild = True
            except Exception as e:
                print(f"读取元数据失败: {e}")
                should_rebuild = True
        else:
            should_rebuild = True
            if args.force_overwrite and os.path.exists(metadata_file):
                backup_dir = f"{output_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"强制覆盖模式，创建备份: {backup_dir}")
                if os.path.exists(output_dir):
                    shutil.move(output_dir, backup_dir)
    else:
        should_rebuild = True
        print("\n【测试模式】使用独立测试目录，直接覆盖")

    if should_rebuild:
        from data_preprocessing.cert_data_loader import CERTDataLoader

        print("\n【步骤1】加载数据...")
        data_loader = CERTDataLoader()

        if args.build_full:
            print("加载完整数据集...")
            data = data_loader.load_all(use_sample=False)
        else:
            print(f"加载采样数据 (每源 {args.sample_size} 条)...")
            data = data_loader.load_all(use_sample=True, sample_size=args.sample_size)

        print("\n数据加载统计:")
        total_memory = 0
        for key, df in data.items():
            if df is not None:
                mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                total_memory += mem_mb
                print(f"  {key}: {len(df):,} 条记录, {mem_mb:.2f} MB")
        print(f"  总内存占用: {total_memory:.2f} MB")

        print("\n【步骤2】构建用户行为序列...")
        builder = UserBehaviorSequenceBuilder(data, include_content=args.include_content)

        all_users = builder._get_all_users()
        print(f"总用户数: {len(all_users)}")

        import time

        start_time = time.time()

        if args.build_full:
            print("开始构建所有用户的行为序列...")
            sequences = builder.build_all_sequences()
        else:
            print(f"测试模式：构建前 {args.user_limit} 个用户的行为序列...")
            target_users = all_users[:args.user_limit]
            sequences = builder.build_all_sequences(target_users)

        elapsed_time = time.time() - start_time
        print(f"\n构建完成，耗时: {elapsed_time:.2f} 秒")
        print(f"成功构建 {len(sequences)} 个用户的行为序列")
        if sequences:
            sample_user = list(sequences.keys())[0]
            sample_events = sequences[sample_user]
            has_content = any('content' in e for e in sample_events[:10])
            print(f"  序列包含content字段: {'是' if has_content else '否'}")

        print("\n【步骤3】生成序列摘要...")
        summary_df = builder.get_sequence_summary()

        if not summary_df.empty:
            print(f"  总用户数: {len(summary_df)}")
            print(f"  总事件数: {summary_df['total_events'].sum():,}")
            print(f"  平均事件数: {summary_df['total_events'].mean():.2f}")

            print(f"\n  事件类型分布:")
            event_types = ['logon_events', 'usb_events', 'web_events', 'email_events', 'file_events']
            for event_type in event_types:
                total = summary_df[event_type].sum()
                if total > 0:
                    print(f"    {event_type.replace('_events', '')}: {total:,} 次")

        print("\n【步骤4】保存行为序列...")
        saved_files = builder.save_sequences(output_dir=output_dir)

        if saved_files:
            print(f"\n✓ 保存成功: {len(saved_files)} 个用户")

            total_size = 0
            for file_path in saved_files.values():
                total_size += os.path.getsize(file_path)
            print(f"  总文件大小: {total_size / 1024 / 1024:.2f} MB")

            print(f"  保存目录: {output_dir}")

            if args.build_full:
                print("\n" + "=" * 80)
                print("完整构建完成！")
                print("=" * 80)
                print(f"  用户数量: {len(saved_files)}")
                print(f"  事件总数: {summary_df['total_events'].sum():,}")
                print(f"  包含content: {'是' if args.include_content else '否'}")
                print(f"  保存目录: {output_dir}")
                print("\n后续使用:")
                print(f"  加载序列: python {os.path.basename(__file__)} --load-only")
                print(f"  指定目录: python {os.path.basename(__file__)} --load-only --output-dir {output_dir}")
                if not args.include_content:
                    print(f"  如需语义检测，请重新构建含content版本: --build-full --include-content")
        else:
            print("✗ 保存失败")
    if not args.build_full and not args.load_only:
        if should_rebuild and sequences:
            print("\n【测试模式】展示序列详情...")

            sample_user = list(sequences.keys())[0]
            events = sequences[sample_user]

            print(f"\n用户 {sample_user} 行为序列示例:")
            print(f"  总事件数: {len(events)}")

            print(f"\n  前10个事件:")
            for i, event in enumerate(events[:10]):
                timestamp = event.get('timestamp')
                if pd.isna(timestamp):
                    timestamp_str = "NaT"
                else:
                    timestamp_str = str(timestamp)
                event_type = event.get('type', 'unknown')
                desc = event.get('description', '')[:50]
                if 'content' in event:
                    content_len = len(event.get('content', ''))
                    print(f"    {i + 1:2d}. {timestamp_str} | {event_type:5s} | {desc} | content_len={content_len}")
                else:
                    print(f"    {i + 1:2d}. {timestamp_str} | {event_type:5s} | {desc}")

    if should_rebuild:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"\n【内存统计】当前进程内存占用: {memory_mb:.2f} MB")
        except ImportError:
            pass

    print("\n" + "=" * 80)
    print("程序执行完成")
    print("=" * 80)
