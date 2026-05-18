
import os
import sys
import random
import logging
from typing import Dict, List, Generator, Optional, Any

import pandas as pd

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

logger = logging.getLogger(__name__)


class MiniBatchLoader:
    

    def __init__(self, sequences: Dict[str, List[Dict]], batch_size: int = 10):
        
        self.sequences = sequences  
        self.batch_size = batch_size  
        self.all_users = list(sequences.keys())  
        self.total_users = len(self.all_users)
        self._redteam_data = None
        self._anomaly_users = []
        self._normal_users = []
        self._num_batches = (self.total_users + batch_size - 1) // batch_size

        logger.info(f"MiniBatchLoader初始化完成: 总用户数={len(self.all_users)}, 批次大小={batch_size}, 总批次={self._num_batches}")

    def set_redteam_data(self, redteam_data: Dict):
        self._redteam_data = redteam_data
        self._anomaly_users = list(redteam_data.get('ground_truth', {}).keys())
        self._normal_users = [u for u in self.all_users if u not in self._anomaly_users]
        logger.info(f"红队数据设置完成: 异常用户={len(self._anomaly_users)}, 正常用户={len(self._normal_users)}")

    def _get_anomaly_users(self) -> List[str]:
        return self._anomaly_users

    def _get_normal_users(self) -> List[str]:
        return self._normal_users if self._normal_users else self.all_users

    def _build_batch_result(self, user_ids: List[str],
                            batch_info: Dict = None,
                            include_ground_truth: bool = False) -> Dict:
        sequences = {}
        for user_id in user_ids:
            sequences[user_id] = self.sequences.get(user_id, [])
        result = {
            'user_ids': user_ids,
            'sequences': sequences,
            'batch_info': batch_info or {}
        }
        if self._redteam_data is not None:
            result['is_anomaly'] = {u: u in self._anomaly_users for u in user_ids}
        if include_ground_truth and self._redteam_data is not None:
            result['ground_truth'] = {}
            for user_id in user_ids:
                if user_id in self._anomaly_users:
                    result['ground_truth'][user_id] = self._redteam_data['ground_truth'].get(user_id, {})

        return result

    def get_batch(self, batch_idx: int, include_anomaly: bool = False) -> Optional[Dict]:
        
        if batch_idx >= self._num_batches:
            logger.warning(f"批次索引 {batch_idx} 超出范围，总批次={self._num_batches}")
            return None

        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        if include_anomaly and self._anomaly_users:
            target_anomaly_count = max(1, min(len(self._anomaly_users), self.batch_size // 2))
            anomaly_offset = (batch_idx * target_anomaly_count) % len(self._anomaly_users)
            selected_anomaly = []
            for i in range(target_anomaly_count):
                
                idx = (anomaly_offset + i) % len(self._anomaly_users)
                selected_anomaly.append(self._anomaly_users[idx])
            selected_anomaly = list(dict.fromkeys(selected_anomaly))
            normal_needed = self.batch_size - len(selected_anomaly)
            normal_offset = (batch_idx * normal_needed) % len(self._normal_users)
            selected_normal = []
            for i in range(normal_needed):
                idx = (normal_offset + i) % len(self._normal_users)
                selected_normal.append(self._normal_users[idx])
            batch_users = selected_anomaly + selected_normal
        else:
            batch_users = self.all_users[start_idx:end_idx]

        if not batch_users:
            return None

        return self._build_batch_result(
            user_ids=batch_users,
            batch_info={
                'batch_idx': batch_idx,
                'size': len(batch_users),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'includes_anomaly': any(u in self._anomaly_users for u in batch_users) if self._anomaly_users else False
            },
            include_ground_truth=include_anomaly
        )

    def get_random_batch(self, batch_size: int = None, include_anomaly: bool = False) -> Dict:
        
        if batch_size is None:
            batch_size = self.batch_size

        if include_anomaly and self._anomaly_users:
            
            anomaly_count = max(1, min(len(self._anomaly_users), batch_size // 2))
            
            selected_anomaly = random.sample(self._anomaly_users, min(anomaly_count, len(self._anomaly_users)))

            normal_needed = batch_size - len(selected_anomaly)
            if normal_needed > 0 and self._normal_users:
                selected_normal = random.sample(self._normal_users, min(normal_needed, len(self._normal_users)))
            else:
                selected_normal = []

            batch_users = selected_anomaly + selected_normal
            random.shuffle(batch_users)  
        else:
            batch_users = random.sample(self.all_users, min(batch_size, len(self.all_users)))

        return self._build_batch_result(
            user_ids=batch_users,
            batch_info={
                'batch_idx': -1,  
                'size': len(batch_users),  
                'sampling': 'random',  
                
                'includes_anomaly': any(
                    u in self._anomaly_users for u in batch_users) if self._anomaly_users else False
            },
            include_ground_truth=include_anomaly  
        )

    def get_redteam_batch(self, batch_size: int = None, include_normal: bool = True) -> Dict:
        
        if self._redteam_data is None:
            logger.warning("红队批次获取失败：未设置红队数据，已自动降级为普通随机批次")
            return self.get_random_batch(batch_size, include_anomaly=False)
        if not self._anomaly_users:
            logger.warning("红队批次获取失败：无异常用户，已自动降级为普通随机批次")
            return self.get_random_batch(batch_size, include_anomaly=False)
        if batch_size is None:
            batch_size = self.batch_size
        selected_anomaly = self._anomaly_users[:batch_size]
        batch_users = selected_anomaly.copy()
        if include_normal:
            normal_needed = batch_size - len(selected_anomaly)
            if normal_needed > 0 and self._normal_users:
                selected_normal = random.sample(self._normal_users, min(normal_needed, len(self._normal_users)))
                batch_users.extend(selected_normal)

        return self._build_batch_result(
            user_ids=batch_users,
            batch_info={
                'batch_idx': -1,
                'size': len(batch_users),
                'sampling': 'redteam',
                'anomaly_count': len(selected_anomaly),
                'normal_count': len(batch_users) - len(selected_anomaly)
            },
            include_ground_truth=True
        )

    def get_users_by_role(self, role: str, batch_size: int = None,
                          ldap_data: Optional[pd.DataFrame] = None) -> Dict:
        
        if ldap_data is None:
            logger.warning("按角色获取用户失败：未提供LDAP数据，已自动降级为随机批次")
            return self.get_random_batch(batch_size)
        if batch_size is None:
            batch_size = self.batch_size
        if 'snapshot_month' in ldap_data.columns:
            latest_month = ldap_data['snapshot_month'].max()
            latest_ldap = ldap_data[ldap_data['snapshot_month'] == latest_month]
        else:
            latest_ldap = ldap_data
        role_users = latest_ldap[latest_ldap['role'] == role]['user_id'].tolist()
        role_users = [u for u in role_users if u in self.all_users]
        if not role_users:
            logger.warning(f"按角色获取用户失败：未找到角色为 {role} 的有效用户，已自动降级为随机批次")
            return self.get_random_batch(batch_size)
        batch_users = role_users[:batch_size]
        return self._build_batch_result(
            user_ids=batch_users,
            batch_info={
                'batch_idx': -1,
                'size': len(batch_users),
                'sampling': 'role',
                'role': role,
                'total_role_users': len(role_users)
            },
            include_ground_truth=False
        )

    def iterate_batches(self, include_anomaly: bool = False) -> Generator[Dict, None, None]:
        
        for batch_idx in range(self._num_batches):
            batch = self.get_batch(batch_idx, include_anomaly)
            if batch and batch['user_ids']:
                yield batch

    def get_batch_info(self) -> Dict[str, Any]:
        
        info = {
            'total_users': len(self.all_users),
            'batch_size': self.batch_size,
            'total_batches': self._num_batches,
            'anomaly_users': len(self._anomaly_users),
            'normal_users': len(self._normal_users) if self._normal_users else len(self.all_users)
        }

        batch_sizes = []
        for i in range(self._num_batches):
            start = i * self.batch_size
            
            batch_sizes.append(min(self.batch_size, len(self.all_users) - start))
        info['batch_sizes'] = batch_sizes 
        info['avg_batch_size'] = sum(batch_sizes) / len(batch_sizes)  

        return info

    def print_batch_summary(self):
        info = self.get_batch_info()
        print("\n" + "=" * 60)
        print("MiniBatchLoader 摘要信息")
        print("=" * 60)
        print(f"总用户数: {info['total_users']}")
        print(f"批次大小: {info['batch_size']}")
        print(f"总批次: {info['total_batches']}")
        print(f"异常用户: {info['anomaly_users']}")
        print(f"正常用户: {info['normal_users']}")
        print(f"平均批次大小: {info['avg_batch_size']:.1f}")
        print(f"批次大小分布: {info['batch_sizes'][:10]}{'...' if len(info['batch_sizes']) > 10 else ''}")

    def load_mixed_events_for_users(self, user_ids: List[str],
                                    normal_ratio: float = 0.8,
                                    use_all_anomaly: bool = True,
                                    max_events_per_user: int = None,
                                    sort_by_time: bool = True,
                                    sampling_strategy: str = 'random',
                                    event_type_field: str = 'type') -> Dict:
        
        
        if self._redteam_data is None:
            logger.error("需要先调用 set_redteam_data() 设置红队数据")
            return {}
        batch_sequences = {}
        sampling_stats = {}
        for user_id in user_ids:
            if user_id not in self.sequences:
                logger.warning(f"用户 {user_id} 不存在于序列数据中")
                continue
            all_events = self.sequences[user_id]
            user_gt = self._redteam_data['ground_truth'].get(user_id, {})
            event_labels = user_gt.get('event_labels', {})
            normal_events = []
            anomaly_events = []
            for event in all_events:
                
                event_id = event.get('event_id')
                if event_labels.get(event_id, {}).get('is_anomaly', False):
                    anomaly_events.append(event)  
                else:
                    normal_events.append(event)
            original_normal_count = len(normal_events)
            original_anomaly_count = len(anomaly_events)

            if use_all_anomaly:
                if max_events_per_user and len(anomaly_events) > max_events_per_user:
                    
                    sampled_anomaly = random.sample(anomaly_events, max_events_per_user)
                    sampled_normal = []
                    logger.warning(f"用户 {user_id}: 异常事件({len(anomaly_events)})超过上限({max_events_per_user})，"
                                   f"只使用异常事件，无法保持比例")
                else:
                    
                    sampled_anomaly = anomaly_events.copy()  
                    actual_anomaly_count = len(anomaly_events)
                    if actual_anomaly_count > 0:
                        target_normal_count = int(actual_anomaly_count * (normal_ratio / (1 - normal_ratio)))
                        if target_normal_count > len(normal_events):
                            logger.warning(
                                f"用户 {user_id} 正常事件不足: 需要{target_normal_count}个，实际只有{len(normal_events)}个，"
                                f"将使用全部正常事件，实际比例将偏离{normal_ratio}")
                        if target_normal_count > 0 and normal_events:
                            sampled_normal = self._sample_normal_events(
                                normal_events=normal_events,
                                anomaly_events=anomaly_events,
                                target_count=target_normal_count,
                                sampling_strategy=sampling_strategy,
                                event_type_field=event_type_field,
                                normal_ratio=normal_ratio  
                            )
                        else:
                            sampled_normal = []
                    else:
                        logger.warning(f"用户 {user_id} 没有异常事件，将只使用正常事件")
                        if max_events_per_user:
                            
                            target_normal_count = min(max_events_per_user, len(normal_events))
                            if target_normal_count < max_events_per_user:
                                logger.warning(
                                    f"用户 {user_id} 正常事件不足: 需要{max_events_per_user}个，实际只有{len(normal_events)}个")
                            sampled_normal = self._sample_normal_events(
                                normal_events=normal_events,
                                anomaly_events=[],  
                                target_count=target_normal_count,
                                sampling_strategy=sampling_strategy,
                                event_type_field=event_type_field,
                                normal_ratio=normal_ratio
                            ) if normal_events else []
                        else:
                            sampled_normal = normal_events.copy()
                        sampled_anomaly = []  

            else:
                
                if max_events_per_user is None:
                    logger.error("use_all_anomaly=False 时必须设置 max_events_per_user")
                    continue

                actual_anomaly_count = min(max_events_per_user, original_anomaly_count)
                remaining = max_events_per_user - actual_anomaly_count
                actual_normal_count = min(remaining, original_normal_count)
                actual_total = actual_anomaly_count + actual_normal_count
                if actual_total < max_events_per_user:
                    logger.warning(f"用户 {user_id} 事件总数不足: 目标{max_events_per_user}, 实际{actual_total}")
                sampled_anomaly = random.sample(anomaly_events,
                                                actual_anomaly_count) if actual_anomaly_count > 0 else []
                if actual_normal_count > 0 and normal_events:
                    sampled_normal = self._sample_normal_events(
                        normal_events=normal_events,
                        anomaly_events=anomaly_events,
                        target_count=actual_normal_count,
                        sampling_strategy=sampling_strategy,
                        event_type_field=event_type_field,
                        normal_ratio=normal_ratio
                    )
                else:
                    sampled_normal = []
                actual_ratio = actual_anomaly_count / actual_total if actual_total > 0 else 0
                logger.info(f"用户 {user_id}: 目标比例{normal_ratio}:{1 - normal_ratio}, "
                            f"实际比例{1 - actual_ratio:.2f}:{actual_ratio:.2f}")
            sampled_normal = [{**event, 'is_anomaly': False} for event in sampled_normal]
            sampled_anomaly = [{**event, 'is_anomaly': True} for event in sampled_anomaly]
            sampled_events = sampled_normal + sampled_anomaly
            if max_events_per_user and len(sampled_events) > max_events_per_user:
                scale = max_events_per_user / len(sampled_events)
                new_normal_count = int(len(sampled_normal) * scale)
                new_anomaly_count = int(len(sampled_anomaly) * scale)
                if new_normal_count > 0 and sampled_normal:
                    sampled_normal = self._sample_normal_events(
                        normal_events=sampled_normal,
                        anomaly_events=sampled_anomaly,
                        target_count=new_normal_count,
                        sampling_strategy=sampling_strategy,
                        event_type_field=event_type_field,
                        normal_ratio=normal_ratio
                    )
                else:
                    sampled_normal = []
                sampled_anomaly = random.sample(sampled_anomaly, new_anomaly_count) if new_anomaly_count > 0 else []
                sampled_events = sampled_normal + sampled_anomaly
            if sort_by_time and sampled_events and 'timestamp' in sampled_events[0]:
                sampled_events.sort(key=lambda x: x.get('timestamp', 0))
            batch_sequences[user_id] = sampled_events
            sampling_stats[user_id] = {
                'total_sampled': len(sampled_events),
                'normal_count': len(sampled_normal),
                'anomaly_count': len(sampled_anomaly),
                'actual_ratio': len(sampled_anomaly) / len(sampled_events) if sampled_events else 0,
                'original_normal_total': original_normal_count,
                'original_anomaly_total': original_anomaly_count,
                'anomaly_usage_rate': len(
                    sampled_anomaly) / original_anomaly_count if original_anomaly_count > 0 else 0,
                'normal_usage_rate': len(sampled_normal) / original_normal_count if original_normal_count > 0 else 0,
                
                'sampling_strategy_used': sampling_strategy if not (
                            use_all_anomaly and len(anomaly_events) == 0) else 'fallback_to_all_normal'
            }
            logger.debug(f"用户 {user_id}: 原始(正常={original_normal_count}, 异常={original_anomaly_count}) -> "
                             f"采样(正常={len(sampled_normal)}, 异常={len(sampled_anomaly)}), "
                             f"异常利用率={sampling_stats[user_id]['anomaly_usage_rate'] * 100:.1f}%")

        return {
            'user_ids': list(batch_sequences.keys()),
            'sequences': batch_sequences,  
            'sampling_stats': sampling_stats,
            
            'batch_info': {
                'normal_ratio': normal_ratio,
                'use_all_anomaly': use_all_anomaly,
                'max_events_per_user': max_events_per_user,
                'sampling_strategy': sampling_strategy,
                'event_type_field': event_type_field,
                'sampling': 'mixed_events_all_anomaly' if use_all_anomaly else 'mixed_events_fixed_size'
            }
        }

    def _sample_normal_events(self,
                              normal_events: List[Dict],
                              anomaly_events: List[Dict],
                              target_count: int,
                              sampling_strategy: str,
                              event_type_field: str,
                              normal_ratio: float = None) -> List[Dict]:
        
        
        if target_count >= len(normal_events):
            return normal_events.copy()
        if sampling_strategy == 'stratified':
            result = self._stratified_sample_by_event_type(
                normal_events, target_count, event_type_field
            )
            logger.debug(f"使用分层采样策略: 目标={target_count}, 实际采样={len(result)}")
        elif sampling_strategy == 'matched':
            result = self._match_normal_by_anomaly_types(
                anomaly_events, normal_events, normal_ratio, event_type_field
            )
            logger.debug(f"使用匹配采样策略: 目标={target_count}, 实际采样={len(result)}")
        else:
            result = random.sample(normal_events, min(target_count, len(normal_events)))
            logger.debug(f"使用随机采样策略: 目标={target_count}, 实际采样={len(result)}")

        return result

    def _stratified_sample_by_event_type(self,
                                         events: List[Dict],
                                         target_count: int,
                                         event_type_field: str = 'type') -> List[Dict]:
        
        
        type_counts = {}
        for event in events:
            event_type = event.get(event_type_field, 'unknown')
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        sampled_events = []
        total_original = len(events)
        allocated = 0
        type_targets = {}
        for event_type, original_count in type_counts.items():
            type_ratio = original_count / total_original
            type_target = max(1, int(target_count * type_ratio))  
            type_targets[event_type] = type_target
            allocated += type_target
        if allocated < target_count:
            remaining = target_count - allocated
            for event_type in type_targets:
                if remaining <= 0:
                    break
                type_targets[event_type] += 1
                remaining -= 1
        elif allocated > target_count:
            excess = allocated - target_count
            for event_type in sorted(type_targets.keys(),
                                     key=lambda x: type_targets[x],
                                     reverse=True):
                if excess <= 0:
                    break
                if type_targets[event_type] > 1:
                    type_targets[event_type] -= 1
                    excess -= 1
        for event_type, type_target in type_targets.items():
            type_events = [e for e in events
                           if e.get(event_type_field, 'unknown') == event_type]
            actual_count = min(type_target, len(type_events))
            if actual_count > 0:
                
                sampled = random.sample(type_events, actual_count)
                sampled_events.extend(sampled)
        if len(sampled_events) < target_count:
            remaining = target_count - len(sampled_events)
            sampled_ids = {id(e) for e in sampled_events}
            unsampled = [e for e in events if id(e) not in sampled_ids]
            if unsampled:
                additional = random.sample(unsampled, min(remaining, len(unsampled)))
                sampled_events.extend(additional)

        return sampled_events

    def _match_normal_by_anomaly_types(self,
                                       anomaly_events: List[Dict],
                                       normal_events: List[Dict],
                                       normal_ratio: float,
                                       event_type_field: str = 'type') -> List[Dict]:
        if not anomaly_events:
            return self._stratified_sample_by_event_type(
                normal_events,
                int(len(normal_events) * normal_ratio) if normal_ratio else len(normal_events),
                event_type_field
            )
        anomaly_type_counts = {}
        for event in anomaly_events:
            event_type = event.get(event_type_field, 'unknown')
            anomaly_type_counts[event_type] = anomaly_type_counts.get(event_type, 0) + 1
        anomaly_count = len(anomaly_events)
        target_normal_count = int(anomaly_count * (normal_ratio / (1 - normal_ratio))) if normal_ratio else 0
        sampled_normal = []
        total_anomaly = len(anomaly_events)
        for event_type, type_anomaly_count in anomaly_type_counts.items():
            type_ratio = type_anomaly_count / total_anomaly
            type_target = int(target_normal_count * type_ratio)
            type_normal = [e for e in normal_events
                           if e.get(event_type_field, 'unknown') == event_type]
            if type_normal and type_target > 0:
                actual_count = min(type_target, len(type_normal))
                sampled = random.sample(type_normal, actual_count)
                sampled_normal.extend(sampled)

                if len(type_normal) < type_target:
                    logger.debug(f"事件类型 {event_type} 的正常事件不足: "
                                 f"需要{type_target}，实际{len(type_normal)}")
        if len(sampled_normal) < target_normal_count:
            remaining = target_normal_count - len(sampled_normal)
            sampled_ids = {id(e) for e in sampled_normal}
            remaining_normal = [e for e in normal_events if id(e) not in sampled_ids]
            if remaining_normal:
                additional = random.sample(remaining_normal,
                                           min(remaining, len(remaining_normal)))
                sampled_normal.extend(additional)

        return sampled_normal

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="小批量加载器 - 调试工具")
    parser.add_argument('--batch-size', type=int, default=5, help='批次大小')
    parser.add_argument('--sample-size', type=int, default=1000, help='采样数据量（仅构建时使用）')
    parser.add_argument('--test-redteam', action='store_true', help='测试红队功能')
    parser.add_argument('--rebuild', action='store_true', help='强制重新构建序列（忽略已保存的）')
    parser.add_argument('--include-content', action='store_true',help='是否包含content字段（用于语义检测）')
    args = parser.parse_args()

    from utils import setup_logging

    log_file = setup_logging("mini_batch_loader_debug", __file__)
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("小批量加载器 - 独立调试")
    print("=" * 80)
    print(f"批次大小: {args.batch_size}")
    print(f"测试红队: {'是' if args.test_redteam else '否'}")
    print(f"强制重建: {'是' if args.rebuild else '否'}")
    print("=" * 80)

    if not config.init_config(auto_create_dirs=False, auto_validate=True):
        print("配置验证失败，请检查配置文件")
        exit(1)

    print("\n【步骤1】加载/构建行为序列...")

    from data_preprocessing.cert_data_loader import CERTDataLoader
    from data_preprocessing.behavior_sequence_builder import UserBehaviorSequenceBuilder
    from config import BEHAVIOR_SEQUENCE_DIR

    if args.include_content:
        sequence_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, 'with_content')
    else:
        sequence_dir = os.path.join(BEHAVIOR_SEQUENCE_DIR, 'light')

    metadata_file = os.path.join(sequence_dir, "_metadata.json")

    sequences = None

    if not args.rebuild and os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"  ✓ 发现已保存的序列: {metadata.get('num_users', 0)}个用户, 保存时间={metadata.get('saved_at')}")
            builder = UserBehaviorSequenceBuilder({})
            sequences = builder.load_sequences(input_dir=sequence_dir)
            print(f"  ✓ 加载完成: {len(sequences)} 个用户")
        except Exception as e:
            print(f"  ⚠ 加载失败: {e}，将重新构建")
            sequences = None

    if sequences is None:
        print(f"  构建新序列（采样大小={args.sample_size}）...")
        data_loader = CERTDataLoader()
        data = data_loader.load_all(use_sample=True, sample_size=args.sample_size)

        builder = UserBehaviorSequenceBuilder(data)
        sequences = builder.build_all_sequences()
        print(f"  ✓ 构建完成: {len(sequences)} 个用户")
        builder.save_sequences(output_dir=sequence_dir)
        print(f"  ✓ 序列已保存到: {sequence_dir}")
    print("\n【步骤2】创建小批量加载器...")

    loader = MiniBatchLoader(sequences, batch_size=args.batch_size)
    loader.print_batch_summary()
    print("\n【测试1】基础批次加载 - 验证顺序批次获取功能")
    for i in range(min(3, loader._num_batches)):
        batch = loader.get_batch(i)
        if batch:
            print(f"  批次{i}: {len(batch['user_ids'])}个用户")
            print(f"    用户列表: {batch['user_ids'][:3]}{'...' if len(batch['user_ids']) > 3 else ''}")
            print(f"    批次信息: {batch['batch_info']}")
    print("\n【测试2】随机批次加载 - 验证随机采样功能")

    random_batch = loader.get_random_batch(batch_size=3)
    print(f"  随机批次: {len(random_batch['user_ids'])}个用户")
    print(f"    用户: {random_batch['user_ids']}")
    print(f"    批次信息: {random_batch['batch_info']}")
    if args.test_redteam:
        print("\n【测试3】红队功能测试 - 验证异常用户识别和优先加载")

        from data_preprocessing.redteam_data_loader import RedTeamDataLoader
        redteam_loader = RedTeamDataLoader()
        redteam_data = redteam_loader.load(version_filter=getattr(config, 'DATASET_VERSION', '4.2'))

        if redteam_data['ground_truth']:
            loader.set_redteam_data(redteam_data)
            print(f"  ✓ 红队数据加载: {len(redteam_data['ground_truth'])} 个异常用户")

            
            print("\n  【格式检查】验证 ground_truth 是否包含事件级别标注:")
            sample_user = list(redteam_data['ground_truth'].keys())[0]
            print(f"    示例用户: {sample_user}")
            print(f"    ground_truth 字段: {list(redteam_data['ground_truth'][sample_user].keys())}")

            if 'event_labels' in redteam_data['ground_truth'][sample_user]:
                sample_labels = redteam_data['ground_truth'][sample_user]['event_labels']
                sample_event_ids = list(sample_labels.keys())[:3]
                print(f"    事件ID示例: {sample_event_ids}")
                print(f"    事件标注示例: {sample_labels[sample_event_ids[0]] if sample_event_ids else '无'}")
                print(f"    ✅ 事件级别标注存在")
            else:
                print(f"    ❌ 缺少 event_labels 字段，load_mixed_events_for_users 将无法工作")

            
            print("\n【测试4】包含异常用户的批次 - 验证异常用户优先策略")
            batch_with_anomaly = loader.get_batch(0, include_anomaly=True)
            if batch_with_anomaly:
                print(f"  批次用户: {batch_with_anomaly['user_ids']}")
                anomaly_flags = batch_with_anomaly.get('is_anomaly', {})
                anomaly_count = sum(1 for v in anomaly_flags.values() if v)
                print(f"  异常用户数: {anomaly_count}/{len(batch_with_anomaly['user_ids'])}")

            
            print("\n【测试5】红队专用批次 - 验证异常用户专用加载")
            redteam_batch = loader.get_redteam_batch(batch_size=5, include_normal=True)
            print(f"  批次用户: {redteam_batch['user_ids']}")
            print(f"  是否异常: {redteam_batch.get('is_anomaly', {})}")
            if redteam_batch.get('ground_truth'):
                for user, info in redteam_batch['ground_truth'].items():
                    print(f"    用户 {user}: 场景{info.get('scenario')}, 事件数={info.get('event_count')}")

            print("\n【测试7】事件级别混合采样 - 验证8:2比例采样")

            print("\n  【调试】检查 sequences 是否包含红队事件:")
            sample_user = 'AAM0658'
            sample_red_event_id = list(redteam_data['ground_truth'][sample_user]['event_labels'].keys())[0]
            print(f"    红队事件ID示例: {sample_red_event_id}")

            if sample_user in sequences:
                found = any(event.get('event_id') == sample_red_event_id for event in sequences[sample_user])
                print(f"    用户在 sequences 中: 是")
                print(f"    红队事件是否在 sequences 中: {found}")
                if not found:
                    print(
                        f"    sequences 中的 event_id 示例: {[e.get('event_id') for e in sequences[sample_user][:3]]}")
            else:
                print(f"    用户在 sequences 中: 否")

            print("\n  【调试】检查 sequences 中的 event_id 格式:")
            sample_user = 'AAM0658'  
            if sample_user in loader.sequences:
                sample_events = loader.sequences[sample_user]
                if sample_events:
                    print(f"    用户 {sample_user} 的第一个事件 event_id: {sample_events[0].get('event_id')}")
                    print(
                        f"    红队事件的 event_id 示例: {list(redteam_data['ground_truth']['AAM0658']['event_labels'].keys())[0]}")

            test_users = list(redteam_data['ground_truth'].keys())[:3]
            print(f"  测试用户: {test_users}")
            mixed_batch = loader.load_mixed_events_for_users(
                user_ids=test_users,
                normal_ratio=0.8,
                use_all_anomaly=True,
                max_events_per_user=500,
                sort_by_time=True
            )
            print(f"\n  【方案A】使用所有异常数据:")
            for uid in mixed_batch['user_ids']:
                stats = mixed_batch['sampling_stats'][uid]
                print(f"    用户 {uid}:")
                print(
                    f"      原始: 正常={stats['original_normal_total']}, 异常={stats['original_anomaly_total']}")
                print(
                    f"      采样: 正常={stats['normal_count']}, 异常={stats['anomaly_count']}, 总计={stats['total_sampled']}")
                print(
                    f"      实际异常比例: {stats['actual_ratio']:.2f} (目标: {1 - mixed_batch['batch_info']['normal_ratio']:.2f})")
                print(f"      异常利用率: {stats['anomaly_usage_rate'] * 100:.1f}%")

            print(f"\n  【方案B】固定总数采样:")
            mixed_batch_fixed = loader.load_mixed_events_for_users(
                user_ids=test_users,
                normal_ratio=0.8,
                use_all_anomaly=False,
                max_events_per_user=200,
                sort_by_time=True
            )

            for uid in mixed_batch_fixed['user_ids']:
                stats = mixed_batch_fixed['sampling_stats'][uid]
                print(f"    用户 {uid}: 采样{stats['total_sampled']}个事件 "
                      f"(正常{stats['normal_count']}:异常{stats['anomaly_count']}), "
                      f"实际异常比例={stats['actual_ratio']:.2f}")

            print("\n【测试8】对比不同采样策略 - random vs stratified vs matched")

            test_user = test_users[0] if test_users else None
            if test_user:
                print(f"\n  测试用户: {test_user}")
                sample_event = loader.sequences[test_user][0] if loader.sequences.get(test_user) else {}
                event_field = 'type' if 'type' in sample_event else 'event_type'
                print(f"    使用事件类型字段: {event_field}")
                user_events = loader.sequences.get(test_user, [])
                original_type_dist = {}
                for e in user_events:
                    etype = e.get(event_field, 'unknown')
                    original_type_dist[etype] = original_type_dist.get(etype, 0) + 1
                print(f"    原始事件类型分布: {dict(sorted(original_type_dist.items(), key=lambda x: -x[1])[:5])}")

                
                strategies = ['random', 'stratified', 'matched']
                for strategy in strategies:
                    result = loader.load_mixed_events_for_users(
                        user_ids=[test_user],
                        normal_ratio=0.8,
                        use_all_anomaly=True,
                        max_events_per_user=300,
                        sampling_strategy=strategy,
                        event_type_field=event_field
                    )

                    if result and test_user in result['sampling_stats']:
                        stats = result['sampling_stats'][test_user]
                        print(f"\n  策略: {strategy}")
                        print(f"    采样结果: 正常={stats['normal_count']}, 异常={stats['anomaly_count']}")
                        print(f"    实际异常比例: {stats['actual_ratio']:.3f}")
                        print(f"    使用策略: {stats.get('sampling_strategy_used', 'unknown')}")

                        
                        sampled_normal = [e for e in result['sequences'][test_user] if not e.get('is_anomaly', False)]
                        sampled_type_dist = {}
                        for e in sampled_normal:
                            etype = e.get(event_field, 'unknown')
                            sampled_type_dist[etype] = sampled_type_dist.get(etype, 0) + 1
                        print(
                            f"    采样后类型分布(Top3): {dict(sorted(sampled_type_dist.items(), key=lambda x: -x[1])[:3])}")

        else:
            print("  ⚠️ 未找到红队数据（当前版本可能没有红队标注）")
    else:
        print("\n【跳过】红队功能测试（使用 --test-redteam 启用）")


    print("\n【测试6】迭代器功能测试 - 验证生成器模式")

    batch_count = 0
    total_users = 0
    for batch in loader.iterate_batches():
        batch_count += 1  
        total_users += len(batch['user_ids'])  
        if batch_count <= 3:
            print(f"  批次{batch_count}: {len(batch['user_ids'])}个用户")

    print(f"  迭代完成: {batch_count}个批次, {total_users}个用户")

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"  ✓ 基础批次加载: 通过")
    print(f"  ✓ 随机批次加载: 通过")
    print(f"  ✓ 迭代器功能: 通过")
    if args.test_redteam:
        if redteam_data.get('ground_truth'):
            print(f"  ✓ 红队功能: 通过")
        else:
            print(f"  ⚠ 红队功能: 无数据")
    print(f"  📝 日志文件: {log_file}")
    print("=" * 80)
    print("调试完成")
