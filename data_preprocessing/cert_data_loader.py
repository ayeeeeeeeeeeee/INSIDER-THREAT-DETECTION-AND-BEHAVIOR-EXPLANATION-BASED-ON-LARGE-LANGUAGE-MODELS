import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from tqdm import tqdm

_current_dir = os.path.dirname(os.path.abspath(__file__))  
_project_root = os.path.dirname(_current_dir)             
if _project_root not in sys.path:                         
    sys.path.insert(0, _project_root)                     

import config                                             
from data_preprocessing.preprocessing_utils import get_dtype_for_file, safe_read_csv

logger = logging.getLogger(__name__)

class CERTDataLoader:
    
    def __init__(self):
        
        self.data = {}
        self.is_loaded = False
        
    def load_all(self, use_sample: bool = False, sample_size: int = None) -> Dict[str, pd.DataFrame]:
        
        logger.info("开始加载CERT数据集...")
        self.data['logon'] = self._load_csv(config.LOGON_FILE, 'logon', use_sample, sample_size)
        self.data['device'] = self._load_csv(config.DEVICE_FILE, 'device', use_sample, sample_size)
        self.data['http'] = self._load_csv(config.HTTP_FILE, 'http', use_sample, sample_size)  
        self.data['email'] = self._load_csv(config.EMAIL_FILE, 'email', use_sample, sample_size)
        self.data['file'] = self._load_csv(config.FILE_FILE, 'file', use_sample, sample_size)
        self.data['psychometric'] = self._load_csv(config.PSYCHOMETRIC_FILE, 'psychometric', use_sample, sample_size)
        self.data['ldap'] = self._load_ldap_data()
        
        if use_sample and sample_size and self.data['ldap'] is not None:
            self.data['ldap'] = self.data['ldap'][
                self.data['ldap']['user_id'].isin(self._get_sampled_users())
            ]
        self._clean_data()
        self.is_loaded = True
        logger.info(f"数据加载完成，共加载 {len(self.data)} 个数据源")
        
        return self.data

    def _load_csv(self, filepath: str, name: str, use_sample: bool = False, sample_size: int = None) -> Optional[pd.DataFrame]:
        
        if not os.path.exists(filepath):
            logger.warning(f"文件不存在: {filepath}")
            return None

        
        nrows = sample_size if (use_sample and sample_size and sample_size > 0) else None
        dtype_dict = get_dtype_for_file(name)
        df = safe_read_csv(filepath, name, dtype_dict, nrows=nrows)

        return df

    def _load_ldap_data(self) -> Optional[pd.DataFrame]:
        
        if not os.path.exists(config.LDAP_PATH):
            logger.warning(f"LDAP目录不存在: {config.LDAP_PATH}")
            return None
        
        ldap_files = sorted([f for f in os.listdir(config.LDAP_PATH) if f.endswith('.csv')])
        if not ldap_files:
            logger.warning("LDAP目录下无CSV文件")
            return None
        
        logger.info(f"加载LDAP数据，共 {len(ldap_files)} 个月份")
        
        ldap_dtype = {
            'employee_name': 'string',  
            'user_id': 'category',  
            'email': 'string',  
            'role': 'category',  
            'business_unit': 'int8',  
            'functional_unit': 'category',  
            'department': 'category',  
            'team': 'category',  
            'supervisor': 'category'  
        }

        ldap_dfs = []  
        for file in tqdm(ldap_files, desc="加载LDAP文件"):  
            filepath = os.path.join(config.LDAP_PATH, file)
            try:
                df = pd.read_csv(
                    filepath,
                    dtype=ldap_dtype,  
                    encoding='utf-8',  
                    engine='python',  
                )
                
                month = file.replace('.csv', '')
                df['snapshot_month'] = month
                ldap_dfs.append(df)
            except UnicodeDecodeError:
                
                logger.warning(f"UTF-8解码失败，尝试latin-1加载 {file}")
                df = pd.read_csv(
                    filepath,
                    dtype=ldap_dtype,
                    encoding='latin-1',
                    engine='python',
                    low_memory=False
                )
                df['snapshot_month'] = month
                ldap_dfs.append(df)
            except Exception as e:
                logger.error(f"加载LDAP文件失败 {file}: {e}")

        if ldap_dfs:
            result = pd.concat(ldap_dfs, ignore_index=True)
            logger.info(f"✓ 加载LDAP: {len(result)} 条记录，{len(ldap_files)} 个月份")
            return result
        return None

    def _clean_data(self):
        if self.data['logon'] is not None:
            self.data['logon']['date'] = pd.to_datetime(self.data['logon']['date'], errors='coerce')
        if self.data['device'] is not None:
            self.data['device']['date'] = pd.to_datetime(self.data['device']['date'], errors='coerce')
        if self.data['http'] is not None:
            self.data['http']['date'] = pd.to_datetime(self.data['http']['date'], errors='coerce')
            if 'content' in self.data['http'].columns:
                self.data['http']['content'] = self.data['http']['content'].fillna('')
        if self.data['email'] is not None:
            self.data['email']['date'] = pd.to_datetime(self.data['email']['date'])
            self.data['email'] = self.data['email'].fillna({
                'cc': '',
                'bcc': ''
            })
            if 'content' in self.data['email'].columns:
                self.data['email']['content'] = self.data['email']['content'].fillna('')
            if 'attachments' in self.data['email'].columns:
                self.data['email']['attachments'] = self.data['email']['attachments'].fillna(0).astype('int16')
        if self.data['file'] is not None:
            self.data['file']['date'] = pd.to_datetime(self.data['file']['date'], errors='coerce')
            if 'content' in self.data['file'].columns:
                self.data['file']['content'] = self.data['file']['content'].fillna('')
        logger.info("数据清洗完成")

    def _get_sampled_users(self) -> set:
        sampled_users = set()
        for key in ['logon', 'device', 'http', 'email', 'file']:
            if self.data.get(key) is not None and 'user' in self.data[key].columns:
                sampled_users.update(self.data[key]['user'].unique())
        return sampled_users

if __name__ == "__main__":
    
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)  
    log_file = os.path.join(log_dir, f"cert_loader_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  
            logging.StreamHandler()  
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志文件保存路径：{log_file}")

    print("=" * 60)
    print("CERT数据加载器 - 独立调试")
    print("=" * 60)
    if not config.init_config(auto_create_dirs=False, auto_validate=True):
        print("配置验证失败，请检查配置文件")
        exit(1)
    loader = CERTDataLoader()
    print("\n【测试1】加载采样数据（每类1000条）...")
    data = loader.load_all(use_sample=True, sample_size=1000)
    print("\n【测试结果】数据加载统计:")
    for key, df in data.items():
        if df is not None:
            print(f"  {key}: {len(df)} 条记录, 内存: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        else:
            print(f"  {key}: 无数据")
    print("\n【测试2】用户统计:")
    all_users = set()
    for key in ['logon', 'device', 'http', 'email', 'file']:
        if data.get(key) is not None and 'user' in data[key].columns:
            users = set(data[key]['user'].unique())
            print(f"  {key}: {len(users)} 个用户")
            all_users.update(users)

    if data.get('psychometric') is not None:
        psych_users = set(data['psychometric']['user_id'].unique())
        print(f"  psychometric: {len(psych_users)} 个用户")
        all_users.update(psych_users)

    print(f"\n  总用户数: {len(all_users)}")

    print("\n【测试3】时间范围:")
    for key in ['logon', 'device', 'http', 'email', 'file']:
        if data.get(key) is not None and 'date' in data[key].columns:
            min_date = data[key]['date'].min()
            max_date = data[key]['date'].max()
            print(f"  {key}: {min_date} ~ {max_date}")

    print("\n【测试4】LDAP数据:")
    if data.get('ldap') is not None:
        print(f"  总记录数: {len(data['ldap'])}")
        print(f"  月份数: {data['ldap']['snapshot_month'].nunique()}")
        print(f"  月份范围: {data['ldap']['snapshot_month'].min()} ~ {data['ldap']['snapshot_month'].max()}")
        print(f"  角色分布:\n{data['ldap']['role'].value_counts().head(5)}")
        print(f"  采样用户数: {len(data['ldap']['user_id'].unique())}")

    print("\n" + "=" * 60)
    print("调试完成")
