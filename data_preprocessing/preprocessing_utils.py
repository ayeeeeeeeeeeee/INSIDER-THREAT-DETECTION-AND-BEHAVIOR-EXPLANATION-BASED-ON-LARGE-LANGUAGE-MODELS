
import os
import sys
import pandas as pd
import logging


_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

def get_dtype_for_file(name: str) -> dict:
    
    dtypes = {
        
        'logon': {'ID': 'category', 'user': 'category', 'pc': 'category', 'activity': 'category'},
        'device': {'ID': 'category', 'user': 'category', 'pc': 'category', 'activity': 'category'},
        'http': {'ID': 'category', 'user': 'category', 'pc': 'category'},
        'email': {'ID': 'category', 'user': 'category', 'pc': 'category', 'attachments': 'int16'},
        'file': {'ID': 'category', 'user': 'category', 'pc': 'category'},
        'psychometric': {'user_id': 'category', 'O': 'int8', 'C': 'int8', 'E': 'int8', 'A': 'int8', 'N': 'int8'}
        
    }
    return dtypes.get(name, {})


def safe_read_csv(filepath: str, name: str, dtype_dict: dict = None, nrows: int = None) -> pd.DataFrame | None:
    
    if dtype_dict is None:
        dtype_dict = {}

    
    read_kwargs = {
        'dtype': dtype_dict,
        'engine': 'python',
        'encoding': 'utf-8',
    }
    
    if nrows and nrows > 0:
        read_kwargs['nrows'] = nrows
        logger.info(f"📌 采样模式：加载 {name} 前 {nrows} 条记录")

    try:
        df = pd.read_csv(filepath, **read_kwargs)
        logger.info(f"✓ 加载 {name}: {len(df)} 条记录")
        return df
    except UnicodeDecodeError:
        
        read_kwargs['encoding'] = 'latin-1'
        try:
            df = pd.read_csv(filepath, **read_kwargs)
            logger.info(f"✓ 加载 {name} (latin-1编码): {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"latin-1编码加载失败 {filepath}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"加载失败 {filepath}: {str(e)}")
        return None
