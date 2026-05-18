"""
================================================================================
[全局工具模块] - utils.py
================================================================================
功能描述：
    提供项目通用的工具函数，包括日志配置、文件操作、时间处理等。

函数列表：
    setup_logging: 配置日志系统，同时输出到文件和控制台
    ensure_dir: 确保目录存在，不存在则创建
    get_timestamp: 获取格式化的时间戳
    safe_serialize: 安全序列化对象（处理datetime等特殊类型）
================================================================================
"""

import os
import logging
from datetime import datetime


def setup_logging(script_name: str, current_file: str = None,
                  log_level: str = "INFO") -> str:
    """
    配置日志系统，同时输出到文件和控制台

    参数：
        script_name: 脚本名称，用于生成日志文件名
        current_file: 当前文件路径（通常用 __file__），用于确定日志目录
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）

    返回：
        日志文件路径

    使用示例：
        from utils import setup_logging
        log_file = setup_logging("my_script", __file__)
        logger = logging.getLogger(__name__)
        logger.info("日志已配置")
    """
    # 确定日志目录
    if current_file:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(current_file)), "logs")
    else:
        log_dir = "logs"

    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    # 配置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置日志
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志文件保存路径：{log_file}")

    return log_file


def ensure_dir(path: str) -> str:
    """
    确保目录存在，不存在则创建
    参数： path: 目录路径
    返回：目录路径
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    生成当前时间字符串，例如：20260326_153022
    用途：给文件 / 目录做备份、命名日志、命名输出文件，保证名字唯一不重复。

    参数：format_str: 时间格式字符串
    返回：格式化的时间戳字符串
    """
    return datetime.now().strftime(format_str)


def safe_serialize(obj):
    """
    JSON 不能直接存 datetime 时间 / 自定义对象，这个函数帮你自动转换成可保存格式
    时间 → 转成字符串
    对象 → 转成字典
    其他 → 转成字符串
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def safe_read_json(filepath: str, default: dict = None) -> dict:
    """
    安全读取JSON文件，失败时返回默认值
    不会因为文件不存在 / 损坏 / 格式错误而崩溃

    参数：
        filepath: JSON文件路径
        default: 读取失败时的默认值

    返回：
        JSON内容字典
    """
    import json
    if default is None:
        default = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"读取JSON文件失败 {filepath}: {e}")
        return default


def safe_write_json(filepath: str, data: dict, indent: int = 2) -> bool:
    """
    安全写入JSON文件

    参数：
        filepath: JSON文件路径
        data: 要写入的数据
        indent: 缩进空格数

    返回：
        是否写入成功
    """
    import json
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=safe_serialize)
        return True
    except Exception as e:
        logging.error(f"写入JSON文件失败 {filepath}: {e}")
        return False


def get_file_size_mb(filepath: str) -> float:
    """
    获取文件大小（MB）

    参数：
        filepath: 文件路径

    返回：
        文件大小（MB）
    """
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / 1024 / 1024
    return 0.0


def format_number(num: int) -> str:
    """
    格式化数字，添加千位分隔符

    参数：
        num: 数字

    返回：
        格式化后的字符串，如 "1,234,567"
    """
    return f"{num:,}"