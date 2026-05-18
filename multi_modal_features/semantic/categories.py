
from enum import Enum
from typing import Dict, Optional

class SemanticCategory(Enum):

    EMAIL = "email_anomaly"          
    WEB = "web_anomaly"              
    FILE = "file_anomaly"            

    def get_display_name(self) -> str:

        names = {
            SemanticCategory.EMAIL: "邮件语义异常",
            SemanticCategory.WEB: "网页访问异常",
            SemanticCategory.FILE: "文件操作异常",
        }
        return names.get(self, self.value)

    def get_description(self) -> str:

        descriptions = {
            SemanticCategory.EMAIL: "邮件中含有敏感信息、机密数据或违规内容",
            SemanticCategory.WEB: "访问异常网站，访问内容与工作无关、偏离正常习惯或包含违规信息",
            SemanticCategory.FILE: "复制内容敏感的文件、文件主题与岗位工作内容无关",
        }
        return descriptions.get(self, "")

EVENT_TYPE_TO_CATEGORY: Dict[str, SemanticCategory] = {
    'email': SemanticCategory.EMAIL,
    'web': SemanticCategory.WEB,

    'file': SemanticCategory.FILE,
}

def get_category_by_event_type(event_type: str) -> Optional[SemanticCategory]:

    return EVENT_TYPE_TO_CATEGORY.get(event_type.lower())

def get_category_by_name(name: str) -> Optional[SemanticCategory]:

    try:
        return SemanticCategory(name)
    except ValueError:
        return None

def get_all_categories() -> list:

    return list(SemanticCategory)

def get_category_metadata() -> Dict[str, Dict]:

    return {
        cat.value: {
            'name': cat.get_display_name(),
            'description': cat.get_description(),
            'event_types': [et for et, c in EVENT_TYPE_TO_CATEGORY.items() if c == cat]
        }
        for cat in SemanticCategory
    }

if __name__ == "__main__":
    print("=" * 60)
    print("语义异常类别定义")
    print("=" * 60)

    print("\n【类别列表】")
    for cat in get_all_categories():
        print(f"  {cat.value}: {cat.get_display_name()}")
        print(f"    描述: {cat.get_description()}")
        print(f"    对应事件类型: {[et for et, c in EVENT_TYPE_TO_CATEGORY.items() if c == cat]}")

    print("\n【事件类型映射测试】")
    for event_type in ['email', 'web', 'file', 'unknown']:
        cat = get_category_by_event_type(event_type)
        print(f"  {event_type} -> {cat.value if cat else 'None'}")
