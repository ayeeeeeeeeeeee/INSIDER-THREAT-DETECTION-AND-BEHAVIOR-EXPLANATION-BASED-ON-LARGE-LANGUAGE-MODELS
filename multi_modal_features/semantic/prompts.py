from enum import Enum
from multi_modal_features.semantic.categories import get_category_by_event_type

class OutputMode(Enum):
    DETAILED = "detailed"
    SIMPLE = "simple"

def truncate_content(content: str, max_chars: int = 1500) -> str:

    if len(content) <= max_chars:
        return content

    head_size = int(max_chars * 0.7)
    tail_size = max_chars - head_size - 20

    return f"{content[:head_size]}...[内容截断]...{content[-tail_size:]}"

def format_event_fields(event_fields: dict, event_type) -> str:

    event_info = ""
    if event_fields.get('timestamp'):
        event_info += f"- 时间: {event_fields['timestamp']}\n"

    if event_type == 'web':
        if event_fields.get('url'):
            event_info += f"- 访问URL: {event_fields['url']}\n"
    elif event_type == 'email':
        if event_fields.get('to'):
            event_info += f"- 收件人: {event_fields['to']}\n"
        if event_fields.get('from'):
            event_info += f"- 发件人: {event_fields['from']}\n"
        if event_fields.get('cc'):
            event_info += f"- 抄送: {event_fields['cc']}\n"
    elif event_type == 'file':
        if event_fields.get('filename'):
            event_info += f"- 文件名: {event_fields['filename']}\n"

    return event_info if event_info else "无额外信息"

def format_event_fields_v2(event_fields: dict) -> str:

    if not event_fields:
        return "无额外的事件相关信息。"

    parts = []
    if event_fields.get('timestamp'):
        parts.append(f"事件发生时间为 {event_fields['timestamp']}")
    if event_fields.get('url'):
        parts.append(f"访问的 URL 地址是 {event_fields['url']}")
    if event_fields.get('to'):
        parts.append(f"邮件收件人为 {event_fields['to']}")
    if event_fields.get('from'):
        parts.append(f"邮件发件人为 {event_fields['from']}")
    if event_fields.get('cc'):
        parts.append(f"邮件抄送给了 {event_fields['cc']}")
    if event_fields.get('filename'):
        parts.append(f"涉及的文件名为 {event_fields['filename']}")
    if parts:
        return "事件背景信息：" + "；".join(parts) + "。"
    else:
        return "无额外的事件相关信息。"

def format_user_profile(user_profile: dict, timestamp: str = None, event_type: str = None) -> str:

    if not user_profile:
        return ""

    relevant_fields = []
    if user_profile.get('role'):
        relevant_fields.append(f"角色:{user_profile['role']}")

    psych = user_profile.get('personality', user_profile)

    personality_fields = []
    for key, label in [('openness', '开放性'), ('conscientiousness', '尽责性'),
                       ('extraversion', '外向性'), ('agreeableness', '宜人性'),
                       ('neuroticism', '神经质')]:
        val = psych.get(key)
        if val is not None:
            personality_fields.append(f"{label}:{val}")
    if personality_fields:
        relevant_fields.append(f"大五人格:({' | '.join(personality_fields)})")

    if event_type == 'email':
        if user_profile.get('department'):
            relevant_fields.append(f"部门:{user_profile['department']}")
        if user_profile.get('employment_status') == 'Terminated':
            relevant_fields.append("已离职")
        if user_profile.get('supervisor'):
            relevant_fields.append(f"上级:{user_profile['supervisor']}")

    elif event_type == 'web':
        user_type = user_profile.get('User_Type')
        if user_type == 'ITAdmin':
            relevant_fields.append("IT管理员")
        elif user_type == 'Non_Employee':
            relevant_fields.append("非正式员工")
    elif event_type == 'file':
        if user_profile.get('is_it_admin'):
            relevant_fields.append("IT管理员")
        if user_profile.get('global_access'):
            relevant_fields.append("全局访问权限")

    if relevant_fields:
        return f"用户画像: {' | '.join(relevant_fields)}"
    return ""

def get_data_format_note(event_type: str) -> str:
    format_notes = {
        'web': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词与网页主题相关
- 你需要从这些关键词中识别是否存在安全风险（如：求职相关、恶意软件、数据泄露等）
- 注意：域名可能为随机生成，但内容关键词仍具有分析价值""",

        'email': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词来自邮件正文或主题
- 邮件主题和正文未作区分，你需要综合理解关键词含义
- 邮件收件人、发件人信息已在"事件信息"中提供，请结合分析""",

        'file': """

- 待分析内容由**十六进制文件头**和**空格分隔的关键词**组成
- 文件头可以判断文件类型（如.exe表示可执行文件）
- 关键词反映文件内容主题，请结合文件名判断是否存在风险"""
    }
    return format_notes.get(event_type, "")

def get_data_format_note_v2(event_type: str) -> str:
    format_notes = {
        'web': "待分析内容是空格分隔的网页内容关键词（域名可能随机生成）。",
        'email': "待分析内容是空格分隔的、来自邮件正文或主题的关键词。",
        'file': "待分析内容由十六进制编码的文件头和后续空格分隔的内容关键词组成。文件头与文件扩展名相关联。"
    }
    return format_notes.get(event_type, "")

def get_data_format_note_v3(event_type: str) -> str:
    format_notes = {
        'web': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词与网页主题相关
- 你需要从这些关键词中识别是否存在安全风险（如：求职诈骗、恶意软件下载、钓鱼、数据泄露等）

**正常网页示例：**
`project update meeting schedule budget approval team collaboration quarterly report`
→ 关键词涉及会议、共识、协作等，属于正常工作讨论范畴

**恶意/敏感网页示例：**
- `bypass dlp data exfiltration stealth technique how to hide`
- `company confidential leak internal salary database dump`
- `keylogger undetectable free download remote admin tool`
→ 关键词涉及数据泄露、规避检测、恶意工具等，具有明显安全风险

- 注意：域名可能为随机生成，但内容关键词仍具有分析价值
""",

        'email': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词来自邮件正文或主题
- 邮件主题和正文未作区分，你需要综合理解关键词含义

**正常邮件示例：**
`system upgrade scheduled friday evening impact minimal users notified beforehand`
→ 关键词涉及系统维护、时间安排、用户通知等，属于正常工作沟通

**恶意/异常邮件示例：**
- `company will suffer i may leave no gratitude fed up my work not appreciated`
- `confidential client list attached please review strictly internal only`
- `password reset account verify urgent click link now`
→ 关键词涉及不满情绪、敏感数据外发、钓鱼诱导等，需重点关注

- 邮件收件人、发件人信息已在"事件信息"中提供，请结合分析
""",

        'file': """

- 待分析内容由**十六进制文件头**和**空格分隔的关键词**组成
- 文件头可以判断文件类型（如 4D 5A 表示 MZ/可执行文件，50 4B 表示 ZIP）

**正常文件示例：**
`D0-CF-11-E0-A1-B1-1A-E1 quarterly report financial summary revenue projection analysis`
→ 文件头 `D0-CF-11-E0` 为 MS Office 文档，关键词为正常工作文档内容

**恶意文件示例：**
`4D-5A-90-00-03-00-00-00-B8-00-00-00-00-00-00-00-40-00 keylogger stealth capture keystroke undetectable password grabber hidden monitor`
→ 文件头 `4D-5A` 为 Windows 可执行文件(.exe)，关键词包含 `keylogger`、`stealth`、`undetectable`、`password` 等恶意特征词汇

- 请结合文件名和关键词判断是否存在风险

"""
    }

    return format_notes.get(event_type, "")

def get_data_format_note_v4(event_type: str) -> str:

    format_notes = {
        'web': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词与网页主题相关
- 你需要从这些关键词中识别是否存在安全风险（如：求职诈骗、恶意软件下载、钓鱼、数据泄露等）

**关键判断原则：**
⚠️ **宁可误报，不可漏报**。如果关键词中出现任何安全相关术语或异常模式，应倾向判定为风险。
**高风险关键词（出现任一即应警惕）：**
- 数据泄露类：bypass, exfiltration, leak, dump, confidential, internal use only
- 恶意工具类：keylogger, malware, undetectable, stealth, hack, exploit, remote admin
- 钓鱼/欺诈类：verify account, password reset, urgent action, login credentials
- 规避检测类：dlp bypass, evade, hide, cover tracks, anti-forensics

**判断指引：**
1. 优先扫描是否包含上述高风险关键词
2. 即使只有1-2个高风险词，也应标记为可疑
3. 正常业务词汇与高风险词混合出现时，**以高风险词为准**
4. 不确定时，倾向于判定为有风险

- 注意：域名可能为随机生成，但内容关键词仍具有分析价值
""",

        'email': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词来自邮件正文或主题
- 邮件主题和正文未作区分，你需要综合理解关键词含义

**关键判断原则：**
⚠️ **宁可误报，不可漏报**。异常情绪、敏感内容、异常沟通模式均应引起警觉。

**高风险模式（出现任一即应警惕）：**
- 不满/离职倾向：no gratitude, fed up, i may leave, not appreciated, suffer
- 敏感数据外发：confidential, internal only, do not forward, client list, salary
- 异常附件/链接：attached, click here, verify account, download now
- 异常沟通：向非工作邮箱发送、异常收件人组合

**判断指引：**
1. 不满情绪+工作内容 = 潜在内部威胁，标记为高风险
2. 敏感词+附件/链接 = 数据外泄风险，标记为高风险
3. 正常业务词汇与高风险词混合出现时，**以高风险词为准**
4. 不确定时，倾向于判定为有风险

- 邮件收件人、发件人信息已在"事件信息"中提供，请结合分析
""",

        'file': """

- 待分析内容由**十六进制文件头**和**空格分隔的关键词**组成
- 文件头可以判断文件类型（如 4D 5A 表示 MZ/可执行文件，50 4B 表示 ZIP）

**关键判断原则：**
⚠️ **可执行文件+敏感关键词=极高风险**，必须标记。

**高风险组合（出现任一即应警惕）：**
- 文件头为可执行文件（4D 5A） + 任何安全相关关键词
- 大量重复的敏感词（如 undetectable 重复出现）
- 文件类型与关键词内容不符（如伪装成文档的可执行文件）

**高风险关键词：**
- 监控/窃取类：keylogger, spyware, capture, monitor, steal, grabber
- 隐藏/规避类：undetectable, stealth, hidden, covert, bypass, evade
- 密码/凭证类：password, credential, account, login
- 恶意软件类：malware, trojan, backdoor, rat, exploit

**判断指引：**
1. 首先检查文件头：4D 5A 开头 = 可执行文件，基础风险提升
2. 扫描关键词：出现任一高风险词 = 可疑
3. 可执行文件 + 伪装性文件名（如 .doc.exe） = 极高风险
4. 关键词大量重复（如 undetectable 出现5次以上）= 强烈恶意信号
5. 不确定时，倾向于判定为有风险

- 请结合文件名和关键词判断是否存在风险

"""
    }

    return format_notes.get(event_type, "")

def get_data_format_note_v5(event_type: str) -> str:

    format_notes = {
        'web': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词与网页主题相关
- 你需要从这些关键词中识别是否存在安全风险（如：求职诈骗、恶意软件下载、钓鱼、数据泄露等）

**恶意/敏感网页示例：**
- `bypass dlp data exfiltration stealth technique how to hide`
- `company confidential leak internal salary database dump`
- `keylogger undetectable free download remote admin tool`
→ 关键词涉及数据泄露、规避检测、恶意工具等，具有明显安全风险

- 注意：域名可能为随机生成，但内容关键词仍具有分析价值
""",

        'email': """

- 待分析内容为**空格分隔的关键词列表**，这些关键词来自邮件正文或主题
- 邮件主题和正文未作区分，你需要综合理解关键词含义

**恶意/异常邮件示例：**
- `company will suffer i may leave no gratitude fed up my work not appreciated`
- `confidential client list attached please review strictly internal only`
- `password reset account verify urgent click link now`
→ 关键词涉及不满情绪、敏感数据外发、钓鱼诱导等，需重点关注

- 邮件收件人、发件人信息已在"事件信息"中提供，请结合分析
""",

        'file': """

- 待分析内容由**十六进制文件头**和**空格分隔的关键词**组成
- 文件头可以判断文件类型（如 4D 5A 表示 MZ/可执行文件，50 4B 表示 ZIP）

**恶意文件示例：**
`4D-5A-90-00-03-00-00-00-B8-00-00-00-00-00-00-00-40-00 keylogger stealth capture keystroke undetectable password grabber hidden monitor`
→ 文件头 `4D-5A` 为 Windows 可执行文件(.exe)，关键词包含 `keylogger`、`stealth`、`undetectable`、`password` 等恶意特征词汇

- 请结合文件名和关键词判断是否存在风险

"""
    }

    return format_notes.get(event_type, "")

# def get_data_format_note_v6(event_type: str) -> str:
#
#     keywords = {
#         'web': """高风险关键词参考：bypass, exfiltration, leak, dump, keylogger, undetectable, stealth, hack, exploit, malware""",
#         'email': """高风险关键词参考：confidential, fed up, i may leave, not appreciated, password reset, verify account, attached, internal only""",
#         'file': """高风险关键词参考：keylogger, stealth, undetectable, password, malware, bypass, covert, hidden, monitor, capture"""
#     }
#     return f"

def _build_prompt_core(content: str, event_type: str, event_fields: dict,
                       user_profile, max_content_chars: int) -> str:

    truncated_content = truncate_content(content, max_content_chars)
    timestamp = event_fields.get('timestamp') if event_fields else None

    category = get_category_by_event_type(event_type)
    category_name = category.get_display_name() if category else "通用语义异常"
    category_desc = category.get_description() if category else "检测内容是否存在安全异常"

    event_info = format_event_fields(event_fields, event_type) if event_fields else "无"
    profile_info = f"{format_user_profile(user_profile, timestamp, event_type)}" if user_profile else ""
    data_format_note = get_data_format_note_v2(event_type)

    return f"""你是一名企业内部威胁检测分析师。请分析以下内容是否属于指定类型的安全异常行为。

检测类别：{category_name}
类别说明：{category_desc}
{profile_info}
事件类型：{event_type}
事件信息：{event_info}
{data_format_note}
待分析内容：{truncated_content}"""

def _build_prompt_core_analysis(content: str, event_type: str, event_fields: dict,
                       user_profile, max_content_chars: int) -> str:

    truncated_content = truncate_content(content, max_content_chars)
    timestamp = event_fields.get('timestamp') if event_fields else None

    category = get_category_by_event_type(event_type)
    category_name = category.get_display_name() if category else "通用语义异常"
    category_desc = category.get_description() if category else "检测内容是否存在安全异常"

    event_info = format_event_fields(event_fields, event_type) if event_fields else "无"
    profile_info = f"用户信息：{format_user_profile(user_profile, timestamp, event_type)}" if user_profile else ""
    data_format_note = get_data_format_note_v3(event_type)

    return f"""你是一名企业内部威胁检测分析师。请分析以下内容是否属于指定类型的安全异常行为。

检测类别：{category_name}
类别说明：{category_desc}
{profile_info}
事件类型：{event_type}
事件信息：{event_info}
{data_format_note}
待分析内容：{truncated_content}"""

def build_semantic_analysis_prompt(content: str, event_type: str,
                                   event_fields: dict = None,
                                   user_profile=None,
                                   mode: OutputMode = OutputMode.SIMPLE,
                                   max_content_chars: int = 1500) -> str:

    core_prompt = _build_prompt_core(content, event_type, event_fields, user_profile, max_content_chars)

    if mode == OutputMode.SIMPLE:
        output_instruction = """请评估以上内容的异常程度，输出一个0.0到1.0之间的数字，表示异常可能性。
必须严格遵守：只输出一个数字，不要任何前缀、后缀、解释或标点符号。"""
    else:
        output_instruction = """
1. 只输出一行文本，使用竖线 | 分隔四个部分
2. 四个部分依次为：判定、分数、关键证据、解释说明
3. 判定只能是"异常"或"正常"
4. 分数是0.0-1.0之间的数字
5. 关键证据不超过30字，如果没有异常可写"无"
6. 解释说明不超过100字
7. 不要输出任何其他内容

正确示例：
异常 | 0.85 | 数据库密码泄露 | 邮件中包含明文密码信息
正常 | 0.12 | 无 | 内容为正常业务沟通

现在请直接输出："""

    prompt = f"{core_prompt}\n\n{output_instruction}"
    return core_prompt

def build_simplified_prompt(content: str, event_type: str,
                            mode: OutputMode = OutputMode.DETAILED) -> str:

    if mode == OutputMode.SIMPLE:
        output_format = '{"is_anomaly": true/false, "anomaly_score": 0.0-1.0}'
    else:
        output_format = '{"is_anomaly": true/false, "anomaly_score": 0.0-1.0, "key_evidence": "...", "explanation": "..."}'

    return f"""分析以下内容是否存在安全异常。

事件类型: {event_type}
内容: {content[:1000]}

输出JSON格式:
{output_format}"""

def build_training_prompt(content, event_type: str,
                          event_fields=None, user_profile=None, max_content_chars: int = 1500):

    core_prompt = _build_prompt_core(content, event_type, event_fields, user_profile, max_content_chars)
    return core_prompt

if __name__ == "__main__":
    import sys
    import os

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_current_dir))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    print("=" * 80)
    print("语义异常检测提示词模板测试")
    print("=" * 80)
    print("\n【测试1】邮件语义异常")
    email_content = """company will suffer i may leave no gratitude too much too much company will suffer i may leave fed up my work not appreciated i work holidays my work not appreciated fed up i may leave i work after-hours fed up i work holidays fed up company will suffer too much i work holidays fed up i may leave no gratitude i work after-hours i work holidays i work after-hours i may leave my work not appreciated i work weekends i may leave no gratitude i work after-hours company will suffer i work after-hours i work after-hours too much company will suffer fed up complaints i may leave complaints company will suffer complaints fed up i work holidays my work not appreciated my work not appreciated my work not appreciated no gratitude"""
    email_context = {
        'user_id': 'BSS0369',
        'timestamp': '09/30/2010 13:31:56',
        'to': 'Francis.Brian.Armstrong@dtaa.com',
        'from': 'Brenden.Samuel.Shaffer@dtaa.com',
        'cc': 'Brenden.Samuel.Shaffer@dtaa.com'
    }
    print("\n--- 详细模式 ---")
    prompt = build_semantic_analysis_prompt(email_content, 'email', email_context, OutputMode.DETAILED)
    print(prompt)
    print("\n--- 简化模式 ---")
    prompt = build_semantic_analysis_prompt(email_content, 'email', email_context, OutputMode.SIMPLE)
    print(prompt)

    print("\n" + "=" * 80)
    print("\n【测试2】网页访问异常")

    web_content = """customer opening management technologies salary process job on-time equivalent multitask visual required permanent passion technologies degree degree sales platform resume technologies analyze strong equivalent process engineer equivalent contribute contribute opening expert interface contribute multiple interface responsibilities opening strong experience responsibilities engineer contribute equivalent benefits job equivalent concepts self equivalent experience customer growth passion experience growth concepts passion guidance concepts opening customer management team"""
    web_context = {
        'user_id': 'ABC0174',
        'timestamp': '10/27/2010 14:11:03',
        'url': 'http://boeing.com/WboUhagvat1904327536.htm'
    }

    print("\n--- 详细模式 ---")
    prompt = build_semantic_analysis_prompt(web_content, 'web', web_context, OutputMode.DETAILED)
    print(prompt)

    print("\n--- 简化模式 ---")
    prompt = build_semantic_analysis_prompt(web_content, 'web', web_context, OutputMode.SIMPLE)
    print(prompt)

    print("\n" + "=" * 80)
    print("\n【测试3】文件操作异常")

    file_content = """4D-5A-90-00-03-00-00-00-04-00-00-00-FF-FF-00-00-B8-00-00-00-00-00-00-00-40-00 easy username free monitor password advanced covert gui program undetectable email activity malware keyboard undetectable undetectable undetectable secure everything captured secure monitor surveillance stealth everything covert malware malware keylogging keylogging free everything pc file captured password gui recommend email effective hidden pc easy activity username keyboard free covert easy download download"""
    file_context = {
        'user_id': 'JLM0364',
        'timestamp': '04/28/2011 16:06:52',
        'filename': '7ESLQOMY.exe'
    }

    print("\n--- 详细模式 ---")
    prompt = build_semantic_analysis_prompt(file_content, 'file', file_context, OutputMode.DETAILED)
    print(prompt)

    print("\n--- 简化模式 ---")
    prompt = build_semantic_analysis_prompt(file_content, 'file', file_context, OutputMode.SIMPLE)
    print(prompt)

    print("\n" + "=" * 80)
    print("\n【测试4】正常内容 - 普通工作邮件（对比）")

    normal_content = """项目进度汇报：本周完成了用户认证模块的开发，目前正在进行单元测试。预计下周可以提交代码审查。项目进度正常，无异常情况。"""

    normal_context = {
        'user_id': 'NORMAL01',
        'timestamp': '2024-03-28 10:30:00',
        'to': 'team@dtaa.com',
        'from': 'employee@dtaa.com'
    }

    print("\n--- 详细模式 ---")
    prompt = build_semantic_analysis_prompt(normal_content, 'email', normal_context, OutputMode.DETAILED)
    print(prompt)

    print("\n" + "=" * 80)
    print("【测试5】训练提示词（无输出格式指令）")
    print("=" * 80)

    training_prompt = build_training_prompt(
        email_content,
        'email',
        email_context,
        max_content_chars=1500
    )
    print(training_prompt)
    print("\n注意：训练提示词不包含输出格式指令，用于微调时的输入部分")
    print("\n" + "=" * 80)
    print("调试完成")
    print("\n提示：以上提示词可直接复制到LLM中进行测试")
    print("=" * 80)
