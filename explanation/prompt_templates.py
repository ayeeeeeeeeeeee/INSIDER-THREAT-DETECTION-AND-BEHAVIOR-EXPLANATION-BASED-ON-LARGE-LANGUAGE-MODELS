
from typing import Dict, List, Optional, Union

OUTPUT_JSON_SCHEMA = """
{
  "user_id": "string",
  "threat_conclusion": {"is_threat": bool, "threat_type": "string", "threat_score": float, "confidence": float},
  "reasoning_chain": [{"step": int, "step_name": "string", "analysis": "string", "evidence_refs": []}],
  "core_anomalies": [{"evidence_id": "string", "anomaly_description": "string", "severity": "string", "quantitative_validation": {}}],
  "module_contributions": {"semantic_module": {"weight": float, "triggered": bool, "key_findings": ["string"], "contribution_reason": "string"}, "statistical_module": {"weight": float, "triggered": bool, "key_findings": ["string"], "contribution_reason": "string"}},
  "natural_language_explanation": {"normal_pattern_summary": "string", "anomaly_comparison": "string", "typical_pattern_comparison": "string"},
  "security_recommendations": {"level": "string", "actions": [], "reason": "string"}
}
"""

SYSTEM_PROMPT = f"""你是内部威胁分析专家。你必须严格按照以下格式输出，先输出<think>标签内的推理过程，再输出<output>标签内的JSON结果。
【强制规则】
1. 必须输出完整的<think>和<output>标签。
2. 所有结论必须引用输入数据中实际存在的evidence_id，如无实际证据则明确标注"无直接证据"。
3. 量化数据必须从输入的metrics字段提取，禁止编造数值。如无metrics数据，必须设为null或0。
4. 必须生成下方【输出JSON结构】定义的所有字段。如果证据不足，也要输出完整JSON，但将confidence设为0.3以下，threat_type设为"无威胁"。
5. 在输出前，你必须确认6个推理步骤全部完成，如果存在某步骤未完成，请返回步骤1重新推理。

【输出JSON结构】
{OUTPUT_JSON_SCHEMA}
"""

COT_OUTPUT_FORMAT = """
**输出结构要求：**
先输出<think>推理过程，再输出 <output> JSON。

<think>
步骤1: 正常模式推断 - 结合用户属性与基线，描述该用户正常行为预期[必须注明岗位、登录时段、日均操作量]。如无法推断正常模式，必须注明基线数据不足。
步骤2: 异常点定位 - 按严重程度排序，必须引用evidence_id
步骤3: 量化指标验证 - 提取metrics中的实际数值与基线/阈值对比，明确与阈值的差距
步骤4: 模块贡献评估 - 分析语义和统计模块的证据是否互相印证？双模块同时告警=强威胁信号；仅统计告警=可信度较高不应忽视；仅语义告警=需结合证据分数与典型威胁模式匹配度判断。
步骤5: 威胁判定 - 结合异常密度、峰值、类型分布给出is_threat，threat_score和threat_type。重要：如果证据不足以判定威胁，必须判定is_threat=false，confidence<=0.3，threat_type="无威胁"，threat_score=0，输出符合指定的Schema结构的JSON。
步骤6: 处置建议 - 根据威胁等级给出分级建议
</think>

<output>
（在此处开始直接输出JSON对象，不要包含任何解释、标记或占位符，仅输出合法的JSON）
</output>
"""

THREAT_PATTERNS_REFERENCE = """
=== 内部威胁典型模式（仅供参考，**禁止复述**）=== 
【数据泄露】
某用户此前并无使用可移动驱动器或非工作时间工作的习惯，开始在下班后登录系统，使用可移动驱动器，并向wikileaks.org网站上传数据。此后不久便离开了该组织。
→ 特征提取：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测异常组合：non_work_hour_login + sudden_usb_usage/usb_usage_spike + web_anomaly
【知识产权盗取】
某用户开始浏览求职网站并向竞争对手求职。在离开公司前，该用户使用U盘（使用频率显著高于其过往活动水平）窃取数据。
→ 特征提取：求职信号 + U盘使用暴增 + 文件操作异常 + 离职时序
→ 可观测异常组合：web_anomaly + usb_usage_spike + file_anomaly
【系统破坏】
某系统管理员心生不满。他下载了一个键盘记录器，并使用U盘将其转移至其主管的电脑。次日，他利用收集到的键盘记录数据登录其主管的账户，并发送了一封引发组织内部恐慌的大规模群发邮件。随后，他立即离开了该组织。
→ 特征提取：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 群发恶意邮件 + 即时离职
→ 可观测异常组合：file_anomaly + usb_usage_spike/sudden_usb_usage + login_other_pc + email_anomaly
"""

THREAT_PATTERNS_REFERENCE_v2 = """
=== 内部威胁典型模式（仅供参考，**禁止复述**）=== 
【数据泄露】
某用户此前并无使用可移动驱动器或非工作时间工作的习惯，开始在下班后登录系统，使用可移动驱动器，并向wikileaks.org网站上传数据。此后不久便离开了该组织。
→ 特征提取：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测异常组合：non_work_hour_login + usb_usage_spike + web_anomaly
【知识产权盗取】
某用户开始浏览求职网站并向竞争对手求职。在离开公司前，该用户使用U盘（使用频率显著高于其过往活动水平）窃取数据。
→ 特征提取：浏览求职网站 + U盘使用激增 + 文件操作异常 + 离职时序
→ 可观测异常组合：web_anomaly + usb_usage_spike + file_anomaly
【系统破坏】
某系统管理员心生不满。他下载了一个键盘记录器，并使用U盘将其转移至其主管的电脑。次日，他利用收集到的键盘记录数据登录其主管的账户，并发送了一封引发组织内部恐慌的大规模群发邮件。随后，他立即离开了该组织。
→ 特征提取：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 邮件语义异常 + 离职时序
→ 可观测异常组合：file_anomaly + usb_usage_spike + login_other_pc + email_anomaly
"""

THREAT_PATTERNS_REFERENCE_v3 = """
=== 内部威胁典型模式（仅供参考，**禁止复述**）=== 
【数据泄露】
→ 特征：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测异常组合：non_work_hour_login + usb_usage_spike + web_anomaly
【知识产权盗取】
→ 特征：浏览求职网站 + U盘使用激增 + 文件操作异常 + 离职时序
→ 可观测异常组合：web_anomaly + usb_usage_spike + file_anomaly
【系统破坏】
→ 特征：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 邮件语义异常 + 离职时序
→ 可观测异常组合：file_anomaly + usb_usage_spike + login_other_pc + email_anomaly
"""

THREAT_PATTERNS_REFERENCE_v4 = """
=== 内部威胁典型模式（仅供参考，**禁止在输出中复述**）===

【数据泄露】
某用户此前并无使用可移动驱动器或非工作时间工作的习惯，开始在下班后登录系统，使用可移动驱动器，并向wikileaks.org网站上传数据。此后不久便离开了该组织。
→ 特征提取：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测统计异常（匹配 evidence.categories 字段）：
  - abnormal_login_time（登录时间异常）
  - usb_usage_spike（U盘使用激增）
  - email_count_anomaly（邮件数量异常）
  - post_termination_login / post_termination_usb / post_termination_email（离职后活动）
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - email_anomaly（邮件异常）
  - web_anomaly（网页异常）

【知识产权盗取】
某用户开始浏览求职网站并向竞争对手求职。在离开公司前，该用户使用U盘（使用频率显著高于其过往活动水平）窃取数据。
→ 特征提取：浏览求职网站 + U盘使用激增 + 文件操作异常 + 离职时序
→ 可观测统计异常（匹配 evidence.categories 字段）：
  - http_count_anomaly（网页浏览激增）
  - usb_usage_spike（U盘使用激增）
  - file_count_anomaly（文件操作激增）
  - post_termination_usb / post_termination_file（离职后U盘/文件操作）
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - web_anomaly（网页异常）
  - file_anomaly（文件异常）

【系统破坏】
某系统管理员心生不满。他下载了一个键盘记录器，并使用U盘将其转移至其主管的电脑。次日，他利用收集到的键盘记录数据登录其主管的账户，并发送了一封引发组织内部恐慌的大规模群发邮件。随后，他立即离开了该组织。
→ 特征提取：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 邮件语义异常 + 离职时序
→ 可观测统计异常（匹配 evidence.categories 字段）：
  - abnormal_login_time（登录时间异常）
  - login_count_anomaly（登录次数异常）
  - usb_usage_spike（U盘使用激增）
  - email_count_anomaly（邮件数量异常）
  - post_termination_*（离职后各类活动）
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - file_anomaly（文件异常）
  - email_anomaly（邮件异常）
"""

THREAT_PATTERNS_REFERENCE_v5 = """
=== 内部威胁典型模式（仅供参考，**禁止在输出中复述**）===

【数据泄露】
某用户此前并无使用可移动驱动器或非工作时间工作的习惯，开始在下班后登录系统，使用可移动驱动器，并向wikileaks.org网站上传数据。此后不久便离开了该组织。
→ 特征提取：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测到的统计异常（匹配 evidences.categories 字段）：
  - abnormal_login_time
  - usb_usage_spike
  - 离职后各类活动post_termination_*
→ 可观测到的语义异常（匹配 evidences.categories 字段）：
  - web_anomaly

【知识产权盗取】
某用户开始浏览求职网站并向竞争对手求职。在离开公司前，该用户使用U盘（使用频率显著高于其过往活动水平）窃取数据。
→ 特征提取：浏览求职网站 + U盘使用激增 + 文件操作异常 + 离职时序
→ 可观测到的统计异常（匹配 evidences.categories 字段）：
  - http_count_anomaly
  - usb_usage_spike
  - 离职后各类活动post_termination_*
→ 可观测到的语义异常（匹配 evidence.categories 字段）：
  - web_anomaly
  - file_anomaly

【系统破坏】
某系统管理员心生不满。他下载了一个键盘记录器，并使用U盘将其转移至其主管的电脑。次日，他利用收集到的键盘记录数据登录其主管的账户，并发送了一封引发组织内部恐慌的大规模群发邮件。随后，他立即离开了该组织。
→ 特征提取：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 邮件语义异常 + 离职时序
→ 可观测到的统计异常（匹配 evidence.categories 字段）：
  - login_other_pc
  - usb_usage_spike
  - email_count_anomaly
  - 离职后各类活动post_termination_*
→ 可观测到的语义异常（匹配 evidence.categories 字段）：
  - file_anomaly
  - email_anomaly
"""
THREAT_PATTERNS_REFERENCE_v6 = """
=== 内部威胁典型模式（仅供参考，**禁止在输出中复述**）===

【数据泄露】
某用户此前并无使用可移动驱动器或非工作时间工作的习惯，开始在下班后登录系统，使用可移动驱动器，并向wikileaks.org网站上传数据。此后不久便离开了该组织。
→ 特征提取：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测统计异常（匹配 evidences.categories 字段）：
  - abnormal_login_time
  - non_work_hour_login
  - usb_usage_spike
  - non_work_hour_usb
  - leak_site_visit_spike
  - 离职后各类活动post_termination_*
→ 可观测语义异常（匹配 evidences.categories 字段）：
  - web_anomaly

【知识产权盗取】
某用户开始浏览求职网站并向竞争对手求职。在离开公司前，该用户使用U盘（使用频率显著高于其过往活动水平）窃取数据。
→ 特征提取：浏览求职网站 + U盘使用激增 + 文件操作异常 + 离职时序
→ 可观测统计异常（匹配 evidences.categories 字段）：
  - job_site_visit_spike
  - http_count_anomaly
  - usb_usage_spike
  - non_work_hour_usb
  - non_work_hour_file
  - 离职后各类活动post_termination_*
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - web_anomaly
  - file_anomaly

【系统破坏】
某系统管理员心生不满。他下载了一个键盘记录器，并使用U盘将其转移至其主管的电脑。次日，他利用收集到的键盘记录数据登录其主管的账户，并发送了一封引发组织内部恐慌的大规模群发邮件。随后，他立即离开了该组织。
→ 特征提取：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 邮件语义异常 + 离职时序
→ 可观测统计异常（匹配 evidence.categories 字段）：
  - non_work_hour_login
  - login_other_pc
  - usb_usage_spike
  - non_work_hour_usb
  - email_count_anomaly
  - non_work_hour_email
  - 离职后各类活动post_termination_*
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - file_anomaly
  - email_anomaly
"""

THREAT_PATTERNS_REFERENCE_v7 = """
=== 内部威胁典型模式（仅供参考，**禁止在输出中复述**）===

【数据泄露】
某用户此前并无使用可移动驱动器或非工作时间工作的习惯，开始在下班后登录系统，使用可移动驱动器，并向wikileaks.org网站上传数据。此后不久便离开了该组织。
→ 特征提取：基线突变 + 非工作时段登录 + U盘使用 + 外传数据到可疑网站 + 离职关联
→ 可观测统计异常（匹配 evidences.categories 字段）：
  - abnormal_login_time
  - non_work_hour_login
  - usb_usage_spike
  - non_work_hour_usb
  - leak_site_visit_spike
  - 离职后各类活动post_termination_*
→ 可观测语义异常（匹配 evidences.categories 字段）：
  - web_anomaly

【知识产权盗取】
某用户开始浏览求职网站并向竞争对手求职。在离开公司前，该用户使用U盘（使用频率显著高于其过往活动水平）窃取数据。
→ 特征提取：浏览求职网站 + U盘使用激增 + 文件操作异常 + 离职时序
→ 可观测统计异常（匹配 evidences.categories 字段）：
  - job_site_visit_spike
  - http_count_anomaly
  - usb_usage_spike
  - file_count_anomaly
  - non_work_hour_usb
  - non_work_hour_file
  - 离职后各类活动post_termination_*
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - web_anomaly
  - file_anomaly

【系统破坏】
某系统管理员心生不满。他下载了一个键盘记录器，并使用U盘将其转移至其主管的电脑。次日，他利用收集到的键盘记录数据登录其主管的账户，并发送了一封引发组织内部恐慌的大规模群发邮件。随后，他立即离开了该组织。
→ 特征提取：高权限角色 + 恶意工具下载 + U盘文件转移 + 冒充上级 + 邮件语义异常 + 离职时序
→ 可观测统计异常（匹配 evidence.categories 字段）：
  - non_work_hour_login
  - login_other_pc
  - usb_usage_spike
  - non_work_hour_usb
  - email_count_anomaly
  - non_work_hour_email
  - 离职后各类活动post_termination_*
→ 可观测语义异常（匹配 evidence.categories 字段）：
  - file_anomaly
  - email_anomaly
"""

RISK_THRESHOLDS_REFERENCE = """
=== 风险判定阈值参考（仅供参考，**禁止复述**）=== 
- 语义异常分数: 
  - <0.5: 正常波动
  - 0.5-0.8: 需关注
  - ≥0.8: 强烈异常信号
- 统计Z-score: 
  - <2.0: 正常范围
  - 2.0-3.0: 偏离基线
  - ≥3.0: 明显偏离
- 统计置信度: 
  - <0.5: 显著性不足
  - 0.5-0.7: 有一定统计意义
  - ≥0.7: 统计意义较强
"""

def build_user_prompt(
        multimodal_evidence: Dict,
        user_profile: Union[Dict, str],
        include_threshold_ref: bool = False,
        include_pattern_ref: bool = True,
        include_module_reliability_ref: bool = False
) -> str:

    import json

    prompt_parts = []

    prompt_parts.append("请分析以下用户的多模态异常证据，按思维链步骤推理，并输出指定格式。")
    prompt_parts.append("")

    prompt_parts.append("=== 用户属性 ===")
    if isinstance(user_profile, str):
        prompt_parts.append(user_profile)
    else:
        prompt_parts.append(json.dumps(user_profile, ensure_ascii=False, indent=2))
    prompt_parts.append("")

    normal_pattern = multimodal_evidence.get('evidence', {}).get('normal_pattern')
    if normal_pattern:
        prompt_parts.append("=== 用户正常行为基线（统计模块） ===")
        prompt_parts.append(f"- 典型登录时段: {normal_pattern.get('typical_login_hours', '未知')}")
        prompt_parts.append(f"- 日均U盘操作: {normal_pattern.get('avg_usb_per_day', 0)} 次")
        prompt_parts.append(f"- 日均邮件: {normal_pattern.get('avg_email_per_day', 0)} 封")
        prompt_parts.append(f"- 日均文件操作: {normal_pattern.get('avg_file_per_day', 0)} 次")
        prompt_parts.append("")

    prompt_parts.append("")
    prompt_parts.append("**在查看异常证据前，请先回答：**")
    prompt_parts.append("1. 根据用户岗位和正常基线，该用户的典型工作日应该是什么样的？")
    prompt_parts.append("2. 该用户是否有合理的理由进行以下操作：")
    prompt_parts.append("   - 在非典型时段活动（如深夜、周末）？")
    prompt_parts.append("   - 使用U盘等可移动存储设备？")
    prompt_parts.append("   - 发送大量邮件或访问非常规网站？")
    prompt_parts.append("3. 异常事件发生日期与用户入职/离职时间的关系？是否处于离职前敏感期或已离职后？")
    prompt_parts.append("")

    prompt_parts.append("=== 多模态异常证据 ===")
    prompt_parts.append("【证据开始】" + json.dumps(multimodal_evidence, ensure_ascii=False, indent=2) + "【证据结束】")
    prompt_parts.append("")

    if include_threshold_ref:
        prompt_parts.append(RISK_THRESHOLDS_REFERENCE)
        prompt_parts.append("")

    if include_pattern_ref:
        prompt_parts.append("【背景知识】")
        prompt_parts.append(THREAT_PATTERNS_REFERENCE_v7)
        prompt_parts.append("【背景知识结束】")
        prompt_parts.append("")

    prompt_parts.append(COT_OUTPUT_FORMAT)
    prompt_parts.append("")

    prompt_parts.append(
        )

    return "\n".join(prompt_parts)

def get_output_schema_dict() -> Dict:

    return {
        "type": "object",
        "required": ["user_id", "threat_conclusion", "reasoning_chain", "core_anomalies",
                     "module_contributions", "natural_language_explanation", "security_recommendations"],
        "properties": {
            "user_id": {"type": "string"},
            "threat_conclusion": {
                "type": "object",
                "required": ["is_threat", "threat_type", "threat_score", "confidence"],
                "properties": {
                    "is_threat": {"type": "boolean"},
                    "threat_type": {"type": "string", "enum": ["数据泄露", "知识产权盗取", "系统破坏", "无威胁"]},
                    "threat_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "reasoning_chain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step", "step_name", "analysis", "evidence_refs"],
                    "properties": {
                        "step": {"type": "integer"},
                        "step_name": {"type": "string"},
                        "analysis": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                        "metrics_used": {"type": "object"}
                    }
                }
            },
            "core_anomalies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["evidence_id", "anomaly_description", "severity", "quantitative_validation"],
                    "properties": {
                        "evidence_id": {"type": "string"},
                        "anomaly_description": {"type": "string"},
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "quantitative_validation": {"type": "object"}
                    }
                }
            },
            "module_contributions": {
                "type": "object",
                "required": ["semantic_module", "statistical_module"],
                "properties": {
                    "semantic_module": {
                        "type": "object",
                        "required": ["weight", "triggered", "key_findings", "contribution_reason"]
                    },
                    "statistical_module": {
                        "type": "object",
                        "required": ["weight", "triggered", "key_findings", "contribution_reason"]
                    }
                }
            },
            "natural_language_explanation": {
                "type": "object",
                "required": ["normal_pattern_summary", "anomaly_comparison", "typical_pattern_comparison"]
            },
            "security_recommendations": {
                "type": "object",
                "required": ["level", "actions", "reason"],
                "properties": {
                    "level": {"type": "string", "enum": ["监控", "审计", "告警", "阻断"]},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"}
                }
            }
        }
    }

if __name__ == "__main__":
    import json

    print("=" * 80)
    print("Prompt模板模块 - 提示词示例")
    print("=" * 80)

    mock_multimodal_evidence = {
        "user_id": "BSS0369",
        "time_range": {"first": "2010-01-03", "last": "2010-07-13"},
        "summary": {
            "total_count": 2,
            "semantic_count": 1,
            "statistical_count": 1,
            "monitoring_days": 2,
            "semantic_stats": {
                "total_anomalies": 5,
                "avg_score": 0.7164,
                "max_score": 0.7344,
                "risk_distribution": {"high": 0, "medium": 5, "low": 0}
            },
            "statistical_stats": {
                "total_anomaly_days": 1,
                "max_confidence": 0.6069,
                "avg_confidence": 0.6069
            }
        },
        "daily_evidences": [
            {
                "date": "2010-01-03",
                "evidence_count": 1,
                "evidences": [
                    {
                        "evidence_id": "sem_BSS0369_2010-01-03_file_0",
                        "date": "2010-01-03",
                        "source": "semantic",
                        "event_type": "file",
                        "categories": ["file_anomaly"],
                        "metrics": {"anomaly_score": 0.703125},
                        "details": {"pc": "PC-8884", "filename": "8UXZMQOU.pdf"}
                    }
                ]
            }
        ]
    }

    mock_user_profile = {
        "user_id": "BSS0369",
        "role": "研发工程师",
        "department": "技术研发中心",
        "employment_date": "2009-03-15",
        "risk_history": "low",
        "access_level": "standard"
    }

    print("\n【1. 系统提示词】")
    print("-" * 80)
    print(SYSTEM_PROMPT)
    print("-" * 80)

    print("\n【3. 输出JSON Schema约束（LLM提示词部分）】")
    print("-" * 80)
    print(OUTPUT_JSON_SCHEMA[:1000] + "...")  
    print("-" * 80)

    print("\n【5. 完整用户提示词】")
    print("-" * 80)
    full_prompt = build_user_prompt(mock_multimodal_evidence, mock_user_profile)

    print(full_prompt[:2000] + "...")
    print("-" * 80)
    print(f"\n提示词总长度: {len(full_prompt)} 字符")

    print("\n【7. JSON Schema验证字典（程序验证用）】")
    print("-" * 80)
    schema = get_output_schema_dict()
    print(json.dumps(schema, indent=2, ensure_ascii=False)[:1500] + "...")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("提示词模板测试完成")
    print("=" * 80)
