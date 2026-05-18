# explanation/report_generator.py
from typing import Dict, List, Optional
from datetime import datetime

try:
    from .llm_reasoning_engine import ReasoningResult
except ImportError:
    from llm_reasoning_engine import ReasoningResult

class ReportGenerator:

    def __init__(self, company_name: str = "公司", report_title: str = "内部威胁分析报告"):
        self.company_name = company_name
        self.report_title = report_title

    def generate_markdown(
            self,
            result: ReasoningResult,
            multimodal_evidence: Dict,
            user_profile: str
    ) -> str:

        raw = result.raw_output
        threat = raw.get('threat_conclusion', {})
        anomalies = raw.get('core_anomalies', [])
        modules = raw.get('module_contributions', {})
        explanation = raw.get('natural_language_explanation', {})
        recommendations = raw.get('security_recommendations', {})

        lines = []

        # 标题
        lines.append(f"# {self.report_title}")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**分析对象**: {result.user_id}")
        lines.append(
            f"**验证状态**: {self._status_icon(result.validation_status.value)} {result.validation_status.value}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 1. 用户概况
        lines.append("## 1. 用户概况")
        lines.append("")
        lines.append(user_profile)
        lines.append("")

        # 2. 威胁判定
        lines.append("## 2. 威胁判定")
        lines.append("")

        threat_level = self._get_threat_level(threat.get('threat_score', 0))
        threat_icon = self._threat_icon(threat.get('is_threat', False), threat.get('threat_score', 0))

        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 威胁判定 | {threat_icon} {'存在威胁' if threat.get('is_threat') else '无威胁'} |")
        lines.append(f"| 威胁类型 | {threat.get('threat_type', '未知')} |")
        lines.append(
            f"| 威胁分数 | {self._progress_bar(threat.get('threat_score', 0))} {threat.get('threat_score', 0):.2f} ({threat_level}) |")
        lines.append(
            f"| 置信度 | {self._progress_bar(threat.get('confidence', 0))} {threat.get('confidence', 0):.2f} |")
        lines.append("")

        # 3. 核心异常点
        lines.append("## 3. 核心异常点")
        lines.append("")

        for i, anomaly in enumerate(anomalies, 1):
            severity = anomaly.get('severity', 'medium')
            severity_icon = self._severity_icon(severity)
            lines.append(f"### 3.{i} {severity_icon} {anomaly.get('anomaly_description', '未知异常')}")
            lines.append("")
            lines.append(f"- **证据ID**: `{anomaly.get('evidence_id', 'N/A')}`")
            lines.append(f"- **严重程度**: {severity}")

            qv = anomaly.get('quantitative_validation', {})
            if qv:
                lines.append("- **量化验证**:")
                for key, value in qv.items():
                    lines.append(f"  - {key}: {value}")
            lines.append("")

        # 4. 量化指标验证详情
        lines.append("## 4. 量化指标验证")
        lines.append("")
        lines.append(explanation.get('anomaly_comparison', '无详细对比数据'))
        lines.append("")

        # 5. 正常模式与典型模式对比
        lines.append("## 5. 行为模式分析")
        lines.append("")
        lines.append("### 正常行为模式")
        lines.append(explanation.get('normal_pattern_summary', '无正常模式数据'))
        lines.append("")
        lines.append("### 与典型威胁模式对比")
        lines.append(explanation.get('typical_pattern_comparison', '无对比数据'))
        lines.append("")

        # 6. 检测模块贡献度
        lines.append("## 6. 检测模块贡献度")
        lines.append("")
        lines.append("| 模块 | 状态 | 贡献权重 | 关键发现 |")
        lines.append("|------|------|----------|----------|")

        semantic = modules.get('semantic_module', {})
        stat = modules.get('statistical_module', {})

        sem_status = "✅ 已触发" if semantic.get('triggered') else "❌ 未触发"
        stat_status = "✅ 已触发" if stat.get('triggered') else "❌ 未触发"

        sem_findings = "、".join(semantic.get('key_findings', [])) or "无"
        stat_findings = "、".join(stat.get('key_findings', [])) or "无"

        lines.append(f"| 语义异常检测 | {sem_status} | {semantic.get('weight', 0):.0%} | {sem_findings} |")
        lines.append(f"| 统计异常检测 | {stat_status} | {stat.get('weight', 0):.0%} | {stat_findings} |")
        lines.append("")

        # 模块贡献说明
        lines.append(f"**贡献分析**: 语义模块贡献 {semantic.get('contribution_reason', '无')}；")
        lines.append(f"统计模块贡献 {stat.get('contribution_reason', '无')}")
        lines.append("")

        # 7. 安全处置建议
        lines.append("## 7. 安全处置建议")
        lines.append("")

        rec_level = recommendations.get('level', '监控')
        level_icon = self._recommendation_icon(rec_level)
        lines.append(f"**建议等级**: {level_icon} {rec_level}")
        lines.append("")
        lines.append("**具体措施**:")
        for action in recommendations.get('actions', []):
            lines.append(f"- {action}")
        lines.append("")
        lines.append(f"**建议理由**: {recommendations.get('reason', '无')}")
        lines.append("")

        # 8. 推理链摘要（可折叠）
        lines.append("---")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary><b>📋 推理链详情（点击展开）</b></summary>")
        lines.append("")
        for step in raw.get('reasoning_chain', []):
            lines.append(f"**步骤{step.get('step')}: {step.get('step_name')}**")
            lines.append(f"{step.get('analysis', '')}")
            refs = step.get('evidence_refs', [])
            if refs:
                lines.append(f"证据引用: {', '.join([f'`{r}`' for r in refs])}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

        # 9. 附录：证据统计
        lines.append("## 附录：证据统计")
        lines.append("")
        summary = multimodal_evidence.get('summary', {})
        lines.append(f"- 证据总数: {summary.get('total_count', 0)} 条")
        lines.append(f"- 语义证据: {summary.get('semantic_count', 0)} 条")
        lines.append(f"- 统计证据: {summary.get('statistical_count', 0)} 条")
        lines.append(f"- 监控天数: {summary.get('monitoring_days', 0)} 天")
        lines.append(
            f"- 时间范围: {multimodal_evidence.get('time_range', {}).get('first', 'N/A')} ~ {multimodal_evidence.get('time_range', {}).get('last', 'N/A')}")
        lines.append("")

        # 验证警告
        if result.validation_warnings:
            lines.append("**验证警告**:")
            for w in result.validation_warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")

        return "\n".join(lines)

    def generate_html(
            self,
            result: ReasoningResult,
            multimodal_evidence: Dict,
            user_profile: str
    ) -> str:
        import html

        raw = result.raw_output
        threat = raw.get('threat_conclusion', {})
        anomalies = raw.get('core_anomalies', [])
        explanation = raw.get('natural_language_explanation', {})
        recommendations = raw.get('security_recommendations', {})
        modules = raw.get('module_contributions', {})

        threat_score = threat.get('threat_score', 0)
        score_color = self._score_color(threat_score)

        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.report_title} - {html.escape(result.user_id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        h3 {{ color: #555; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
        .threat-card {{ background: #fafafa; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .score-bar {{ background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden; width: 200px; display: inline-block; }}
        .score-fill {{ background: {score_color}; height: 100%; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
        .badge-high {{ background: #ffebee; color: #c62828; }}
        .badge-medium {{ background: #fff3e0; color: #ef6c00; }}
        .badge-low {{ background: #e8f5e9; color: #2e7d32; }}
        .evidence-id {{ font-family: monospace; background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .anomaly-item {{ background: #fff8e1; border-left: 4px solid #ff9800; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        .warning {{ color: #e65100; }}
        details {{ margin: 20px 0; }}
        summary {{ cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.report_title}</h1>
        <div class="meta">
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            分析对象: {html.escape(result.user_id)}<br>
            验证状态: {result.validation_status.value}</p>
        </div>

        <h2>1. 用户概况</h2>
        <p>{html.escape(user_profile)}</p>

        <h2>2. 威胁判定</h2>
        <div class="threat-card">
            <p><strong>判定结果</strong>: {'⚠️ 存在威胁' if threat.get('is_threat') else '✅ 无威胁'}</p>
            <p><strong>威胁类型</strong>: {html.escape(threat.get('threat_type', '未知'))}</p>
            <p><strong>威胁分数</strong>: 
                <span class="score-bar"><span class="score-fill" style="width:{threat_score * 100}%"></span></span>
                {threat_score:.2f} ({self._get_threat_level(threat_score)})
            </p>
            <p><strong>置信度</strong>: {threat.get('confidence', 0):.2f}</p>
        </div>
"""

        # 核心异常点
        html_content += "<h2>3. 核心异常点</h2>\n"
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'medium')
            badge_class = f"badge-{severity}"
            html_content += f"""
        <div class="anomaly-item">
            <h3><span class="badge {badge_class}">{severity.upper()}</span> {html.escape(anomaly.get('anomaly_description', '未知'))}</h3>
            <p><strong>证据ID</strong>: <span class="evidence-id">{html.escape(anomaly.get('evidence_id', 'N/A'))}</span></p>
"""
            qv = anomaly.get('quantitative_validation', {})
            if qv:
                html_content += "<p><strong>量化验证</strong>:</p><ul>\n"
                for k, v in qv.items():
                    html_content += f"<li>{html.escape(str(k))}: {html.escape(str(v))}</li>\n"
                html_content += "</ul>\n"
            html_content += "</div>\n"

        # 行为模式分析
        html_content += f"""
        <h2>4. 行为模式分析</h2>
        <h3>正常行为模式</h3>
        <p>{html.escape(explanation.get('normal_pattern_summary', '无'))}</p>
        <h3>异常对比</h3>
        <p>{html.escape(explanation.get('anomaly_comparison', '无'))}</p>
        <h3>典型威胁模式对比</h3>
        <p>{html.escape(explanation.get('typical_pattern_comparison', '无'))}</p>

        <h2>5. 模块贡献度</h2>
        <table>
            <tr><th>模块</th><th>状态</th><th>权重</th><th>关键发现</th></tr>
"""
        semantic = modules.get('semantic_module', {})
        stat = modules.get('statistical_module', {})
        html_content += f"""
            <tr><td>语义异常检测</td><td>{'✅' if semantic.get('triggered') else '❌'}</td><td>{semantic.get('weight', 0):.0%}</td><td>{html.escape('、'.join(semantic.get('key_findings', [])) or '无')}</td></tr>
            <tr><td>统计异常检测</td><td>{'✅' if stat.get('triggered') else '❌'}</td><td>{stat.get('weight', 0):.0%}</td><td>{html.escape('、'.join(stat.get('key_findings', [])) or '无')}</td></tr>
        </table>

        <h2>6. 安全处置建议</h2>
        <p><strong>建议等级</strong>: {html.escape(recommendations.get('level', '监控'))}</p>
        <p><strong>具体措施</strong>:</p>
        <ul>
"""
        for action in recommendations.get('actions', []):
            html_content += f"<li>{html.escape(action)}</li>\n"
        html_content += f"""
        </ul>
        <p><strong>理由</strong>: {html.escape(recommendations.get('reason', '无'))}</p>
"""

        # 推理链
        html_content += """
        <details>
            <summary><b>📋 推理链详情</b></summary>
"""
        for step in raw.get('reasoning_chain', []):
            html_content += f"""
            <p><strong>步骤{step.get('step')}: {html.escape(step.get('step_name', ''))}</strong><br>
            {html.escape(step.get('analysis', ''))}</p>
"""
        html_content += """
        </details>
    </div>
</body>
</html>"""

        return html_content

    def generate_text(self, result: ReasoningResult, user_profile: str) -> str:
        """
        生成纯文本格式报告
        """
        raw = result.raw_output
        threat = raw.get('threat_conclusion', {})
        anomalies = raw.get('core_anomalies', [])
        recommendations = raw.get('security_recommendations', {})

        lines = []
        lines.append("=" * 60)
        lines.append(self.report_title)
        lines.append("=" * 60)
        lines.append(f"用户: {result.user_id}")
        lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("【威胁判定】")
        lines.append(f"  判定: {'存在威胁' if threat.get('is_threat') else '无威胁'}")
        lines.append(f"  类型: {threat.get('threat_type', '未知')}")
        lines.append(f"  分数: {threat.get('threat_score', 0):.2f}")
        lines.append("")
        lines.append("【核心异常点】")
        for anomaly in anomalies:
            lines.append(f"  - {anomaly.get('anomaly_description')}")
            lines.append(f"    证据ID: {anomaly.get('evidence_id')}")
        lines.append("")
        lines.append("【处置建议】")
        lines.append(f"  等级: {recommendations.get('level', '监控')}")
        for action in recommendations.get('actions', []):
            lines.append(f"  - {action}")

        return "\n".join(lines)

    # ========== 辅助方法 ==========
    def _status_icon(self, status: str) -> str:
        icons = {'passed': '✅', 'partial': '⚠️', 'failed': '❌'}
        return icons.get(status, '❓')

    def _threat_icon(self, is_threat: bool, score: float) -> str:
        if not is_threat:
            return "✅"
        if score >= 0.7:
            return "🔴"
        elif score >= 0.4:
            return "🟡"
        return "🟢"

    def _severity_icon(self, severity: str) -> str:
        icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        return icons.get(severity, '⚪')

    def _recommendation_icon(self, level: str) -> str:
        icons = {'阻断': '🚫', '告警': '🚨', '审计': '📋', '监控': '👁️'}
        return icons.get(level, '📌')

    def _get_threat_level(self, score: float) -> str:
        if score >= 0.7:
            return "高危"
        elif score >= 0.4:
            return "中危"
        return "低危"

    def _progress_bar(self, value: float, width: int = 10) -> str:
        filled = int(value * width)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"`{bar}`"

    def _score_color(self, score: float) -> str:
        if score >= 0.7:
            return "#c62828"
        elif score >= 0.4:
            return "#ef6c00"
        return "#2e7d32"

def generate_report(
        result: ReasoningResult,
        multimodal_evidence: Dict,
        user_profile: str,
        format: str = "markdown"
) -> str:
    """
    便捷函数：生成指定格式的报告

    参数：
        result: 推理结果
        multimodal_evidence: 原始证据
        user_profile: 用户属性
        format: "markdown", "html", "text"
    """
    generator = ReportGenerator()
    if format == "html":
        return generator.generate_html(result, multimodal_evidence, user_profile)
    elif format == "text":
        return generator.generate_text(result, user_profile)
    else:
        return generator.generate_markdown(result, multimodal_evidence, user_profile)


if __name__ == "__main__":
    import sys
    import os

    # 添加项目根目录到路径
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    sys.path.insert(0, _project_root)
    from explanation.llm_reasoning_engine import ReasoningResult, ValidationStatus

    print("=" * 80)
    print("报告生成器 - 测试")
    print("=" * 80)

    # 创建Mock结果
    mock_result = ReasoningResult(
        user_id="BSS0369",
        is_threat=True,
        threat_type="数据窃取",
        threat_score=0.68,
        confidence=0.85,
        raw_output={
            "user_id": "BSS0369",
            "threat_conclusion": {
                "is_threat": True,
                "threat_type": "数据窃取",
                "threat_score": 0.68,
                "confidence": 0.85
            },
            "reasoning_chain": [
                {"step": 1, "step_name": "正常模式", "analysis": "用户正常文件操作约10次/天", "evidence_refs": []}
            ],
            "core_anomalies": [
                {
                    "evidence_id": "stat_BSS0369_2010-07-13_0",
                    "anomaly_description": "非工作时段文件操作暴增",
                    "severity": "high",
                    "quantitative_validation": {"实际值": 28, "基线值": 10.29, "偏离": "172%"}
                }
            ],
            "module_contributions": {
                "semantic_module": {"weight": 0.4, "triggered": True, "key_findings": ["文件内容异常"],
                                    "contribution_reason": "发现可疑内容"},
                "statistical_module": {"weight": 0.6, "triggered": True, "key_findings": ["数量暴增"],
                                       "contribution_reason": "量化偏离显著"}
            },
            "natural_language_explanation": {
                "normal_pattern_summary": "每日约10次文件操作",
                "anomaly_comparison": "异常当天28次，超出172%",
                "typical_pattern_comparison": "符合数据窃取特征"
            },
            "security_recommendations": {
                "level": "审计",
                "actions": ["审计文件操作", "检查U盘日志"],
                "reason": "行为符合数据窃取模式"
            }
        },
        reasoning_log="Mock推理日志",
        validation_status=ValidationStatus.PASSED,
        validation_errors=[],
        validation_warnings=[],
        retry_count=0
    )

    mock_evidence = {
        "time_range": {"first": "2010-01-03", "last": "2010-07-13"},
        "summary": {"total_count": 2, "semantic_count": 1, "statistical_count": 1, "monitoring_days": 2}
    }

    mock_profile = "用户 BSS0369，岗位为研发工程师，在职，入职于2009-03。"

    generator = ReportGenerator()

    print("\n【Markdown报告】")
    print("-" * 80)
    md_report = generator.generate_markdown(mock_result, mock_evidence, mock_profile)
    print(md_report[:1500] + "...")

    print("\n【纯文本报告】")
    print("-" * 80)
    text_report = generator.generate_text(mock_result, mock_profile)
    print(text_report)

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)