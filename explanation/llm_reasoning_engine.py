
import json
import re
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    from .prompt_templates import get_output_schema_dict, build_user_prompt, SYSTEM_PROMPT
except ImportError:
    from prompt_templates import get_output_schema_dict, build_user_prompt, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"  

def _extract_last_valid_output(text: str) -> tuple[str, str]:

    thinking = ""
    output = ""

    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    else:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()

    json_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_block_match:
        output = json_block_match.group(1).strip()
        logger.info("从 ```json 代码块中提取到输出")
    else:
        output_match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
        if output_match:
            output = output_match.group(1).strip()
            logger.info("从 <output> 标签中提取到输出")
        else:

            think_end = text.find('</think>')
            if think_end != -1:
                after_think = text[think_end + 7:]  

                json_start = after_think.find('{')
                if json_start != -1:

                    brace_count = 0
                    in_string = False
                    escape = False
                    for i in range(json_start, len(after_think)):
                        ch = after_think[i]
                        if in_string:
                            if escape:
                                escape = False
                            elif ch == '\\':
                                escape = True
                            elif ch == '"':
                                in_string = False
                        else:
                            if ch == '"':
                                in_string = True
                            elif ch == '{':
                                brace_count += 1
                            elif ch == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    output = after_think[json_start:i + 1].strip()
                                    logger.info("从 </think> 后通过括号配对提取 JSON 对象")
                                    break

                    if not output:

                        json_obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', after_think[json_start:],
                                                   re.DOTALL)
                        if json_obj_match:
                            output = json_obj_match.group(0).strip()
                            logger.warning("从 </think> 后使用正则提取 JSON 对象")

            if not output:

                json_start = text.find('{')
                if json_start != -1:

                    brace_count = 0
                    in_string = False
                    escape = False
                    for i in range(json_start, len(text)):
                        ch = text[i]
                        if in_string:
                            if escape:
                                escape = False
                            elif ch == '\\':
                                escape = True
                            elif ch == '"':
                                in_string = False
                        else:
                            if ch == '"':
                                in_string = True
                            elif ch == '{':
                                brace_count += 1
                            elif ch == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    output = text[json_start:i + 1].strip()
                                    logger.warning("通过括号配对提取第一个完整JSON对象")
                                    break

                    if not output:
                        output = text.strip()
                        logger.warning("未找到任何 JSON 结构，使用原始文本作为输出")
                else:
                    output = text.strip()
                    logger.warning("未找到任何 JSON 结构，使用原始文本作为输出")

    output = re.sub(r'^[:\s]+', '', output)
    return thinking, output

@dataclass
class ReasoningResult:

    user_id: str
    is_threat: bool
    threat_type: str
    threat_score: float
    confidence: float
    raw_output: Dict  
    reasoning_log: str  
    validation_status: ValidationStatus
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    llm_usage: Optional[Dict] = None  

    def to_dict(self) -> Dict:

        return {
            'user_id': self.user_id,
            'is_threat': self.is_threat,
            'threat_type': self.threat_type,
            'threat_score': self.threat_score,
            'confidence': self.confidence,
            'raw_output': self.raw_output,
            'reasoning_log': self.reasoning_log,
            'validation_status': self.validation_status.value,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'retry_count': self.retry_count,
            'llm_usage': self.llm_usage
        }

    def is_valid(self) -> bool:

        return self.validation_status in (ValidationStatus.PASSED, ValidationStatus.PARTIAL)

class LLMReasoningEngine:

    def __init__(
            self,
            model: str = "gpt-4",
            temperature: float = 0.1,
            max_tokens: int = 8192,
            max_retries: int = 2,
            custom_llm_call: Optional[Callable] = None
    ):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.custom_llm_call = custom_llm_call
        self.output_schema = get_output_schema_dict()

        logger.info(f"LLMReasoningEngine初始化完成: model={model}, max_retries={max_retries}")

    def reason(
            self,
            multimodal_evidence: Dict,
            user_profile: Union[Dict, str],
            include_threshold_ref: bool = True
    ) -> ReasoningResult:

        user_prompt = build_user_prompt(
            multimodal_evidence,
            user_profile,
            include_threshold_ref=include_threshold_ref,
            include_pattern_ref=True  
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        last_error = None
        for retry in range(self.max_retries):
            try:
                logger.info(f"开始LLM推理 (尝试 {retry + 1}/{self.max_retries})")

                if retry > 0 and last_error:
                    error_feedback = f"\n\n⚠️ 上次输出验证失败：{str(last_error)}。请确保输出完整的JSON结构，特别是必须包含 threat_conclusion 对象。"
                    messages[-1]["content"] = user_prompt + error_feedback

                if "deepseek" in self.model.lower():
                    messages[0][
                        "content"] += "\n\n🔴 关键要求：你的输出必须包含完整的 threat_conclusion 对象，格式如：{\"is_threat\": bool, \"threat_type\": str, \"threat_score\": float, \"confidence\": float}"

                llm_output, usage = self._call_llm(messages)

                parsed_json, reasoning_log = self._parse_output(llm_output)

                if 'reasoning_chain' in parsed_json:
                    rc = parsed_json['reasoning_chain']
                    if isinstance(rc, dict):

                        rc = [rc[k] for k in sorted(rc.keys(), key=lambda x: int(x) if x.isdigit() else 0)]
                        parsed_json['reasoning_chain'] = rc
                        logger.warning("reasoning_chain 从字典转换为列表")
                    elif not isinstance(rc, list):
                        parsed_json['reasoning_chain'] = []

                if 'core_anomalies' in parsed_json:
                    ca = parsed_json['core_anomalies']
                    if isinstance(ca, dict):
                        ca = list(ca.values())
                        parsed_json['core_anomalies'] = ca
                        logger.warning("core_anomalies 从字典转换为列表")
                    elif not isinstance(ca, list):
                        parsed_json['core_anomalies'] = []

                validation_status, errors, warnings = self._validate_output(
                    parsed_json, multimodal_evidence
                )
                if validation_status == ValidationStatus.FAILED:
                    logger.warning(f"输出验证失败: {errors}")

                    critical_keywords = ['threat_score', 'is_threat', 'confidence', 'threat_type']
                    critical_errors = [e for e in errors if any(kw in e for kw in critical_keywords)]

                    if critical_errors:

                        raise ValueError(f"关键字段验证失败: {critical_errors}")
                    elif 'threat_conclusion' in parsed_json:

                        logger.warning("非关键验证失败，使用已解析的威胁结论")
                        validation_status = ValidationStatus.PARTIAL
                    else:
                        raise ValueError(f"输出验证失败且无法恢复: {errors}")

                tc = parsed_json.get('threat_conclusion', {})

                is_threat = tc.get('is_threat', False)
                if not isinstance(is_threat, bool):
                    is_threat = bool(is_threat)
                    warnings.append(f"is_threat 类型错误，已转换为 {is_threat}")
                    validation_status = ValidationStatus.PARTIAL

                threat_type = tc.get('threat_type', '无威胁')
                if not isinstance(threat_type, str):
                    threat_type = str(threat_type)
                    warnings.append("threat_type 类型错误，已转换为字符串")
                    validation_status = ValidationStatus.PARTIAL

                threat_score = tc.get('threat_score', 0.0)
                try:
                    threat_score = float(threat_score) if threat_score is not None else 0.0
                    if not (0 <= threat_score <= 1):
                        logger.warning(f"threat_score 超出范围 [0,1]: {threat_score}，进行裁剪")
                        threat_score = max(0.0, min(1.0, threat_score))
                        warnings.append(f"threat_score 超出范围，已裁剪为 {threat_score}")
                        validation_status = ValidationStatus.PARTIAL
                except (ValueError, TypeError) as e:
                    logger.error(f"threat_score 转换失败: {threat_score}，使用 0.0")
                    threat_score = 0.0
                    warnings.append(f"threat_score 转换失败: {e}，使用 0.0")
                    validation_status = ValidationStatus.PARTIAL

                confidence = tc.get('confidence', 0.0)
                try:
                    confidence = float(confidence) if confidence is not None else 0.0
                    if not (0 <= confidence <= 1):
                        logger.warning(f"confidence 超出范围 [0,1]: {confidence}，进行裁剪")
                        confidence = max(0.0, min(1.0, confidence))
                        warnings.append(f"confidence 超出范围，已裁剪为 {confidence}")
                        validation_status = ValidationStatus.PARTIAL
                except (ValueError, TypeError) as e:
                    logger.error(f"confidence 转换失败: {confidence}，使用 0.0")
                    confidence = 0.0
                    warnings.append(f"confidence 转换失败: {e}，使用 0.0")
                    validation_status = ValidationStatus.PARTIAL

                if is_threat and threat_score < 0.3:
                    logger.warning(f"威胁判定为 True 但分数过低 ({threat_score})，可能有误")
                    warnings.append(f"威胁判定与分数不一致: is_threat=True, score={threat_score}")
                    validation_status = ValidationStatus.PARTIAL

                if not is_threat and threat_score > 0.5:
                    logger.warning(f"威胁判定为 False 但分数过高 ({threat_score})，可能有误")
                    warnings.append(f"威胁判定与分数不一致: is_threat=False, score={threat_score}")
                    validation_status = ValidationStatus.PARTIAL

                result = ReasoningResult(
                    user_id=parsed_json.get('user_id', multimodal_evidence.get('user_id', 'unknown')),
                    is_threat=is_threat,
                    threat_type=threat_type,
                    threat_score=threat_score,
                    confidence=confidence,
                    raw_output=parsed_json,
                    reasoning_log=reasoning_log,
                    validation_status=validation_status,
                    validation_errors=errors,
                    validation_warnings=warnings,
                    retry_count=retry,
                    llm_usage=usage
                )

                logger.info(f"推理完成: is_threat={result.is_threat}, score={result.threat_score}")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"推理失败 (尝试 {retry + 1}): {e}")

                error_msg = str(e)
                if "回显输入" in error_msg and "证据" in error_msg:
                    last_error = ValueError(
                        "你重复了输入中的证据内容，未进行分析。请基于输入证据直接进行威胁分析，"
                        "输出包含 threat_conclusion 对象的完整推理结果，不要复述输入内容。"
                    )
                elif "回显输入" in error_msg:
                    last_error = ValueError(
                        "你重复了输入中的提示词内容，未进行分析。请基于用户属性和证据直接进行威胁分析，"
                        "输出包含 threat_conclusion 的完整结果。"
                    )
                elif "threat_conclusion" in error_msg:
                    last_error = ValueError(
                        "输出缺少 threat_conclusion 对象。请确保JSON中包含 is_threat、threat_type、threat_score、confidence 四个字段。"
                    )
                elif "无法从响应中提取" in error_msg:
                    last_error = ValueError(
                        "响应中未包含任何JSON输出内容。请在<output>标签按【输出JSON结构】输出JSON对象。"
                    )
                elif "JSON解析失败" in error_msg or "解析结果不是字典" in error_msg:
                    last_error = ValueError(
                        "输出格式错误。请按【输出JSON结构】输出完整JSON，不要添加额外说明文字。"
                    )
                if retry < self.max_retries - 1:
                    self.temperature = min(0.3, self.temperature + 0.15)  

        raise RuntimeError(f"LLM推理失败，已重试{self.max_retries}次。最后错误: {last_error}")

    def _call_llm(self, messages: List[Dict]) -> tuple[str, Optional[Dict]]:

        if self.custom_llm_call:

            import copy
            enhanced_messages = copy.deepcopy(messages)  

            if "deepseek" in self.model.lower():
                for msg in enhanced_messages:
                    if msg["role"] == "system":
                        msg["content"] = msg["content"] + "\n\n⚠️ 你必须输出完整的<think>和<output>标签。不要在标签外添加任何文字。"
                        break  

            output = self.custom_llm_call(
                enhanced_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            logger.info("=" * 80)
            logger.info("【调试】LLM原始输出:")
            logger.info("-" * 80)
            logger.info(output)
            logger.info("=" * 80)
            return output, None
        raise RuntimeError("未配置LLM调用方式，请提供api_key或custom_llm_call")

    def _repair_json(self, json_str: str) -> str:

        import re

        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

        json_str = re.sub(r"(?<!\\)'([^']*)'(?=\s*[,\]}])", r'"\1"', json_str)

        json_str = re.sub(r'}\s*{', '}, {', json_str)
        json_str = re.sub(r']\s*\[', '], [', json_str)
        json_str = re.sub(r'}\s*"', '}, "', json_str)
        json_str = re.sub(r']\s*"', '], "', json_str)

        json_str = re.sub(r'}\s*([a-zA-Z_])', r'}, \1', json_str)
        json_str = re.sub(r']\s*([a-zA-Z_])', r'], \1', json_str)

        json_str = re.sub(r'"\s+([a-zA-Z_])', r'", \1', json_str)

        json_str = re.sub(r'(\d)\s+([a-zA-Z_])', r'\1, \2', json_str)

        json_str = re.sub(r'(true|false)\s+([a-zA-Z_])', r'\1, \2', json_str)

        json_str = re.sub(r'null\s+([a-zA-Z_])', r'null, \1', json_str)

        json_str = json_str.strip()
        start = json_str.find('{')
        if start == -1:
            start = json_str.find('[')
        if start != -1:

            stack = []
            in_string = False
            escape = False
            end = -1
            for i, ch in enumerate(json_str[start:], start):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{' or ch == '[':
                        stack.append(ch)
                    elif ch == '}' or ch == ']':
                        if stack and ((stack[-1] == '{' and ch == '}') or (stack[-1] == '[' and ch == ']')):
                            stack.pop()
                            if not stack:
                                end = i
                                break
            if end != -1:
                json_str = json_str[start:end + 1]

        lines = json_str.split('\n')
        if lines:

            last_complete_idx = len(lines) - 1

            for i in range(len(lines) - 1, -1, -1):
                stripped = lines[i].strip()
                if not stripped:
                    continue

                is_complete = False
                if stripped.endswith(('}', ']', 'true', 'false', 'null')):
                    is_complete = True
                elif stripped and stripped[-1].isdigit():
                    is_complete = True
                elif stripped.endswith('"') and ':' in stripped:
                    is_complete = True
                elif stripped in ('{', '[', '{', '['):
                    is_complete = True

                if is_complete:
                    last_complete_idx = i
                    break

            if last_complete_idx < len(lines) - 1:
                lines = lines[:last_complete_idx + 1]
                json_str = '\n'.join(lines)
                logger.warning(f"检测到截断，保留前 {last_complete_idx + 1} 行")

            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')

            if open_braces > 0 or open_brackets > 0:
                json_str += '\n' + ']' * open_brackets
                json_str += '\n' + '}' * open_braces
                logger.warning(f"补全括号: +{open_braces}个}}, +{open_brackets}个]")

        if json_str.count('{') > json_str.count('}'):
            json_str += '}' * (json_str.count('{') - json_str.count('}'))
        if json_str.count('[') > json_str.count(']'):
            json_str += ']' * (json_str.count('[') - json_str.count(']'))

        return json_str

    def _parse_output(self, llm_output: str) -> tuple[Dict, str]:

        reasoning_log, output_content = _extract_last_valid_output(llm_output)

        if "=== 用户属性 ===" in llm_output and "<thinking>" not in llm_output and "<think>" not in llm_output:
            logger.error("检测到模型回显输入内容，未生成有效推理")
            raise ValueError("回显输入")

        input_signatures = ['"evidence_id"', '"date":', '"source":', '"event_type":', '"categories":', '"metrics":',
                            '"details":']
        signature_hits = sum(1 for sig in input_signatures if sig in output_content)
        if signature_hits >= 3 and '"threat_conclusion"' not in output_content:
            logger.error(f"检测到输出内容疑似回显证据片段（命中{signature_hits}个特征），未生成有效推理")
            raise ValueError("回显证据")

        if output_content.startswith('<think>') or output_content.startswith('<thinking>'):

            json_start = output_content.find('{')
            if json_start != -1:
                output_content = output_content[json_start:]
                logger.warning("检测到输出以think标签开头，已截取JSON部分")

        if "login, connect USB" in output_content:

            json_start = output_content.find('{')
            if json_start != -1:
                output_content = output_content[json_start:]
                logger.warning("检测到输出中包含原始证据，已截取JSON部分")

        logger.info("=" * 80)
        logger.info("【调试】提取的推理过程 (think):")
        logger.info("-" * 80)
        thinking_text = reasoning_log if reasoning_log else "（未提取到think内容）"
        logger.info(thinking_text)
        logger.info(f"  [thinking token 估算: ~{len(thinking_text) // 2}]")
        logger.info("=" * 80)
        logger.info("【调试】提取的输出内容 (output):")
        logger.info("-" * 80)
        output_text = output_content[:1000] + "..." if len(output_content) > 1000 else output_content
        logger.info(output_text)
        logger.info(f"  [output token 估算: ~{len(output_content) // 2}]")
        logger.info(f"  [总生成 token 估算: ~{(len(reasoning_log or '') + len(output_content)) // 2}]")
        logger.info("=" * 80)

        if not output_content:
            raise ValueError("无法从响应中提取任何输出内容")

        json_str = output_content
        import re

        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        json_str = self._repair_json(json_str)
        try:
            parsed = json.loads(json_str)
            logger.info(f"JSON解析成功: is_threat={parsed.get('threat_conclusion', {}).get('is_threat')}")
        except json.JSONDecodeError as e:

            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

            fixed_str = re.sub(r',\s*}', '}', json_str)
            fixed_str = re.sub(r',\s*]', ']', fixed_str)

            fixed_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_str)

            fixed_str = re.sub(r'"evidence_refs":\s*\[(.*?)\]',
                               lambda m: '"evidence_refs": [' + re.sub(r'\{[^}]*\}', '""', m.group(1)) + ']', fixed_str,
                               flags=re.DOTALL)
            try:
                parsed = json.loads(fixed_str)
                logger.warning("通过移除尾部逗号修复JSON成功")
            except:

                start = fixed_str.find('{')
                if start != -1:
                    count = 0
                    for i, char in enumerate(fixed_str[start:], start):
                        if char == '{':
                            count += 1
                        elif char == '}':
                            count -= 1
                            if count == 0:
                                extracted_json = fixed_str[start:i + 1]
                                try:
                                    parsed = json.loads(extracted_json)
                                    logger.warning("从花括号中成功提取JSON")
                                    break
                                except:
                                    continue
                    else:
                        parsed = self._extract_key_fields_from_broken_json(json_str, str(e))
                else:
                    parsed = self._extract_key_fields_from_broken_json(json_str, str(e))

        if not isinstance(parsed, dict):
            logger.error(f"解析结果不是字典: {type(parsed)}")

            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                parsed = parsed[0]
                logger.warning("从列表中提取第一个元素作为字典")
            else:
                raise ValueError(f"解析结果不是字典，类型为: {type(parsed)}")
        return parsed, reasoning_log

    def _extract_key_fields_from_broken_json(self, json_str: str, original_error: str) -> Dict:

        import re
        score_match = re.search(r'"threat_score":\s*([0-9.]+)', json_str)
        is_threat_match = re.search(r'"is_threat":\s*(true|false)', json_str)
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', json_str)
        type_match = re.search(r'"threat_type":\s*"([^"]+)"', json_str)

        if score_match and is_threat_match:
            parsed = {
                'threat_conclusion': {
                    'is_threat': is_threat_match.group(1) == 'true',
                    'threat_score': float(score_match.group(1)),
                    'confidence': float(confidence_match.group(1)) if confidence_match else 0.3,
                    'threat_type': type_match.group(1) if type_match else '无法判定'
                }
            }
            logger.warning(f"JSON损坏，正则提取成功: is_threat={parsed['threat_conclusion']['is_threat']}, score={parsed['threat_conclusion']['threat_score']}")
            return parsed
        else:
            raise ValueError(f"JSON解析失败且无法提取关键字段: {original_error}\n原始内容: {json_str[:500]}...")

    def _validate_output(
            self,
            output: Dict,
            multimodal_evidence: Dict
    ) -> tuple[ValidationStatus, List[str], List[str]]:

        import traceback
        try:
            errors = []
            warnings = []

            if 'user_id' not in output:
                output['user_id'] = multimodal_evidence.get('user_id', 'unknown')
                warnings.append("自动补充 user_id")

            if 'threat_conclusion' not in output:

                if 'is_threat' in output:
                    output['threat_conclusion'] = {
                        'is_threat': output.get('is_threat', False),
                        'threat_type': output.get('threat_type', '无威胁'),
                        'threat_score': output.get('threat_score', 0.0),
                        'confidence': output.get('confidence', 0.0)
                    }
                    warnings.append("从扁平结构构建 threat_conclusion")
                else:
                    errors.append("threat_conclusion完全缺失，且无扁平字段可构建")
                    return ValidationStatus.FAILED, errors, warnings

            if 'security_recommendations' in output:
                rec = output['security_recommendations']

                if isinstance(rec, str):
                    rec = {'reason': rec, 'actions': []}
                    output['security_recommendations'] = rec
                    warnings.append("security_recommendations 从字符串转换为字典")
                elif isinstance(rec, list):
                    if len(rec) > 0:
                        if isinstance(rec[0], dict):
                            rec = rec[0]
                        else:

                            rec = {'reason': str(rec[0]), 'actions': []}
                    else:
                        rec = {}
                    output['security_recommendations'] = rec
                    warnings.append("security_recommendations 从列表转换为字典")
                elif not isinstance(rec, dict):
                    rec = {}
                    output['security_recommendations'] = rec
                    warnings.append("security_recommendations 类型未知，重置为空")
                if 'reason' not in rec:
                    rec['reason'] = "模型未提供原因"
                    warnings.append("自动补充 security_recommendations.reason")
                if 'actions' not in rec:
                    rec['actions'] = []
                    warnings.append("自动补充 security_recommendations.actions")

            required_threat = ['is_threat', 'threat_type', 'threat_score', 'confidence']
            threat = output.get('threat_conclusion', {})
            for field in required_threat:
                if field not in threat:
                    errors.append(f"threat_conclusion缺少字段: {field}")

            if errors:
                return ValidationStatus.FAILED, errors, warnings

            threat_score = threat.get('threat_score', 0)
            if threat_score is None:
                threat_score = 0.0  
                threat['threat_score'] = 0.0
            elif isinstance(threat_score, str):
                score_map = {'high': 0.8, 'medium': 0.5, 'low': 0.3}
                threat_score = score_map.get(threat_score.lower(), 0.5)
                threat['threat_score'] = threat_score
            if not (0 <= threat_score <= 1):
                errors.append(f"threat_score超出范围[0,1]: {threat_score}")

            confidence = threat.get('confidence', 0)
            if confidence is None:
                confidence = 0.3
                threat['confidence'] = 0.3
            if not (0 <= confidence <= 1):
                errors.append(f"confidence超出范围[0,1]: {confidence}")

            reasoning_chain = output.get('reasoning_chain', [])
            if not reasoning_chain:
                errors.append("缺少 reasoning_chain，模型未进行有效推理")
                return ValidationStatus.FAILED, errors, warnings

            input_evidence_ids = set()
            for day in multimodal_evidence.get('daily_evidences', []):
                for ev in day.get('evidences', []):
                    if 'evidence_id' in ev:
                        input_evidence_ids.add(ev['evidence_id'])

            core_anomalies = output.get('core_anomalies', [])
            if isinstance(core_anomalies, dict):

                core_anomalies = list(core_anomalies.values())
                warnings.append("core_anomalies 是字典格式，已转换为列表")
                output['core_anomalies'] = core_anomalies
            try:
                for anomaly in core_anomalies:
                    if not isinstance(anomaly, dict):
                        errors.append(f"core_anomalies中存在非字典元素: {type(anomaly)}")
                        continue
                    ev_id = anomaly.get('evidence_id')
                    if not ev_id:
                        errors.append("core_anomalies中存在缺少evidence_id的条目")
                    elif ev_id not in input_evidence_ids:
                        errors.append(f"evidence_id不存在于输入证据中: {ev_id}")
            except Exception as e:
                logger.error(f"core_anomalies 处理失败: type={type(core_anomalies)}, value={core_anomalies}")
                raise

            reasoning_chain = output.get('reasoning_chain', [])
            if isinstance(reasoning_chain, dict):
                reasoning_chain = list(reasoning_chain.values())
                warnings.append("reasoning_chain 是字典格式，已转换为列表")
                output['reasoning_chain'] = reasoning_chain

            for step in reasoning_chain:
                if not isinstance(step, dict):
                    continue
                refs = step.get('evidence_refs', [])

                if isinstance(refs, dict):
                    refs = list(refs.values()) if refs else []
                    warnings.append("evidence_refs 是字典格式，已转换")
                str_refs = []
                for r in refs:
                    if isinstance(r, str):
                        str_refs.append(r)
                    elif isinstance(r, dict):
                        str_refs.append(str(r.get('evidence_id', r)))
                    else:
                        str_refs.append(str(r))
                refs = str_refs
                for ref in refs:

                    if isinstance(ref, dict):
                        ref = ref.get('evidence_id', '')
                    if ref and isinstance(ref, str) and ref not in input_evidence_ids:
                        warnings.append(f"reasoning_chain引用了不存在的evidence_id: {ref}")

            modules = output.get('module_contributions', {})

            for module_name in ['semantic_module', 'statistical_module']:
                if module_name in modules:
                    module = modules[module_name]
                    if 'key_findings' not in module:

                        if module.get('triggered', False):
                            module['key_findings'] = ['检测到异常行为模式']
                        else:
                            module['key_findings'] = []
                        warnings.append(f"自动补充 {module_name}.key_findings")

                    if 'contribution_reason' not in module:
                        if module.get('triggered', False):
                            module[
                                'contribution_reason'] = f"{'语义' if 'semantic' in module_name else '统计'}模块检测到异常"
                        else:
                            module[
                                'contribution_reason'] = f"{'语义' if 'semantic' in module_name else '统计'}模块未检测到显著异常"
                        warnings.append(f"自动补充 {module_name}.contribution_reason")

            semantic_weight = modules.get('semantic_module', {}).get('weight', 0)
            if semantic_weight is None:
                semantic_weight = 0
            statistical_weight = modules.get('statistical_module', {}).get('weight', 0)
            if statistical_weight is None:
                statistical_weight = 0

            semantic_triggered = modules.get('semantic_module', {}).get('triggered', False)
            if semantic_triggered is None:
                semantic_triggered = False
            statistical_triggered = modules.get('statistical_module', {}).get('triggered', False)
            if statistical_triggered is None:
                statistical_triggered = False

            if not semantic_triggered and semantic_weight > 0:
                warnings.append("语义模块未触发但权重>0")
            if not statistical_triggered and statistical_weight > 0:
                warnings.append("统计模块未触发但权重>0")

            if semantic_triggered and statistical_triggered:
                total_weight = semantic_weight + statistical_weight
                if abs(total_weight - 1.0) > 0.1:
                    warnings.append(f"两模块权重之和({total_weight})偏离1.0")

            if errors:
                return ValidationStatus.FAILED, errors, warnings
            elif warnings:
                return ValidationStatus.PARTIAL, errors, warnings
            else:
                return ValidationStatus.PASSED, errors, warnings
        except Exception as e:
            logger.error(f"验证崩溃: {e}\n{traceback.format_exc()}")
            raise

def create_mock_engine() -> LLMReasoningEngine:
    """创建Mock推理引擎（用于测试）"""

    MOCK_OUTPUT = """
    <thinking>
    步骤1: 正常模式推断 - 研发工程师，日均文件操作约10次
    步骤2: 异常点定位 - stat_BSS0369_2010-07-13_0 文件操作暴增
    步骤3: 量化验证 - 实际28次 vs 基线10.29次，超出172%
    步骤4: 模块贡献 - 统计模块0.6，语义模块0.4
    步骤5: 威胁判定 - 数据窃取，分数0.68
    步骤6: 处置建议 - 审计级别
    </thinking>
    <output>
    {
      "user_id": "BSS0369",
      "threat_conclusion": {
        "is_threat": true,
        "threat_type": "数据窃取",
        "threat_score": 0.68,
        "confidence": 0.85
      },
      "core_anomalies": [
        {
          "evidence_id": "stat_BSS0369_2010-07-13_0",
          "anomaly_description": "非工作时段文件操作暴增",
          "severity": "high",
          "quantitative_validation": {
            "metric_name": "file_count",
            "value": 28,
            "baseline_or_threshold": 10.29,
            "deviation": "超出172%"
          }
        }
      ],
      "module_contributions": {
        "semantic_module": {
          "weight": 0.4,
          "triggered": true,
          "key_findings": ["文件内容异常"],
          "contribution_reason": "发现可疑文件内容"
        },
        "statistical_module": {
          "weight": 0.6,
          "triggered": true,
          "key_findings": ["文件操作暴增"],
          "contribution_reason": "量化偏离显著"
        }
      },
      "reasoning_chain": [
        {"step": 1, "step_name": "正常模式推断", "analysis": "研发工程师，日均文件操作约10次", "evidence_refs": []},
        {"step": 2, "step_name": "异常点定位", "analysis": "文件操作暴增", "evidence_refs": ["stat_BSS0369_2010-07-13_0"]},
        {"step": 3, "step_name": "量化验证", "analysis": "实际28次 vs 基线10.29次", "evidence_refs": ["stat_BSS0369_2010-07-13_0"]}
      ],
      "natural_language_explanation": {
        "normal_pattern_summary": "用户通常每日进行约10次文件操作",
        "anomaly_comparison": "异常当天28次，超出172%",
        "typical_pattern_comparison": "符合数据窃取的典型模式"
      },
      "security_recommendations": {
        "level": "审计",
        "actions": ["审计文件操作记录", "检查U盘使用情况"],
        "reason": "行为符合数据窃取模式"
      }
    }
    </output>
    """

    def mock_llm_call(messages, **kwargs):
        return MOCK_OUTPUT

    return LLMReasoningEngine(custom_llm_call=mock_llm_call)

if __name__ == "__main__":
    import sys
    import os

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    sys.path.insert(0, _project_root)
    from prompt_templates import build_user_prompt, SYSTEM_PROMPT, get_output_schema_dict

    print("=" * 80)
    print("LLM推理引擎 - 测试")
    print("=" * 80)

    mock_evidence = {
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
                "max_score": 0.7344
            },
            "statistical_stats": {
                "total_anomaly_days": 1,
                "max_confidence": 0.6069
            }
        },
        "daily_evidences": [
            {
                "date": "2010-01-03",
                "evidences": [
                    {
                        "evidence_id": "sem_BSS0369_2010-01-03_file_0",
                        "source": "semantic",
                        "metrics": {"anomaly_score": 0.703125}
                    }
                ]
            },
            {
                "date": "2010-07-13",
                "evidences": [
                    {
                        "evidence_id": "stat_BSS0369_2010-07-13_0",
                        "source": "statistical",
                        "metrics": {"max_z_score": 3.04, "confidence": 0.6069},
                        "details": {
                            "value_baseline_comparison": {
                                "file_count_anomaly_count": 28,
                                "file_count_anomaly_baseline": 10.291
                            }
                        }
                    }
                ]
            }
        ]
    }

    mock_user_profile = "用户 BSS0369，岗位为研发工程师，隶属于技术研发中心，用户类型为正式员工；在职，入职于2009-03；专属设备为PC-8884。"

    print("\n【使用Mock引擎测试】")
    engine = create_mock_engine()
    result = engine.reason(mock_evidence, mock_user_profile)

    print(f"\n推理结果:")
    print(f"  用户: {result.user_id}")
    print(f"  威胁判定: {result.is_threat}")
    print(f"  威胁类型: {result.threat_type}")
    print(f"  威胁分数: {result.threat_score}")
    print(f"  置信度: {result.confidence}")
    print(f"  验证状态: {result.validation_status.value}")
    print(f"  重试次数: {result.retry_count}")

    if result.validation_warnings:
        print(f"  警告: {result.validation_warnings}")

    print(f"\n推理日志摘要:\n{result.reasoning_log[:500]}...")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)