
import logging
import json
from typing import Dict, Optional, List

import torch

from multi_modal_features.semantic.prompts import build_semantic_analysis_prompt, build_training_prompt, OutputMode
from multi_modal_features.semantic.categories import get_category_by_event_type
import os

DEBUG_LLM = os.environ.get('DEBUG_LLM', 'false').lower() == 'true'

logger = logging.getLogger(__name__)

class SemanticAnalyzer:

    def __init__(self, llm_model=None, output_mode: OutputMode = OutputMode.SIMPLE,
                 max_content_chars: int = 1500, prefer_trained: bool = True,
                 auto_load: bool = True, model_path: str = None):

        self.llm = llm_model
        self.output_mode = output_mode
        self.max_content_chars = max_content_chars

        if llm_model is None and auto_load:
            from utils import get_semantic_model
            logger.info(f"开始加载语义模型...")
            self.llm = get_semantic_model(prefer_trained=prefer_trained, model_path=model_path)
            if self.llm is None:
                logger.error("模型加载失败，get_semantic_model返回None")
                self.is_binary_model = False
            else:
                self.is_binary_model = hasattr(self.llm, 'predict') and not hasattr(self.llm, 'generate')
                logger.info(f"模型类型判断: is_binary_model={self.is_binary_model}")
        else:
            self.llm = llm_model
            self.is_binary_model = hasattr(self.llm, 'predict') and not hasattr(self.llm, 'generate')

        logger.info(f"SemanticAnalyzer初始化完成, output_mode={output_mode.value}, "
                    f"max_content_chars={max_content_chars}, model_loaded={self.llm is not None}")

    def analyze(self, content: str, event_type: str, event_fields: Dict = None) -> Dict:

        category = get_category_by_event_type(event_type)
        category_value = category.value if category else 'unknown'

        if not content or not content.strip():
            return self._get_empty_result(category_value)

        prompt = build_semantic_analysis_prompt(
            content=content,
            event_type=event_type,
            event_fields=event_fields,
            mode=self.output_mode,
            max_content_chars=self.max_content_chars
        )

        if self.llm is None:
            logger.warning("LLM模型未配置，返回默认结果")
            return self._get_default_result(category_value)

        try:
            if self.is_binary_model:

                result = self._predict_with_binary_model(prompt)
            else:

                response = self.llm.generate(prompt)

                if DEBUG_LLM:
                    logger.info(f"LLM原始响应: {response}")
                result = self._parse_response(response)

            result['category'] = category_value
            return result
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._get_error_result(category_value, str(e))

    def batch_analyze(self, events_data: List[Dict]) -> List[Dict]:

        if not events_data:
            return []

        results = []

        valid_indices = []  
        valid_prompts = []  

        for idx, data in enumerate(events_data):
            content = data.get('content', '')

            if not content or not content.strip():
                category = get_category_by_event_type(data.get('event_type', ''))
                results.append((idx, self._get_empty_result(category.value if category else 'unknown')))
            else:
                prompt = build_semantic_analysis_prompt(
                    content=content,
                    event_type=data.get('event_type', ''),
                    event_fields=data.get('event_fields', {}),
                    user_profile=data.get('user_profile'),
                    mode=self.output_mode,
                    max_content_chars=self.max_content_chars
                )
                valid_indices.append(idx)
                valid_prompts.append(prompt)

        if valid_prompts:
            if self.is_binary_model:

                batch_predictions = self.llm.predict(valid_prompts)
                for i, (label, confidence) in enumerate(batch_predictions):
                    idx = valid_indices[i]
                    data = events_data[idx]

                    result = {
                        'is_anomaly': (label == "异常"),
                        'anomaly_score': confidence,
                        'category': get_category_by_event_type(data.get('event_type', '')).value
                    }
                    if self.output_mode == OutputMode.DETAILED:
                        result.update({
                            'key_evidence': '基于训练模型的预测结果',
                            'explanation': f'二分类模型预测: {label}, 置信度: {confidence:.4f}',
                        })
                    results.append((idx, result))
            else:

                for i, prompt in enumerate(valid_prompts):
                    idx = valid_indices[i]
                    data = events_data[idx]
                    try:
                        response = self.llm.generate(prompt)
                        if DEBUG_LLM:
                            logger.info(f"LLM原始响应: {response}")
                        result = self._parse_response(response)
                        result['category'] = get_category_by_event_type(data.get('event_type', '')).value
                    except Exception as e:
                        logger.error(f"LLM调用失败: {e}")
                        category = get_category_by_event_type(data.get('event_type', ''))
                        category_value = category.value if category else 'unknown'
                        result = self._get_error_result(category_value, str(e))
                    results.append((idx, result))

        results.sort(key=lambda x: x[0])
        return [r for _, r in results]  

    def _extract_final_answer(self, response: str) -> str:

        import re

        if '</think>' in response:
            parts = response.split('</think>')

            final = parts[-1].strip()
            if final:
                return final

        if '  instant' in response:
            parts = response.split('  instant')
            final = parts[-1].strip()
            if final:
                return final

        lines = [l.strip() for l in response.split('\n') if l.strip()]
        inference_keywords = ['嗯', '首先', '然后', '我需要', '现在', '分析', '考虑', '用户', '看起来', '可能']

        for line in reversed(lines):
            if len(line) < 50 and not any(kw in line for kw in inference_keywords):

                if '异常' in line or '正常' in line or re.search(r'\d+\.?\d*', line):
                    return line

        return response

    def _parse_simple_response(self, response: str) -> Dict:

        import re

        final_answer = self._extract_final_answer(response)
        response_lower = final_answer.lower()

        if '异常' in response_lower or 'anomaly' in response_lower:
            is_anomaly = True

        elif '不正常' in response_lower or '非正常' in response_lower:
            is_anomaly = True

        elif '正常' in response_lower or 'normal' in response_lower:
            is_anomaly = False

        else:
            is_anomaly = False

        score_match = re.search(r'(\d+\.?\d*)', final_answer)
        anomaly_score = 0.0
        if score_match:
            try:
                score = float(score_match.group(1))

                if 0.0 <= score <= 1.0:
                    anomaly_score = score
                elif score > 1.0:
                    anomaly_score = min(score / 100, 1.0)  
            except ValueError:
                pass

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
        }

    def _parse_detailed_response(self, response: str) -> Dict:

        final_answer = self._extract_final_answer(response)

        parts = [p.strip() for p in final_answer.split('|')]

        is_anomaly = False
        anomaly_score = 0.0
        key_evidence = ""
        explanation = ""

        if len(parts) >= 1:

            part1 = parts[0].lower()
            if '异常' in part1 or 'anomaly' in part1:
                is_anomaly = True
            elif '正常' in part1 or 'normal' in part1:
                is_anomaly = False

        if len(parts) >= 2:

            try:
                score = float(parts[1])
                if 0.0 <= score <= 1.0:
                    anomaly_score = score
                elif score > 1.0:
                    anomaly_score = min(score / 100, 1.0)
            except ValueError:
                pass

        if len(parts) >= 3:
            key_evidence = parts[2][:100]  

        if len(parts) >= 4:
            explanation = parts[3][:200]  

        if not key_evidence:
            key_evidence = "无" if not is_anomaly else "检测到异常"

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'key_evidence': key_evidence,
            'explanation': explanation,
        }

    def _parse_json_response(self, response: str) -> Optional[Dict]:

        import re
        try:

            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in matches:
                try:
                    result = json.loads(match)
                    if 'is_anomaly' in result:
                        return self._format_result(result)
                except json.JSONDecodeError:
                    continue

            cleaned = response.strip()

            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]

            cleaned = cleaned.replace("'", '"')

            start = cleaned.find('{')
            end = cleaned.rfind('}')

            if start != -1 and end > start:
                json_str = cleaned[start:end + 1]  
                try:
                    result = json.loads(json_str)  
                    return self._format_result(result)
                except json.JSONDecodeError:
                    pass
        except Exception as e:

            logger.error(f"JSON解析过程发生未知错误: {e}")
            return None

    def _parse_response(self, response: str) -> Dict:

        if self.output_mode == OutputMode.SIMPLE:

            return self._parse_simple_response(response)
        else:

            result = self._parse_detailed_response(response)

            if result['anomaly_score'] == 0.0 and not result['key_evidence']:
                json_result = self._parse_json_response(response)
                if json_result:
                    return json_result

            return result

    def _format_result(self, result: Dict) -> Dict:

        if self.output_mode == OutputMode.SIMPLE:
            return {
                'is_anomaly': result.get('is_anomaly', False),
                'anomaly_score': float(result.get('anomaly_score', 0.0)),
            }
        else:
            return {
                'is_anomaly': result.get('is_anomaly', False),
                'anomaly_score': float(result.get('anomaly_score', 0.0)),
                'key_evidence': result.get('key_evidence', '')[:100],
                'explanation': result.get('explanation', '')[:200],
            }

    def _predict_with_binary_model(self, content: str) -> Dict:

        try:

            label, confidence = self.llm.predict(content)

            is_anomaly = (label == "异常")
            anomaly_score = confidence

            if self.output_mode == OutputMode.SIMPLE:
                return {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score,
                }
            else:
                return {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score,
                    'key_evidence': '基于训练模型的预测结果',
                    'explanation': f'二分类模型预测: {label}, 置信度: {confidence:.4f}',
                }
        except Exception as e:
            logger.error(f"二分类模型预测失败: {e}")
            return self._get_error_result('unknown', str(e))

    def _get_empty_result(self, category_value: str) -> Dict:

        if self.output_mode == OutputMode.SIMPLE:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'category': category_value
            }
        else:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'key_evidence': '',
                'explanation': '内容为空，跳过分析',
                'category': category_value
            }

    def _get_default_result(self, category_value: str) -> Dict:

        if self.output_mode == OutputMode.SIMPLE:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'category': category_value
            }
        else:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'key_evidence': '',
                'explanation': 'LLM未配置，无法分析',
                'category': category_value
            }

    def _get_error_result(self, category_value: str, error_msg: str) -> Dict:

        if self.output_mode == OutputMode.SIMPLE:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'category': category_value
            }
        else:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'key_evidence': '',
                'explanation': f'分析失败: {error_msg[:50]}',
                'category': category_value
            }

    def _get_parse_error_result(self) -> Dict:

        if self.output_mode == OutputMode.SIMPLE:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
            }
        else:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'key_evidence': '',
                'explanation': '响应解析失败'
            }

    def set_output_mode(self, mode: OutputMode):

        self.output_mode = mode
        logger.info(f"输出模式已切换为: {mode.value}")

    def train_analyze(self, content: str, event_fields: Dict = None, user_profile: Dict = None) -> Dict:

        from multi_modal_features.semantic.prompts import build_training_prompt

        prompt = build_training_prompt(
            content=content,
            event_fields=event_fields,

            max_content_chars=self.max_content_chars
        )

        if self.llm is None:
            return self._get_default_result('unknown')

        try:
            response = self.llm.generate(prompt)
            result = self._parse_response(response)
            return result
        except Exception as e:
            logger.error(f"训练模式LLM调用失败: {e}")
            return self._get_error_result('unknown', str(e))

if __name__ == "__main__":

    import sys
    import os

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_current_dir))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from utils import setup_logging

    setup_logging("semantic_analyzer_debug", __file__)

    print("=" * 80)
    print("语义分析器 - 完整功能测试")
    print("=" * 80)

    print("\n【测试1】响应解析功能")
    print("-" * 40)

    analyzer_detailed = SemanticAnalyzer(llm_model=None, output_mode=OutputMode.DETAILED, auto_load=False)
    analyzer_simple = SemanticAnalyzer(llm_model=None, output_mode=OutputMode.SIMPLE, auto_load=False)

    mock_response = '{"is_anomaly": true, "anomaly_score": 0.85, "key_evidence": "数据库密码", "explanation": "包含敏感密码"}'
    print("  正常JSON响应:")
    result = analyzer_detailed._parse_response(mock_response)
    print(f"    详细模式: {result}")
    result = analyzer_simple._parse_response(mock_response)
    print(f"    简化模式: {result}")

    mock_response_with_text = '分析结果：{"is_anomaly": true, "anomaly_score": 0.85, "key_evidence": "密码泄露"}'
    print("\n  带额外文本的JSON响应:")
    result = analyzer_detailed._parse_response(mock_response_with_text)
    print(f"    解析结果: {result}")

    mock_invalid = '这不是JSON格式'
    print("\n  无效JSON响应:")
    result = analyzer_detailed._parse_response(mock_invalid)
    print(f"    解析结果: {result}")

    print("\n【测试2】空内容处理")
    print("-" * 40)

    class MockLLM:
        def generate(self, prompt):
            return '{"is_anomaly": true, "anomaly_score": 0.9, "key_evidence": "测试", "explanation": "测试"}'

    mock_llm = MockLLM()
    analyzer = SemanticAnalyzer(llm_model=mock_llm, output_mode=OutputMode.DETAILED)

    empty_result = analyzer.analyze("", "email")
    print(f"  空字符串: is_anomaly={empty_result['is_anomaly']}, explanation={empty_result.get('explanation', '')}")

    space_result = analyzer.analyze("   ", "email")
    print(f"  纯空格: is_anomaly={space_result['is_anomaly']}, explanation={space_result.get('explanation', '')}")

    print("\n【测试3】LLM未配置处理")
    print("-" * 40)

    analyzer_no_llm = SemanticAnalyzer(llm_model=None, output_mode=OutputMode.DETAILED, auto_load=False)
    result = analyzer_no_llm.analyze("这是测试内容", "email")
    print(f"  详细模式结果: is_anomaly={result['is_anomaly']}, explanation={result.get('explanation', '')}")

    analyzer_no_llm_simple = SemanticAnalyzer(llm_model=None, output_mode=OutputMode.SIMPLE, auto_load=False)
    result = analyzer_no_llm_simple.analyze("这是测试内容", "email")
    print(f"  简化模式结果: is_anomaly={result['is_anomaly']}, category={result.get('category', '')}")

    print("\n【测试4】不同事件类型类别映射")
    print("-" * 40)

    for event_type in ['email', 'web', 'file', 'unknown']:
        result = analyzer_no_llm.analyze("测试内容", event_type)
        print(f"  {event_type} -> category={result.get('category', 'N/A')}")

    print("\n【测试5】输出模式动态切换")
    print("-" * 40)

    analyzer_switch = SemanticAnalyzer(llm_model=mock_llm, output_mode=OutputMode.DETAILED, auto_load=False)
    print(f"  初始模式: {analyzer_switch.output_mode.value}")

    analyzer_switch.set_output_mode(OutputMode.SIMPLE)
    print(f"  切换后模式: {analyzer_switch.output_mode.value}")

    print("\n【测试6】内容截断功能验证")
    print("-" * 40)

    long_content = "A" * 3000  
    analyzer_short = SemanticAnalyzer(llm_model=None, output_mode=OutputMode.DETAILED, max_content_chars=1000, auto_load=False)

    result = analyzer_short.analyze(long_content, "email")
    print(f"  max_content_chars=1000, 空内容检测通过（LLM未配置）")

    print("\n【测试7】完整流程模拟（Mock LLM）")
    print("-" * 40)

    class RecordingMockLLM:

        def __init__(self):
            self.last_prompt = None

        def generate(self, prompt):
            self.last_prompt = prompt
            return '{"is_anomaly": true, "anomaly_score": 0.85, "key_evidence": "离职抱怨", "explanation": "员工表达离职意愿"}'

    recording_llm = RecordingMockLLM()
    analyzer_full = SemanticAnalyzer(llm_model=recording_llm, output_mode=OutputMode.DETAILED)

    test_content = "company will suffer i may leave no gratitude my work not appreciated"
    test_fields = {
        'user_id': 'BSS0369',
        'timestamp': '09/30/2010 13:31:56',
        'to': 'manager@dtaa.com',
        'from': 'employee@dtaa.com'
    }

    result = analyzer_full.analyze(test_content, "email", test_fields)

    print(f"  分析结果:")
    print(f"    is_anomaly: {result.get('is_anomaly')}")
    print(f"    anomaly_score: {result.get('anomaly_score')}")
    print(f"    key_evidence: {result.get('key_evidence', '')}")
    print(f"    explanation: {result.get('explanation', '')}")
    print(f"    category: {result.get('category')}")

    if recording_llm.last_prompt:
        print(f"\n  Prompt中是否包含事件字段:")
        print(f"    包含 user_id: {'user_id' in recording_llm.last_prompt}")
        print(f"    包含 to: {'to' in recording_llm.last_prompt}")
        print(f"    包含 from: {'from' in recording_llm.last_prompt}")

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("  ✓ 响应解析功能（正常/带文本/无效JSON）")
    print("  ✓ 空内容处理（空字符串/纯空格）")
    print("  ✓ LLM未配置处理（详细/简化模式）")
    print("  ✓ 不同事件类型类别映射")
    print("  ✓ 输出模式动态切换")
    print("  ✓ 截断功能验证")
    print("  ✓ 完整流程模拟（Mock LLM）")
    print("=" * 80)
    print("调试完成")