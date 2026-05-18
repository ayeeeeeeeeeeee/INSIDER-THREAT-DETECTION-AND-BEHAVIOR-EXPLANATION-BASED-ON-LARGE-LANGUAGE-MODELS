from transformers import BitsAndBytesConfig
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from peft import PeftModel
from tqdm import tqdm
import torch
from config import BASE_MODEL_PATH

logger = logging.getLogger(__name__)

class ModelLoader:

    _instance = None  
    _models: Dict[str, Any] = {}  

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._loaded_models = {}  

    @staticmethod
    def _get_model_path(model_type: str, prefer_trained: bool = True) -> Optional[str]:
        try:
            from config import get_model_path, ModelType, check_trained_model_exists

            type_map = {
                'semantic': ModelType.SEMANTIC,
                'cot': ModelType.COT,

            }

            mapped_type = type_map.get(model_type)
            if mapped_type is None:
                logger.warning(f"未知的模型类型: {model_type}")
                return None

            if prefer_trained and check_trained_model_exists(mapped_type):
                return get_model_path(mapped_type, prefer_trained=True)

            return get_model_path(mapped_type, prefer_trained=False)

        except ImportError as e:
            logger.error(f"导入配置失败: {e}")
            return None
        except Exception as e:
            logger.error(f"获取模型路径失败: {e}")
            return None

    @staticmethod
    def _load_transformers_model(model_path: str, model_type: str = 'semantic'):
        try:
            from transformers import AutoTokenizer
            from config.model_training import TRAINING_CONFIG
            classification_max_length = CLASSIFICATION_MAX_LENGTH
            generation_max_length = GENERATION_MAX_LENGTH

            logger.info(f"加载模型: {model_path}")

            if model_type == 'semantic':
                from transformers import AutoModelForSequenceClassification
                from config import BASE_MODEL_PATH

                adapter_config_path = os.path.join(model_path, 'adapter_config.json')
                if os.path.exists(adapter_config_path):

                    logger.info(f"检测到 LoRA 适配器，加载基础模型 {BASE_MODEL_PATH} + 适配器 {model_path}")
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        BASE_MODEL_PATH,
                        num_labels=2,
                        trust_remote_code=True,
                        device_map="auto",
                        torch_dtype=torch.float32  
                    )
                    model = PeftModel.from_pretrained(base_model, model_path)
                else:

                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path, trust_remote_code=True,
                        device_map="auto",
                        torch_dtype=torch.float32  
                    )
            else:
                from transformers import AutoModelForCausalLM

                use_flash_attn = False
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    if capability >= (8, 0):

                        try:
                            import flash_attn
                            use_flash_attn = True
                            logger.info(f"✓ Flash Attention 2 可用 (GPU: {torch.cuda.get_device_name()})")
                        except ImportError:
                            logger.warning(
                                f"GPU ({torch.cuda.get_device_name()}) 支持 Flash Attention 2，"
                                f"但 flash_attn 未安装。使用标准注意力。"
                                f"安装命令: pip install flash-attn==2.7.3 --no-build-isolation"
                            )
                    else:
                        logger.info(f"GPU计算能力 {capability} < 8.0，Flash Attention 2 不可用")
                model_kwargs = {
                    'trust_remote_code': True,
                    'device_map': 'auto',
                    'torch_dtype': torch.float16  
                }

                if use_flash_attn:
                    model_kwargs['attn_implementation'] = 'flash_attention_2'
                    logger.info("✓ 启用 Flash Attention 2")

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs  
                )

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if model_type == 'semantic':

                class ClassificationWrapper:
                    def __init__(self, model, tokenizer, batch_size=64):

                        self.model = model
                        self.tokenizer = tokenizer
                        self.batch_size = batch_size
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        if self.model.config.pad_token_id is None:
                            self.model.config.pad_token_id = self.tokenizer.pad_token_id

                        if torch.cuda.is_available():
                            self.model = self.model.cuda()
                            logger.info(f"模型已移至 GPU: {torch.cuda.get_device_name()}")
                        self.model.eval()
                        logger.info("模型已设置为评估模式 (eval mode)")

                    def predict(self, text: str, batch_size=None) -> tuple:

                        is_single = isinstance(text, str)

                        texts = [text] if is_single else text

                        if not texts:
                            return [] if not is_single else ("正常", 0.0)

                        truncation_count = 0
                        max_length_seen = 0
                        for t in texts:

                            temp_inputs = self.tokenizer(t, return_tensors="pt", truncation=False)
                            actual_length = temp_inputs['input_ids'].shape[1]
                            max_length_seen = max(max_length_seen, actual_length)
                            if actual_length > classification_max_length:
                                truncation_count += 1

                        if truncation_count > 0:
                            logger.warning(
                                f"批次中有 {truncation_count}/{len(texts)} 个样本会被截断 "
                                f"(最大长度={max_length_seen}, 限制={classification_max_length})"
                            )
                        if batch_size is None:
                            base_seq_len = 576
                            base_batch = 128

                            effective_seq_len = min(max_length_seen, classification_max_length)

                            if effective_seq_len < base_seq_len:  
                                scale_factor = (base_seq_len / effective_seq_len) ** 2
                                dynamic_batch = min(BATCH_SIZE, int(base_batch * scale_factor))
                                logger.info(
                                    f"动态batch: {dynamic_batch} (max_len={max_length_seen}, effective_len={effective_seq_len})")
                            else:
                                dynamic_batch = base_batch  
                        else:
                            dynamic_batch = BATCH_SIZE

                        all_results = []

                        for i in range(0, len(texts), dynamic_batch):

                            batch_texts = texts[i:i + dynamic_batch]

                            inputs = self.tokenizer(
                                batch_texts,
                                return_tensors="pt",
                                truncation=True,
                                padding='longest', 
                                max_length=classification_max_length  
                            )

                            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                with torch.amp.autocast('cuda'):
                                    outputs = self.model(**inputs)

                                    probs = torch.softmax(outputs.logits, dim=-1)

                                    anomaly_probs = probs[:, 1]

                                    anomaly_probs = anomaly_probs.cpu().tolist()
                            for anomaly_prob in anomaly_probs:
                                label = "异常" if anomaly_prob > 0.5 else "正常"
                                all_results.append((label, anomaly_prob))
                            reserved = torch.cuda.memory_reserved()
                            allocated = torch.cuda.memory_allocated()
                            if reserved - allocated > 500 * 1024 * 1024:  
                                torch.cuda.empty_cache()

                        torch.cuda.empty_cache()

                        return all_results[0] if is_single else all_results

                return ClassificationWrapper(model, tokenizer)

            else:
                class TransformersWrapper:
                    def __init__(self, model, tokenizer):

                        self.model = model
                        self.tokenizer = tokenizer
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        if self.model.config.pad_token_id is None:
                            self.model.config.pad_token_id = self.tokenizer.pad_token_id
                        try:
                            device = next(self.model.parameters()).device
                            logger.info(f"模型设备: {device}")
                        except:
                            logger.info("模型设备由 accelerate/device_map 管理")

                        self.model.eval()
                        logger.info("模型已设置为评估模式 (eval mode)")

                    def generate(self, prompt: str, max_new_tokens: int = 512,
                                 temperature: float = 0.1, **kwargs) -> str:

                        if "System:" in prompt or "<|im_start|>" in prompt:

                            text = prompt
                        else:

                            messages = [{"role": "user", "content": prompt}]
                            try:
                                text = self.tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )
                            except Exception as e:
                                logger.warning(f"apply_chat_template失败: {e}，使用手动格式")
                                text = f"<|User|>\n{prompt}\n<|Assistant|>\n"

                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",  
                            max_length=GENERATION_MAX_LENGTH,  
                            truncation=True  
                        )

                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                        generation_kwargs = {
                            'max_new_tokens': max_new_tokens,       
                            'do_sample': True if temperature > 0 else False,  
                            'use_cache': True,                      

                            'pad_token_id': self.tokenizer.eos_token_id,  
                            'repetition_penalty': 1.15,  
                            **kwargs
                        }
                        if temperature and temperature > 0:
                            generation_kwargs['temperature'] = temperature

                            generation_kwargs['top_p'] = kwargs.get('top_p', 0.95)

                        logger.info(f"实际生成参数: do_sample={generation_kwargs.get('do_sample')}, "
                                    f"temperature={generation_kwargs.get('temperature')}, "
                                    f"top_p={generation_kwargs.get('top_p')}")
                        with torch.amp.autocast('cuda'):

                            outputs = self.model.generate(**inputs, **generation_kwargs)

                        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                        if not response.strip():
                            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            if text in full_response:
                                response = full_response.replace(text, "").strip()

                        return response if response.strip() else "正常 0.0"

                return TransformersWrapper(model, tokenizer)

        except ImportError as e:
            logger.error(f"transformers导入失败: {e}")
            return None

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None

    def load_model(self, model_type: str = 'semantic', 
                   prefer_trained: bool = True,
                   force_reload: bool = False,
                   model_path: str = None) -> Optional[Any]:

        if model_path:
            cache_key = f"{model_type}_custom_{model_path}"
        else:
            cache_key = f"{model_type}_{prefer_trained}"

        if not force_reload and cache_key in self._loaded_models:
            logger.debug(f"从缓存加载模型: {cache_key}")
            return self._loaded_models[cache_key]

        if model_path is None:
            model_path = self._get_model_path(model_type, prefer_trained)
            if not model_path:
                logger.error(f"无法获取模型路径: {model_type}")
                return None

        if not os.path.exists(model_path):
            logger.warning(f"模型路径不存在: {model_path}")
            return None

        model = self._load_transformers_model(model_path, model_type)

        if model:
            self._loaded_models[cache_key] = model
            logger.info(f"模型加载成功: {model_type} from {model_path}")
        else:
            logger.error(f"模型加载失败: {model_type}")

        return model

    def unload_model(self, model_type: str = None):

        if model_type is None:
            self._loaded_models.clear()
            logger.info("已卸载所有模型")
        else:
            keys_to_remove = [k for k in self._loaded_models.keys() if k.startswith(model_type)]
            for key in keys_to_remove:
                del self._loaded_models[key]
            logger.info(f"已卸载模型: {model_type}")

    def get_model_status(self) -> Dict[str, bool]:

        status = {}
        for key in self._loaded_models.keys():
            model_type = key.split('_')[0]
            status[model_type] = True
        return status

_loader = ModelLoader()

def load_model(model_type: str = 'semantic', prefer_trained: bool = True,
               model_path: Optional[str] = None) -> Optional[Any]:

    return _loader.load_model(model_type, prefer_trained, model_path=model_path)

def get_semantic_model(prefer_trained: bool = True, model_path: str = None):

    return load_model('semantic', prefer_trained, model_path=model_path)

def get_cot_model(prefer_trained: bool = True, model_path: str = None):

    return load_model('cot', prefer_trained, model_path=model_path)

def get_explanation_model(prefer_trained: bool = True, model_path: str = None):

    return load_model('explanation', prefer_trained, model_path=model_path)

def unload_models(model_type: str = None):

    _loader.unload_model(model_type)

def get_model_status() -> Dict[str, bool]:

    return _loader.get_model_status()

if __name__ == "__main__":
    import sys
    import os

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from utils import setup_logging
    setup_logging("model_loader_test", __file__)

    print("=" * 80)
    print("模型加载器测试")
    print("=" * 80)

    print("\n【测试1】获取模型路径")
    from config import get_semantic_model_path, check_trained_model_exists

    semantic_path = get_semantic_model_path()
    print(f"  语义模型路径: {semantic_path}")
    print(f"  路径是否存在: {os.path.exists(semantic_path) if semantic_path else False}")
    print(f"  训练模型是否存在: {check_trained_model_exists('semantic')}")

    print("\n【测试2】加载模型测试")
    model = load_model('semantic', prefer_trained=False)

    if model:
        print("  ✓ 模型加载成功")

        if hasattr(model, 'model'):
            device = model.model.device
            print(f"  模型设备: {device}")
    else:
        print("  ✗ 模型加载失败")
        sys.exit(1)

    print("\n【测试3】基础生成测试")

    test_cases = [
        ("你好", "简单问候"),
        ("请输出：正常 0.12", "格式输出测试"),
        ("请回答：异常 0.85", "异常输出测试"),
    ]

    for prompt, description in test_cases:
        print(f"\n  测试: {description}")
        print(f"    Prompt: {prompt}")
        try:
            response = model.generate(prompt, max_new_tokens=512)
            print(f"    响应: {response}")

            if prompt in response and len(response) < len(prompt) + 20:
                print(f"    ⚠️ 警告: 模型似乎只返回了输入内容")
        except Exception as e:
            print(f"    ✗ 生成失败: {e}")

    print("\n【测试4】语义异常检测格式测试")

    test_prompts = [
        {
            "name": "简化模式输出测试",
            "prompt": 

        },
        {
            "name": "详细模式输出测试",
            "prompt": 

        }
    ]

    for test in test_prompts:
        print(f"\n  测试: {test['name']}")
        try:
            label, confidence = model.predict(test['prompt'])  
            print(f"    预测结果: {label}, 置信度: {confidence:.4f}")

        except Exception as e:
            print(f"    ✗ 生成失败: {e}")

    print("\n【测试5】参数兼容性测试")

    try:

        response1 = model.generate("输出：正常 0.12", temperature=0.5, max_new_tokens=20)
        response2 = model.generate("输出：正常 0.12", max_new_tokens=20)
        print(f"  带temperature参数: {response1[:50]}")
        print(f"  不带temperature: {response2[:50]}")
        if response1 == response2:
            print("  ✅ temperature参数被正确忽略，输出一致")
        else:
            print("  ⚠️ 两次输出不一致")
    except Exception as e:
        print(f"  ✗ 参数测试失败: {e}")

    print("\n【测试6】模型缓存状态")
    print(f"  已加载模型: {get_model_status()}")

    print("\n  测试重复加载（应从缓存获取）:")
    model2 = load_model('semantic', prefer_trained=False)
    if model2 is model:
        print("    ✅ 模型从缓存加载，实例相同")
    else:
        print("    ⚠️ 模型重新加载了（缓存可能失效）")

    print("\n【测试7】卸载模型")
    unload_models()
    print(f"  卸载后状态: {get_model_status()}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    print("\n【建议】")
    print("  1. 如果模型返回 prompt 本身 → 检查 apply_chat_template 是否生效")
    print("  2. 如果模型输出乱码 → 检查 tokenizer 配置")
    print("  3. 如果格式不符合预期 → 调整 prompt 或解析逻辑")
    print("  4. 如果 GPU 显存不足 → 设置 device_map='cpu' 或使用量化")