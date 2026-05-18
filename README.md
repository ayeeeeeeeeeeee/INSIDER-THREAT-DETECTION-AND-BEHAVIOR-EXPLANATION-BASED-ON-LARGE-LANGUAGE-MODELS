# INSIDER THREAT DETECTION AND BEHAVIOR EXPLANATION BASED ON LARGE LANGUAGE MODELS
本项目是一个基于大语言模型的内部威胁检测与行为解释框架。

## 项目目录结构
``` 
├── test_all.py
├── data_preprocessing/
│   ├── behavior_sequence_builder.py
│   ├── cert_data_loader.py
│   ├── mini_batch_loader.py
│   ├── preprocessing_utils.py
│   └── redteam_data_loader.py
├── evidence_aggregation/
│   ├── behavior_abstractor_4W.py
│   ├── multimodal_aggregator.py
│   ├── semantic_aggregator.py
│   └── statistical_aggregator.py
├── explanation/
│   ├── llm_reasoning_engine.py
│   ├── prompt_templates.py
│   └── report_generator.py
├── multi_modal_features/
│   ├── behavior_statistics.py
│   ├── user_profile.py
│   ├── utils.py
│   └── semantic/
│       ├── analyzer.py
│       ├── categories.py
│       ├── detector_wrapper.py
│       └── prompts.py
└── utils/
    ├── common.py
    └── model_loader.py
```

## 项目声明

- **项目名称**：基于大语言模型的内部威胁检测与行为解释系统
- **项目作者**：Ye Xining
- **作者单位**：暨南大学网络空间安全学院
- **开发语言**：Python
- **核心技术**：多视角用户行为表示、多粒度异常检测、基于思维链的可解释推理
