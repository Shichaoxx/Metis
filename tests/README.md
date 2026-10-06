# 测试索引

这里验证 `metis` 的软件契约。所有模型 fixture 均为本地构造的 tiny 或替身模型；测试不需要预训练权重、下载数据或 GPU。运行命令与结果记录约定见[贡献指南](../CONTRIBUTING.md)。

| 文件 | 关注的行为 |
|---|---|
| [test_core.py](test_core.py) | schema、配置、注册、指标、决策与产物完整性 |
| [test_distribution.py](test_distribution.py) | wheel/sdist 的分发范围、必需模块与禁止内容 |
| [test_nfcorpus.py](test_nfcorpus.py) | 合成 NFCorpus 数据准备、split 与 qrels、BM25 和输入校验 |
| [test_attention.py](test_attention.py) | tree 可见性、逻辑位置和无效输入拒绝 |
| [test_model.py](test_model.py) | yes/no 模型的 pairs/tree 分数、梯度、候选换序与重载 |
| [test_score_head.py](test_score_head.py) | 独立 ScoreHead、读出、布局与保存加载 |
| [test_training.py](test_training.py) | 损失、采样、full/LoRA 更新、完整 dev 选优、checkpoint 恢复契约 |
| [test_cli.py](test_cli.py) | 数据 CLI 与 tiny 训练、预测、迁移加载、评测及产物完整性 |
| [test_retrieval.py](test_retrieval.py) | pooling 与替身 embedding 的固定检索结果 |
| [test_task_model_reports.py](test_task_model_reports.py) | 无模型推理的评测协议、报告比较与拒绝条件 |
| [test_artifact_evaluator_integration.py](test_artifact_evaluator_integration.py) | tiny ScoreHead 真实前向、完整 qrels、续评幂等和篡改拒绝 |
| [test_readout_ablation.py](test_readout_ablation.py) | 冻结表示研究的真实优化、dev 选优、指标与研究头重载 |
| [test_boolq_feature_protocol.py](test_boolq_feature_protocol.py) | 研究数据的 passage 隔离与不依赖标签的固定抽样 |
| [test_boolq_lora_ablation.py](test_boolq_lora_ablation.py) | 联合 LoRA 研究的梯度、参数更新、基座冻结与独立格式重载 |
| [test_boolq_readout_protocol.py](test_boolq_readout_protocol.py) | 第二轮研究的标签盲采样、历史样本排除和冻结协议 |
| [test_boolq_readout_model.py](test_boolq_readout_model.py) | 批处理隔离、共同正文预算、token 边界与候选池化 mask |
| [test_boolq_readout_training.py](test_boolq_readout_training.py) | 真实 tiny LoRA 更新、选定模型重载与全部选优冻结后评测 |
| [test_boolq_readout_summary.py](test_boolq_readout_summary.py) | 完整证据校验、开发集选择、预测指标重算与配对区间 |

模型相关测试在缺少可选依赖时可能跳过，测试报告应注明跳过项。测试产物写入临时目录；回归测试使用独立 fixture，不依赖外部训练 run 或真实留出数据。

目前保持扁平测试目录，便于按模块选择。新增独立子系统形成稳定边界后再引入分组；迁移前核对现有脚本和 fixture 的相对路径依赖。
