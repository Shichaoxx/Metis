# Cookbook 索引

Cookbook 记录具体任务从数据、训练到导出、评测与调用的完整协议。Metis 的目标是复用这条链路；NFCorpus 是首个实例，后续任务仍需独立定义输入、监督、损失和验收。

| 文档 | 当前状态 | 阅读用途 |
|---|---|---|
| [score-head-plan.md](score-head-plan.md) | **已完成真实后训练**：Base + 独立 ScoreHead + LoRA；训练、dev 选优、完整 test、重载和 workflow 已完成 | 首轮冻结协议、数据边界、参数设置与实际结果 |
| [nfcorpus.md](nfcorpus.md) | **混合历史状态**：BM25 和原始 reranker 基线已运行；yes/no 微调及 embedding top100 路线未运行 | 理解数据准备、原始基线与保留的备选模板 |

首轮工程闭环已完成，模型质量仍落后同协议强基线；它没有证明 tree 加速或通用框架扩展已完成。完整证据入口见 [项目进度](../docs/project/status.md)，配置入口见 [recipes](../recipes/README.md)。不要因为文档包含命令而重新执行已经完成的训练或 test。

## 新 cookbook 应写清什么

1. **任务与数据**：输入、候选 ID、标签含义、数据来源与许可、固定 split、弱负例及未标注样本处理。
2. **可复现配置**：对应 recipe、基座版本、输入预算、布局、目标、采样、训练预算和环境。
3. **选优与评测**：基线、完整 dev 的选择规则、冻结后的 test 协议；限制、未知项和负结果也保留。
4. **训练与交付证据**：实际参数更新、日志、checkpoint 恢复依赖、封存产物及独立加载结果。
5. **调用与范围**：使用训练产物的具体示例，区分接口可调用与下游业务效果已验证。

通用接口说明放在 [框架文档](../docs/architecture/framework.md)，学习材料放在 [张量说明](../docs/learning/tensors.md)。大量运行产物保留在数据工作区，cookbook 链接证据并解释协议，不复制 checkpoint 或原始数据。
