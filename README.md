# Metis · 墨提斯

Metis 面向需要在 Agent 和业务流程中使用专用模型的开发者，提供排序、选择和分类任务的后训练、评测与调用工具。

目前主要适配 Qwen3：Base 模型可以接独立 ScoreHead，Reranker 则沿用 yes/no 读出。Python 包和命令行统一使用 `metis`。

## 能做什么

- 用 full fine-tuning 或 LoRA 训练模型，再按 validation 指标选择 checkpoint。
- 使用 pairs 或 dense tree 输入，支持 eager 和 SDPA。当前 tree 使用 dense mask，主要用于核对计算语义，还没有稀疏加速。
- 保存和加载训练产物，通过 Predictor 或开发用 HTTP 接口调用。
- 参考 NFCorpus 配方完成训练、评测，并将模型接入 Agent workflow。

通用任务与模型注册接口、稀疏 attention、强化学习和多卡训练目前还没有实现。

## NFCorpus 首轮结果

我们用 NFCorpus 跑完了首轮 ScoreHead 微调，包括训练、模型选择、重载和一次 Agent workflow 调用。

| 模型 | Test nDCG@10 |
|---|---:|
| Qwen3-Reranker-0.6B | 0.3548 |
| Qwen3-0.6B-Base + ScoreHead + LoRA | 0.2910 |

ScoreHead 这次没有超过 reranker。两边使用不同的读出、输入模板和训练设置，这组分数只能作为当前方案的阶段性对比。

## 安装

需要 Python 3.10 或更新版本。安装训练依赖后查看命令行帮助：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,peft]"
metis --help
```

## 调用模型

```python
from metis import Predictor

predictor = Predictor("path/to/run/exports/best")
results = predictor.predict(samples, top_k=5)
```

样本格式见[使用指南](docs/usage/getting-started.md)。

## 文档

- [安装与使用](docs/usage/getting-started.md)
- [NFCorpus cookbook](cookbooks/score-head-plan.md)
- [模型与 ScoreHead](docs/architecture/score-head.md)
- [张量教程](docs/learning/tensors.md)
- [示例](examples/README.md)
- [贡献指南](CONTRIBUTING.md)

## 许可与致谢

仓库目前没有 LICENSE 文件，源码再分发许可尚未设定。NFCorpus、Qwen 模型和软件依赖沿用各自许可。Tree Mask 的学习参考见[来源与致谢](ACKNOWLEDGEMENTS.md)。
