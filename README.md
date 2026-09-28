# Metis · 墨提斯

**任务专用 SystemOne 后训练工具包。** 将排序、列选择、分类等明确任务，训练成可评估、导出、加载并被 Agent 调用的模型模块。

Metis 源于作者的 decoder-only 列排序经验，目标是通过配置和适配器复用任务后训练流程。当前包名、Python import 与 CLI 为 `cometa`。本仓库公开提供源码与文档。当前未附软件许可证，因此没有授予源码再分发许可；第三方数据、模型和引用材料各自遵守其原有条款。

## 能力与边界

| 能力 | 当前实现 |
|---|---|
| 模型与读出 | Qwen3 base + 独立 MLP ScoreHead；Qwen3 reranker yes/no |
| 监督训练 | full / LoRA、独立 head 学习率、混合精度、候选采样 |
| 实验管理 | 数据身份、日志、checkpoint 恢复、完整 dev 选优、best/final 导出与完整性校验 |
| 输入布局 | pairs 与 dense tree，eager / SDPA；tree 仍是正确性参考实现 |
| 调用 | Predictor、结构化候选分数/ID、Python 工具示例、开发 HTTP 服务 |

通用 task/objective/backend 注册、稀疏算子、RL、多卡和更多 backbone 尚未完成。完整状态见 [项目进度](docs/project/status.md)，产品目标与验收见 [核心约定](docs/project/core.md)。

## 开始使用

Python 3.10+。在新的开发环境中，核心数据与配置接口可直接安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cometa --help
```

模型训练和推理需要 `.[train,peft]` 可选依赖。安装、数据契约与调用见 [使用指南](docs/usage/getting-started.md)，软件检查见 [贡献指南](CONTRIBUTING.md)。

```python
from cometa import Predictor

predictor = Predictor("/absolute/path/to/run/exports/best")
result = predictor.predict([sample], top_k=5)[0]
# sample 包含 query/context 和带稳定 ID 的候选；格式见使用指南。
# 返回 scores、ranked_ids、selected_ids 等结构化结果。
```

## 第一个真实 cookbook

NFCorpus 上的 Qwen3-0.6B-Base + ScoreHead + LoRA 已完成 2 epochs、648 次更新、完整 dev 选优、选定 best 的一次完整 test、新进程重载和真实产物 workflow。

| NFCorpus test nDCG@10 | 结果 |
|---|---:|
| 原始 Qwen3-Reranker-0.6B | 0.354783 |
| 本轮 Base + ScoreHead + LoRA | 0.290978 |

工程闭环已跑通，质量落后于强基线。两者模板、读出与训练历史不同；结果不构成单因素消融或 SOTA 证据。本轮使用 pairs/SDPA，尚未验证 tree 加速。完整协议、数据边界与结果见 [ScoreHead cookbook](cookbooks/score-head-plan.md)。NFCorpus 是工具包的首个实例。

## 导航

| 位置 | 用途 |
|---|---|
| [docs/](docs/README.md) | 产品、使用、设计、教学、开发和研究文档 |
| [src/cometa/](src/cometa/) | 可安装运行库与 CLI |
| [tests/](tests/README.md) | 软件契约与 tiny 模型集成验证 |
| [examples/](examples/README.md) | 训练 smoke、Agent/Workflow 与评测脚本 |
| [recipes/](recipes/README.md) | 按任务组织的训练配置及执行状态 |
| [cookbooks/](cookbooks/README.md) | 公开任务的数据、训练、评测和调用协议 |

目录维护约定见 [仓库结构](docs/development/repository-layout.md)。数据、权重和运行输出不随源码分发。

## 来源

Tree Mask 与张量教学参考 Bilibili「五道口纳什」的视频及 [modern_genai_bilibili](https://github.com/wdkns/modern_genai_bilibili)。项目使用独立实现与原创教学例子，保留 [来源致谢与版权边界](ACKNOWLEDGEMENTS.md)；原课件、企业源码和企业数据不随项目分发。数据、模型与软件依赖分别遵守各自条款。
