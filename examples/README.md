# 示例与实验入口

示例展示如何训练、评测和调用模型；可复用实现位于 `src/metis/`。

## 了解训练与工具调用

| 入口 | 用途与边界 |
|---|---|
| [smoke_train.py](smoke_train.py) | 本地随机 tiny Qwen3 的 CPU 训练、导出与重载；无需下载权重 |
| [agent_tool.py](agent_tool.py) | 将封存产物包装为结构化候选评分工具；`--show-schema` 只打印契约，不加载模型 |
| [nfcorpus_workflow.py](nfcorpus_workflow.py) | 在固定 validation 样本上调用产物，按返回 ID 组装证据上下文；不调用通用 LLM |

可先查看不依赖模型的工具契约：

```bash
python examples/agent_tool.py --show-schema
```

tiny 示例与列选择合成示例不代表真实任务质量。NFCorpus 排序模型也不能直接视为已验证的列选择模型。

## 模型实验与报告工具

| 入口 | 用途 |
|---|---|
| [check_score_head.py](benchmarks/check_score_head.py) | 真实权重的 ScoreHead 前向、梯度、pairs/tree 与重载预检 |
| [check_invariance.py](benchmarks/check_invariance.py) | 本地权重的候选换序与数值一致性检查 |
| [benchmark_layouts.py](benchmarks/benchmark_layouts.py) | 比较布局的数值、时间和内存；结果只适用于实测环境 |
| [evaluate_pretrained.py](benchmarks/evaluate_pretrained.py) | 已下载原始 reranker 的评测与严格续评 |
| [evaluate_artifact.py](benchmarks/evaluate_artifact.py) | 封存训练产物的评测与严格续评 |
| [compare_results.py](benchmarks/compare_results.py) | 读取既有报告，执行同协议比较与配对 bootstrap |
| [compare_task_models.py](benchmarks/compare_task_models.py) | 比较不同读出/模板的完整方法，显式核对公共协议 |

真实模型脚本需要已准备的权重、manifest 或产物。部分脚本默认使用 `test` 或特定设备；运行时应显式指定 split 和设备，开发调试使用 validation，test 保留给固定方案的最终评测。

实验协议见 [ScoreHead cookbook](../cookbooks/score-head-plan.md)，结果见 [项目状态](../docs/project/status.md)。每次实验使用独立输出目录；报告比较脚本读取已有结果，不执行模型推理。
