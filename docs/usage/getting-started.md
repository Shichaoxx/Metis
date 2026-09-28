# 开始使用

Metis 当前包与 CLI 名称为 `cometa`。先看 [能力与进度](../project/status.md)，再阅读对应 cookbook。

## 环境与安装

Python 3.10+。在新环境中安装核心接口：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cometa --help
```

核心数据、配置、注册和指标模块不依赖 Torch。训练、模型推理和 LoRA 使用可选依赖：

```bash
python -m pip install -e '.[train,peft]'
```



软件验证与 tiny 示例见 [测试说明](../development/testing.md)；真实任务的运行步骤见 [cookbook 索引](../../cookbooks/README.md)。

## 数据契约

每个样本包含输入与具有稳定 ID 的候选；推理时不需要监督字段。

```json
{
  "schema_version": "1.0",
  "id": "query-001",
  "task_id": "schema_linking",
  "input": {"query": "统计城市订单数量", "context": ""},
  "candidates": [
    {"id": "orders.city", "text": "订单所在城市"},
    {"id": "orders.amount", "text": "订单金额"}
  ],
  "supervision": {
    "kind": "candidate_labels",
    "labels": {"orders.city": 1, "orders.amount": 0},
    "unjudged_policy": "ignore"
  }
}
```

任务种类由 manifest/config 定义，`task_id` 标识业务任务，尚不是通用插件入口。未知标签不自动等于负例；labels、metadata、gold qrels 不进入模型文本。

数据集 manifest 保存 split 路径、数量及内容哈希；相对路径按 manifest 解析。配置相对自身位置解析，未知字段报错。已有冻结数据不重复准备或覆盖；新数据流程见 [NFCorpus 数据说明](../../cookbooks/nfcorpus.md)。

## 使用训练产物

```python
from cometa import Predictor

sample = {
    "schema_version": "1.0",
    "id": "query-001",
    "task_id": "schema_linking",
    "input": {"query": "统计城市订单数量", "context": ""},
    "candidates": [
        {"id": "orders.city", "text": "订单所在城市"},
        {"id": "orders.amount", "text": "订单金额"},
    ],
}
predictor = Predictor("/absolute/path/to/run/exports/best")
result = predictor.predict([sample], top_k=1)[0]
```

这是合成列选择的调用格式，具体效果取决于已加载产物的任务训练与验证。模型返回候选分数、ID 与决定，调用方负责后续 workflow。首轮 NFCorpus 产物的已验证流程是排序后构建证据上下文；没有验证通用 LLM 最终答案质量。

命令行示例：

```bash
python examples/agent_tool.py --artifact /absolute/path/to/exports/best
cometa serve --artifact /absolute/path/to/exports/best
```

HTTP 是单 worker 开发服务：`POST /v1/decisions` 接收 `samples` 和可选 `policy`，`GET /health` 检查加载状态。加载依赖、封存规则和 checkpoint 恢复见 [产物说明](artifacts.md)。
