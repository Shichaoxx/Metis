# 目录说明

`main` 是当前开发分支。运行库、示例、配方和文档按职责组织：

| 位置 | 职责 |
|---|---|
| `README.md` | 项目介绍、安装与入口 |
| `CONTRIBUTING.md` | 开发环境、检查与贡献约定 |
| `ACKNOWLEDGEMENTS.md` | 方法与教学来源 |
| `pyproject.toml` / `MANIFEST.in` | 依赖、CLI、构建和分发范围 |
| `src/metis/` | 可安装运行库、训练与推理接口 |
| `src/metis/tasks/` | 内置任务的监督目标与候选决策；不负责模型加载和训练生命周期 |
| `tests/` | 软件契约与 tiny 模型测试 |
| `examples/` | 调用、训练 smoke 与评测示例 |
| `recipes/<task>/` | 按任务组织的训练配置 |
| `cookbooks/` | 任务的数据、训练、评测与调用流程 |
| `docs/project/` | 核心设计与当前状态 |
| `docs/usage/` | 数据契约、产物、恢复和使用指南 |
| `docs/architecture/` | 框架、模型与读出设计 |
| `docs/learning/` | 张量教学 |
| `docs/research/` | 参考资料与研究说明 |
| `scripts/` | 开发检查工具 |
| `.github/` | 核心、CPU 模型与分发检查，以及 PR 模板 |

数据、权重、checkpoint、原始报告、日志、历史资料和服务器运维文件保存在本地，不随源码分发。维护备份放 `.local/`，不作为另一套开发版本。

新增可复用代码放 `src/metis/`，调用示例放 `examples/`；训练参数放 `recipes/`，任务过程放 `cookbooks/`。文档按职责归档，当前进度只维护 `docs/project/status.md`。

`training.py` 管理训练、选优与恢复，监督损失位于 `tasks/objectives.py`；`api.py` 管理 Predictor，候选选择位于 `tasks/decisions.py`。旧的 `metis.training.supervised_loss` 和 `metis.api.decide` 导入路径保持有效。核心导入不加载 Torch；张量损失按需加载模型依赖。

检查命令见[贡献指南](../../CONTRIBUTING.md)。
