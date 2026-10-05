# 目录说明

Metis 只维护一套当前源码和一个 `main` 分支，统一在仓库根目录编辑和提交。

| 位置 | 职责 |
|---|---|
| `README.md` | 项目介绍、安装与入口 |
| `CONTRIBUTING.md` | 开发环境、检查与贡献约定 |
| `ACKNOWLEDGEMENTS.md` | 方法与教学来源 |
| `pyproject.toml` / `MANIFEST.in` | 依赖、CLI、构建和分发范围 |
| `src/metis/` | 可安装运行库、训练与推理接口 |
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

数据、权重、checkpoint、原始报告、日志、历史资料和服务器运维文件保存在本地，不随源码分发。维护备份放 `.local/`，不作为另一套开发版本。

新增可复用代码放 `src/metis/`，调用示例放 `examples/`；训练参数放 `recipes/`，任务过程放 `cookbooks/`。文档按职责归档，当前进度只维护 `docs/project/status.md`。

检查命令见[贡献指南](../../CONTRIBUTING.md)。
