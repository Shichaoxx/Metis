# 参与开发

欢迎提交问题反馈和代码改进。项目结构见[目录说明](docs/development/repository-layout.md)。

## 开发环境

需要 Python 3.10 或更新版本。建议使用独立虚拟环境安装开发依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,peft,dev]"
```

## 检查改动

```bash
python scripts/check_repository.py
```

纯文档改动检查本地链接和配置即可。代码改动按影响范围选择[测试文件](tests/README.md)；完整软件回归命令为：

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false \
python -m pytest -q
```

测试使用合成数据、tiny 模型和临时产物，无需预训练权重、下载数据或 GPU。模型与 LoRA 测试分别需要 `train`、`peft` 依赖；缺失依赖时会跳过，报告结果时请写明运行范围及通过、失败、跳过数量。真实训练与质量评测按 [cookbook](cookbooks/README.md) 单独安排。

提交时说明改动和验证方式，保留 [来源说明](ACKNOWLEDGEMENTS.md)。数据、权重、checkpoint、原始报告和日志保存在本地；企业代码、业务数据和未经许可的第三方课件不要提交。

## 自动检查与分发

[CI](.github/workflows/checks.yml) 分为三组：无模型依赖的核心检查（Python 3.10/3.12）、离线 CPU tiny 模型与 LoRA 测试、wheel/sdist 构建检查。模型测试使用合成数据，真实 cookbook 的质量结果单独记录。

只开发数据、配置或文档时，可安装 `.[dev]`；`train` 和 `peft` 是模型相关的可选依赖。构建与检查分发包：

```bash
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist/*.whl dist/*.tar.gz
```

分发检查校验必要模块并拒绝本地运行资料、运维入口、缓存与链接。可重复指定 `--forbid-text` 检查额外的不应发布的字面字符串；PDF 和图片中的可见内容仍需单独检查。检查规则应区分公开作者署名与私人信息，不将作者姓名或署名账号本身列为禁用字符串。

## 重建架构图

论文架构图的英文与中文版本共用几何和模块关系，图中只描述已实现路径。安装 XeLaTeX（含 `ctex`/Fandol、TikZ、standalone）和 Poppler 后执行：

```bash
python scripts/build_figures.py
```

输出为可编辑 LaTeX、矢量 PDF 和 PNG；命令更新 PDF/PNG，源文件位于 `docs/architecture/figures/`。可用 `--figure metis-architecture-en` 单独导出，`--engine` 指定 XeLaTeX 路径。提交前检查中英文标签、箭头、图例和 PDF 元数据。

## 文档约定

公开文档面向首次接触项目的使用者与贡献者，应能脱离聊天记录独立阅读。说明用途、接口、使用步骤、实现状态和证据；不保留对话回复、个人指令、交接记录或内部验收口吻。示例使用可替换路径，实验结果注明数据、模型和评测协议；计划能力与已实现能力分开描述。

Metis 是个人维护的项目。公开源码、文档、示例与图表使用 Metis 项目标识，保留项目作者署名、贡献者信息及公开的作者链接；现有张量图署名为 `shichao`，图表及 PDF 元数据可包含作者署名。不写入私人联系方式、个人经历、机器用户名、主机名或实际工作目录；示例路径使用通用占位值。第三方作者署名、学术引用和来源链接应完整保留。
