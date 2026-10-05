# 参与开发

欢迎提交问题反馈和代码改进。项目结构见[目录说明](docs/development/repository-layout.md)。

## 开发环境

需要 Python 3.10 或更新版本。在新环境中安装开发依赖；已有环境直接复用，按需补齐依赖，保持现有训练栈版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,peft,test]"
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
