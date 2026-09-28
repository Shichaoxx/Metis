# 参与 Metis 开发

Metis 使用 Python src 布局。包与 CLI 名称为 cometa。

## 开发环境

Python 3.10+。核心数据和配置接口不依赖 Torch；模型训练与推理需要 train，LoRA 需要 peft，pytest 需要 test。

在虚拟环境中执行：python -m pip install -e '.[train,peft,test]'，然后运行 python scripts/check_repository.py 和 pytest -q。

## 代码与文档

- 可复用实现放在 src/cometa/，示例放在 examples/，配置放在 recipes/，任务协议放在 cookbooks/。
- 数据、模型权重、checkpoint 和运行日志不得提交到源码仓库。
- 保留 ACKNOWLEDGEMENTS.md 中的来源说明，不复制企业数据、代码或第三方课件。
- 新能力应明确实现状态、验证范围和未覆盖边界；tiny 测试不代表真实任务质量。

请在 pull request 中说明问题、行为变化、验证命令和风险边界。
