# 参与开发

欢迎提交问题反馈和代码改进。开始前，可以先从 README 和相关模块文档了解项目结构。

## 开发环境

需要 Python 3.10 或更新版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,peft,test]"
```

## 检查改动

```bash
python scripts/check_repository.py
pytest -q
```

运行库位于 src/cometa，调用示例在 examples，训练配置在 recipes，任务流程和评测说明在 cookbooks。数据集缓存、模型权重、checkpoint 和运行日志不要提交到仓库。

提交改动时，请说明改了什么、如何验证，并保留 ACKNOWLEDGEMENTS.md 中的来源说明。不要提交企业代码、业务数据或未经许可的第三方课件。
