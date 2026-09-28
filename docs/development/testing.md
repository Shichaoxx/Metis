# 软件检查

运行轻量仓库结构检查：python scripts/check_repository.py

代码改动按影响范围运行测试：pytest -q

涉及模型的测试可能需要安装 train 可选依赖。tiny 模型只验证软件链路，不代表真实任务的模型质量。完整任务质量评测见对应 cookbook。
