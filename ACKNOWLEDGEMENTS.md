# 来源、致谢与版权边界

Metis 的问题起点是作者在列排序任务中的工程经验；本项目对 Tree Mask Attention、共享前缀、分支位置编码及张量化讲解的探索，**明确参考了 Bilibili 创作者「五道口纳什」的视频和其 GitHub 教学资料**。感谢其对理解问题与组织讲解的启发。这项署名不是对 tree attention 算法首创者的判定；相关论文与已有实现另见 [参考资料](docs/research/references.md)。

## 五道口纳什的教学来源

- 视频：[Bilibili「五道口纳什」作者主页](https://space.bilibili.com/59807853)，参考其关于 Jev 架构与 Tree Mask 的讲解。作者主页链接经[公开博主索引](https://github.com/kaixindelele/2025-Awesome-AI-Bloggers/blob/78852b28d7dfb1b48719984a2ece7f911b3a176e/README.md)定位，主页不是具体视频链接。该参考关系由项目作者的学习经历确认；**具体视频标题、发布日期与 BV 链接尚待核实补齐**，不填写推测的视频编号。
- GitHub：[wdkns/modern_genai_bilibili](https://github.com/wdkns/modern_genai_bilibili)。链接来自本地克隆的 `origin`，本次远端读取未成功，不将其写成在线状态已验证。
- 参考课件：[slides/jev_architecture-v2.pdf](https://github.com/wdkns/modern_genai_bilibili/blob/6b8729b923d17863014ac6e0ff7715fc84473300/slides/jev_architecture-v2.pdf)。记录本地参考版本 `6b8729b923d17863014ac6e0ff7715fc84473300`，便于后续定位，不把后续更新混入同一来源版本。

## 本项目做了什么

| 内容 | 来源关系与处理 |
|---|---|
| Jev / Tree Mask 的问题提出与教学组织 | 参考五道口纳什的视频与课件，保留署名和来源；其中对 Jev 内部结构的推测仍是推测 |
| token 例子、矩阵、流程图和 PDF 绘图代码 | 本项目重新构造并独立绘制；教学表达受上述资料启发，不将参考思路表述为本项目首创 |
| tree mask、独立逻辑位置、共享前缀计算 | 结合公开方法独立实现；另引用 RoPE、Hydragen、Flash Preference 等已有工作 |
| yes/no 读出与输入模板 | 参考 Qwen 官方 Reranker 示例，保留原始预训练能力和来源说明 |
| Qwen3 Base + ScoreHead | 基座权重来自 [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，使用固定 revision；MLP 与候选读出是围绕列排序经验实现的任务适配，不声称首创使用 decoder 做判别任务。模型权重条款独立于本项目代码许可 |
| 训练、保存、LoRA | 使用 Transformers Trainer / PEFT 等公开依赖；不声称这些通用方法由本项目提出 |
| 列排序业务示例 | 从个人工程经验抽取通用要求；新项目使用自行编写的合成示例，没有复制企业代码或数据 |

## 权利与分发

原视频、原课件及其中素材的版权归原作者及相关权利人所有。致谢与链接不等于获得转载、改编或再许可授权，也不表示原作者认可本项目。检查上述本地 Git 版本时，未发现明确的 `LICENSE`、`COPYING` 或 `NOTICE`；因此不把该教学仓库默认视为已获 MIT / Apache 等开源授权。

本项目目录不附带原视频、原课件 PDF、截图或其图像素材，只提供来源链接和独立实现。若以后需要引入这些内容，应先核实对应许可或取得所需授权，并保留适用的版权说明。本项目尚处于本地私有阶段；未来选定的代码许可证只覆盖有权授权的部分，不自动覆盖第三方素材、数据与模型权重。

NFCorpus 数据条款、Qwen 模型条款及软件依赖许可证分别适用，见 [cookbook](cookbooks/nfcorpus.md) 和 [参考资料](docs/research/references.md)。公开数据缓存和训练产物由 Git 忽略。
