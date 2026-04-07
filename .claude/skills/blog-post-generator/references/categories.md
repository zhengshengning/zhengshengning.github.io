# 博客分类映射表

## 分类体系

| 一级分类 | 二级分类 | 适用主题 | 已有文章 |
|---------|---------|---------|---------|
| AI Infra | 学习路线 | AI Infra 整体学习路径、知识图谱 | AI Infra学习路线 |
| AI Infra | 前置知识 | Transformer架构、PyTorch框架、线性代数、概率论、Python/C++基础、Linux | 编程语言入门、Transformer架构入门、PyTorch框架入门、通信拓扑入门、AI Infra工程师为什么必须懂Transformer、Transformer全貌及代码实现、Tokenization与词嵌入、Self-Attention机制深入理解、Transformer前馈网络FFN深入理解、Transformer位置编码深入理解、LayerNorm与残差连接深入理解、Transformer Decoder Block完整解析、从Transformer到LLM自回归生成深入理解 |
| AI Infra | CUDA编程与算子优化 | CUDA编程、算子开发、Kernel优化、FlashAttention、Triton、GPU架构 | CUDA编程入门指南、高效CUDA编程速查、Thread Block Cluster 架构特性、GPU架构与存储体系 |
| AI Infra | 分布式训练 | DDP、FSDP、3D并行、ZeRO、DeepSpeed、Megatron-LM | 分布式训练入门 |
| AI Infra | 推理与部署 | LLM推理、vLLM、SGLang、TensorRT-LLM、量化、KV Cache | 大模型推理与部署入门 |
| AI Infra | 性能分析 | Nsight Systems、Nsight Compute、Profiling、Roofline、Benchmark | Nsight Systems性能分析实战指南、Nsight Compute性能分析实战指南 |
| 路飞玩AI | AI编程 | AI辅助编程、Claude Code、Copilot、AI IDE、Agent Skills | ClaudeCode使用指南、OpenCode入门使用指南、如何创建自定义Agent Skills、Claude多智能体系统构建指南 |
| 路飞玩AI | Agent开发 | AI Agent框架、多智能体系统、Agent SDK | Agent框架设计与实现 |
| 编程技能包 | Python | Python开发、包管理、CLI工具、uv、pip、PyPI | 使用uv构建Python命令行工具、构建并发布Python包到PyPI完整指南 |
| 编程技能包 | C++基础 | C++语法、STL、面向对象、现代C++ | C++入门教程 |
| 编程技能包 | Web开发 | Next.js、React、前端开发、Web框架 | Nextjs网页开发入门指南 |

## 选择规则

1. 每篇文章只属于一个 `[一级分类, 二级分类]` 组合
2. 优先根据文章核心技术领域选择，而非辅助工具
3. 若内容跨领域，选择最核心的分类，其他领域用 tags 标注
4. 无法匹配时询问用户
