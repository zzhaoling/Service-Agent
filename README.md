# 智能客服系统 - Service Agent

基于 LangGraph、Neo4j和 DeepSeek/Qwen 构建的多轮对话智能客服系统，支持结构化产品查询、非结构化说明书问答，并具备多层幻觉检测机制。

## ✨ 主要功能

- **意图识别与路由**：自动识别用户意图（产品咨询、售后服务、混合问题），选择最佳处理路径。
- **结构化产品查询（Cypher）**：通过 Neo4j 存储的产品信息（品牌、型号、能效、容量等），利用 Text2Cypher Agent 实现自然语言转图查询。
- **非结构化知识检索（Vector）**：将 PDF 说明书拆分为结构化章节并生成向量索引，支持语义检索。
- **混合检索（GraphRAG）**：同时利用图结构和向量检索，融合两个知识源的回答。
- **多层幻觉检测**：知识溯源、数值一致性校验、Qwen 小模型裁判，降低生成幻觉。

## 📦 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 图数据库 | Neo4j |
| 向量存储 | Neo4j Vector Index |
| LLM 基础 | DeepSeek / OpenAI 兼容 API |
| 嵌入模型 | all-MiniLM-L6-v2 |
| 裁判模型 | Qwen2.5-7B-Instruct |
| 工作流编排 | LangGraph |
| PDF 解析 | PyMuPDF |
| 文档索引 | LangChain + Sentence-Transformers |

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- Neo4j 数据库（5.x 及以上，需安装 GDS 库）
- 可选：SiliconFlow API Key（用于 Qwen 裁判）

### 2. 克隆项目

```bash
git clone https://github.com/your-username/assistgen.git
cd assistgen

**### 3. 安装依赖**

**### 4. 配置环境变量**
