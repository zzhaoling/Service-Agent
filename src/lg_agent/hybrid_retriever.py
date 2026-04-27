import re
import os
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseChatModel
from lg_agent.text2cypher.cypher_create import create_text2cypher_agent
from lg_agent.retrievers.base import BaseCypherExampleRetriever
from lg_agent.state import InputState
from lg_agent.state import CypherInputState

class HybridRetriever:
    """
    混合检索器：根据问题自动选择图查询（Cypher）或向量检索，并融合结果。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        graph: Neo4jGraph,
        cypher_example_retriever: BaseCypherExampleRetriever,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_index_name: str = "manual_section_vectors",
    ):
        self.llm = llm
        self.graph = graph
        self.cypher_example_retriever = cypher_example_retriever

        # 初始化 embedding 模型
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 连接已有的向量索引
        self.vector_store = Neo4jVector.from_existing_index(
            embedding=self.embedding,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
            index_name=vector_index_name,
            text_node_property="content",
            embedding_node_property="embedding",
        )

    def _route(self, query: str) -> str:
        """
        简单的关键词路由，返回 "graph", "vector", "both"
        """
        graph_kw = ["品牌", "型号", "能效", "容量", "参数", "列表", "多少", "哪些", "价格", "功率"]
        vector_kw = ["怎么", "如何", "故障", "错误", "清洁", "安装", "保养", "代码", "步骤", "解决", "为什么"]

        is_graph = any(kw in query for kw in graph_kw)
        is_vector = any(kw in query for kw in vector_kw)

        if is_graph and is_vector:
            return "both"
        elif is_graph:
            return "graph"
        elif is_vector:
            return "vector"
        else:
            return "both"  # 默认两者都查

    async def _cypher_search(self, query: str) -> str:
        """调用 text2cypher agent 执行图查询（Cypher），返回结果文本。"""
        try:
            agent = create_text2cypher_agent(
                llm=self.llm,
                graph=self.graph,
                cypher_example_retriever=self.cypher_example_retriever,
                llm_cypher_validation=True,
                max_attempts=2,
                attempt_cypher_execution_on_final_attempt=True,
            )
            input_state = CypherInputState(task=[query])
            result = await agent.ainvoke(input_state)
            # 提取答案
            if "answer" in result and result["answer"]:
                return str(result["answer"])
            if "summary" in result and result["summary"]:
                return str(result["summary"])
            if "cyphers" in result and result["cyphers"]:
                records = result["cyphers"][0].get("records", [])
                if records:
                    lines = []
                    for rec in records[:10]:
                        lines.append(str(rec))
                    return "\n".join(lines)
            return "未查询到相关信息"
        except Exception as e:
            return f"图查询出错: {str(e)}"

    def _vector_search(self, query: str) -> str:
        """向量检索，返回相关章节内容。"""
        docs = self.vector_store.similarity_search(query, k=3)
        if not docs:
            return "未在说明书中找到相关信息"

        parts = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get("title", f"段落{i+1}")
            content = doc.page_content[:1000]  # 限制长度
            parts.append(f"【{title}】\n{content}")
        return "\n\n".join(parts)

    async def retrieve(self, query: str) -> str:
        """混合检索主入口。"""
        mode = self._route(query)
        print(f"[HybridRetriever] 路由模式: {mode}")

        graph_result = ""
        vector_result = ""

        if mode in ("graph", "both"):
            graph_result = await self._cypher_search(query)
        if mode in ("vector", "both"):
            vector_result = self._vector_search(query)

        if not graph_result and not vector_result:
            return "抱歉，我没有找到相关信息。请尝试更具体的问题。"

        # 融合 prompt
        fusion_prompt = f"""
        用户问题：{query}

        【产品/参数信息】（可能为空）
        {graph_result}

        【说明书信息】（可能为空）
        {vector_result}

        请综合以上信息，给出准确、有帮助、亲切的回答。如果某个部分为空，请忽略它。
        """
        response = await self.llm.ainvoke(fusion_prompt)
        # 确保返回字符串
        content = response.content
        if isinstance(content, list):
            # 处理多模态内容，提取文本
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
                else:
                    texts.append(str(part))
            return "\n".join(texts)
        return str(content)