"""
综合测试脚本：验证 GraphRAG + 幻觉检测的完整链路
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 调整路径，确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------- 导入项目模块 ----------
from lg_builder import classify_intent, get_llm
from lg_agent.neo4j_client import get_neo4j_graph
from lg_agent.retrievers.northwind_retriever import NorthwindCypherRetriever
from lg_agent.hybrid_retriever import HybridRetriever
from lg_agent.hallucination_detector import HallucinationDetector
from lg_agent.text2cypher.cypher_create import create_text2cypher_agent
from lg_agent.state import InputState
from lg_agent.state import CypherInputState

# ---------- 配置 ----------
TEST_QUESTIONS = {
    "product": "能效5星的洗衣机有哪些？",
    "manual": "如何清洁泵过滤器？",
    "hybrid": "海尔8kg洗衣机怎么清洁？",
    "error_code": "E4错误代码是什么意思？",
}

# ---------- 1. 测试意图识别 ----------
async def test_intent_recognition():
    print("\n=== 测试意图识别 ===")
    for name, q in TEST_QUESTIONS.items():
        result = await classify_intent([{"role": "user", "content": q}], temperature=0.7)
        print(f"问题({name}): {q}\n  意图: {result.type}\n  逻辑: {result.logic}\n")

# ---------- 2. 测试商品查询 (text2cypher) ----------
async def test_product_query():
    print("\n=== 测试商品查询 (Cypher) ===")
    llm = get_llm(temperature=0.2)
    graph = get_neo4j_graph()
    retriever = NorthwindCypherRetriever()  
    agent = create_text2cypher_agent(
        llm=llm,
        graph=graph,
        cypher_example_retriever=retriever,
        llm_cypher_validation=True,
        max_attempts=2,
    )
    question = TEST_QUESTIONS["product"]
    input_state = CypherInputState(task=[question])
    result = await agent.ainvoke(input_state)
    print(f"问题: {question}")
    print(f"答案: {result.get('answer', '无答案')}\n")

# ---------- 3. 测试说明书向量检索 ----------
async def test_manual_search():
    print("\n=== 测试说明书向量检索 ===")
    graph = get_neo4j_graph()
    llm = get_llm(temperature=0.0)
    retriever = NorthwindCypherRetriever()  # 向量检索不需要这个，但HybridRetriever要求传入
    hybrid = HybridRetriever(
        llm=llm,
        graph=graph,
        cypher_example_retriever=retriever,
        embedding_model="all-MiniLM-L6-v2",
        vector_index_name="manual_section_vectors"
    )
    question = TEST_QUESTIONS["manual"]
    # 直接调用向量检索部分
    result = hybrid._vector_search(question)
    print(f"问题: {question}")
    print(f"检索结果:\n{result[:500]}...\n")

# ---------- 4. 测试混合检索 ----------
async def test_hybrid_retrieval():
    print("\n=== 测试混合检索 (GraphRAG) ===")
    graph = get_neo4j_graph()
    llm = get_llm(temperature=0.2)
    retriever = NorthwindCypherRetriever()
    hybrid = HybridRetriever(
        llm=llm,
        graph=graph,
        cypher_example_retriever=retriever,
    )
    for name, q in TEST_QUESTIONS.items():
        print(f"\n[{name}] 问题: {q}")
        answer = await hybrid.retrieve(q)
        print(f"答案: {answer}\n")

# ---------- 5. 测试幻觉检测 ----------
async def test_hallucination_detection():
    print("\n=== 测试幻觉检测 ===")
    graph = get_neo4j_graph()
    llm = get_llm(temperature=0.0)
    # 模拟一些证据（从之前的检索结果中取，这里构造几个）
    evidences = [
        "HW80-B14979 能效等级为 A+++，容量 8kg，年耗电 111 kWh。",
        "清洁泵过滤器：打开底部面板，旋转过滤器，清除杂物后装回。",
        "E4 错误代码：水位未达到，请检查水龙头和水压。"
    ]
    detector = HallucinationDetector(
        graph=graph,
        llm=llm,
        qwen_config={
            "model_id": os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
            "base_url": os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1"),
            "api_key": os.getenv("QWEN_API_KEY", None),
        }
    )
    # 测试一个有幻觉的答案
    question = "如何清洁泵过滤器？"
    answer_correct = "打开底部面板，旋转过滤器，清除杂物后装回。"
    answer_hallucinated = "需要拆开整个机器，使用漂白剂清洗。"
    
    print("正确答案检测:")
    result = await detector.detect(question, answer_correct, evidences)
    print(f"通过: {result['passed']}, 详情: {result['details']}")
    
    print("\n幻觉答案检测:")
    result = await detector.detect(question, answer_hallucinated, evidences)
    print(f"通过: {result['passed']}, 详情: {result['details']}")

# ---------- 主函数 ----------
async def main():
    print("开始执行综合测试...")
    await test_intent_recognition()
    await test_product_query()
    await test_manual_search()
    await test_hybrid_retrieval()
    await test_hallucination_detection()
    print("\n所有测试完成！")

if __name__ == "__main__":
    asyncio.run(main())