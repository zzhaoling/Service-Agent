import os
import logging
from pydantic import BaseModel, Field, SecretStr
from typing import cast, Literal, Dict, List, Any, Optional
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

from lg_state import AgentState, InputState, Router, GradeHallucinations

from lg_agent.neo4j_client import get_neo4j_graph
from lg_agent.text2cypher.schema_utils import retrieve_and_parse_schema_from_graph_for_prompts
from lg_agent.retrievers.northwind_retriever import NorthwindCypherRetriever
from multi_tool import create_multi_tool_workflow
from lg_agent.hallucination_detector import HallucinationDetector

from lg_prompts import (
    GENERAL_QUERY_SYSTEM_PROMPT,
    GET_ADDITIONAL_SYSTEM_PROMPT,
    GUARDRAILS_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    ROUTER_SYSTEM_PROMPT,   
)
qwen_config = {
    "model_id": os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
    "base_url": os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1"),
    "api_key": os.getenv("QWEN_API_KEY", None),
}

class AdditionalGuardrailsOutput(BaseModel):
    """格式化输出，用于判断用户的问题是否与图谱内容相关"""
    decision: Literal["end", "continue"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )
    
logger = logging.getLogger(__name__)
def get_llm(temperature: float = 0.7, tags: Optional[List[str]] = None) -> ChatDeepSeek:
    """统一创建 DeepSeek 模型实例，所有节点都用这个"""
    api_key_str = os.getenv("LLM_API_KEY")
    # 将普通字符串转换为 SecretStr，如果未设置则为 None
    api_key = SecretStr(api_key_str) if api_key_str else None
    
    return ChatDeepSeek(
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        temperature=temperature,
        tags=tags or [],
    )

# ---------- 意图识别核心函数（与节点分离，便于测试和不确定性采样）----------
async def classify_intent(messages: List[Dict[str, str]], temperature: float = 0.7) -> Router:
    """
    对对话历史进行意图分类。
    messages 格式: [{"role": "user", "content": "..."}, ...]
    """
    model = get_llm(temperature=temperature, tags=["intent_router"])
     # 使用 json_mode 强制输出 JSON 并自动解析为 Pydantic 模型
    structured_model = model.with_structured_output(Router, method='json_mode')
    full_messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + messages
    result = await structured_model.ainvoke(full_messages)
    # 防御性处理：如果结果仍是字典，手动构造 Router 实例
    if isinstance(result, dict):
        result = Router(**result)
    return result

# ---------- LangGraph 节点：意图识别 ----------
async def analyze_and_route_query(state: AgentState, config: RunnableConfig) -> Dict[str, Router]:
    """LangGraph 节点：将 state.messages 转换为字典并调用 classify_intent"""
    conv = []
    for m in state.messages:
        if isinstance(m, BaseMessage):
            conv.append({"role": m.type, "content": m.content})
        else:
            conv.append(m)   # 假设已是 dict
    
    router_result = await classify_intent(conv, temperature=0.7)
    return {"router": router_result}
    
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    logger.info("-----generate general-query response-----")
    model = get_llm(temperature=0.7, tags=["general_query"])
    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(logic=state.router.logic)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def get_additional_info(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    logger.info("------continue to get additional info------")
    model = get_llm(temperature=0.7, tags=["additional_info"])
    # 如果用户的问题是电商相关，但与自己的业务无关，则需要返回"无关问题"

    # 首先连接 Neo4j 图数据库
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    # 定义电商经营范围
    scope_description = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）
    
    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """
    
    # 动态从 Neo4j 图表中获取图表结构
    graph_context = (
        f"\n参考图表结构来回答:\n{retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)}"
        if neo4j_graph is not None
        else ""
    )
    
    message_template = scope_description + graph_context + "\nQuestion: {question}"
    full_system_prompt = ChatPromptTemplate.from_messages([
        ("system", GUARDRAILS_SYSTEM_PROMPT),
        ("human", message_template),
    ])
    
    # 使用 json_mode 确保返回 Pydantic 模型
    structured_model = model.with_structured_output(AdditionalGuardrailsOutput, method='json_mode')
    guardrails_chain = full_system_prompt | structured_model
    
    last_question = state.messages[-1].content if state.messages else ""
    guardrails_output = await guardrails_chain.ainvoke({"question": last_question})
    
    # 类型安全处理（可选，因为 method='json_mode' 通常返回模型实例）
    if isinstance(guardrails_output, dict):
        guardrails_output = AdditionalGuardrailsOutput(**guardrails_output)
    
    if guardrails_output.decision == "end":
        logger.info("-----Fail to pass guardrails check-----")
        return {"messages": [AIMessage(content="抱歉，我家暂时没有这方面的商品，可以在别家看看哦~")]}
    else:
        logger.info("-----Pass guardrails check-----")
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(logic=state.router.logic)  # 注意改用点号
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = await model.ainvoke(messages)
        return {"messages": [response]}

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    logger.info("------execute local knowledge base query------")
    model = get_llm(temperature=0.7, tags=["research_plan"])
    # 1. Neo4j图数据库连接 - 使用配置中的连接信息
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    # 2. 创建自定义检索器实例，根据 Graph Schema 创建 Cypher 示例，用来引导大模型生成正确的Cypher 查询语句
    cypher_retriever = NorthwindCypherRetriever()

    # step 3. 定义工具模式列表    
    from lg_agent.kg_tools_list import cypher_query, predefined_cypher, hybrid_query
    tool_schemas = [cypher_query, predefined_cypher, hybrid_query]

    # 3. 预定义的Cypher查询 - 为电商场景定义有用的查询
    from lg_agent.cypher_dict import predefined_cypher_dict

    # 定义电商经营范围
    scope_description = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）
    
    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """

    # 创建多工具工作流
    multi_tool_workflow = create_multi_tool_workflow(
        llm=model,
        graph=neo4j_graph,
        tool_schemas=tool_schemas,
        predefined_cypher_dict=predefined_cypher_dict,
        cypher_example_retriever=cypher_retriever,
        scope_description=scope_description,
        llm_cypher_validation=True,
    )
    
    # return multi_tool_workflow
    # 准备输入状态
    last_question = state.messages[-1].content if state.messages else ""
    input_state = {"question": last_question, "data": [], "history": []}
    # 执行多工具工作流
    response = await multi_tool_workflow.ainvoke(input_state)

    answer = response["answer"]
    # 收集证据：从 response 中的 cyphers 获取记录，以及可能从向量检索中获取的文本
    evidences = []
    for cypher in response.get("cyphers", []):
        # 将 Cypher 查询结果转换为文本描述
        records = cypher.get("records", [])
        for rec in records:
            evidences.append(str(rec))
    # 如果有向量检索的文档节点，也可以通过其他方式添加，这里简化
    
    # 幻觉检测
    detector = HallucinationDetector(
    graph=neo4j_graph,
    llm=get_llm(temperature=0.0),
    qwen_config=qwen_config,
    )
    detection = await detector.detect(
        question=last_question,
        answer=answer,
        evidences=evidences,
        context={"product_models": []}   # 可扩展
    )

    if not detection["passed"]:
        # 处理失败：可以返回一个安全答案
        logger.warning(f"幻觉检测未通过: {detection['details']}")
        answer = "抱歉，我无法确认这个答案的准确性，请稍后再试或联系人工客服。"

    return {"messages": [AIMessage(content=answer)]}

async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, Any]:
    model = get_llm(temperature=0.0, tags=["hallucination_check"])
    documents = getattr(state, "documents", "无检索文档")
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=documents,
        generation=state.messages[-1].content if state.messages else ""
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))
    return {"hallucination": response}

# ---------- 路由函数 ----------
def route_query(state: AgentState) -> Literal["respond_to_general_query", "get_additional_info", "create_research_plan"]:
    if state.router.type == "general-query":
        return "respond_to_general_query"
    elif state.router.type == "additional-query":
        return "get_additional_info"
    elif state.router.type == "graphrag-query":
        return "create_research_plan"
    else:
        raise ValueError(f"Unknown router type {state.router.type}")
    

# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("respond_to_general_query", respond_to_general_query)
builder.add_node("get_additional_info", get_additional_info)
builder.add_node("create_research_plan", create_research_plan)  # 这里是子图
builder.add_node("check_hallucinations", check_hallucinations)

# 添加边
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)

# 添加
builder.add_edge("respond_to_general_query", END)
builder.add_edge("get_additional_info", END)
builder.add_edge("create_research_plan", END)

# 定义持久化存储，也可以使用SQLiteSaver()、PostgresSaver()等
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 导出 classify_intent 供外部测试脚本使用
__all__ = ["graph", "classify_intent", "get_llm"]