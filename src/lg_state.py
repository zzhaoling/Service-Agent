from typing import Literal, Annotated
from dataclasses import dataclass, field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class Router(BaseModel):
    
    logic: str = Field(description="Brief reasoning about why this query belongs to the chosen type")
    type: Literal["general-query", "additional-query", "graphrag-query"] = Field(
        description="The category of the user's query"
    )
    question: str = Field(
        default="",
        description="If type is 'additional-query', ask the user for missing information (e.g., order number, product model). Otherwise empty."
    )

class GradeHallucinations(BaseModel):
    """评估生成的答案是否基于事实（binary_score 为 "1" 或 "0"），用于幻觉检测或重试。"""

    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )

@dataclass(kw_only=True)
class InputState:
    """对话历史，LangGraph 的 add_messages 会自动合并新消息。"""
    messages: Annotated[list[AnyMessage], add_messages]

# @dataclass(kw_only=True)： 强制要求数据类中的所有字段必须以关键字参数的形式提供。即不能以位置参数的方式传递。
@dataclass(kw_only=True)
class AgentState(InputState):
    """意图分类结果，后续节点可根据 router.type 选择不同分支。"""
    router: Router = field(default_factory=lambda: Router(type="general-query", logic="", question=""))
    """记录处理步骤（如检索到的文档 ID、调用的工具等），便于调试和追踪。"""
    steps: list[str] = field(default_factory=list)
    """当前用户问题的清晰表述（可能经过重写或简化）。"""
    question: str = field(default_factory=str) 
    """最终生成的回答内容。"""
    answer: str = field(default_factory=str)  
    """ hallucination 评分结果。"""
    hallucination: GradeHallucinations = field(default_factory=lambda: GradeHallucinations(binary_score="0"))
