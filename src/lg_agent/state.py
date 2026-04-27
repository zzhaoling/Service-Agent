from operator import add
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# ---------- Task 定义（简化版）----------
# class Task(BaseModel):
#     """子任务，由 Planner 生成"""
#     description: str = Field(..., description="子任务的自然语言描述")
#     tool: str = Field(..., description="要使用的工具名称，如 'predefined_cypher', 'graphrag_query'")
#     parameters: Dict[str, Any] = Field(default_factory=dict, description="工具参数")

# ---------- 历史记录（保留原逻辑）----------
class CypherInputState(TypedDict):
    # task: str
    task: Annotated[list, add]


class CypherState(TypedDict):
    # task: str
    task: Annotated[list, add]
    statement: str
    parameters: Optional[Dict[str, Any]]
    errors: List[str]
    records: List[Dict[str, Any]]
    next_action_cypher: str
    attempts: int
    steps: Annotated[List[str], add]
    
class CypherOutputState(TypedDict):
    # task: str
    task: Annotated[list, add]
    statement: str
    parameters: Optional[Dict[str, Any]]
    errors: List[str]
    records: List[Dict[str, Any]]
    steps: List[str]
    
class CypherHistoryRecord(TypedDict):
    """简化表示一次 Cypher 查询历史"""
    task: str
    statement: str
    records: List[Dict[str, Any]]

class HistoryRecord(TypedDict):
    """用户历史问答记录"""
    question: str
    answer: str
    cyphers: List[CypherHistoryRecord]

def update_history(history: List[HistoryRecord], new: List[HistoryRecord]) -> List[HistoryRecord]:
    """更新历史记录，最多保留 SIZE 条"""
    SIZE: int = 5
    history.extend(new)
    return history[-SIZE:]

class Task(BaseModel):
    question: str = Field(..., description="The question to be addressed.")
    parent_task: str = Field(
        ..., description="The parent task this task is derived from."
    )
    requires_visualization: bool = Field(
        default=False,
        description="Whether this task requires a visual to be returned.",
    )
    data: Optional[CypherOutputState] = Field(
        default=None, description="The Cypher query result details."
    )

# ---------- 工作流状态（使用简化字段）----------
class InputState(TypedDict):
    """子图输入状态"""
    question: str
    data: List[Dict[str, Any]]
    history: Annotated[List[HistoryRecord], update_history]

class OverallState(TypedDict):
    """The main state in multi agent workflows."""

    question: str
    tasks: Annotated[List[Task], add]
    next_action: str
    cyphers: Annotated[List[CypherOutputState], add]
    summary: str
    steps: Annotated[List[str], add]
    history: Annotated[List[HistoryRecord], update_history]


class OutputState(TypedDict):
    """The final output for multi agent workflows."""

    answer: str
    question: str
    steps: List[str]
    cyphers: List[CypherOutputState]
    history: Annotated[List[HistoryRecord], update_history]

# ---------- 其他辅助状态（保留但简化）----------
class TaskState(TypedDict):
    """单个任务的状态（用于 Map 节点）"""
    question: str
    parent_task: str
    requires_visualization: bool
    data: Dict[str, Any]        # 简化
    visualization: Any

class PredefinedCypherInputState(TypedDict):
    """预定义 Cypher 节点的输入状态"""
    task: str
    query_name: str
    query_parameters: Dict[str, Any]
    steps: List[str]

class ToolSelectionInputState(TypedDict):
    """工具选择节点的输入状态"""
    question: str
    parent_task: str
    context: Any

class ToolSelectionOutputState(TypedDict):
    tool_selection_task: str
    tool_call: Optional[Any]    # 避免引入 ToolCall
    steps: List[str]

class ToolSelectionErrorState(TypedDict):
    """工具选择错误处理状态"""
    task: str
    errors: List[str]
    steps: List[str]

