from langgraph.types import Send
from typing import List, Dict, Any, Literal
from .state import Task, OverallState, ToolSelectionOutputState  

def guardrails_conditional_edge(
    state: OverallState,
) -> Literal["planner", "final_answer"]:
    match state.get("next_action"):
        case "final_answer":
            return "final_answer"
        case "end":
            return "final_answer"
        case "planner":
            return "planner"
        case _:
            return "final_answer"

def map_reduce_planner_to_tool_selection(state: Dict[str, Any]) -> List[Send]:
    """
    为 Planner 输出的每个任务生成一个 Send 请求，发送到 'tool_selection' 节点。
    tool_selection 节点的输入状态应包含单个任务的信息。
    """
    tasks: List[Task] = state.get("tasks", [])
    sends = []
    for task in tasks:
        # 将单个任务包装成 ToolSelectionInputState 需要的格式
        sends.append(
            Send(
                "tool_selection",
                {
                    "task": task.dict() if hasattr(task, "dict") else task,
                    "parent_task": state.get("question", ""),
                    "context": None  # 可以传递一些上下文
                }
            )
        )
    return sends