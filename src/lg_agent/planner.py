from typing import Any, Callable, Coroutine, Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 导入简化版 Task 和状态（注意路径）
from .state import Task, InputState  

# 定义 Planner 的输出结构
class PlannerOutput(BaseModel):
    tasks: List[Task] = Field(
        default=[],
        description="A list of subtasks to be executed in parallel. Each task includes description, tool name, and parameters."
    )

# 系统提示词
PLANNER_SYSTEM = """
你必须分析输入问题并将其分解为单独的子任务。
如果存在适当的独立任务，则将其作为列表提供，否则返回空列表。
任务不应该相互依赖。
返回要完成的任务列表。
"""

def create_planner_prompt_template() -> ChatPromptTemplate:
    """创建 planner 的提示模板"""
    message = """Rules:
* Ensure that the tasks are not returning duplicated or similar information.
* Ensure that tasks are NOT dependent on information gathered from other tasks!
* tasks that are dependent on each other should be combined into a single question.
* tasks that return the same information should be combined into a single question.

question: {question}
"""
    return ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM),
        ("human", message)
    ])

planner_prompt = create_planner_prompt_template()

def create_planner_node(
    llm: BaseChatModel,
    ignore_node: bool = False,
    next_action: str = "tool_selection"
) -> Callable[[InputState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    创建 planner 节点。
    """
    planner_chain = planner_prompt | llm.with_structured_output(PlannerOutput)

    async def planner(state: InputState) -> Dict[str, Any]:
        print("----- Planner: 开始任务分解 -----")
        if not ignore_node:
            planner_output: PlannerOutput = await planner_chain.ainvoke({
                "question": state.get("question", "")
            })
            print(f"planner_output: {planner_output}")
        else:
            # 忽略分解，直接返回一个包含原问题的任务（工具留空，后续可能由 tool_selection 处理）
            
            planner_output = PlannerOutput(tasks=[])
            print(f"planner_output: {planner_output}")
            print("Planner: 忽略分解，返回空/默认任务列表")

        tasks= planner_output.tasks
        print(f"Planner 输出任务数量: {len(tasks)}")
        return {
            "next_action": next_action,
            "tasks": planner_output.tasks
            or [
                Task(
                    question=state.get("question", ""),
                    parent_task=state.get("question", ""),
                )
            ],
            "steps": ["planner"],
        }
    return planner