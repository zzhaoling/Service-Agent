from typing import Dict, List, Literal, Optional, Any, Callable, Coroutine, Set, Annotated

from langchain_core.language_models import BaseChatModel
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel
from operator import add
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command, Send
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import PydanticToolsParser
from lg_agent.state import CypherOutputState, ToolSelectionErrorState, ToolSelectionInputState, InputState, OverallState, OutputState, Task, HistoryRecord, update_history
from lg_agent.planner import create_planner_node
from lg_agent.text2cypher.cypher_create import create_text2cypher_agent
from lg_agent.retrievers.base import BaseCypherExampleRetriever
from lg_agent.kg_tools_list import hybrid_query   
from lg_agent.hybrid_retriever import HybridRetriever  
from lg_agent.edges import (
    guardrails_conditional_edge,
    map_reduce_planner_to_tool_selection,
)
# 需要跳过toolselection
def create_error_tool_selection_node() -> (
    Callable[[ToolSelectionErrorState], Coroutine[Any, Any, Dict[str, Any]]]
):
    """
    创建错误处理节点，用于工具选择失败时记录错误。
    输入 state 应包含: task (str), errors (List[str]), steps (List[str])。
    输出将向 cyphers 列表追加一条错误记录。
    """

    async def error_tool_selection(state: ToolSelectionErrorState) -> Dict[str, Any]:
        """
        Handle errors in tool selection node.
        """
        errors: List[str] = list()
        steps = ["error_tool_selection"]

        errors.extend(state.get("errors", list()))

        return {
            "cyphers": [
                CypherOutputState(
                    **{
                        "task": state.get("task", ""),
                        "statement": "",
                        "parameters": None,
                        "errors": errors,
                        "records": list(),
                        "steps": steps,
                    }
                )
            ],
            "steps": steps,
        }

    return error_tool_selection


def create_final_answer_node() -> (
    Callable[[OverallState], Coroutine[Any, Any, dict[str, Any]]]
):
    """
    Create a final_answer node for a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM do perform processing.

    Returns
    -------
    Callable[[OverallState], OutputState]
        The LangGraph node.
    """

    async def final_answer(state: OverallState) -> dict[str, Any]:
        """
        Construct a final answer.
        """

        ERROR = "Unable to answer the question."

        answer = state.get("summary", ERROR)

        history_record = {
            "question": state.get("question", ""),
            "answer": answer,
            "cyphers": [
                {
                    "task": c.get("task", ""),
                    "statement": c.get("statement", ""),
                    "records": c.get("records", list()),
                }
                for c in state.get("cyphers", list())
            ],
        }

        return {
            "answer": answer,
            "steps": ["final_answer"],
            "history": [history_record],
        }

    return final_answer

def create_summarization_prompt_template() -> ChatPromptTemplate:
    """
    Create a summarization prompt template.

    Returns
    -------
    ChatPromptTemplate
        The prompt template.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant",
            ),
            (
                "human",
                (
                    """Fact: {results}

    * Summarise the above fact as if you are answering this question "{question}"
    * When the fact is not empty, assume the question is valid and the answer is true
    * Do not return helpful or extra text or apologies
    * Just return summary to the user. DO NOT start with "Here is a summary"
    * List the results in rich text format if there are more than one results
    * Don't report empty String results, but include results that are 0 or 0.0."""
                ),
            ),
        ]
    )

generate_summary_prompt = create_summarization_prompt_template()

def create_summarization_node(
    llm: BaseChatModel,
) -> Callable[[OverallState], Coroutine[Any, Any, dict[str, Any]]]:
    """
    Create a Summarization node for a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM do perform processing.

    Returns
    -------
    Callable[[OverallState], OutputState]
        The LangGraph node.
    """

    generate_summary = generate_summary_prompt | llm | StrOutputParser()

    async def summarize(state: OverallState) -> Dict[str, Any]:
        """
        Summarize results of the performed Cypher queries.
        """

        results = [
            cypher.get("records")
            for cypher in state.get("cyphers", list())
            if cypher.get("records") is not None
        ]

        if results:
            summary = await generate_summary.ainvoke(
                {
                    "question": state.get("question"),
                    "results": results,
                }
            )

        else:
            summary = "No data to summarize."

        return {"summary": summary, "steps": ["summarize"]}

    return summarize

def create_hybrid_query_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
):
    async def hybrid_query_node(state: ToolSelectionInputState) -> Dict[str, Any]:
        retriever = HybridRetriever(
            llm=llm,
            graph=graph,
            cypher_example_retriever=cypher_example_retriever,
        )
        answer = await retriever.retrieve(state.get("question", ""))
        dummy_state = CypherOutputState(
            task=[state.get("question", "")],
            statement="",
            parameters=None,
            errors=[],
            records=[{"answer": answer}],
            steps=["hybrid_query"],
        )
        return {"cyphers": [dummy_state], "steps": ["hybrid_query"]}
    return hybrid_query_node

system = """
You are responsible for choosing the appropriate tool for the given question. Use only the tools available to you.
You should select the text2cypher tool, unless another tool exactly matches what the question is asking for.
"""


def create_tool_selection_prompt_template() -> ChatPromptTemplate:
    """
    Create a tool selection prompt template.

    Returns
    -------
    ChatPromptTemplate
        The prompt template.
    """

    message = "Question: {question}"

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system,
            ),
            (
                "human",
                (message),
            ),
        ]
    )

tool_selection_prompt = create_tool_selection_prompt_template()


def create_tool_selection_node(
    llm: BaseChatModel,
    tool_schemas: List[type[BaseModel]],
    default_to_text2cypher: bool = True,
) -> Callable[[ToolSelectionInputState], Coroutine[Any, Any, Command[Any]]]:
    """
    Create a tool_selection node to be used in a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM used to process data.
    tool_schemas : Sequence[Union[Dict[str, Any], type, Callable, BaseTool]
        tools schemas that inform the LLM which tools are available.
    default_to_text2cypher : bool, optional
        Whether to attempt Text2Cypher if no tool calls are returned by the LLM, by default True

    Returns
    -------
    Callable[[ToolSelectionInputState], ToolSelectionOutputState]
        The LangGraph node.
    """

    tool_selection_chain: Runnable[Dict[str, Any], Any] = (
        tool_selection_prompt
        | llm.bind_tools(tools=tool_schemas)   # 将工具 schema 绑定到 LLM，让 LLM 能够输出工具调用。
        | PydanticToolsParser(tools=tool_schemas, first_tool_only=True)  # 将 LLM 输出解析为 Pydantic 模型实例（工具参数）。
    )

    async def tool_selection(
        state: ToolSelectionInputState,
    ) -> Command[Literal["text2cypher", "hybrid_query", "error_tool_selection", "predefined_cypher"]]:
        """
        Choose the appropriate tool for the given task.
        """
        # 如果工具只有一个且是 text2cypher，直接跳转
        if len(tool_schemas) == 1 and tool_schemas[0].__name__ == "text2cypher":
            return Command(goto=Send("text2cypher", {"task": state.get("question", ""), "steps": ["tool_selection"]}))

        # use LLM to determine tool
        tool_output: BaseModel = await tool_selection_chain.ainvoke(
            {"question": state.get("question", "")}
        )
        if tool_output is None:
                if default_to_text2cypher:
                    return Command(goto=Send("text2cypher", {"task": state.get("question", ""), "steps": ["tool_selection"]}))
                else:
                    return Command(goto=Send("error_tool_selection", {"task": state.get("question", ""), "errors": ["No tool selected"], "steps": ["tool_selection"]}))
                
        tool_name: str = tool_output.model_json_schema().get("title", "")
        tool_args: Dict[str, Any] = tool_output.model_dump()

        # 路由到对应节点
        if tool_name == "hybrid_query":
            return Command(goto=Send("hybrid_query", {"question": state.get("question", ""), "steps": ["tool_selection"]}))
        elif tool_name == "text2cypher":
            return Command(goto=Send("text2cypher", {"task": state.get("question", ""), "steps": ["tool_selection"]}))
        # elif tool_name == "predefined_cypher":
        #     return Command(goto=Send("predefined_cypher", {"task": state.get("question", ""), "query_name": tool_name, "query_parameters": tool_args, "steps": ["tool_selection"]}))
        else:
            if default_to_text2cypher:
                return Command(goto=Send("text2cypher", {"task": state.get("question", ""), "steps": ["tool_selection"]}))
            else:
                return Command(goto=Send("error_tool_selection", {"task": state.get("question", ""), "errors": [f"Unknown tool: {tool_name}"], "steps": ["tool_selection"]}))

    return tool_selection

def create_multi_tool_workflow(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    tool_schemas: List[type[BaseModel]],     
    predefined_cypher_dict: Dict[str, str],
    cypher_example_retriever: BaseCypherExampleRetriever,
    scope_description: Optional[str] = None,
    llm_cypher_validation: bool = True,
    max_attempts: int = 3,
    attempt_cypher_execution_on_final_attempt: bool = False,
    default_to_text2cypher: bool = True,
) -> CompiledStateGraph:
    """
    Create a multi tool Agent workflow using LangGraph.
    This workflow allows an agent to select from various tools to complete each identified task.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM to use for processing
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    tool_schemas : List[BaseModel]
        A list of Pydantic class defining the available tools.
    predefined_cypher_dict : Dict[str, str]
        A Python dictionary of Cypher query names as keys and Cypher queries as values.
    scope_description: Optional[str], optional
        A short description of the application scope, by default None
    cypher_example_retriever: BaseCypherExampleRetriever
        The retriever used to collect Cypher examples for few shot prompting.
    llm_cypher_validation : bool, optional
        Whether to perform LLM validation with the provided LLM, by default True
    max_attempts: int, optional
        The max number of allowed attempts to generate valid Cypher, by default 3
    attempt_cypher_execution_on_final_attempt, bool, optional
        THIS MAY BE DANGEROUS.
        Whether to attempt Cypher execution on the last attempt, regardless of if the Cypher contains errors, by default False
    default_to_text2cypher : bool, optional
        Whether to attempt Text2Cypher if no tool calls are returned by the LLM, by default True

    Returns
    -------
    CompiledStateGraph
        The workflow.
    """

    # guardrails = create_guardrails_node(
    #     llm=llm, graph=graph, scope_description=scope_description
    # )

    planner = create_planner_node(llm=llm)
    text2cypher = create_text2cypher_agent(
        llm=llm,
        graph=graph,
        cypher_example_retriever=cypher_example_retriever,
        llm_cypher_validation=llm_cypher_validation,
        max_attempts=max_attempts,
        attempt_cypher_execution_on_final_attempt=attempt_cypher_execution_on_final_attempt,
    )
    # predefined_cypher = create_predefined_cypher_node(
    #     graph=graph, predefined_cypher_dict=predefined_cypher_dict
    # )
    tool_selection = create_tool_selection_node(
        llm=llm,
        tool_schemas=tool_schemas,
        default_to_text2cypher=default_to_text2cypher,
    )
    error_tool_selection = create_error_tool_selection_node()
    summarize = create_summarization_node(llm=llm)

    final_answer = create_final_answer_node()

    main_graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)

    # main_graph_builder.add_node(guardrails)
    main_graph_builder.add_node(planner)
    main_graph_builder.add_node("text2cypher", text2cypher)
    # main_graph_builder.add_node(predefined_cypher)
    main_graph_builder.add_node(summarize)
    main_graph_builder.add_node(tool_selection)
    main_graph_builder.add_node(error_tool_selection)
    main_graph_builder.add_node(final_answer)

    main_graph_builder.add_edge(START, "guardrails")
    main_graph_builder.add_conditional_edges(
        "guardrails",
        guardrails_conditional_edge,
    )
    main_graph_builder.add_conditional_edges(
        "planner",
        map_reduce_planner_to_tool_selection,  # type: ignore[arg-type, unused-ignore]
        ["tool_selection"],
    )
    main_graph_builder.add_edge("error_tool_selection", "summarize")
    main_graph_builder.add_edge("text2cypher", "summarize")
    # main_graph_builder.add_edge("predefined_cypher", "summarize")
    main_graph_builder.add_edge("hybrid_query", "summarize")
    main_graph_builder.add_edge("summarize", "final_answer")

    main_graph_builder.add_edge("final_answer", END)

    return main_graph_builder.compile()
