import re
from langchain_neo4j import Neo4jGraph

def retrieve_and_parse_schema_from_graph_for_prompts(graph: Neo4jGraph) -> str:
    """
    从 Neo4j 图中提取结构描述（Schema），过滤内部节点（如 CypherQuery），
    并将花括号替换为方括号，避免与 LangChain 的 Prompt 模板语法冲突。
    最终得到一个干净的、可嵌入 LLM Prompt 的图结构文本。
    """
    schema = graph.get_schema

    # 删除以 "- **CypherQuery**" 开头的段落（如果存在）
    # 匹配从 "- **CypherQuery**" 到下一个 "Relationship properties" 或 "- *" 之前的内容
    pattern = r"^(- \*\*CypherQuery\*\*[\s\S]+?)(^Relationship properties|- \*)"
    schema = re.sub(pattern, r"\2", schema, flags=re.MULTILINE)

    # 替换花括号为方括号，避免与 LangChain 模板变量语法冲突
    schema = schema.replace("{", "[").replace("}", "]")

    return schema