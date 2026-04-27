import logging
from langchain_neo4j import Neo4jGraph
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 把 src 加入路径
from config import config  

# 抑制 Neo4j 驱动的 INFO/WARNING 日志，减少输出干扰
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("langchain_neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.io").setLevel(logging.ERROR)
logging.getLogger("neo4j.bolt").setLevel(logging.ERROR)

def get_neo4j_graph() -> Neo4jGraph:
    """
    创建并返回一个 Neo4jGraph 实例，使用 config 中的连接参数。
    """
    # 默认 database 为 "neo4j"，如果 config 中没有 NEO4J_DATABASE 则使用默认值
    database = getattr(config, "NEO4J_DATABASE", "neo4j")

    print(f"Connecting to Neo4j at {config.NEO4J_URI}")  # 可选，实际项目中建议用 logging.info

    try:
        neo4j_graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=database
        )
        return neo4j_graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        raise