from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv()

# 连接 Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j"
)

# 加载 embedding 模型
embedding = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL"),
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 从已存在的 ManualSection 节点创建向量索引
vector_store = Neo4jVector.from_existing_graph(
    embedding=embedding,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j",
    index_name="manual_section_vectors",   # 索引名称
    node_label="ManualSection",            # 节点标签
    text_node_properties=["content"],      # 要索引的属性（可多个，这里只索引 content）
    embedding_node_property="embedding",   # 存储向量的属性名
)

print(f"✅ 向量索引 '{vector_store.index_name}' 已创建")