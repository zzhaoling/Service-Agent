import pandas as pd
from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j",
    # refresh_schema=False   # 跳过 APOC 检查
)

# 请确认文件路径
df = pd.read_csv("data/clothes-washers.csv").fillna("")

records = []
for _, row in df.iterrows():
    model = str(row.get("Model No", "")).strip()
    if not model:
        continue
    
    brand = row.get("Brand", "").strip()
    if not brand:
        brand = "Unknown"
    
    # 容量可能包含 "5.5" 等数字，转为 float
    try:
        cap = float(row.get("Cap", 0)) if row.get("Cap") else None
    except:
        cap = None
    
    # 能效星级
    try:
        star = float(row.get("New Star", 0)) if row.get("New Star") else None
    except:
        star = None
    
    consumption = row.get("Labelled energy consumption (kWh/year)", "")
    ptype = row.get("Type", "")          # Front / Top
    action = row.get("MachineAction", "") # Drum / Agitator / Impeller
    
    records.append({
        "model": model,
        "model_clean": model.replace("-", "").upper(),
        "brand": brand,
        "capacity": cap,
        "star": star,
        "consumption": consumption,
        "type": ptype,
        "action": action
    })

query = """
UNWIND $rows AS row

MERGE (p:Product {model_number: row.model})
SET p.model_clean = row.model_clean,
    p.capacity_kg = row.capacity,
    p.annual_consumption_kwh = row.consumption,
    p.type = row.type,
    p.machine_action = row.action,
    p.category = "Washing Machine"

WITH p, row

MERGE (b:Brand {name: row.brand})
MERGE (p)-[:MANUFACTURED_BY]->(b)

WITH p, row

FOREACH (_ IN CASE WHEN row.star IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s:EnergyStar {rating: row.star})
    MERGE (p)-[:RATED_AS]->(s)
)
"""

graph.query(query, {"rows": records})
print(f"✅ 成功导入 {len(records)} 条产品记录")
