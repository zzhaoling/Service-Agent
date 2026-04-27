import re
import fitz
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

PDF_PATH = "data/Washing Machine Manual.pdf"      # PDF 文件路径

# ================= 1. 读取 PDF 全文 =================
def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ================= 2. 按标题切分章节 =================
def split_sections_by_heading(text):
    """
    匹配类似 "1. General information" 或 "9. Troubleshooting" 的标题行
    返回列表 [(标题, 内容), ...]
    """
    # 匹配以数字加点空格开头，且后续非数字的行作为章节标题
    pattern = r'\n(\d+\.\s+[^\n]+)\n'
    matches = list(re.finditer(pattern, text))
    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        if len(content) > 50:   # 过滤太短的章节
            sections.append((title, content))
    return sections

# ================= 3. 抽取错误代码表 =================
def extract_error_codes(text):
    """
    从内容中提取表格形式的错误代码（基于你 PDF 中的第 22-23 页格式）
    示例格式：
    Code    Cause    Solution
    E2      Lock error    Shut the door properly.
    E4      Water level not reached   Check tap...
    """
    error_pattern = r'\n(E\d+|CLR FLTR|Unb|F[0-9]|FC0?[0-9])\s+([^\n]+)\s+([^\n]+)'
    matches = re.findall(error_pattern, text)
    codes = []
    for code, cause, solution in matches:
        codes.append({
            "code": code.strip(),
            "cause": cause.strip(),
            "solution": solution.strip()
        })
    return codes

# ================= 4. 规则判断动作和部件 =================
def classify_action(content):
    keywords = {
        "清洁": "clean",
        "安装": "install",
        "故障": "troubleshoot",
        "保养": "maintain",
        "存放": "store",
        "运输": "move"
    }
    for kw, action in keywords.items():
        if kw in content:
            return action
    return "general"

def extract_components(content):
    """
    从内容中识别提到的部件名称
    """
    component_keywords = ["过滤器", "滚筒", "进水阀", "排水管", "控制面板", "运输螺栓", "泵", "洗涤剂盒"]
    found = [comp for comp in component_keywords if comp in content]
    return found if found else ["通用"]

# ================= 5. 存储到 Neo4j =================
def store_sections(sections):
    """
    sections: list of (title, content)
    """
    for title, content in sections:
        action = classify_action(content)
        components = extract_components(content)
        for comp in components:
            graph.query("""
                MERGE (s:ManualSection {title: $title})
                SET s.content = $content
                MERGE (a:Action {name: $action})
                MERGE (c:Component {name: $comp})
                MERGE (s)-[:HAS_ACTION]->(a)
                MERGE (s)-[:ABOUT]->(c)
            """, {
                "title": title,
                "content": content[:5000],
                "action": action,
                "comp": comp
            })
def store_error_codes(codes):
    for code_info in codes:
        graph.query("""
            MERGE (e:ErrorCode {code: $code})
            SET e.cause = $cause,
                e.solution = $solution
        """, code_info)

# ================= 主流程 =================
if __name__ == "__main__":
    print("读取 PDF...")
    full_text = load_pdf_text(PDF_PATH)

    print("切分章节...")
    sections = split_sections_by_heading(full_text)
    print(f"共发现 {len(sections)} 个章节")

    print("存入章节、动作、部件...")
    store_sections(sections)

    print("抽取并存入错误代码...")
    error_codes = extract_error_codes(full_text)
    print(f"发现 {len(error_codes)} 个错误代码")
    if error_codes:
        store_error_codes(error_codes)

    print("✅ PDF 结构化入库完成")