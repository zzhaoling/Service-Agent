import re
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseChatModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_openai import ChatOpenAI
import torch

class HallucinationDetector:
    """
    多层幻觉检测器：知识溯源 + 数值一致性 + Qwen 裁判
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        llm: BaseChatModel,               # 主 LLM（如 deepseek）
        qwen_config: dict
    ):
        self.graph = graph
        self.llm = llm
        self.qwen_llm = ChatOpenAI(
            model=qwen_config.get("model_id", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=qwen_config.get("base_url", "https://api.siliconflow.cn/v1"),
            api_key=qwen_config.get("api_key", None),
            temperature=0.0,
            model_kwargs={"max_tokens": 200},
        )

    # ---------- 1. 知识溯源 ----------
    def extract_citations(self, answer: str) -> List[str]:
        """从答案中提取引用标记，例如 [1], [2] 或 (来源：xxx)"""
        # 简单提取方括号内的数字
        citations = re.findall(r'\[(\d+)\]', answer)
        # 也可以提取其他格式，例如（说明书第X页）
        return citations

    async def check_source_grounding(self, answer: str, evidences: List[str]) -> Tuple[bool, str]:
        """使用 LLM 判断答案是否完全基于证据（不依赖引用标号）"""
        if not evidences:
            return False, "无证据可进行溯源验证。"
        
        all_evidence = "\n\n".join(evidences)
        prompt = f"""你是一个严格的事实核查员。请判断以下答案的内容是否**完全**基于提供的证据，没有编造或超出证据范围的信息。

    证据：
    {all_evidence}

    答案：
    {answer}

    请只回答“是”或“否”。"""
        
        response = await self.llm.ainvoke(prompt)
        content = response.content.strip()
        if "是" in content:
            return True, "答案完全基于证据。"
        else:
            return False, "答案包含证据中未出现的内容。"

    # ---------- 2. 数值一致性校验 ----------
    def extract_numbers_with_units(self, text: str) -> List[Tuple[float, str]]:
        """提取数值及其后的单位，例如 8 kg, 5星, 240 kWh"""
        pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z\u4e00-\u9fa5]+)'
        matches = re.findall(pattern, text)
        return [(float(num), unit) for num, unit in matches]

    async def check_numerical_consistency(self, answer: str, context: Dict) -> Tuple[bool, str]:
        """
        提取答案中的数值，在数据库中查询真实值比对。
        context 应包含产品型号等信息，或者从 Cypher 查询结果中获取。
        """
        # 这里假设 context 中包含一个字段 'product_models' 列表
        # 你需要从之前的检索结果中提取涉及的产品型号
        # 简化：调用一个 Cypher 查询获取所有产品的真实数值
        # 实际实现中，需要更智能地确定答案中提到了哪个产品
        numbers = self.extract_numbers_with_units(answer)
        if not numbers:
            return True, "无数值信息，跳过一致性校验。"

        # 例如从答案中提取可能的产品型号
        model_match = re.search(r'([A-Z0-9\-]{5,})', answer)
        if not model_match:
            return True, "未识别到具体产品型号，跳过数值校验。"
        model = model_match.group(1)

        # 查询数据库中的真实值
        query = """
        MATCH (p:Product {model_number: $model})
        RETURN p.capacity_kg, p.annual_consumption_kwh, p.energy_rating
        """
        result = self.graph.query(query, {"model": model})
        if not result:
            return True, f"未找到型号 {model}，跳过数值校验。"
        real = result[0]

        errors = []
        for num, unit in numbers:
            if "kg" in unit and real.get("p.capacity_kg") is not None:
                if abs(num - real["p.capacity_kg"]) > 0.1:
                    errors.append(f"容量 {num}{unit} ≠ {real['p.capacity_kg']}kg")
            elif "kwh" in unit and real.get("p.annual_consumption_kwh"):
                # 去除可能的单位差异
                try:
                    real_num = float(real["p.annual_consumption_kwh"])
                    if abs(num - real_num) > 1:
                        errors.append(f"耗电量 {num}{unit} ≠ {real_num}kWh")
                except:
                    pass
            elif "星" in unit and real.get("p.energy_rating"):
                try:
                    real_star = float(real["p.energy_rating"])
                    if abs(num - real_star) > 0.1:
                        errors.append(f"能效等级 {num}{unit} ≠ {real_star}星")
                except:
                    pass

        if errors:
            return False, "数值不一致: " + "; ".join(errors)
        return True, "数值校验通过。"

    # ---------- 3. Qwen 小模型裁判 ----------
    async def qwen_judge(self, question: str, answer: str, evidences: List[str]) -> Tuple[bool, float]:
        prompt = f"""你是一个公正的评判员。请根据提供的证据判断以下答案是否完全基于证据，没有编造信息。
        证据：
        {chr(10).join(evidences)}

        问题：{question}

        答案：{answer}

        请先给出判断（是/否），然后给出置信度分数（0-1）。输出格式：
        判断：是/否
        置信度：<分数>
        """
        response = await self.qwen_llm.ainvoke(prompt)
        text = response.content
        is_grounded = "判断：是" in text
        score_match = re.search(r"置信度：(\d\.?\d*)", text)
        confidence = float(score_match.group(1)) if score_match else 0.5
        return is_grounded, confidence

    # ---------- 主检测入口 ----------
    async def detect(
        self,
        question: str,
        answer: str,
        evidences: List[str],
        context: Dict = None,
    ) -> Dict[str, Any]:
        """
        执行多层检测。
        evidences: 检索到的文本块列表（可以是 Cypher 结果转成的文本，也可以是向量片段）。
        context: 可能包含产品型号等辅助信息。
        返回字典包含：
            - passed: bool
            - details: dict 每层结果
            - final_answer: str (如果通过则原样，否则可尝试修正)
        """
        results = {}
        final_passed = True

        # 1. 知识溯源
        source_ok, source_msg = await self.check_source_grounding(answer, evidences)
        results["source_grounding"] = {"passed": source_ok, "message": source_msg}
        if not source_ok:
            final_passed = False

        # 2. 数值一致性
        num_ok, num_msg = await self.check_numerical_consistency(answer, context or {})
        results["numerical_consistency"] = {"passed": num_ok, "message": num_msg}
        if not num_ok:
            final_passed = False

        # 3. Qwen 裁判
        qwen_ok, confidence = await self.qwen_judge(question, answer, evidences)
        results["qwen_judge"] = {"passed": qwen_ok, "confidence": confidence}
        if not qwen_ok:
            final_passed = False

        return {"passed": final_passed, "details": results}