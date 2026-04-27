# active_learning.py
import asyncio
from collections import Counter
from typing import List, Dict
import json
from lg_builder import classify_intent  

async def get_uncertainty_score(query: str, n_samples: int = 5, temperature: float = 0.3) -> float:
    """
    通过多次采样计算模型对某条 query 的不确定性分数。
    Args:
        query: 用户输入
        n_samples: 采样次数
        temperature: 采样温度（建议 0.3 ~ 0.5，不能为 0）
    Returns:
        不确定性分数，范围 0~1，越高表示模型越纠结。
    """
    results = []
    for _ in range(n_samples):
        # 调用 classify_intent 时传入 temperature
        result = await classify_intent([{"role": "user", "content": query}], temperature=temperature)
        results.append(result.type)
    
    counter = Counter(results)
    max_count = max(counter.values())
    consistency = max_count / n_samples
    uncertainty = 1 - consistency
    return uncertainty

async def find_hard_samples(
    test_data_path: str,
    threshold: float = 0.6,
    n_samples: int = 5,
    temperature: float = 0.3,
    max_samples: int = 10
) -> List[Dict]:
    """
    从测试集中找出模型纠结的样本。
    Args:
        test_data_path: JSON 测试集文件路径，格式 [{"query": "...", "expected": "..."}]
        threshold: 不确定性阈值，高于此值视为困难样本
        n_samples: 采样次数
        temperature: 采样温度
        max_samples: 最多返回的困难样本数
    Returns:
        困难样本列表，按不确定性降序排列。
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    hard_samples = []
    for item in test_data:
        query = item["query"]
        expected = item["expected"]
        
        # 先正常分类一次（temperature=0.7）
        predicted = await classify_intent([{"role": "user", "content": query}], temperature=0.7)
        if predicted.type != expected:
            # 预测错误，再计算不确定性
            uncertainty = await get_uncertainty_score(query, n_samples=n_samples, temperature=temperature)
            if uncertainty > threshold:
                hard_samples.append({
                    "query": query,
                    "expected": expected,
                    "predicted": predicted.type,
                    "uncertainty": uncertainty
                })
    
    hard_samples.sort(key=lambda x: x["uncertainty"], reverse=True)
    return hard_samples[:max_samples]

# 示例运行
async def main():
    
    hard = await find_hard_samples(".../data/test_data.json", threshold=0.5)
    for h in hard:
        print(f"Query: {h['query']}")
        print(f"Expected: {h['expected']}, Predicted: {h['predicted']}, Uncertainty: {h['uncertainty']:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())