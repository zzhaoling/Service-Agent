import asyncio
import json
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support, classification_report
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from lg_builder import classify_intent
from lg_state import Router

class IntentEvaluator:
    def __init__(self, test_data_path: str):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)  # [{"query": "...", "expected": "graphrag-query"}, ...]
    
    async def evaluate_single(self, query: str) -> str:
        """单条意图识别"""
        messages = [{"role": "user", "content": query}]
        result: Router = await classify_intent(messages)
        return result.type
    
    async def evaluate_all(self) -> Dict:
        """批量评估"""
        y_true, y_pred = [], []
        for item in self.test_data:
            expected = item["expected"]
            predicted = await self.evaluate_single(item["query"])
            y_true.append(expected)
            y_pred.append(predicted)
        
        # 计算各类别的 Precision、Recall、F1
        labels = ["general-query", "additional-query", "graphrag-query"]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        # 计算 Macro F1（所有类别F1的平均值）
        macro_f1 = sum(f1) / len(f1)
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, labels=labels)
        
        return {
            "macro_f1": macro_f1,
            "per_class": {label: {"precision": p, "recall": r, "f1": f} 
                          for label, p, r, f in zip(labels, precision, recall, f1)},
            "full_report": report
        }

# 运行评估
async def main():
    evaluator = IntentEvaluator("data/test_data.json")
    results = await evaluator.evaluate_all()
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(results["full_report"])

if __name__ == "__main__":
    asyncio.run(main())