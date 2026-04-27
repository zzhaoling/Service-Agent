import asyncio
import uuid
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from lg_builder import graph  
from lg_builder import get_llm  


class CustomerServiceBot:
    def __init__(self):
        self.graph = graph
        # 每个会话使用独立的 thread_id 来保持记忆
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    async def chat(self, user_input: str) -> str:
        """处理单条用户消息，返回客服回答"""
        # 构造输入状态（与 InputState 结构一致）
        input_state = {
            "messages": [HumanMessage(content=user_input)],
        }
        # 调用图，使用配置中的 thread_id 自动管理历史
        final_state = await self.graph.ainvoke(input_state, config=self.config)
        # 提取最后一条 AI 消息作为答案
        messages = final_state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return "抱歉，我无法回答。"

    async def interactive_loop(self):
        """命令行交互循环"""
        print("智能客服系统已启动 (输入 'quit' 退出)")
        print("-" * 50)
        while True:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break
            if not user_input:
                continue
            answer = await self.chat(user_input)
            print(f"客服: {answer}")


async def main():
    bot = CustomerServiceBot()
    await bot.interactive_loop()


if __name__ == "__main__":
    asyncio.run(main())