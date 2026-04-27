import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    GRAPHRAG_PROJECT_DIR = os.getenv("GRAPHRAG_PROJECT_DIR")
    GRAPHRAG_DATA_DIR = os.getenv("GRAPHRAG_DATA_DIR", "data")

    QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1")
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", None)

config = Config()