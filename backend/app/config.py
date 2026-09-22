import os
from typing import List
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

class Settings:

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "local")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen3:8b")

    DENSE_VECTOR_SIZE: int = 1024
    SPARSE_VECTOR_SIZE:int = 250000  

    # Safety cap (in characters) when a query needs the FULL document
    # (e.g. "list every question in this document") instead of a top-k
    # chunk slice. Keeps huge documents from blowing the LLM's context
    # window while letting small/medium docs (like a 5-page exam paper)
    # through in their entirety.
    MAX_FULL_DOC_CONTEXT_CHARS: int = int(os.getenv("MAX_FULL_DOC_CONTEXT_CHARS", 20000))
    
    MAX_FILE_SIZE : int = int(os.getenv("MAX_FILE_SIZE", 10485760))
    ALLOWED_EXTENSIONS: List[str] = ['.pdf', '.docx', '.txt']
    
    WORKSPACE_DIR :  Path = Path(os.getenv("WORKSPACE_DIR", "./workspace"))
    
    ALLOWED_READ_EXTENSIONS: List[str] = [
        '.txt', '.md', '.json', '.yaml', '.yml', 
        '.xml', '.csv', '.log', '.ini', '.cfg'
    ]
    
    ALLOWED_WRITE_EXTENSIONS: List[str] = [
        '.txt', '.md', '.json', '.csv', '.xml'
    ]
    MAX_READ_SIZE: int = 10 * 1024 * 1024   # 10 Mb
    MAX_WRITE_SIZE: int = 10 * 1024 * 1024  


    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

    def __init__(self):
        self.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        (self.WORKSPACE_DIR / "inputs").mkdir(exist_ok=True)
        (self.WORKSPACE_DIR / "outputs").mkdir(exist_ok=True)
        (self.WORKSPACE_DIR / "temp").mkdir(exist_ok=True)
        

settings = Settings() 