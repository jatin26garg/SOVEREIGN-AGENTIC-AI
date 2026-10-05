from fastapi import FastAPI, UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional

from app.config import settings
from app.models import QueryRequest,QueryResponse,Documentinfo
from app.services.rag_service import get_rag_service
from app.agent.orchestrator import run_agent
from pydantic import BaseModel
from pathlib import Path

rag = get_rag_service()

app = FastAPI(
    title="Document RAG API",
    description="Upload documents and ask questions with RAG (Gemini + Qdrant)",
    version='3.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return{
        "status" : "healthy",
        "embedding_model": "BAAI/bge-m3",
        "vector_db": "Qdrant",
        "embedding_dimension": 1024,
        "service" : "Document RAG API",
        "version" : "3.0.0",
        "rag_tool": "rag-tool"
    }
class AgentExecuteRequest(BaseModel):
    task: str
    file_path: Optional[str] = None
    max_steps: Optional[int] = 6
    # 0 disables the feature; None means "use the agent's default (2)".
    max_replans: Optional[int] = None
    max_critiques: Optional[int] = None

@app.post("/agent/execute")
async def execute_agent(request: AgentExecuteRequest):
    if not request.task or not request.task.strip():
        raise HTTPException(400, detail="task is required")
    
    try:
        result = run_agent(
            task=request.task,
            file_path=request.file_path,
            max_steps=request.max_steps or 6,
            max_replans=2 if request.max_replans is None else request.max_replans,
            max_critiques=2 if request.max_critiques is None else request.max_critiques,
        )

        # Return the most useful fields for the frontend
        return {
            "status": "success",
            "final_answer": result.get("final_answer", ""),
            "file_path": result.get("file_path"),
            "tool_calls": result.get("tool_calls", []),
            "rag_results": result.get("rag_results", []),
            "errors": result.get("errors", []),
            "memory": result.get("memory", []),
            "plan": result.get("plan", []),
            "route": result.get("route", ""),
            "route_reason": result.get("route_reason", ""),
            "replan_count": result.get("replan_count", 0),
            "critique_count": result.get("critique_count", 0),
            "critique_score": result.get("critique_score"),
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Agent execution failed: {str(e)}")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_name = file.filename
        if not file_name:
            raise HTTPException(400,"no fileName provided")
        
        import os
        ext = os.path.splitext(file_name)[1].lower()
        if not ext in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"file type {ext} not allowed ")
        
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(400,f"file is too large")
        
        if file_size == 0:
            raise HTTPException(400,"file is empty")
        
        inputs_dir = settings.WORKSPACE_DIR / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        saved_path = inputs_dir / file_name
        saved_path.write_bytes(content)
        print(f"📁 Saved upload to: {saved_path}")
        doc_id = rag.process_document(content,file_name)
        
        return{
            "status" : "success",
            "document_id" : doc_id,
            "filename" : file_name,
            "message" : f"Successfully proccessed {file_name}"
            
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        print(f"upload error : {str(e)}")
        raise HTTPException(400, detail=str(e))
    
@app.post("/query",response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(400,"Question is empty")
        result = rag.query(question =request.question, top_k = request.top_k)
        print(f"the result is {result}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f" Query Error : {str(e)}")
        raise HTTPException(500, detail=f"internal error {str(e)}")



@app.get("/documents", response_model=List[Documentinfo])
async def list_documents():
    documents =  rag.get_documents()
    
    output_dir = Path("workspace/outputs")

    for file in output_dir.iterdir():
        if file.is_file():
            documents.append({
                "file_name": file.name,
                "file_path": str(file),
                "type": "agent_output"
            })

    return documents

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id:str):
    try:
        success = rag.delete_document(doc_id)
        if not success:
            raise HTTPException(404, f" document {doc_id} not found")
        return {
            "status" : "success",
            "message" : f" document {doc_id} deleted successfully!"
        }
    except Exception as e:
        print(f" couldnt delete")
        
        raise HTTPException(500, detail=f" internal error : {str(e)}")
    
    