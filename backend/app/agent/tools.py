"""
Agent Tool Wrappers
====================
Thin wrappers around the project's existing tools:
  - RAGTool         (app/tools/rag_tool.py)   -> retrieval + answer generation
  - FileSystemTool  (app/tools/file_tools.py) -> sandboxed file read/write

No LLM calls live here. Nodes decide WHEN to call these; these functions
only decide HOW to call the underlying tool safely and return a uniform,
loggable result shape (always a dict with "success").
"""

from typing import Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.services.rag_service import RAGService
from app.tools.file_tools import create_file_tool
from app.tools.rag_tool import RAGTool

# Both RAGService (loads BGE-M3 + connects to Qdrant/Ollama) and
# FileSystemTool are expensive/stateful - build once per process and
# reuse across every graph run instead of re-instantiating per request.

_rag_tool = RAGTool()
_file_tool = create_file_tool()


def rag_search(query: str, top_k: int = 6) -> Dict[str, Any]:
    """Retrieve context from indexed PDFs and get a generated answer."""
    return _rag_tool.search(query=query, top_k=top_k)


def write_output(path: str, content: str) -> Dict[str, Any]:
    """Persist text to the sandboxed workspace via FileSystemTool."""
    return _file_tool.write_file(path=path, content=content, overwrite=True)


def read_input(path: str) -> Dict[str, Any]:
    """Read a text file that already lives inside the sandboxed workspace."""
    return _file_tool.read_file(path=path)


class RagSearchInput(BaseModel):
    """Validated arguments for the agent-facing RAG search tool."""
    
    query: str = Field(..., min_length=1, description="The search query to run against the indexed documents.")
    top_k: int = Field(6, ge=1, le=20, description="Number of relevant chunks to retrieve.")


class WriteOutputInput(BaseModel):
    """Validated arguments for the agent-facing file-writing tool."""

    path: str = Field(..., min_length=1, description="Workspace-relative output path, for example outputs/report.md.")
    content: str = Field(..., min_length=1, description="The exact text to write to the output file.")


RAG_SEARCH_TOOL = StructuredTool.from_function(
    func=rag_search,
    name="rag_search",
    description=(
        "Search the user's indexed documents using semantic/hybrid retrieval. "
        "Use this when information must be retrieved from the uploaded documents."
    ),
    args_schema=RagSearchInput,
)


WRITE_OUTPUT_TOOL = StructuredTool.from_function(
    func=write_output,
    name="write_output",
    description=(
        "Write text to a file inside the sandboxed workspace. Use this when the "
        "user asked to save, export, or write the result to a file."
    ),
    args_schema=WriteOutputInput,
)

AGENT_TOOLS = [RAG_SEARCH_TOOL, WRITE_OUTPUT_TOOL]

AGENT_TOOL_REGISTRY = {
    tool.name: tool
    for tool in AGENT_TOOLS
}

# Single place to look up a tool by name - lets nodes.py log which tool a
# step used, and lets you register a new tool in exactly one spot.
TOOL_REGISTRY = {
    "rag_search": rag_search,
    "write_output": write_output,
    "read_input": read_input,
}
