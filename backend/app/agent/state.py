
from typing import TypedDict,List, Dict, Any, Optional
from datetime import datetime

class AgentState(TypedDict, total = False):
    
    task : str
    plan : List[str]
    current_step : int 
    max_steps : int
    
    rag_results: List[Dict[str, Any]]
    file_path : str
    final_answer : str
    
    memory : List[Dict[str,Any]]
    conversation_history: List[Dict[str, str]]
    last_answer: str
    last_file_path: str
    preserve_last_answer: bool
    turn_history_saved: bool
    thread_id: str
    turn_count: int
    new_turn: bool
    errors: List[str]
    start_time : str
    
    iteration: int                     
    tool_calls: List[str]

    # Router (router_node / route_after_router in nodes.py)
    route: str            # "documents" | "direct" | "clarify"
    route_reason: str

    # Reflection loop (critique_node / should_reflect in nodes.py)
    critique_score: int
    critique_feedback: str
    critique_count: int
    max_critiques: int
    needs_retry: bool

    # Adaptive replanning (replan_node / route_after_execute in nodes.py)
    needs_replan: bool
    replan_count: int
    max_replans: int
    last_failure: Dict[str, Any]
    search_query: str
    tried_queries: List[str]
    unrecoverable: bool
