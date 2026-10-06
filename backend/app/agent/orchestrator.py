import atexit
import os
import sqlite3
from typing import Optional


os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK","true")


from langgraph.graph import StateGraph,END
from app.agent.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver
from app.config import settings

from app.agent.nodes import (
    _init_node, router_node, route_after_router, plan_node, execute_step_node, should_continue,
    route_after_execute, replan_node,
    finalize_node, critique_node, should_reflect,
)

_checkpoint_db_path = settings.CHECKPOINT_DB_PATH

_checkpoint_conn = sqlite3.connect(
    str(_checkpoint_db_path),
    check_same_thread=False,
    timeout=30,
)
_checkpoint_conn.execute("PRAGMA journal_mode=WAL")
_checkpoint_conn.commit()

_checkpointer = SqliteSaver(_checkpoint_conn)
_checkpointer.setup()

def close_agent_checkpointer() -> None:
    """Close the shared SQLite checkpoint connection on process exit."""
    try:
        _checkpoint_conn.close()
    except Exception:
        pass
    
atexit.register(close_agent_checkpointer)

def build_agent():
    
    graph = StateGraph(AgentState)
    
    graph.add_node("init",_init_node)
    graph.add_node("router", router_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step",execute_step_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("critique", critique_node)
    
    graph.set_entry_point("init")
    graph.add_edge("init", "router")
    # Only real document questions go through plan -> search; greetings and
    # too-vague messages are answered by the router itself and stop here.
    graph.add_conditional_edges("router", route_after_router, {
        "plan": "plan",
        "end": END,
    })
    graph.add_edge("plan","execute_step")
    
    # After each step: recover from a failed/empty search, or continue/finish.
    graph.add_conditional_edges("execute_step", route_after_execute, {
        "replan" : "replan",
        "execute_step" : "execute_step",
        "finalize" : "finalize"
    })
    graph.add_conditional_edges("replan", should_continue, {
        "execute_step" : "execute_step",
        "finalize" : "finalize"
    })
    
    graph.add_edge("finalize", "critique")
    graph.add_conditional_edges("critique", should_reflect, {
        "retry": "plan",
        "done": END,
    })
    
    return graph.compile(checkpointer=_checkpointer)




_agent = build_agent()

def run_agent(task :str, file_path : Optional[str] = None, max_steps:int = 6, max_critiques: int = 2, max_replans: int = 2,thread_id: str = "default")->AgentState:
    """
    Run the agent end-to-end on a single task.

    Args:
        task: natural-language request, e.g. "Summarize the refund policy
              and save it to refund_summary.md"
        file_path: optional explicit output path (relative to workspace).
                   If omitted and a write step runs, defaults to
                   outputs/agent_answer.md
        max_steps: hard cap on execute_step iterations (loop safety net,
                   independent of how many steps the planner proposed, and
                   shared across all reflection retries - not reset per retry)
        max_critiques: how many times the agent may replan after a
                   reflection cycle judges the answer insufficient, before
                   giving up and returning its best attempt.
        max_replans: how many times the agent may recover from a failed or
                   empty search step (rewriting the query / skipping /
                   giving up) within a single planning attempt.
        thread_id: stable conversation identifier. Calls using the same
                   thread_id share checkpointed state.

    Returns:
        The final AgentState - `final_answer`, `rag_results`, `tool_calls`,
        `critique_score`, `critique_count`, and `errors` are the fields
        most useful for callers/observability.
    """
    
    initial_state: AgentState = {
        "task": task,
        "file_path": file_path,
        "max_steps": max_steps,
        "max_critiques": max_critiques,
        "max_replans": max_replans,
        "thread_id" : thread_id,
        "new_turn": True,
    }
    config = {"configurable" : {"thread_id": thread_id}}
    print(f"\n\nAGENT CALLED -- {thread_id} \n\n\n\n\n\n\n")
    return _agent.invoke(initial_state, config)

def get_agent_state(thread_id: str) -> dict:
    """Read the latest checkpointed state for a conversation thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _agent.get_state(config)
    return snapshot.values if snapshot else {}


def delete_agent_thread(thread_id: str) -> None:
    """Delete all checkpointed state for a conversation thread."""
    _checkpointer.delete_thread(thread_id)