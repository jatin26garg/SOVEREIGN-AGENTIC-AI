import re
import json

from langchain_groq import ChatGroq
from app.config import settings
from app.agent.state import AgentState
from app.agent.tools import rag_search,write_output,AGENT_TOOL_REGISTRY,AGENT_TOOLS,RAG_SEARCH_TOOL

from datetime import datetime


# response_mime_type="application/json" is Gemini's equivalent of Ollama's
# format='json' - constrains the output to valid JSON at the API level, so
# _clean_json()/_extract_plan_list() below still matter only as a defensive
# second layer (e.g. in case the model wraps JSON in markdown anyway), not
# as the primary mechanism.
_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0.2,
    reasoning_effort="medium",
    reasoning_format="hidden",
    max_retries=2,
    request_timeout=60,
    model_kwargs={"response_format": {"type": "json_object"}},
) 

_tool_llm_base = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0.1,
    reasoning_effort="low",
    reasoning_format="hidden",
    max_retries=2,
    request_timeout=60,
)

_tool_llm = _tool_llm_base.bind_tools(
    AGENT_TOOLS,
    tool_choice="any",
)

_retry_rag_llm = _tool_llm_base.bind_tools(
    [RAG_SEARCH_TOOL],
    tool_choice="rag_search",
)

_MAX_CONVERSATION_MESSAGES = 12
_MAX_CONVERSATION_CHARS = 8000

def _append_conversation_turn(state: AgentState) -> None:
    """Persist the user/assistant exchange for the current API turn.

    A critique retry can reach finalize more than once in one turn. In that
    case update the existing assistant message instead of duplicating the user
    message in conversation history.
    """
    
    print(f"---ENTERED _append_conversation_turn FUNCTION \n\n\n\n")
    
    history = state.setdefault("conversation_history", [])
    task = str(state.get("task", "")).strip()
    answer = str(state.get("final_answer", "")).strip()

    if not task:
        return

    if state.get("turn_history_saved"):
        for index in range(len(history) - 1, -1, -1):
            message = history[index]
            if (
                message.get("role") == "user"
                and message.get("content") == task[:_MAX_CONVERSATION_CHARS]
            ):
                if index + 1 < len(history) and history[index + 1].get("role") == "assistant":
                    history[index + 1]["content"] = answer[:_MAX_CONVERSATION_CHARS]
                elif answer:
                    history.insert(index + 1, {
                        "role": "assistant",
                        "content": answer[:_MAX_CONVERSATION_CHARS],
                    })
                return
        return

    history.append({
        "role": "user",
        "content": task[:_MAX_CONVERSATION_CHARS],
    })
    if answer:
        history.append({
            "role": "assistant",
            "content": answer[:_MAX_CONVERSATION_CHARS],
        })

    # Bound both prompt size and checkpoint growth while keeping recent turns.
    if len(history) > _MAX_CONVERSATION_MESSAGES:
        del history[:-_MAX_CONVERSATION_MESSAGES]

    state["turn_history_saved"] = True

def _conversation_context(state: AgentState) -> str:
    
    """Format recent conversation turns for the planner."""
    history = state.get("conversation_history", [])
    if not history:
        return "No previous conversation turns."

    lines = []
    for message in history[-_MAX_CONVERSATION_MESSAGES:]:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No previous conversation turns."

    

def _clean_json(raw) -> str:
    """
    Normalize an LLM response to a string, then extract a JSON object/list.
    Handles plain strings and provider content-block lists.
    """
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        raw = "\n".join(parts)
    elif not isinstance(raw, str):
        raw = str(raw)

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    return match.group(1) if match else raw



def _init_node(state : AgentState) ->AgentState:
    
    print("---ENTERED INTIT_NODE --- \n\n\n")
    
    state.setdefault("conversation_history", [])
    state.setdefault("last_answer", "")
    state.setdefault("last_file_path", "")
    state.setdefault("preserve_last_answer", False)
    state.setdefault("turn_history_saved", False)
    state.setdefault("thread_id", "default")
    state.setdefault("turn_count", 0)
    
    
    
    # A checkpointer restores the previous graph state before the new
    # invocation starts. Reset per-turn execution state, but deliberately keep
    # conversation_history and last_answer so follow-ups can refer to them.
    if state.get("new_turn", False):
        if state.get("final_answer") and not state.get("last_answer"):
            state["last_answer"] = state["final_answer"]
            
    state["plan"] = []
    state["current_step"] = 0
    state["rag_results"] = []
    state["final_answer"] = ""
    state["preserve_last_answer"] = False
    state["turn_history_saved"] = False
    state["memory"] = []
    state["errors"] = []
    state["iteration"] = 0
    state["tool_calls"] = []
    state["route"] = ""
    state["route_reason"] = ""
    state["critique_count"] = 0
    state["critique_feedback"] = ""
    state["needs_retry"] = False
    state["needs_replan"] = False
    state["replan_count"] = 0
    state["last_failure"] = {}
    state["search_query"] = ""
    state["tried_queries"] = []
    state["unrecoverable"] = False
    state["turn_count"] = state.get("turn_count", 0) + 1
    state["new_turn"] = False
    
    state.setdefault("plan" ,[])
    state.setdefault("current_step",0)
    state.setdefault("max_steps",6)
    state.setdefault("rag_results", [])
    state.setdefault("final_answer", "")
    state.setdefault("memory",[])
    state.setdefault("errors",[])
    state.setdefault("iteration",0)
    state.setdefault("tool_calls", [])
    state.setdefault("route", "")
    state.setdefault("route_reason", "")
    state.setdefault("critique_count", 0)
    state.setdefault("max_critiques", 2)
    state.setdefault("critique_feedback", "")
    state.setdefault("needs_replan", False)
    state.setdefault("replan_count", 0)
    state.setdefault("max_replans", 2)
    state.setdefault("last_failure", {})
    state.setdefault("search_query", "")
    state.setdefault("tried_queries", [])
    state.setdefault("unrecoverable", False)
    state["start_time"] = datetime.now().isoformat()
    
    print("---EXITED INTIT_NODE --- \n\n\n")
    return state

def _extract_plan_list(parsed):
    """
    Ollama's format="json" guarantees valid JSON, but NOT that the top-level
    shape is a bare array - instruct-tuned models often wrap a requested
    array in an object anyway, e.g. {"steps": [...]} or {"plan": [...]}.
    Accept the bare-array case (what we asked for) and the common
    dict-wrapped cases, so a well-formed answer never gets thrown away
    just because of the wrapper.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("steps", "plan", "actions", "tasks"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        # last resort 1: a dict with exactly one list-valued key
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        # last resort 2: qwen3 sometimes turns "give me a list" into an
        # object whose keys AND values are both the step text itself
        # (e.g. {"Search the PDFs": "Search the PDFs", "Save it": "Save it"})
        # to satisfy an object-only JSON mode while still trying to comply.
        # If every value is a plain non-empty string, treat the values as
        # the ordered step list rather than throwing a good plan away.
        str_values = [v for v in parsed.values() if isinstance(v, str) and v.strip()]
        if parsed and len(str_values) == len(parsed):
            return str_values
    return None

_FILENAME_PATTERN = re.compile(
    r'(?:to|as|into|save\s+to|write\s+to|save\s+it\s+to|write\s+it\s+to)\s+'
    r'([\w\-\.]+\.(?:md|txt|json|csv|docx|xlsx))',
    re.IGNORECASE
)

def _extract_filename(text: str) -> str | None:
    """Extract filename from text like 'save it to report.md'."""
    match = _FILENAME_PATTERN.search(text)
    return match.group(1) if match else None

# ===================================================
# ROUTER
# ===================================================
# First decision point, before any planning: should this message go
# through the (expensive) plan -> search -> finalize pipeline at all?
#
#   documents : the normal pipeline. DEFAULT whenever unsure.
#   direct    : greeting / thanks / "what can you do" - answered instantly.
#   clarify   : no identifiable request ("explain me") - ask instead of
#               running a pointless search.
#
# Cost control: cheap regex rules decide the obvious cases with ZERO LLM
# calls; only genuinely ambiguous messages reach the model.

_GREETING_RE = re.compile(
    r"^\s*(hi+|hello|hey+|greetings|good\s+(morning|afternoon|evening))\b[\s!.,?]*$", re.IGNORECASE)
_THANKS_RE = re.compile(
    r"^\s*(thanks|thank\s+you|thx|ty)(\s+(a\s+lot|so\s+much))?[\s!.]*$", re.IGNORECASE)
_BYE_RE = re.compile(r"^\s*(bye|goodbye|see\s+you)[\s!.]*$", re.IGNORECASE)

_DOC_SIGNAL_RE = re.compile(
    r"\b(documents?|docs?|pdfs?|files?|pages?|sections?|chapters?|summar\w*|extract\w*|"
    r"policy|policies|save|write|export|uploaded|resume|report|paper)\b", re.IGNORECASE)

_CAPABILITIES = (
    "I answer questions about the documents you've uploaded, summarize them, "
    "list their contents, and save results to files. What would you like to know?"
)

_ROUTES = ("documents", "direct", "clarify")
_MAX_WORDS_CLARIFY = 6    # a long message is never "too vague to act on"
_MAX_WORDS_DIRECT = 12    # ...nor is it just small talk


def _route_by_rules(task: str) -> dict | None:
    """Deterministic fast path. Returns None when the message is ambiguous
    and the LLM should decide."""
    if _GREETING_RE.match(task):
        return {"route": "direct", "reply": f"Hello! {_CAPABILITIES}", "reason": "greeting (rule)"}
    if _THANKS_RE.match(task):
        return {"route": "direct",
                "reply": "You're welcome! Let me know if you'd like anything else from your documents.",
                "reason": "thanks (rule)"}
    if _BYE_RE.match(task):
        return {"route": "direct", "reply": "Goodbye!", "reason": "farewell (rule)"}
    if _DOC_SIGNAL_RE.search(task):
        return {"route": "documents", "reply": "", "reason": "document/file keyword (rule)"}
    return None


_ROUTER_SYSTEM = (
    "You are the routing module of a document question-answering agent. "
    "Classify the user's message into exactly ONE route:\n"
    '- "documents": the message asks about, refers to, or could plausibly be '
    "answered from uploaded documents, or asks to save something. THIS IS THE "
    "DEFAULT - choose it whenever you are unsure.\n"
    '- "direct": ONLY a greeting, thanks, or a question about the assistant '
    'itself (e.g. "what can you do"). Put a short friendly reply in "reply".\n'
    '- "clarify": ONLY when the message has no identifiable subject or request '
    'at all (e.g. "explain me", "help", "do it"). Put ONE short clarifying '
    'question in "reply".\n'
    "Respond with ONLY a JSON object: "
    '{"route": "documents"|"direct"|"clarify", "reply": "<only for direct/clarify, '
    'otherwise empty>", "reason": "<short>"}'
)


def _extract_route(parsed, task: str) -> dict | None:
    print(f"INSIDE _EXTRACT_ROUTE -- \n")
    """Tolerant parse of the router JSON, plus deterministic guards so a
    model misfire can never swallow a real question."""
    print(f"parsed message in _extract_route = {parsed} \n")
    if not isinstance(parsed, dict):
        return None
    route = str(parsed.get("route", "")).strip().lower()
    if route not in _ROUTES:
        return None
    reply = str(parsed.get("reply") or "").strip()
    reason = str(parsed.get("reason") or "").strip()
    words = len(task.split())
    
   
    print(f"route = {route}\n")
    print(f"reply = {reply}\n")
    print(f"reason = {reason}\n")
    print(f"words = {words}\n")
    
    
    if route == "clarify" and (not reply or words > _MAX_WORDS_CLARIFY):
        return {"route": "documents", "reply": "", "reason": f"clarify downgraded ({reason or 'guard'})"}
    if route == "direct" and (not reply or words > _MAX_WORDS_DIRECT):
        return {"route": "documents", "reply": "", "reason": f"direct downgraded ({reason or 'guard'})"}
    
    print(f"END OF __EXTRACT_ROUTE")
    
    return {"route": route, "reply": reply if route != "documents" else "", "reason": reason}


def _extract_text(content) -> str:
    """Normalize LLM content (str or list of blocks) into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)

def router_node(state: AgentState) -> AgentState:
    """
    Decide the route for this request. Rules first (free), LLM only for
    ambiguous messages, and on ANY failure default to "documents" - i.e.
    exactly the behaviour the agent had before the router existed.
    """
    
    print("--ENTERD ROUTER_NODE \n\n\n")
    
    task = state["task"]
    print(f"task = {task} \n")
    decision = _route_by_rules(task)
    print(f"Decision in router_node = {decision} \n")
    llm_used = False
    raw_content = None

    if decision is None:
        print(f"enter here ***\n")
        llm_used = True
        try:
            resp = _llm.invoke([
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": task},
            ])
            
            raw_content = _extract_text(resp.content)
            print(f"response content by llm in router_node = {resp.content} \n")
            decision = _extract_route(json.loads(_clean_json(raw_content)), task)
            print(f"Decision in router_node = {decision}")
            
            if decision is None:
                raise ValueError(f"router returned malformed output (raw: {raw_content!r})")
        except Exception as e:
            print(f"exception occured in router_node = {e}\n")
            decision = {"route": "documents", "reply": "", "reason": "router unavailable - default route"}
            state["errors"].append(f"router_node fallback used : {e}")

    state["route"] = decision["route"]
    state["route_reason"] = decision["reason"]
    
    if decision["route"] in ("direct", "clarify"):
        state["final_answer"] = decision["reply"]
        state["last_answer"] = state["final_answer"]
        _append_conversation_turn(state)

    state["memory"].append({
        "node": "router",
        "route": decision["route"],
        "reason": decision["reason"],
        "llm_used": llm_used,
        "raw_llm_response": raw_content,
    })
    print("--EXITED ROUTER_NODE \n\n\n")
    return state


def route_after_router(state: AgentState) -> str:
    """Pure router: only the documents route runs the plan/search pipeline."""
    print("---ENTERED  AND LEFT route_after_router NODE \n\n\n")
    return "plan" if state.get("route", "documents") == "documents" else "end"


def plan_node(state: AgentState)->AgentState:
    
    print(f"PLAN NODE START -- \n")
    
    """
    Ask the LLM to break the task into a short ordered list of concrete
    steps. One planning call up front is far cheaper than re-planning on
    every step, and is sufficient for this agent's two-tool workflow
    (retrieve from PDFs, optionally save the result to a file).
    """
    
    system = (
        "Break the user's task into 1-4 short, concrete steps for an agent "
        "with exactly two tools:\n"
        "  - rag_search: retrieves an answer from the user's uploaded PDFs\n"
        "  - write_output: saves text to a file\n\n"
        "Only include a write_output step if the user explicitly asked to "
        "save, export, or write the result to a file.\n"
        'Respond with ONLY a JSON object of the form {"steps": [...]}, '
        "where the value is a list of short step strings, nothing else. "
        'Example: {"steps": ["Search the PDFs for the refund policy", '
        '"Save the answer to refund_policy.md"]}.\n'
        "The user may refer to an earlier answer with words such as 'it', 'that', "
        "'this', or 'the previous answer'. Resolve those references using the "
        "conversation context. A request such as 'save that' should become a "
        "write_output step that uses the previous assistant answer."
    )
    
    raw_content = None
    try:
        task_lower = state["task"].lower()
        reference_words = re.search(
            r"\b(that|it|this|previous\s+answer|last\s+answer|same)\b",
            task_lower,
        )
        
        print(f"reference_words = {bool(reference_words)} \n")
        print(f"bool state get last_answer  = {bool(state.get("last_answer"))}")
        
        # The most important conversational follow-up can be handled
        # deterministically: "save/write/export that/it/this" refers to the
        # checkpointed previous answer, so there is no need to ask the planner
        # LLM to rediscover that reference.
        is_reference_write = (
            bool(state.get("last_answer"))
            and bool(reference_words)
            and any(kw in task_lower for kw in _WRITE_KEYWORDS)
        )
        
        if is_reference_write:
            plan = ["save the previous assistant answer to a file"]
            print("entered is_refrence block of plan node \n")
        else:
            
            user_content = (
                "Conversation so far:\n"
                f"{_conversation_context(state)}\n\n"
                "Previous answer available for follow-up requests:\n"
                f"{state.get('last_answer', '') or '(none)'}\n\n"
                f"Current request:\n{state['task']}"
            )
            
            # print(f"user_content = {user_content} \n\n")
        
            if state.get("critique_feedback"):
                print(f"Entered critique_feedback in PLAN NODE \n")
                # This is a retry after a reflection cycle judged the previous
                # answer insufficient - tell the planner what was wrong so it
                # doesn't just regenerate the same plan and fail the same way.
                user_content += (
                    f"\n\nNote: a previous attempt at this task was judged "
                    f"insufficient. Feedback: {state['critique_feedback']}\n"
                    f"Try a different or more thorough approach this time."
                )

            resp = _llm.invoke([
                {"role" : "system",  "content" : system},
                {"role" : "user" , "content" : user_content}
            ])
            raw_content = resp.content
            
            print(f"\nRAW_CONNTENT = {raw_content} ")
    
            parsed = json.loads(_clean_json(raw_content))
            print(f"PARSED_CONNTENT = {parsed} ")
            
            plan = _extract_plan_list(parsed)
            print(f"PLAN = {plan} \n")
    
            if not plan:
                raise ValueError(f"planner returned an empty or malformed plan (raw: {raw_content!r})")
    except Exception as e:
        print(f"\n ERROR OCCURED IN PLAN_NODE {e}\n\n")
        plan = ["search the pdfs to answer the task"]
        state['errors'].append(f"plan_node fallback used : {e}")
        
    state['plan'] = plan
    state['memory'].append({"node" : "plan" , "plan" : plan, "raw_llm_response": raw_content})
    if not state.get("file_path"):
        filename = _extract_filename(state["task"])
        print(f"FILE NAME = {filename} \n\n")
        if filename:
            state["file_path"] = f"outputs/{filename}"
            
   
    
    print(f"PLAN NODE END -- \n")
    return state
    
 
_WRITE_KEYWORDS = ("save", "write", "export", "store", "persist")


def _classify_search_failure(result: dict) -> dict | None:
    """
    Cheap, deterministic check (no LLM call) for whether a rag_search
    result is a failure or an empty retrieval worth replanning around.
    Returns None when the search actually produced usable content.
    """
    if not result.get("success"):
        return {"kind": "search_error", "detail": str(result.get("error", "unknown error"))}
    if result.get("total_found", 1) == 0:
        if "no documents" in (result.get("answer") or "").lower():
            return {"kind": "no_documents", "detail": "no documents have been uploaded"}
        return {"kind": "empty_results", "detail": "search returned no relevant chunks"}
    return None


def _usable_answers(state: AgentState) -> list:
    """Answers from retrievals that actually found something - a
    'no documents uploaded' / 'nothing relevant' reply is not content."""
    return [
        r["answer"] for r in state["rag_results"]
        if r.get("answer") and r.get("total_found", 1) > 0
    ]

def _extract_native_tool_calls(response) -> list:
    """Return normalized native tool calls from a LangChain AIMessage."""
    calls = getattr(response, "tool_calls", None) or []
    # print(f"calls = {calls} \n")
    if not calls:
        return []

    normalized = []
    for call in calls:
        name = str(call.get("name") or "").strip()
        args = call.get("args") or {}
        call_id = str(call.get("id") or "")
        if not name:
            continue
        if not isinstance(args, dict):
            raise ValueError(f"Tool arguments for {name!r} must be an object")
        normalized.append({"id": call_id, "name": name, "args": args})
    return normalized


def _safe_tool_args_for_memory(tool_name: str, args: dict) -> dict:
    """Keep tool traces useful without storing huge write payloads."""
    safe_args = dict(args)
    if tool_name == "write_output" and "content" in safe_args:
        content = str(safe_args["content"])
        safe_args["content_preview"] = content[:240]
        safe_args["content_chars"] = len(content)
        safe_args.pop("content", None)
    return safe_args


def _execute_native_tool_call(tool_call: dict) -> tuple[str, dict, dict]:
    """Validate and execute one model-selected tool call."""
    tool_name = tool_call["name"]
    print(f"tool_name = {tool_name}\n")
    tool = AGENT_TOOL_REGISTRY.get(tool_name)
    print(f"tool = {tool}\n")
    if tool is None:
        raise ValueError(f"Unknown agent tool requested by Groq: {tool_name}")

    args = tool_call["args"]
    # print(f"args = {args}\n")
    result = tool.invoke(args)
   
    if not isinstance(result, dict):
        result = {"success": True, "result": result}
    return tool_name, args, result


def execute_step_node(state: AgentState) -> AgentState:
    """
    Run exactly one plan step using native Groq function calling.

    The plan step is context for the model; Python no longer decides
    "search vs. write" by looking for words such as "save" or "write".
    Groq receives the actual tool schemas, chooses the tool, and produces
    structured arguments. The selected StructuredTool then validates and
    executes those arguments locally.

    After a rag_search call, the existing deterministic failure detection
    decides whether the graph should enter the replan path.
    """
    print("execute_step_node START \n")
    idx = state["current_step"]
    step = state["plan"][idx]
    state["needs_replan"] = False

    print(f"INDEX = {idx} \n")
    print(f"STEP = {step} \n")

    # Give the model the exact state it needs to select and parameterize a
    # tool, while keeping tool choice itself fully native/structured.
    content = (
        "You are executing exactly one step of a document agent.\n"
        "Choose exactly one tool from the available tools and call it with "
        "complete, valid arguments. Do not answer with prose.\n\n"
        f"Overall user task (context only):\n{state['task']}\n\n"
        f"CURRENT PLAN STEP — THIS IS THE ONLY ACTION TO EXECUTE:\n{step}\n\n"
        f"Current search query from recovery logic:\n{state.get('search_query') or '(none)'}\n\n"   
        f"Previous answer:\n{state.get('last_answer', '') or '(none)'}\n\n"
        f"Current answer:\n{state.get('final_answer', '') or '(none)'}\n\n"
        f"Output path hint from the application:\n{state.get('file_path') or '(none)'}\n\n"
        f"Retrieved answers so far:\n"
        f"{_usable_answers(state) or ['(none)']}\n\n"
        "Rules:\n"   
        "-PERFORM ONLY THE CURRENT PLAN STEP NOT THE OVERALL USER TASK\n"
        "- Execute ONLY the CURRENT PLAN STEP.\n"
        "- Do NOT execute a later or earlier plan step.\n"
        "- The overall user task is context only; it must NOT override the current plan step.\n"
        "- Use rag_search when this step requires retrieving information from uploaded documents.\n"
        "- Use write_output when this step requires saving/exporting/writing a result.\n"
        "- For rag_search, provide the best focused query and a reasonable top_k.\n"
        "- For write_output, use the requested/output filename when one is available. "
        "For content, use the current or previous substantive answer rather than inventing a new answer.\n"
    )
    
    # print(f"content = {content} \n")

    raw_response = None
    result = {"success": False, "error": "tool execution did not run"}
    failure = None
    tool_name = None
    tool_args = {}
    selected_call = None
    forced_retry = bool(state.get("search_query"))

    try:
        tool_llm = _retry_rag_llm if forced_retry else _tool_llm
        raw_response = tool_llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are the execution controller. Use native function calling only. "
                    "Never return a plain-text answer."
                ),
            },
            {"role": "user", "content": content},
        ])
        
        # print(f"raw_response = {raw_response} \n")

        tool_calls = _extract_native_tool_calls(raw_response)
        # print(f"tool_calls = {tool_calls} \n")
        if not tool_calls:
            raise ValueError(
                "Groq returned no native tool call for execute_step; "
                "the executor will not fall back to keyword routing"
            )
        if len(tool_calls) > 1:
            raise ValueError(
                f"Groq returned {len(tool_calls)} tool calls for one plan step; "
                "parallel tool execution is not enabled in this executor"
            )

        selected_call = tool_calls[0]
        tool_name, tool_args, result = _execute_native_tool_call(selected_call)
        
        # print(f"NATIVE TOOL CALL = {tool_name}({tool_args})")
        

    except Exception as e:
        result = {"success": False, "error": str(e)}
        state["errors"].append(f"native tool execution failed: {e}")
        print(f"native tool execution error = {e} \n")

    # Store both a compact human-readable call record and structured memory.
    if tool_name:
        if tool_name == "rag_search":
            query = str(tool_args.get("query", ""))
            state.setdefault("tried_queries", []).append(query)
            state["tool_calls"].append(f"rag_search({query[:60]!r})")
        elif tool_name == "write_output":
            path = str(tool_args.get("path", ""))
            state["tool_calls"].append(f"write_output({path})")

    failure = _classify_search_failure(result) if tool_name == "rag_search" else None

    if tool_name == "rag_search":
        if result.get("success"):
            state["rag_results"].append(result)
            if result.get("answer"):
                state["final_answer"] = result["answer"]
        else:
            state["errors"].append(result.get("error", "rag_search failed"))

        # The recovery query is a one-shot override for this execution. If
        # another search is needed, replan_node will install a fresh query.
        state["search_query"] = ""

    elif tool_name == "write_output":
        if result.get("success"):
            state["file_path"] = result["path"]
            state["last_file_path"] = result["path"]
            if not state.get("final_answer") and state.get("last_answer"):
                state["preserve_last_answer"] = True
                state["final_answer"] = f"Saved the previous answer to {result['path']}."
        else:
            state["errors"].append(result.get("error", "write_output failed"))

    state["memory"].append({
        "node": "execute_step",
        "step": step,
        "tool": tool_name,
        "tool_call_id": selected_call.get("id") if selected_call else None,
        "tool_args": _safe_tool_args_for_memory(tool_name or "", tool_args),
        "result": result,
        "native_function_call": bool(tool_name),
    })

    state["current_step"] += 1
    state["iteration"] += 1

    if failure:
        has_budget = (
            state.get("replan_count", 0) < state.get("max_replans", 2)
            and state["iteration"] < state["max_steps"]
        )
        if has_budget:
            state["needs_replan"] = True
            query_for_failure = str(tool_args.get("query", state.get("search_query", "")))
            state["last_failure"] = {
                "step": step,
                "query": query_for_failure,
                "kind": failure["kind"],
                "detail": failure["detail"],
            }

    print(f"failure = {failure} \n\n")
    
    print("execute_step_node END \n")
    return state

def should_continue(state:AgentState)->str:
    """Router: keep executing plan steps, or move to finalize."""
    if state["current_step"] >= len(state["plan"]):
        return "finalize"
    
    if state["iteration"] >= state["max_steps"]:
        # Loop guard - guarantees termination even if the plan is
        # pathologically long or execute_step somehow never finishes it.
        return "finalize"
    return "execute_step"


def route_after_execute(state:AgentState)->str:
    """Pure router: replan if execute_step_node flagged a failure,
    otherwise fall through to the normal continue/finalize decision."""
    
    print("\n ENTERED ROUTE_AFTER_EXEUTE \n\n\n")
    
    if state.get("needs_replan"):
        return "replan"
    return should_continue(state)


_REPLAN_SYSTEM = (
    "You are the recovery module of a document-search agent. A step just "
    "failed or came back empty. Decide what to do next.\n"
    "Respond with ONLY a JSON object of the form "
    '{"action": "retry"|"skip"|"give_up", "new_query": "<rewritten search '
    'query, only for retry>", "reason": "<one short sentence>"}.\n'
    "- retry: the search likely failed because of wording. Provide a "
    "genuinely different new_query (synonyms, key terms, a narrower or "
    "broader phrasing) - NOT one of the queries already tried.\n"
    "- skip: this step is not essential; continue with the rest of the plan.\n"
    "- give_up: the task cannot be completed with the available documents.\n"
    "Prefer retry with a rewritten query when the failure was an empty "
    "search and few queries have been tried."
)


def _extract_replan(parsed) -> dict | None:
    """Tolerant parse of the replan JSON (same qwen3 JSON-mode quirks as
    the plan/critique parsers)."""
    if not isinstance(parsed, dict):
        return None
    action = str(parsed.get("action", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if action not in ("retry", "skip", "give_up"):
        return None
    return {
        "action": action,
        "new_query": str(parsed.get("new_query") or "").strip(),
        "reason": str(parsed.get("reason") or "").strip(),
    }


def replan_node(state: AgentState) -> AgentState:
    """
    Recovery step after a failed/empty search. Decides retry (with a
    rewritten query), skip, or give up. All state mutation happens here;
    the routers stay pure.

    'No documents uploaded' is decided deterministically with no LLM
    call - no query rewrite can conjure a document, so asking the model
    would just burn latency to reach the same answer.
    """
    failure = state.get("last_failure") or {}
    kind = failure.get("kind", "unknown")
    state["needs_replan"] = False
    state["replan_count"] = state.get("replan_count", 0) + 1

    raw_content = None
    if kind == "no_documents":
        decision = {
            "action": "give_up",
            "new_query": "",
            "reason": "No documents have been uploaded yet.",
        }
    else:
        try:
            resp = _llm.invoke([
                {"role": "system", "content": _REPLAN_SYSTEM},
                {"role": "user", "content": (
                    f"Task: {state['task']}\n"
                    f"Plan: {state['plan']}\n"
                    f"Failed step: {failure.get('step')}\n"
                    f"Failure: {kind} - {failure.get('detail')}\n"
                    f"Queries already tried: {state.get('tried_queries', [])}"
                )},
            ])
            raw_content = resp.content
            decision = _extract_replan(json.loads(_clean_json(raw_content)))
            if decision is None:
                raise ValueError(f"replan returned malformed output (raw: {raw_content!r})")
        except Exception as e:
            # Fail safe: if recovery itself breaks, behave like the old
            # agent (carry on with the plan) rather than erroring out.
            decision = {"action": "skip", "new_query": "", "reason": "replan unavailable"}
            state["errors"].append(f"replan_node fallback used : {e}")

    # A "retry" is only meaningful with a genuinely new query.
    if decision["action"] == "retry":
        tried = {q.strip().lower() for q in state.get("tried_queries", [])}
        new_query = decision["new_query"]
        if not new_query or new_query.strip().lower() in tried:
            decision = {
                "action": "skip",
                "new_query": "",
                "reason": decision["reason"] or "no new query to try",
            }

    if decision["action"] == "retry":
        state["search_query"] = decision["new_query"]
        state["current_step"] = max(0, state["current_step"] - 1)  # redo the failed step
    elif decision["action"] == "skip":
        # The failed retrieval's "nothing found" text must not survive as
        # the answer (or get written to a file by a later step).
        if kind in ("no_documents", "empty_results"):
            state["final_answer"] = ""
    else:  # give_up
        state["current_step"] = len(state["plan"])  # end execution -> finalize
        state["unrecoverable"] = True
        if kind == "no_documents":
            state["final_answer"] = (
                "No documents are uploaded yet. Please upload a document "
                "first, then ask again."
            )
        else:
            state["final_answer"] = (
                "I couldn't find relevant information in the uploaded "
                f"documents for this task. ({decision['reason'] or failure.get('detail', '')})"
            )

    state["memory"].append({
        "node": "replan",
        "failure": failure,
        "decision": decision,
        "raw_llm_response": raw_content,
    })
    return state

def finalize_node(state:AgentState)->AgentState:
    """
    Compose the final answer from everything gathered. If retrieval never
    produced an answer, say so honestly instead of inventing one.
    """
    print("\n ENTERED FINALISE NODE \n\n\n")
    
    if not state["final_answer"]:
        usable = _usable_answers(state)
        if usable:
            state["final_answer"] = usable[-1]
        else:
            state["final_answer"] = (
                "I wasn't able to retrieve relevant information from the "
                "PDFs to answer this task."
            )
    if not state.get("preserve_last_answer"):
        state["last_answer"] = state["final_answer"]
        
    _append_conversation_turn(state)

    state["memory"].append({"node": "finalize"})
    
    print("\n EXITED  FINALISE NODE \n\n\n")
    
    return state


_CRITIQUE_SYSTEM = (
    "You are a strict reviewer checking whether an AI agent's answer "
    "actually satisfies the user's task.\n"
    "Respond with ONLY a JSON object of the form "
    '{"score": <1-5>, "sufficient": true|false, "feedback": "<short note on '
    'what is missing, if anything>"}.\n'
    "score 5 = fully answers the task with concrete detail from real content. "
    'score 1 = does not address the task at all (e.g. "I don\'t have '
    'information about that").\n'
    "Mark sufficient=false only if the answer is genuinely incomplete, "
    "vague, or evasive - not just because it could theoretically be more "
    "detailed. A short but accurate and complete answer is sufficient."
    "For an action-only task such as saving or exporting a previous answer, "
    "a clear confirmation that the action succeeded is sufficient."
)


def _extract_critique(parsed) -> dict | None:
    """Tolerant parse of the critique JSON - mirrors _extract_plan_list's
    defensiveness, since this hits the same qwen3 JSON-mode quirks."""
    if not isinstance(parsed, dict):
        return None
    score = parsed.get("score")
    if not isinstance(score, (int, float)):
        return None
    sufficient = parsed.get("sufficient")
    return {
        "score": int(score),
        "sufficient": bool(sufficient) if sufficient is not None else int(score) >= 4,
        "feedback": str(parsed.get("feedback") or ""),
    }


def critique_node(state: AgentState) -> AgentState:
    """
    Reflection step: ask the LLM to judge whether final_answer actually
    satisfies state["task"]. If not, and there's retry budget left, this
    node sets state up for another plan -> execute pass (with the
    critique's feedback folded into the next planning call) instead of
    silently returning a weak answer.

    All state mutation happens HERE, not in the router below - a routing
    function that mutates state is fragile with LangGraph and has bitten
    this project before (see should_continue's history). should_reflect
    only ever reads what this node already decided.
    """
    
    print(f"ENTERED critique_node \n\n\n\n")
    
    state.setdefault("critique_count", 0)
    state.setdefault("max_critiques", 2)

    if state.get("unrecoverable"):
        print("STATE IS unrecoverable ")
        # replan_node already concluded nothing more can be done (e.g. no
        # documents uploaded). Reflecting would only score the honest
        # apology poorly and burn retries on a problem no replan can fix.
        state["needs_retry"] = False
        state["memory"].append({
            "node": "critique",
            "skipped": "replan gave up - nothing further to try",
            "will_retry": False,
        })
        return state

    raw_content = None
    try:
        resp = _llm.invoke([
            {"role": "system", "content": _CRITIQUE_SYSTEM},
            {"role": "user", "content": (
                f"Task: {state['task']}\n\nAnswer given: {state['final_answer']}"
            )},
        ])
        raw_content = resp.content
        parsed = json.loads(_clean_json(raw_content))
        critique = _extract_critique(parsed)
        print(f"critique = {critique}")
        if critique is None:
            raise ValueError(f"critique returned malformed output (raw: {raw_content!r})")
    except Exception as e:
        # Fail safe: if the critique step itself breaks, don't block the
        # user from getting the answer they already have.
        critique = {"score": 5, "sufficient": True, "feedback": ""}
        state["errors"].append(f"critique_node fallback used : {e}")

    state["critique_score"] = critique["score"]
    state["critique_feedback"] = critique["feedback"]

    sufficient = critique["sufficient"] and critique["score"] >= 4
    retries_left = (
        state["critique_count"] < state["max_critiques"]
        and state["iteration"] < state["max_steps"]
    )
    should_retry = (not sufficient) and retries_left

    state["needs_retry"] = should_retry
    if should_retry:
        state["critique_count"] += 1
        state["current_step"] = 0
        # Each critique-driven attempt gets its own replan budget and
        # starts from the original query, not a previous attempt's rewrite
        # (tried_queries is kept so the replanner won't repeat old rewrites).
        state["replan_count"] = 0
        state["search_query"] = ""
        state["last_failure"] = {}
        state["needs_replan"] = False
        # Clear final_answer so execute_step_node's write-branch fallback
        # (final_answer or joined rag_results) can't silently reuse the
        # answer that was just judged insufficient if the retry's search
        # step comes back empty too.
        state["final_answer"] = ""

    state["memory"].append({
        "node": "critique",
        "critique": critique,
        "will_retry": should_retry,
        "raw_llm_response": raw_content,
    })
    
    print(f"end of CRITIQUE NODE \n\n\n\n\n")
    return state


def should_reflect(state: AgentState) -> str:
    """Pure router - reads the retry decision critique_node already made."""
    return "retry" if state.get("needs_retry") else "done"
    
        