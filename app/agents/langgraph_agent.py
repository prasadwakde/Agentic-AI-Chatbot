from typing import Literal

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_tavily import TavilySearch
from app.rag.rag_service import rag_search_tool

Provider = Literal["Groq", "OpenAI"]

DEFAULT_SYSTEM_PROMPT = """
You are an AI/ML Study Assistant.

CRITICAL RULES:
1) For any question related to machine learning, deep learning, LLMs, transformers,
   LangChain, LangGraph, RAG, or optimization:
   - You MUST call `rag_search` first.
2) Use web_search ONLY for recent or version-specific information.
3) If you do not find relevant info in rag_search, say so.

RESPONSE FORMAT (MANDATORY):
- Use Markdown.
- Always include a "Sources" section.
- Keep [SOURCE: ...] tags exactly as provided by rag_search.

MATH FORMATTING RULES (MANDATORY):
- Do NOT use Unicode math symbols like θ, ∇, η in plain text equations.
- Write ALL equations ONLY in LaTeX.
- Put every standalone equation inside a display block using $$ ... $$.
Example:
$$
\\theta_{\\text{new}} = \\theta_{\\text{old}} - \\eta \\cdot \\nabla_\\theta L(\\theta_{\\text{old}})
$$
""".strip()


def get_llm(model_name: str, provider: Provider):
    """
    Return the correct LLM instance based on provider and model name
    Uses API keys from env via settings
    """
    if provider == "Groq":
        return ChatGroq(model=model_name)
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name)
    raise ValueError(f"Unsupported provider: {provider}")


@tool("rag_search")
def rag_search(query: str) -> str:
    """
    MANDATORY: Use this for AI/ML + GenAI questions from the local knowledge base
    text is returned with [SOURCE: ...] blocks
    """
    #user_id=None means "global KB"
    return rag_search_tool(query=query, user_id=None, k=4)


def response_from_ai_agent(
    model_name: str,
    messages: list[str],
    allow_search: bool,
    system_prompt: str | None,
    provider: Provider,
) -> str:
    llm = get_llm(model_name, provider)

    tools = [rag_search]

    # if allow_search:
    #     tools.append(TavilySearch(max_results=2))

    if allow_search:
        tools.append(TavilySearchResults(max_results=2))

    prompt_to_use = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        #state_modifier=prompt_to_use,
    )

    chat_messages = [SystemMessage(content=prompt_to_use)]
    chat_messages += [HumanMessage(content=m) for m in messages if m and m.strip()]

    # human_messages = [HumanMessage(content=m) for m in messages if m and m.strip()]
    # if not human_messages:
    #     return "Please ask a non-empty question."

    result = agent.invoke({"messages": chat_messages})
    all_messages = result.get("messages", [])

    #Return last non-empty AI response
    for m in reversed(all_messages):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            return m.content

    #show last tool output
    for m in reversed(all_messages):
        if isinstance(m, ToolMessage) and (m.content or "").strip():
            return m.content

    print("----- GRAPH MESSAGES -----")
    for m in all_messages:
        print(type(m).__name__, getattr(m, "content", None))
    print("--------------------------")

    return ""