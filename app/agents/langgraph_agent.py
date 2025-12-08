from typing import Literal


#Setting up API Keys
from dotenv import load_dotenv
load_dotenv()
#print(OPENAI_API_KEY)
# Setting LLM & Tools
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# AI Agent with search tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from app.core.config import settings

Provider = Literal["Groq", "OpenAI"]
DEFAULT_SYSTEM_PROMPT = "Act as an AI chatbot who is smart and friendly."

def get_llm(model_name: str, provider: Provider):
    """
    Return the correct LLM instance based on provider and model name.
    Uses API keys from env via settings.
    """
    if provider == "Groq":
        # ChatGroq reads GROQ_API_KEY from env
        return ChatGroq(model=model_name)

    if provider == "OpenAI":
        # ChatOpenAI reads OPENAI_API_KEY from env
        return ChatOpenAI(model=model_name)

    raise ValueError(f"Unsupported provider: {provider}")

# Response from Agent
def response_from_ai_agent(
    model_name: str,
    messages: list[str],
    allow_search: bool,
    system_prompt: str | None,
    provider: Provider,
) -> str:
    """
    Agent with optional Tavily aearch tool and returns final Agent Response Message
    """

    llm = get_llm(model_name, provider)

    tools = [TavilySearch(max_results = 2)] if allow_search else []

    prompt_to_use = system_prompt or DEFAULT_SYSTEM_PROMPT

    agent = create_agent(
        model = llm,
        tools= tools,
        system_prompt= prompt_to_use
    )

    state={"messages": messages}
    result = agent.invoke(state)
    #print(result)
    all_messages = result.get("messages", [])
    ai_messages = [m.content for m in all_messages if isinstance(m, AIMessage)]

    if not ai_messages:
        raise RuntimeError("Agent returned no AI messages")

    return ai_messages[-1]