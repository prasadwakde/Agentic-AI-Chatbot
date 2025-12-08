import os


#Setting up API Keys
from dotenv import load_dotenv
load_dotenv()
#print(OPENAI_API_KEY)
#Setting LLM & Tools
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

#openai_llm = ChatOpenAI(model = "gpt-4o-mini")

# search_tool = TavilySearch(
#     max_results=2,
# )

#AI Agent with search tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage



def response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm = ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearch(max_results = 2)] if allow_search else []


    agent = create_agent(
        model = llm,
        tools= tools,
        system_prompt= system_prompt
    )

    # query = "Tell me about passau university in germany"

    # inputs = {
    #     "messages": [
    #         {"role": "user", "content": query}
    #     ]
    # }

    state={"messages": query}

    response = agent.invoke(state)
    #print(response)
    last_msg = response["messages"][-1]
    answer = last_msg.content
    print(answer)
    return answer