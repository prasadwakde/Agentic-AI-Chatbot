from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from app.api.schemas import RequestState
from app.agents.langgraph_agent import response_from_ai_agent
from app.api.routes_docs import router as docs_router
from app.rag.rag_service import bootstrap_global_kb

# Free models
ALLOWED_MODEL_NAMES=["llama3-70b-8192", "llama-3.3-70b-versatile", "gpt-4o-mini"]

app = FastAPI(title="LangGraph AI Agent")
app.include_router(docs_router)

@app.on_event("startup")
def load_knowledge_base():
    project_root = Path(__file__).resolve().parents[2]
    kb_path = project_root / "knowledge_base"
    bootstrap_global_kb(str(kb_path))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """

    if request.model_name not in ALLOWED_MODEL_NAMES:
            raise HTTPException(
                status_code=400,
                detail="Invalid model name. Kindly select a valid AI model.",
        )

    try:
         
         response_text = response_from_ai_agent(
            model_name = request.model_name,
            messages = request.messages,
            allow_search = request.allow_search,
            system_prompt = request.system_prompt,
            provider = request.model_provider,
         )

    except Exception as exc:
        msg = str(exc)
        if "tool_use_failed" in msg or "tavily" in msg.lower():
            # Fallback by retrying without web search
            try:
                response_text = response_from_ai_agent(
                    model_name=request.model_name,
                    messages=request.messages,
                    allow_search=False,
                    system_prompt=request.system_prompt,
                    provider=request.model_provider,
                )
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Web search failed and fallback also failed.",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error in AI agent: {msg}",
            )
        
    #response = response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)

    return {"answer": response_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="127.0.0.1", port=9999, reload=True)
