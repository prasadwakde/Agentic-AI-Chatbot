from typing import List, Literal

from pydantic import BaseModel


Provider = Literal["Groq", "OpenAI"]

# Schema for Request
class RequestState(BaseModel):
    model_name: str
    model_provider: Provider
    system_prompt: str
    messages: List[str]
    allow_search: bool
