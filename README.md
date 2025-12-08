# Agentic-AI-Chatbot

Create pythion env
python -m venv AIAgent

Activate the env
for powershell
.\AIAgent\Scripts\Activate.ps1

Install the dependencies
pip install streamlit uvicorn langgraph langchain fastapi pydantic pytest

Backend run command
python -m app.api.main

Frontend run command
streamlit run app/frontend/streamlit_app.py

Tests run command
pytest tests/test_chat_api.py


# Current Progress Response
![Agent Response](AgentResponse.png)
