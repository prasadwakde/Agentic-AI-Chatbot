
---

# Agentic AI Chatbot (FastAPI, LangGraph, Groq/OpenAI, Streamlit)

A modular, production-ready **Agentic AI Chatbot** built using **LangGraph**, **FastAPI**, **Groq/OpenAI LLMs**, and **Streamlit**.
This project is currently at **Stage 1** of development, featuring a clean architecture, API routing, configurable LLM provider selection, and a fully functional frontend UI.

Future stages will add RAG, memory, multi-tool agents, streaming responses, and deployment workflows.

---

## Current Progress

![Agent Response](AgentResponse.png)

---


## Features (Current – Stage 1)

### ✅ Core Features

* Agentic chatbot powered by **LangGraph's ReAct agent**
* Supports both **Groq** (LLaMA) and **OpenAI** models
* Optional **web search tool** (Tavily)
* Configurable **system prompts**, **model selection**, and **search toggles**
* Clean separation between **backend**, **frontend**, **agent**, and **config**

### ✅ Backend (FastAPI)

* `/chat` endpoint with structured request validation using **Pydantic schemas**
* Centralized configuration with **Pydantic Settings** (`.env`)
* Provider-agnostic LLM selection (Groq / OpenAI)
* Error handling with descriptive messages
* Ready for extensions like RAG, memory, user profiles, etc.

### ✅ Frontend (Streamlit)

* Chat UI with:

  * Model selector
  * Provider selector
  * System prompt input
  * Web search toggle
  * Query box + response display
* Calls backend API via JSON request

### Modular folder structure:

```
app/
  core/        → config & settings
  agents/      → LangGraph agent implementations
  api/         → FastAPI routers, schemas
  frontend/    → Streamlit UI
tests/         → pytest test suite
```

* Added **unit tests** for API health & validation
* Environment-isolated execution using `.env`

---

## Installation & Setup

### 1️⃣ Create Python virtual environment

```bash
python -m venv AIAgent
```

### 2️⃣ Activate environment

**PowerShell:**

```bash
.\AIAgent\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies

```bash
pip install streamlit uvicorn langgraph langchain fastapi pydantic pytest
```

---

## ▶️ Running the Project

### Start Backend (FastAPI)

```bash
python -m app.api.main
```

### Start Frontend (Streamlit)

```bash
streamlit run app/frontend/streamlit_app.py
```

### Run Tests

```bash
pytest tests/test_chat_api.py
```

---

## API Overview (Stage 1)

### **POST /chat**

**Request:**

```json
{
  "model_name": "llama-3.3-70b-versatile",
  "model_provider": "Groq",
  "system_prompt": "You are a helpful assistant.",
  "messages": ["What is overfitting?"],
  "allow_search": false
}
```

**Response:**

```json
{
  "answer": "Overfitting occurs when a model..."
}
```

## Upcoming Milestones

### **Stage 2 — RAG (Retrieval-Augmented Generation)**

* Document ingestion
* Embeddings & vector DB (Chroma)
* RAG search tool inside LangGraph
* “AI/ML Study Assistant” local knowledge base

### **Stage 3 — Memory + Multi-Tool Agents**

* Persistent conversation memory
* Additional tools (todo management, code tools)

### **Stage 4 — Streaming + Auth + Logging**

### **Stage 5 — Frontend Enhancements**

* Chat history
* Knowledge base UI
* Settings panel

### **Stage 6 — Deployment (Docker + Cloud)**

---
