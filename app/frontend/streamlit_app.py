from dotenv import load_dotenv

load_dotenv()

import re
import streamlit as st
import requests

LATEX_BLOCK = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

def render_answer(answer: str):
    answer = normalize_latex(answer)
    parts = LATEX_BLOCK.split(answer)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            st.latex(part.strip())

def normalize_latex(text: str) -> str:
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return text

st.set_page_config(page_title="Langagraph Agent UI", layout="centered")
st.title("AI Chatbot Agent")
st.write("Create and Interact with the AI Agent")

st.sidebar.title("📄 Your Documents")

uploaded = st.sidebar.file_uploader("Upload a TXT file", type=["txt"])

if uploaded:
    resp = requests.post(
        "http://localhost:9999/docs/upload",
        data={"user_id": "user123"},
        files={"file": uploaded}
    )
    st.sidebar.success(f"Uploaded! Chunked into {resp.json()['chunks']} pieces.")


SYSTEM_PROMPT_DEFAULT = """
You are an AI/ML Study Assistant.

CRITICAL RULE:
For AI/ML, DL, LLM, Transformers, LangChain, LangGraph, RAG questions:
- Always call rag_search first and keep [SOURCE: ...] tags.
Answer in Markdown and end with a Sources section.

MATH FORMATTING RULES (MANDATORY):
- Do NOT use Unicode math symbols like θ, ∇, η in plain text equations.
- Write ALL equations ONLY in LaTeX.
- Put every standalone equation inside a display block using $$ ... $$.
Example:
$$
\\theta_{\\text{new}} = \\theta_{\\text{old}} - \\eta \\cdot \\nabla_\\theta L(\\theta_{\\text{old}})
$$
""".strip()


system_prompt = st.text_area(
    "Define your AI Agent: ",
    height = 70,
    placeholder = "Type your system prompt here...",
    value=SYSTEM_PROMPT_DEFAULT,
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", {"Groq", "OpenAI"})

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("Allow Web Search")

def should_use_web_search(query: str) -> bool:
    q = query.lower()
    # basic rule to not do web search for personal queries
    personal_starters = ["should i", "do you think", "what should i do", "how do i deal with"]
    if any(q.startswith(p) for p in personal_starters):
        return False
    return True

user_query = st.text_area(
    "Enter your query: ",
    height=150,
    placeholder="Type your system prompt here..."
)

API_URL = "http://localhost:9999/chat"

if st.button("Ask Agent!"):
    if user_query.strip():

        allow_search_final = allow_web_search and should_use_web_search(user_query)

        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_search_final
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
        except requests.RequestException as e:
            st.error(f"Failed to reach backend: {e}")
        else:
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"Backend error: {detail}")
            else:
                data = resp.json()
                answer = data.get("answer")
                if answer is None:
                    st.error(f"Unexpected backend response: {data}")
                elif answer.strip() == "":
                    st.warning("Backend returned an empty answer. Showing raw response for debugging:")
                    st.code(data)
                else:
                    st.subheader("Agent Response")
                    #st.markdown(answer, unsafe_allow_html=False)
                    render_answer(answer)

    else:
        st.warning("Please enter a query first")